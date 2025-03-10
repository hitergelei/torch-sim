# %%
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

from torchsim.integrators import nvt_langevin
from torchsim.models.mace import MaceModel
from torchsim.optimizers import unit_cell_fire
from torchsim.runners import atoms_to_state
from torchsim.autobatching import HotswappingAutoBatcher, ChunkingAutoBatcher, split_state
from torchsim.units import MetalUnits
from torchsim.state import concatenate_states, BaseState


si_atoms = bulk("Si", "fcc", a=5.43, cubic=True).repeat((3, 3, 3))
fe_atoms = bulk("Fe", "fcc", a=5.43, cubic=True).repeat((3, 3, 3))

device = torch.device("cuda")

mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    periodic=True,
    dtype=torch.float64,
    compute_force=True,
)

si_state = atoms_to_state(si_atoms, device=device, dtype=torch.float64)
fe_state = atoms_to_state(fe_atoms, device=device, dtype=torch.float64)

fire_init, fire_update = unit_cell_fire(mace_model)

si_fire_state = fire_init(si_state)
fe_fire_state = fire_init(fe_state)

fire_states = [si_fire_state, fe_fire_state] * 100
fire_states = [state.clone() for state in fire_states]
for state in fire_states:
    state.positions += torch.randn_like(state.positions) * 0.01


# %%

batcher = HotswappingAutoBatcher(
    model=mace_model,
    states=fire_states,
    metric="n_atoms_x_density",
    # max_metric=400_000,
    max_metric=100_000,
)

def convergence_fn(state: BaseState) -> bool:
    batch_wise_max_force = torch.zeros(state.n_batches, device=state.device)
    max_forces = state.forces.norm(dim=1)
    batch_wise_max_force = batch_wise_max_force.scatter_reduce(
        dim=0,
        index=state.batch,
        src=max_forces,
        reduce="amax",
    )
    return batch_wise_max_force < 1e-1


next_batch = batcher.first_batch()

# %%
all_completed_states = []
while True:
    state = concatenate_states(next_batch)

    print("Starting new batch.")
    # run 10 steps, arbitrary number
    for i in range(10):
        state = fire_update(state)

    convergence_tensor = convergence_fn(state)

    next_batch, completed_states = batcher.next_batch(state, convergence_tensor)

    print("number of completed states", len(completed_states))

    if not next_batch:
        print("No more batches to run.")
        break

    all_completed_states.extend(completed_states)


# %%

nvt_init, nvt_update = nvt_langevin(model=mace_model, dt=0.001, kT=300 * MetalUnits.temperature)


si_state = atoms_to_state(si_atoms, device=device, dtype=torch.float64)
fe_state = atoms_to_state(fe_atoms, device=device, dtype=torch.float64)

si_nvt_state = nvt_init(si_state)
fe_nvt_state = nvt_init(fe_state)

nvt_states = [si_nvt_state, fe_nvt_state] * 100
nvt_states = [state.clone() for state in nvt_states]
for state in nvt_states:
    state.positions += torch.randn_like(state.positions) * 0.01


batcher = ChunkingAutoBatcher(
    model=mace_model,
    states=nvt_states,
    metric="n_atoms_x_density",
    max_metric=100_000,
)

finished_states = []
for batch in batcher:
    print(f"Starting new batch of size {len(batch)}")
    full_state = concatenate_states(batch)
    for _ in range(100):

        full_state = nvt_update(full_state)

    finished_states.extend(split_state(full_state))

# %%
len(finished_states)
