# %%
import copy
from typing import Literal

import numpy as np
import torch
import torch.profiler
from ase.build import bulk
from propfolio.utils import composition_to_random_structure
from pymatgen.core import Composition

from torchsim.models.soft_sphere import SoftSphereModel
from torchsim.optimizers import unit_cell_fire
from torchsim.runners import atoms_to_state, optimize
from torchsim.state import BaseState


def pack_soft_sphere(
    comp: Composition,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    sigma: float = 2.5,
    scale_volume: float = 1.0,
) -> BaseState:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    struct = composition_to_random_structure(comp, scale_volume=scale_volume)

    ss_model = SoftSphereModel(
        sigma=sigma,
        device=device,
        dtype=dtype,
        use_neighbor_list=True,
        compute_stress=True,
    )
    return optimize(
        system=struct,
        model=ss_model,
        optimizer=unit_cell_fire,
    )


def load_fairchem_or_mace(
    model_path: str,
    model_type: Literal["mace", "fairchem"],
    device: torch.device,
    dtype: torch.dtype,
    compute_stress: bool | None = None,
):  # TODO: replace with generic model
    if model_type == "mace":
        from torchsim.models.mace import MaceModel

        model = MaceModel(
            model=torch.load(model_path, map_location=device),
            device=device,
            periodic=True,
            compute_force=True,
            dtype=dtype,
            enable_cueq=True,
            compute_stress=compute_stress if compute_stress is not None else False,
        )
    elif model_type == "fairchem":
        from torchsim.models.fairchem import FairChemModel

        model = FairChemModel(
            model=model_path,
            dtype=dtype,
            compute_stress=compute_stress if compute_stress is not None else True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


# def profile_model(model, input_tensors: dict[str, torch.Tensor]):

#     with torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CUDA],
#         profile_memory=True,
#         record_shapes=True,
#     ) as prof:
#         with torch.no_grad():
#             model(**input_tensors)

#     print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))


def measure_model_memory_forward(model, input_tensors: dict[str, torch.Tensor]):
    torch.cuda.reset_peak_memory_stats()

    import time

    start = time.perf_counter()
    with torch.no_grad():
        model(**input_tensors)
    end = time.perf_counter()

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to MB
    return {"peak_memory": peak_memory, "time": end - start}


def measure_model_memory_optimize(model, state: BaseState):
    def converge_forces(state) -> bool:
        return torch.all(state.forces < 2e-1)

    optimize_state = optimize(
        system=state,
        model=model,
        optimizer=unit_cell_fire,
        convergence_fn=converge_forces,
        cell_factor=10000,
    )
    torch.cuda.reset_peak_memory_stats()

    import time

    start = time.perf_counter()
    model(
        positions=optimize_state.positions,
        cell=optimize_state.cell,
        batch=optimize_state.batch,
        atomic_numbers=optimize_state.atomic_numbers,
    )
    end = time.perf_counter()

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to MB
    return {"peak_memory": peak_memory, "time": end - start}


# %%
# load model
mace_model_path = "/lambdafs/assets/mace_checkpoints/2024-12-03-mace-mp-alex-0.model"
radsim_model_path = (
    "/lambdafs/assets/radsim_checkpoints/radsim-s-v4/FT-BMG-GN-S-OMat-noquad-cutoff6.pt"
)

# model = load_fairchem_or_mace(
#     model_path=mace_model_path,
#     model_type="mace",
#     device=torch.device("cuda"),
#     dtype=torch.float64,
#     compute_stress=True,
# )


# %%
# gc gpu memory
torch.cuda.empty_cache()


# %%
profile_model(
    model,
    {
        "positions": state.positions,
        "cell": state.cell,
        "batch": state.batch,
        "atomic_numbers": state.atomic_numbers,
    },
)


# %%
import gc


gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


atoms = bulk("Al", "fcc", a=4.05).repeat((5, 5, 5))
state = atoms_to_state([atoms] * 60, model.device, model.dtype)

# model.forward(
#     positions=state.positions,
#     cell=state.cell,
#     batch=state.batch,
#     atomic_numbers=state.atomic_numbers,
# )
print("state.n_atoms", state.n_atoms)

stats = measure_model_memory(
    model,
    {
        "positions": state.positions,
        "cell": state.cell,
        "batch": state.batch,
        "atomic_numbers": state.atomic_numbers,
    },
)
print(stats)
del state


# %%
# generate arbitrary compositions in Al Fe Mg space
compositions = [
    Composition("Al10Fe10Mg10"),
    Composition("Al30"),
    Composition("Fe30"),
    Composition("Mg30"),
    Composition("Al15Fe15"),
    Composition("Al15Mg15"),
    Composition("Fe15Mg15"),
]
stats_records = []
for comp in compositions:
    state = pack_soft_sphere(comp * 50, model.device, model.dtype)
    # number density
    n_atoms = state.n_atoms
    volume = (state.cell[0, 0, 0] ** 3 / 1000).item()
    number_density = n_atoms / volume
    stats = measure_model_memory_optimize(
        model,
        state,
        # {
        #     "positions": state.positions,
        #     "cell": state.cell,
        #     "batch": state.batch,
        #     "atomic_numbers": state.atomic_numbers,
        # },
    )
    stats["number_density"] = number_density
    stats["composition"] = str(comp.formula)
    stats_records.append(stats)


# %%
import pandas as pd


df = pd.DataFrame.from_records(stats_records)
df

# use plotly express to plot number density vs peak memory
import plotly.express as px


# make plot origin 0,0
fig = px.scatter(df, x="number_density", y="peak_memory", hover_data=["composition"])
fig.update_xaxes(range=[0, 100])
fig.update_yaxes(range=[0, 10])
fig.show()


# %%
# visualize the number density of the periodic table of
# elements with pymatviz

# ... existing code ...


def measure_model_memory_forward(model, input_tensors: dict[str, torch.Tensor]):
    # Clear GPU memory
    import gc
    import time

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    model(**input_tensors)
    end = time.perf_counter()

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    return {"peak_memory": peak_memory, "time": end - start}


def generate_test_sizes(start: int = 100, max_size: int = 100000, factor: float = 1.4):
    """Generate a sequence of test sizes with exponential growth.

    Args:
        start: Starting number of atoms
        max_size: Maximum number of atoms to test
        factor: Multiplicative factor between sizes

    Yields:
        int: Next test size
    """
    current = start
    while current <= max_size:
        yield current
        current = int(current * factor)


def create_bulk_system(
    element: str,
    structure_type: str,
    repeat_size: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> BaseState:
    """Create a bulk crystal system with the specified parameters.

    Args:
        element: Chemical element symbol
        structure_type: Crystal structure type (fcc, bcc, etc.)
        repeat_size: Tuple of repeat counts along each axis
        device: Torch device
        dtype: Data type for tensors

    Returns:
        BaseState: System state
    """
    # Create bulk structure with ASE
    atoms = bulk(element, structure_type).repeat(repeat_size)

    # Convert to state
    return atoms_to_state([atoms], device, dtype)


def calculate_number_density(state: BaseState) -> float:
    """Calculate number density in atoms/nm³ for a state.

    Args:
        state: System state

    Returns:
        float: Number density in atoms/nm³
    """
    n_atoms = state.n_atoms
    # Calculate volume in nm³ (convert from Å³)
    volume = torch.abs(torch.linalg.det(state.cell[0])) / 1000
    return n_atoms / volume.item()


def test_model_memory_limit(
    model,
    element: str = "Al",
    structure_type: str = "fcc",
    device: torch.device = None,
    dtype: torch.dtype = torch.float64,
    start_size: int = 100,
    max_size: int = 100000,
    size_factor: float = 1.4,
    density_factors: list[float] = None,
    max_memory_gb: float = None,
    safety_factor: float = 0.9,
):
    """Test model memory limits using bulk systems of increasing size.

    Args:
        model: The model to test
        element: Element to use for bulk structure
        structure_type: Crystal structure type
        device: Torch device
        dtype: Data type for tensors
        start_size: Initial number of atoms to test
        max_size: Maximum number of atoms to test
        size_factor: Growth factor between test sizes
        density_factors: List of factors to scale the lattice constant
        max_memory_gb: Maximum GPU memory in GB (defaults to 90% of available)
        safety_factor: Factor to reduce max_memory by

    Returns:
        dict: Analysis results
    """
    # Set defaults
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if density_factors is None:
        density_factors = [0.7, 0.85, 1.0, 1.15, 1.3]

    # Prepare to store results
    results = []

    stop_loop = False
    # Test with different system sizes
    # for n_atoms_target in generate_test_sizes(start_size, max_size, size_factor):
    for repeat_dim in range(5, 30):
        # Determine repeat size to get close to target atom count
        # First get atoms per unit cell
        atoms = bulk(element, structure_type)

        # Calculate repeat dimensions
        # repeat_dim = int(round((n_atoms_target / len(atoms)) ** (1 / 3)))
        repeat_size = (repeat_dim, repeat_dim, repeat_dim)
        atoms = atoms.repeat(repeat_size)
        atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1

        print(f"Testing system with {len(atoms)} atoms")

        # Test each density factor
        for density_factor in density_factors:
            # Create bulk system
            # For density variation, we'll scale the lattice constant
            # Smaller lattice constant = higher density
            atoms = copy.deepcopy(atoms)
            atoms.set_cell(atoms.get_cell() / density_factor)

            # Convert to state
            state = atoms_to_state([atoms], device, dtype)

            # Get actual number of atoms and density
            number_density = calculate_number_density(state)

            try:
                # Measure memory during forward pass
                memory_stats = measure_model_memory_forward(
                    model,
                    {
                        "positions": state.positions,
                        "cell": state.cell,
                        "batch": state.batch,
                        "atomic_numbers": state.atomic_numbers,
                    },
                )

                # Add to results
                results.append(
                    {
                        "n_atoms": len(atoms),
                        "number_density": number_density,
                        "peak_memory_gb": memory_stats["peak_memory"],
                        "time": memory_stats["time"],
                        "element": element,
                        "structure": structure_type,
                        "density_factor": density_factor,
                        "success": True,
                    }
                )

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(
                        f"Out of memory at {len(atoms)} atoms, density factor {density_factor}"
                    )
                    results.append(
                        {
                            "n_atoms": len(atoms),
                            "number_density": number_density,
                            "peak_memory_gb": float("nan"),
                            "time": float("nan"),
                            "element": element,
                            "structure": structure_type,
                            "density_factor": density_factor,
                            "success": False,
                        }
                    )
                    # Break inner loop and try smaller atom count
                    stop_loop = True
                    break
                # Re-raise if it's not an OOM error
                raise

        if stop_loop:
            break

    return results


def analyze_memory_results(results):
    """Analyze memory usage results and fit a predictive model.

    Args:
        results: List of result dictionaries
        max_memory_gb: Maximum GPU memory in GB

    Returns:
        dict: Analysis results including interaction term between atoms and density
    """
    import numpy as np
    import pandas as pd

    # Convert results to DataFrame
    df = pd.DataFrame.from_records(results)

    # Only proceed with model fitting if we have enough successful runs
    successful_runs = df[df["success"]]
    if len(successful_runs) > 3:
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures

            max_memory_gb = df["peak_memory_gb"].max()

            # Create features including the interaction term
            # We want: memory ~ a*n_atoms + b*n_atoms*density + c
            X_base = successful_runs[["n_atoms", "number_density"]].values

            # Create interaction term manually
            successful_runs["interaction"] = (
                successful_runs["n_atoms"] * successful_runs["number_density"]
            )
            X = successful_runs[["n_atoms", "interaction"]].values
            y = successful_runs["peak_memory_gb"].values

            # Fit linear model with interaction term
            model = LinearRegression()
            model.fit(X, y)

            # Calculate estimated maximum atoms at each density
            max_atoms_est = {}
            for density in sorted(df["number_density"].unique()):
                if np.isnan(density):
                    continue

                # Solve for n_atoms where:
                # model.coef_[0] * n_atoms + model.coef_[1] * (n_atoms * density) + model.intercept_ = max_memory_gb
                # Rearranging: n_atoms * (model.coef_[0] + model.coef_[1] * density) = max_memory_gb - model.intercept_

                denominator = model.coef_[0] + model.coef_[1] * density
                if denominator > 0:  # Avoid division by zero or negative values
                    max_atoms = (max_memory_gb - model.intercept_) / denominator
                    max_atoms_est[density] = max(0, int(max_atoms))
                else:
                    max_atoms_est[density] = 0

            # Create formula string
            formula = (
                f"Memory (GB) = {model.coef_[0]:.2e} * n_atoms + "
                f"{model.coef_[1]:.2e} * (n_atoms * density) + "
                f"{model.intercept_:.2f}"
            )

            # define a function that calculates the peak memory from the formula
            def peak_memory_from_formula(n_atoms, density):
                return (
                    model.coef_[0] * n_atoms
                    + model.coef_[1] * (n_atoms * density)
                    + model.intercept_
                )

            return {
                "fitted_model": model,
                "max_atoms_estimated": max_atoms_est,
                "coef_n_atoms": model.coef_[0],
                "coef_interaction": model.coef_[1],
                "intercept": model.intercept_,
                "formula": formula,
                "peak_memory_from_formula": peak_memory_from_formula,
                "max_memory_used": max_memory_gb,
            }
        except ImportError:
            return {"error": "sklearn not available for model fitting"}

    return {"error": "Not enough successful runs to fit model"}


# %%
# Load your model
model = load_fairchem_or_mace(
    model_path=radsim_model_path,
    model_type="fairchem",
    device=torch.device("cuda"),
    dtype=torch.float64,
    compute_stress=False,
)


# %%
# Test memory limits
results = test_model_memory_limit(
    model=model,
    element="Al",
    structure_type="fcc",
    device=torch.device("cuda"),
    dtype=torch.float64,
    start_size=100,
    max_size=50000,
    density_factors=[0.8, 1.0, 1.2],
)


# %%
fit_results = analyze_memory_results(results["dataframe"], 10)

all_results = {**results, **fit_results}

# Print memory scaling formula
# if "formula" in results:
#     print(results["formula"])

#     # Print estimated maximum atoms at different densities
#     print("\nEstimated maximum atoms before OOM:")
#     for density, max_atoms in results["max_atoms_estimated"].items():
#         print(f"  At density {density:.1f}: {max_atoms:,} atoms")

# # Visualize results
# fig1, fig2 = visualize_memory_results(results)
# if fig1:
#     fig1.show()
# if fig2:
#     fig2.show()
fit_results


# %%
fit_results


# %%
df = results["dataframe"]

df["number_density_x_n_atoms"] = df["number_density"] * df["n_atoms"]

# plot peak_memory vs n_atoms
import plotly.express as px


fig = px.scatter(
    df,
    x="number_density_x_n_atoms",
    y="peak_memory_gb",
    title="MACE Memory Usage vs Number Density x Number of Atoms",
)
# width 800
fig.update_layout(width=600)
fig.show()


# %%
fig = px.scatter(df, x="number_density", y="peak_memory_gb")
fig.show()


# %%
fig = px.scatter(
    df,
    x="n_atoms",
    y="peak_memory_gb",
    title="Fairchem Memory Usage vs Number of Atoms",
)

fig.update_layout(width=600, xaxis_range=[0, 5000])
fig.show()


# %%
df


# %%
results
