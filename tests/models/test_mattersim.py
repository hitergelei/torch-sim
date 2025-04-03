# codespell-ignore: convertor

import ase.units
import numpy as np
import pytest
import torch
from ase.build import bulk

from torch_sim.io import atoms_to_state, state_to_atoms
from torch_sim.models.interface import validate_model_outputs
from torch_sim.state import SimState


try:
    from mattersim.datasets.utils.convertor import (
        _compute_threebody as _compute_threebody_cython,
    )
    from mattersim.datasets.utils.convertor import (
        compute_threebody_indices as compute_threebody_indices_numpy,
    )
    from mattersim.forcefield import MatterSimCalculator, Potential
    from pymatgen.optimization.neighbors import find_points_in_spheres

    from torch_sim.models.mattersim import MatterSimModel, _mattersim_vesin_nl_ts
    from torch_sim.models.mattersim import _compute_threebody as _compute_threebody_torch
    from torch_sim.models.mattersim import (
        compute_threebody_indices as compute_threebody_indices_torch,
    )

except ImportError:
    pytest.skip("mattersim not installed", allow_module_level=True)


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture to provide the default dtype for testing."""
    return torch.float32


@pytest.fixture
def model_name() -> str:
    """Fixture to provide the model name for testing. Load smaller 1M model
    for testing purposes.
    """
    return "mattersim-v1.0.0-1m.pth"


@pytest.fixture
def cu_system(dtype: torch.dtype, device: torch.device) -> SimState:
    # Create FCC Copper
    cu_fcc = bulk("Cu", "fcc", a=3.58, cubic=True)
    return atoms_to_state([cu_fcc], device, dtype)


@pytest.fixture
def pretrained_mattersim_model(device: torch.device, model_name: str):
    """Load a pretrained MatterSim model for testing."""
    return Potential.from_checkpoint(
        load_path=model_name,
        model_name="m3gnet",
        device=device,
        load_training_state=False,
    )


@pytest.fixture
def mattersim_model(
    pretrained_mattersim_model: torch.nn.Module, device: torch.device
) -> MatterSimModel:
    """Create an MatterSimModel wrapper for the pretrained model."""
    return MatterSimModel(
        model=pretrained_mattersim_model,
        device=device,
    )


@pytest.fixture
def mattersim_calculator(
    pretrained_mattersim_model: Potential, device: torch.device
) -> MatterSimCalculator:
    """Create an MatterSimCalculator for the pretrained model."""
    return MatterSimCalculator(pretrained_mattersim_model, device=device)


def get_sorted_indices(array: np.ndarray) -> np.ndarray:
    """Get sorted indices of a 2D NumPy array lexicographically."""
    return np.lexsort(array.T[::-1])


def assert_outputs_equal(
    torch_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    np_out: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
):
    """Asserts equality between torch and numpy outputs after conversion."""
    torch_b_idx, torch_n_ij, torch_n_i, torch_n_s = torch_out
    np_b_idx, np_n_ij, np_n_i, np_n_s = np_out

    # Convert torch tensors to numpy arrays (int64)
    torch_b_idx_np = torch_b_idx.cpu().numpy().astype(np.int64)
    torch_n_ij_np = torch_n_ij.cpu().numpy().astype(np.int64)
    torch_n_i_np = torch_n_i.cpu().numpy().astype(np.int64)
    torch_n_s_np = torch_n_s.cpu().numpy().astype(np.int64)

    # Ensure numpy outputs are int64 for direct comparison
    np_b_idx = np_b_idx.astype(np.int64)
    np_n_ij = np_n_ij.astype(np.int64)
    np_n_i = np_n_i.astype(np.int64)
    np_n_s = np_n_s.astype(np.int64)

    # Compare bond_indices (order doesn't matter, sort rows first)
    torch_b_idx_np_indices = get_sorted_indices(torch_b_idx_np)
    np_b_idx_indices = get_sorted_indices(np_b_idx)
    np.testing.assert_array_equal(
        torch_b_idx_np[torch_b_idx_np_indices], np_b_idx[np_b_idx_indices]
    )

    # Compare counts (order matters)
    np.testing.assert_array_equal(torch_n_ij_np, np_n_ij)
    np.testing.assert_array_equal(torch_n_i_np, np_n_i)
    np.testing.assert_array_equal(torch_n_s_np, np_n_s)


def test_compare_neighbor_list_implementations(
    cu_system: SimState,
    device: torch.device,
) -> None:
    # Get system data
    batch_mask = cu_system.batch == 0
    pos = cu_system.positions[batch_mask]
    cell = cu_system.cell[0]

    lattice_matrix = np.ascontiguousarray(cell.cpu().numpy(), dtype=float)
    cart_coords = np.ascontiguousarray(pos.cpu().numpy(), dtype=float)

    cutoff = 3.0  # Example cutoff

    # 1. Compare neighbor lists
    # PyMatGen implementation
    center_indices, neighbor_indices, images, distances = find_points_in_spheres(
        cart_coords,
        cart_coords,
        r=cutoff,
        pbc=np.ones(3, int),
        lattice=lattice_matrix,
        tol=1e-8,
    )

    # torch_sim implementation
    edge_idx, shifts_idx = _mattersim_vesin_nl_ts(
        positions=pos,
        cell=cell,
        pbc=cu_system.pbc,
        cutoff=torch.tensor(cutoff, device=device),
    )

    # Convert torch outputs to numpy for comparison
    edge_idx_np = edge_idx.cpu().numpy().T
    shifts_idx_np = shifts_idx.cpu().numpy()

    # Sort both outputs for comparison
    pymatgen_pairs = np.stack([center_indices, neighbor_indices], axis=1)
    pymatgen_pairs_indices = get_sorted_indices(pymatgen_pairs)
    torch_pairs_indices = get_sorted_indices(edge_idx_np)

    # Compare neighbor lists
    np.testing.assert_array_almost_equal(
        pymatgen_pairs[pymatgen_pairs_indices], edge_idx_np[torch_pairs_indices]
    )
    np.testing.assert_array_almost_equal(
        images[pymatgen_pairs_indices], shifts_idx_np[torch_pairs_indices]
    )


def test_compare_threebody_implementations(
    cu_system: SimState,
    device: torch.device,
) -> None:
    """Compare torch and Cython implementations of three-body computation directly."""
    # Get system data
    batch_mask = cu_system.batch == 0
    pos = cu_system.positions[batch_mask]
    cell = cu_system.cell[0]

    # Get neighbor lists first
    edge_idx, _shifts_idx = _mattersim_vesin_nl_ts(
        positions=pos,
        cell=cell,
        pbc=cu_system.pbc,
        cutoff=torch.tensor(3.0, device=device),
    )

    # Prepare inputs for both implementations
    bond_atom_indices = torch.transpose(edge_idx, 1, 0)
    n_atoms = torch.tensor([pos.shape[0]], device=device)

    # Convert to numpy for Cython implementation
    bond_atom_indices_np = bond_atom_indices.cpu().numpy().astype(np.int32)
    n_atoms_np = n_atoms.cpu().numpy().astype(np.int32)

    # Get torch implementation results
    torch_out = _compute_threebody_torch(bond_atom_indices, n_atoms)

    # Get Cython implementation results
    cython_out = _compute_threebody_cython(
        np.ascontiguousarray(bond_atom_indices_np, dtype="int32"),
        np.array(n_atoms_np, dtype="int32"),
    )

    # Compare outputs
    assert_outputs_equal(torch_out, cython_out)


def test_compare_threebody_indicies_implementations(
    cu_system: SimState,
    device: torch.device,
) -> None:
    """Compare torch and Cython implementations of three-body computation directly."""
    # Get system data
    batch_mask = cu_system.batch == 0
    pos = cu_system.positions[batch_mask]
    cell = cu_system.cell[0]

    two_body_cutoff = 4.0
    three_body_cutoff = 3.0

    # Get neighbor lists first
    edge_idx, shifts_idx = _mattersim_vesin_nl_ts(
        positions=pos,
        cell=cell,
        pbc=cu_system.pbc,
        cutoff=torch.tensor(two_body_cutoff, device=device),
    )

    # Calculate distances for torch implementation
    shifts = torch.mm(shifts_idx, cu_system.cell[0])
    edge_vec = (
        cu_system.positions[edge_idx[0]] - cu_system.positions[edge_idx[1]] - shifts
    )
    distances_torch = torch.norm(edge_vec, dim=-1)

    # Torch implementation
    torch_out = compute_threebody_indices_torch(
        bond_atom_indices=torch.transpose(edge_idx, 1, 0),
        bond_length=distances_torch,
        n_atoms=torch.tensor([pos.shape[0]]),
        atomic_number=cu_system.atomic_numbers[: pos.shape[0]],
        threebody_cutoff=three_body_cutoff,
    )

    # NumPy implementation
    numpy_out = compute_threebody_indices_numpy(
        bond_atom_indices=edge_idx.cpu().numpy().T.astype(np.int32),
        bond_length=distances_torch.cpu().numpy().astype(np.float32),
        n_atoms=np.array([pos.shape[0]], dtype=np.int32),
        atomic_number=cu_system.atomic_numbers[: pos.shape[0]]
        .cpu()
        .numpy()
        .astype(np.int32),
        threebody_cutoff=three_body_cutoff,
    )

    # Compare outputs
    assert_outputs_equal(torch_out, numpy_out)


def test_mattersim_initialization(
    pretrained_mattersim_model: torch.nn.Module, device: torch.device
) -> None:
    """Test that the MatterSim model initializes correctly."""
    model = MatterSimModel(
        model=pretrained_mattersim_model,
        device=device,
    )
    assert model.neighbor_list_fn == _mattersim_vesin_nl_ts
    assert model._device == device  # noqa: SLF001
    assert model.stress_weight == ase.units.GPa


def test_mattersim_calculator_consistency(
    mattersim_model: MatterSimModel,
    mattersim_calculator: MatterSimCalculator,
    cu_system: SimState,
    device: torch.device,
) -> None:
    """Test consistency between MatterSimModel and MatterSimCalculator."""
    # Set up ASE calculator
    cu_fcc = state_to_atoms(cu_system)[0]
    cu_fcc.calc = mattersim_calculator

    # Get MatterSimModel results
    mattersim_results = mattersim_model(cu_system)

    # Get calculator results
    calc_energy = cu_fcc.get_potential_energy()
    calc_forces = torch.tensor(
        cu_fcc.get_forces(),
        device=device,
        dtype=mattersim_results["forces"].dtype,
    )

    # Test consistency with reasonable tolerances
    torch.testing.assert_close(
        mattersim_results["energy"].item(),
        calc_energy,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        mattersim_results["forces"],
        calc_forces,
        rtol=1e-5,
        atol=1e-5,
    )


def test_validate_model_outputs(
    mattersim_model: MatterSimModel, device: torch.device
) -> None:
    """Test that the model passes the standard validation."""
    validate_model_outputs(mattersim_model, device, torch.float32)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
