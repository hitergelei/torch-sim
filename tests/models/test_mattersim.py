# codespell-ignore: convertor

import ase.units
import numpy as np
import pytest
import torch
from ase.build import bulk

from torch_sim.io import atoms_to_state, state_to_atoms
from torch_sim.models.interface import validate_model_outputs
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.state import SimState


try:
    from mattersim.datasets.utils.convertor import (
        compute_threebody_indices as compute_threebody_indices_numpy,
    )
    from mattersim.forcefield import MatterSimCalculator, Potential

    from torch_sim.models.mattersim import MatterSimModel
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


@pytest.fixture(scope="module")
def example_data_torch() -> dict[str, torch.Tensor]:
    """Provides the example data as torch tensors."""
    return {
        "n_atoms": torch.tensor([5]),
        "atomic_number": torch.tensor([1, 1, 8, 6, 6]),  # Example
        "bond_atom_indices": torch.tensor(
            [
                [0, 1],  # Bond 0
                [0, 2],  # Bond 1
                [1, 0],  # Bond 2
                [1, 2],  # Bond 3
                [3, 4],  # Bond 4
                [4, 3],  # Bond 5
            ]
        ),
        "bond_length": torch.tensor([1.0, 1.5, 1.0, 0.9, 2.1, 2.1], dtype=torch.float32),
    }


@pytest.fixture(scope="module")
def example_data_numpy(
    example_data_torch: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    """Provides the example data as numpy arrays."""
    data_np = {}
    for key, tensor in example_data_torch.items():
        data_np[key] = tensor.numpy()
    # Ensure specific dtypes expected by NumPy version if necessary
    data_np["n_atoms"] = data_np["n_atoms"].astype(np.int32)
    data_np["atomic_number"] = data_np["atomic_number"].astype(np.int32)
    data_np["bond_atom_indices"] = data_np["bond_atom_indices"].astype(np.int32)
    return data_np


def sort_rows(array: np.ndarray) -> np.ndarray:
    """Sorts rows of a 2D NumPy array lexicographically."""
    if array.shape[0] == 0:
        return array
    # Use lexsort for stable row sorting
    return array[np.lexsort(array.T[::-1])]


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
    np.testing.assert_array_equal(sort_rows(torch_b_idx_np), sort_rows(np_b_idx))

    # Compare counts (order matters)
    np.testing.assert_array_equal(torch_n_ij_np, np_n_ij)
    np.testing.assert_array_equal(torch_n_i_np, np_n_i)
    np.testing.assert_array_equal(torch_n_s_np, np_n_s)


def test_torch_example_no_cutoff(example_data_torch: dict[str, torch.Tensor]):
    """Test torch implementation against expected values without cutoff."""
    b_idx, n_ij, n_i, n_s = compute_threebody_indices_torch(**example_data_torch)

    expected_b_idx = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]])
    expected_n_ij = torch.tensor([1, 1, 1, 1, 0, 0])
    expected_n_i = torch.tensor([2, 2, 0, 0, 0])
    expected_n_s = torch.tensor([4])

    # Sort bond indices for comparison
    b_idx_np_sorted = sort_rows(b_idx.cpu().numpy())
    expected_b_idx_np_sorted = sort_rows(expected_b_idx.cpu().numpy())

    np.testing.assert_array_equal(b_idx_np_sorted, expected_b_idx_np_sorted)
    assert torch.equal(n_ij.cpu(), expected_n_ij.cpu())
    assert torch.equal(n_i.cpu(), expected_n_i.cpu())
    assert torch.equal(n_s.cpu(), expected_n_s.cpu())


def test_compare_no_cutoff(
    example_data_torch: dict[str, torch.Tensor],
    example_data_numpy: dict[str, np.ndarray],
):
    """Compare torch and numpy implementations without cutoff."""
    torch_out = compute_threebody_indices_torch(**example_data_torch)
    np_out = compute_threebody_indices_numpy(**example_data_numpy)
    assert_outputs_equal(torch_out, np_out)


def test_compare_with_cutoff(
    example_data_torch: dict[str, torch.Tensor],
    example_data_numpy: dict[str, np.ndarray],
):
    """Compare torch and numpy implementations with cutoff."""
    cutoff = 1.6
    torch_out = compute_threebody_indices_torch(
        **example_data_torch, threebody_cutoff=cutoff
    )
    np_out = compute_threebody_indices_numpy(
        **example_data_numpy, threebody_cutoff=cutoff
    )
    assert_outputs_equal(torch_out, np_out)


def test_edge_no_bonds(
    example_data_torch: dict[str, torch.Tensor],
    example_data_numpy: dict[str, np.ndarray],
):
    """Test edge case with zero bonds."""
    n_atoms_t = example_data_torch["n_atoms"]
    atomic_num_t = example_data_torch["atomic_number"]
    n_atoms_np = example_data_numpy["n_atoms"]
    atomic_num_np = example_data_numpy["atomic_number"]

    # Torch
    bonds_t = torch.empty((0, 2), dtype=torch.long)
    lengths_t = torch.empty((0,), dtype=torch.float32)
    torch_out = compute_threebody_indices_torch(
        bonds_t, lengths_t, n_atoms_t, atomic_num_t
    )

    # NumPy
    bonds_np = np.empty((0, 2), dtype=np.int32)
    lengths_np = np.empty((0,), dtype=np.float32)
    np_out = compute_threebody_indices_numpy(
        bonds_np, lengths_np, n_atoms_np, atomic_num_np
    )

    assert_outputs_equal(torch_out, np_out)
    # Also check shapes explicitly
    assert torch_out[0].shape == (0, 2)
    assert torch_out[1].shape == (0,)
    assert torch_out[2].shape == (atomic_num_t.shape[0],)
    assert torch_out[3].shape == (n_atoms_t.shape[0],)
    assert torch.all(torch_out[2] == 0)
    assert torch.all(torch_out[3] == 0)


def test_edge_no_angles(
    example_data_torch: dict[str, torch.Tensor],
    example_data_numpy: dict[str, np.ndarray],
):
    """Test edge case with bonds but no central atom having > 1 bond."""
    # Linear chain 0-1, 1-2, 3-4 (no atom is center of >1 bond)
    bonds_t = torch.tensor([[0, 1], [1, 2], [3, 4]])
    lengths_t = torch.tensor([1.0, 1.0, 1.0])
    n_atoms_t = example_data_torch["n_atoms"]
    atomic_num_t = example_data_torch["atomic_number"]

    bonds_np = bonds_t.numpy().astype(np.int32)
    lengths_np = lengths_t.numpy()
    n_atoms_np = example_data_numpy["n_atoms"]
    atomic_num_np = example_data_numpy["atomic_number"]

    torch_out = compute_threebody_indices_torch(
        bonds_t, lengths_t, n_atoms_t, atomic_num_t
    )
    np_out = compute_threebody_indices_numpy(
        bonds_np, lengths_np, n_atoms_np, atomic_num_np
    )

    assert_outputs_equal(torch_out, np_out)
    # Check explicit values for no angles
    assert torch_out[0].shape == (0, 2)  # No triple indices
    assert torch.all(torch_out[1] == 0)  # n_ij all zero
    assert torch.all(torch_out[2] == 0)  # n_i all zero
    assert torch.all(torch_out[3] == 0)  # n_s all zero


def test_edge_cutoff_filters_all(
    example_data_torch: dict[str, torch.Tensor],
    example_data_numpy: dict[str, np.ndarray],
):
    """Test edge case where cutoff filters all bonds."""
    cutoff = 0.5  # Smaller than any bond length in example
    torch_out = compute_threebody_indices_torch(
        **example_data_torch, threebody_cutoff=cutoff
    )
    np_out = compute_threebody_indices_numpy(
        **example_data_numpy, threebody_cutoff=cutoff
    )

    assert_outputs_equal(torch_out, np_out)
    # Check explicit values for no bonds after filtering
    n_bond_orig = example_data_torch["bond_atom_indices"].shape[0]
    n_atom_total = example_data_torch["atomic_number"].shape[0]
    n_struct = example_data_torch["n_atoms"].shape[0]

    assert torch_out[0].shape == (0, 2)  # No triple indices
    assert torch_out[1].shape == (n_bond_orig,)
    assert torch.all(torch_out[1] == 0)  # n_ij all zero
    assert torch_out[2].shape == (n_atom_total,)
    assert torch.all(torch_out[2] == 0)  # n_i all zero
    assert torch_out[3].shape == (n_struct,)
    assert torch.all(torch_out[3] == 0)  # n_s all zero


def test_mattersim_initialization(
    pretrained_mattersim_model: torch.nn.Module, device: torch.device
) -> None:
    """Test that the MatterSim model initializes correctly."""
    model = MatterSimModel(
        model=pretrained_mattersim_model,
        device=device,
    )
    assert model.neighbor_list_fn == vesin_nl_ts
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
