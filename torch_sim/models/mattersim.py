"""TorchSim wrapper for MatterSim models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.state import SimState, StateDict
from torch_sim.units import MetalUnits


try:
    import torch
    from mattersim.forcefield.potential import batch_to_dict
    from torch_geometric.data import Data
    from torch_geometric.loader.dataloader import Collater

except ImportError:

    class MatterSimModel(torch.nn.Module, ModelInterface):
        """MatterSim model wrapper for torch_sim.

        This class is a placeholder for the MatterSimModel class.
        It raises an ImportError if sevenn is not installed.
        """

        def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
            """Dummy constructor to raise ImportError."""
            raise ImportError("sevenn must be installed to use this model.")


if TYPE_CHECKING:
    from collections.abc import Callable

    from mattersim.forcefield import Potential


def _mattersim_vesin_nl_ts(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrapper for MatterSim vesin_nl_ts function to add self-edges."""
    edge_idx, shifts_idx = vesin_nl_ts(
        positions=positions,
        cell=cell,
        pbc=pbc,
        cutoff=cutoff,
    )

    self_idx = torch.arange(len(positions), device=positions.device)
    self_edge_idx = self_idx.unsqueeze(0).repeat(2, 1)
    self_shifts_idx = torch.zeros((len(positions), 3))

    edge_idx = torch.cat((self_edge_idx, edge_idx), dim=1)
    shifts_idx = torch.cat((self_shifts_idx, shifts_idx), dim=0)

    return edge_idx, shifts_idx


def _compute_threebody(
    bond_atom_indices: torch.Tensor, n_atoms: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core computation of three-body topology based on bond indices."""
    device = bond_atom_indices.device
    n_bond = bond_atom_indices.shape[0]
    n_struct = n_atoms.shape[0]
    n_atom = torch.sum(n_atoms).item()

    # Initialize arrays exactly like Cython
    n_bond_per_atom = torch.zeros(n_atom, dtype=torch.long, device=device)
    n_triple_i = torch.zeros(n_atom, dtype=torch.long, device=device)
    n_triple_ij = torch.zeros(n_bond, dtype=torch.long, device=device)
    n_triple_s = torch.zeros(n_struct, dtype=torch.long, device=device)

    # Count bonds per atom
    for i in range(n_bond):
        n_bond_per_atom[bond_atom_indices[i, 0]] += 1

    # Calculate n_triple_i and n_triple_ij
    n_triple = 0
    start = 0
    for i in range(n_atom):
        n_triple_temp = n_bond_per_atom[i] * (n_bond_per_atom[i] - 1)
        for j in range(n_bond_per_atom[i]):
            n_triple_ij[start + j] = n_bond_per_atom[i] - 1
        n_triple += n_triple_temp
        n_triple_i[i] = n_triple_temp
        start += n_bond_per_atom[i]

    # Generate triple bond indices
    triple_bond_indices = torch.zeros((n_triple, 2), dtype=torch.long, device=device)
    start = 0
    index = 0
    for i in range(n_atom):
        for j in range(n_bond_per_atom[i]):
            for k in range(n_bond_per_atom[i]):
                if j != k:
                    triple_bond_indices[index, 0] = start + j
                    triple_bond_indices[index, 1] = start + k
                    index += 1
        start += n_bond_per_atom[i]

    # Calculate n_triple_s
    start = 0
    end = start
    for i in range(n_struct):
        end += n_atoms[i]
        for j in range(start, end):
            n_triple_s[i] += n_triple_i[j]
        start = end

    return triple_bond_indices, n_triple_ij, n_triple_i, n_triple_s


def compute_threebody_indices(
    bond_atom_indices: torch.Tensor,
    bond_length: torch.Tensor,
    n_atoms: torch.Tensor,
    atomic_number: torch.Tensor,
    threebody_cutoff: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes three-body topology, optionally filtering bonds by a cutoff first.

    Args:
        bond_atom_indices (torch.Tensor): Shape (n_bond, 2). Atom indices for bonds.
        bond_length (torch.Tensor): Shape (n_bond). Length of each bond.
        n_atoms (torch.Tensor): Shape (n_struct,). Number of atoms in each structure.
        atomic_number (torch.Tensor): Shape (n_atom_total,). Atomic numbers.
        threebody_cutoff (Optional[float]): Cutoff distance to filter bonds before
                                             calculating three-body topology.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - bond_indices: Shape (n_triple, 2). Pairs of ORIGINAL bond indices.
            - n_triple_ij: Shape (n_bond,). Angles per ORIGINAL bond index.
            - n_triple_i: Shape (n_atom_total,). Angles per atom.
            - n_triple_s: Shape (n_struct,). Angles per structure.
    """
    if (
        not isinstance(bond_atom_indices, torch.Tensor)
        or not isinstance(bond_length, torch.Tensor)
        or not isinstance(n_atoms, torch.Tensor)
        or not isinstance(atomic_number, torch.Tensor)
    ):
        raise TypeError("All array/list inputs must be torch.Tensors")

    device = bond_atom_indices.device
    bond_atom_indices = bond_atom_indices.long()
    bond_length = bond_length.to(device)
    n_atoms = n_atoms.long().to(device)
    atomic_number = atomic_number.long().to(device)
    n_atom_total = torch.sum(n_atoms).item()

    if atomic_number.shape[0] != n_atom_total:
        raise ValueError(
            f"Shape of atomic_number ({atomic_number.shape[0]}) does not match "
            f"total atoms from n_atoms ({n_atom_total})"
        )
    if bond_atom_indices.shape[0] != bond_length.shape[0]:
        raise ValueError(
            "bond_atom_indices and bond_length must have the same first dimension. "
            f"Got {bond_atom_indices.shape=} and {bond_length.shape=}."
        )

    n_bond = bond_atom_indices.shape[0]
    original_indices = torch.arange(n_bond, device=device)

    apply_filter = n_bond > 0 and threebody_cutoff is not None
    if apply_filter:
        valid_three_body_mask = bond_length <= threebody_cutoff
        original_indices_filtered = original_indices[valid_three_body_mask]
        bond_atom_indices_core = bond_atom_indices[valid_three_body_mask]
    else:
        original_indices_filtered = original_indices
        bond_atom_indices_core = bond_atom_indices

    filtered_list_size = bond_atom_indices_core.shape[0]

    if filtered_list_size > 0:
        bond_indices_relative, n_triple_ij_relative, n_triple_i, n_triple_s = (
            _compute_threebody(bond_atom_indices_core, n_atoms)
        )

        if apply_filter:
            n_triple_ij = torch.zeros(n_bond, dtype=torch.long, device=device)
            n_triple_ij[original_indices_filtered] = n_triple_ij_relative

            bond_indices = original_indices_filtered[bond_indices_relative]
        else:
            n_triple_ij = n_triple_ij_relative
            bond_indices = bond_indices_relative

    else:
        bond_indices = torch.empty((0, 2), dtype=torch.long, device=device)
        n_triple_ij = torch.zeros(n_bond, dtype=torch.long, device=device)
        n_triple_i = torch.zeros(n_atom_total, dtype=torch.long, device=device)
        n_triple_s = torch.zeros(n_atoms.shape[0], dtype=torch.long, device=device)

    return bond_indices, n_triple_ij, n_triple_i, n_triple_s


class MatterSimModel(torch.nn.Module, ModelInterface):
    """Computes atomistic energies, forces and stresses using an MatterSim model.

    This class wraps an MatterSim model to compute energies, forces, and stresses for
    atomistic systems. It handles model initialization, configuration, and
    provides a forward pass that accepts a SimState object and returns model
    predictions.

    Examples:
        >>> model = MatterSimModel(model=loaded_matersim_model)
        >>> results = model(state)
    """

    def __init__(
        self,
        model: Potential,
        *,  # force remaining arguments to be keyword-only
        neighbor_list_fn: Callable = _mattersim_vesin_nl_ts,
        stress_weight: float = MetalUnits.pressure * 1e4,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the MatterSimModel with specified configuration.

        Loads an MatterSim model from either a model object or a model path.
        Sets up the model parameters for subsequent use in energy and force calculations.

        Args:
            model (Potential): The MatterSim model to wrap.
            neighbor_list_fn (Callable): Neighbor list function to use. The
                implementation must match that of from
                `pymatgen.optimization.neighbors.find_points_in_spheres` in order
                to work with the pretrained model weights. This function adds a
                self-edge to the neighbor list to ensure that the model can handle
                systems with a single atom.
            stress_weight (float): Stress weight to use to scale the stress units.
                Defaults to value of ase.units.GPa to match MatterSimCalculator default.
            device (torch.device | str | None): Device to run the model on
            dtype (torch.dtype | None): Data type for computation
        """
        super().__init__()

        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if isinstance(self._device, str):
            self._device = torch.device(self._device)

        self._dtype = dtype or torch.float32
        self._memory_scales_with = "n_atoms"  # scale memory with n_atoms due to triplets
        self._compute_stress = True
        self._compute_forces = True

        self.neighbor_list_fn = neighbor_list_fn
        self.stress_weight = stress_weight

        self.model = model
        self.model.eval()
        self.model.to(self.device)
        if self._dtype is not None:
            self.model = self.model.to(dtype=self._dtype)

        model_args = self.model.model.model_args
        self.two_body_cutoff = torch.tensor(model_args["cutoff"])
        self.three_body_cutoff = torch.tensor(model_args["threebody_cutoff"])

        self.implemented_properties = [
            "energy",
            "forces",
            "stress",
        ]

    def forward(self, state: SimState | StateDict) -> dict[str, torch.Tensor]:
        """Perform forward pass to compute energies, forces, and other properties.

        Takes a simulation state and computes the properties implemented by the model,
        such as energy, forces, and stresses.

        Args:
            state (SimState | StateDict): State object containing positions, cells,
                atomic numbers, and other system information. If a dictionary is provided,
                it will be converted to a SimState.

        Returns:
            dict: Dictionary of model predictions, which may include:
                - energy (torch.Tensor): Energy with shape [batch_size]
                - forces (torch.Tensor): Forces with shape [n_atoms, 3]
                - stress (torch.Tensor): Stress tensor with shape [batch_size, 3, 3],
                    if compute_stress is True

        Notes:
            The state is automatically transferred to the model's device if needed.
            All output tensors are detached from the computation graph.
        """
        if isinstance(state, dict):
            state = SimState(**state, masses=torch.ones_like(state["positions"]))

        if state.device != self._device:
            state = state.to(self._device)

        data_list = []
        for b in range(state.batch.max().item() + 1):
            batch_mask = state.batch == b

            pos = state.positions[batch_mask]
            cell = state.cell[b]
            pbc = state.pbc
            atomic_number = state.atomic_numbers[batch_mask]

            edge_idx, shifts_idx = self.neighbor_list_fn(
                positions=pos,
                cell=cell,
                pbc=pbc,
                cutoff=self.two_body_cutoff,
            )

            shifts = torch.mm(shifts_idx, cell)

            edge_vec = pos[edge_idx[0]] - pos[edge_idx[1]] - shifts
            distances = torch.norm(edge_vec, dim=-1)

            (
                triple_bond_index,
                n_triple_ij,
                _n_triple_i,
                _n_triple_s,
            ) = compute_threebody_indices(
                bond_atom_indices=torch.transpose(edge_idx, 1, 0),
                bond_length=distances,
                n_atoms=torch.tensor([pos.shape[0]]),
                atomic_number=atomic_number,
                threebody_cutoff=self.three_body_cutoff,
            )

            data = {
                "num_atoms": torch.tensor(len(atomic_number)),
                "num_bonds": torch.tensor(len(edge_idx[0])),
                "atom_attr": atomic_number.unsqueeze(-1),  # [n_atoms, 1] expected
                "atom_pos": pos,
                "edge_index": edge_idx,
                "pbc_offsets": shifts_idx,
                "cell": cell.unsqueeze(0),
                "three_body_indices": triple_bond_index,
                "num_three_body": triple_bond_index.shape[0],
                "num_triple_ij": n_triple_ij.unsqueeze(-1),
            }

            data = Data(**data)
            data_list.append(data)

        batched_data = Collater([], follow_batch=None, exclude_keys=None)(data_list)
        output = self.model.forward(
            batch_to_dict(batched_data, device=self.device),
            include_forces=self.compute_forces,
            include_stresses=self.compute_stress,
        )

        results = {}
        results["energy"] = output["total_energy"].detach()
        results["forces"] = output["forces"].detach()
        results["stress"] = self.stress_weight * output["stresses"].detach()

        return results
