"""Utilities for batching and memory management in torchsim."""

from collections.abc import Iterator
from itertools import chain
from typing import Literal

import binpacking
import torch
from ase.build import bulk

from torchsim.models.interface import ModelInterface
from torchsim.runners import atoms_to_state
from torchsim.state import BaseState, concatenate_states, slice_substate


def measure_model_memory_forward(model: ModelInterface, state: BaseState) -> float:
    """Measure peak GPU memory usage during model forward pass.

    Args:
        model: The model to measure memory usage for.
        state: The input state to pass to the model.

    Returns:
        Peak memory usage in GB.
    """
    # Clear GPU memory

    # gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()

    model(
        positions=state.positions,
        cell=state.cell,
        batch=state.batch,
        atomic_numbers=state.atomic_numbers,
    )

    return torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB


def determine_max_batch_size(
    model: ModelInterface, state: BaseState, max_atoms: int = 20000
) -> int:
    """Determine maximum batch size that fits in GPU memory.

    Args:
        model: The model to test with.
        state: The base state to replicate.
        max_atoms: Maximum number of atoms to try.

    Returns:
        Maximum number of batches that fit in GPU memory.
    """
    # create a list of integers following the fibonacci sequence
    fib = [1, 2]
    while fib[-1] < max_atoms:
        fib.append(fib[-1] + fib[-2])

    for i in range(len(fib)):
        n_batches = fib[i]
        concat_state = concatenate_states([state] * n_batches)

        try:
            measure_model_memory_forward(model, concat_state)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                return fib[i - 2]
            raise

    return fib[-2]


def split_state(state: BaseState) -> list[BaseState]:
    """Split a state into a list of states, each containing a single batch element."""
    # TODO: make this more efficient
    return [slice_substate(state, i) for i in range(state.n_batches)]


def calculate_baseline_memory(model: ModelInterface) -> float:
    """Calculate baseline memory usage of the model.

    Args:
        model: The model to measure baseline memory for.

    Returns:
        Baseline memory usage in GB.
    """
    # Create baseline atoms with different sizes
    baseline_atoms = [bulk("Al", "fcc").repeat((i, 1, 1)) for i in range(1, 9, 2)]
    baseline_states = [
        atoms_to_state(atoms, model.device, model.dtype) for atoms in baseline_atoms
    ]

    # Measure memory usage for each state
    memory_list = [
        measure_model_memory_forward(model, state) for state in baseline_states
    ]

    # Calculate number of atoms in each baseline state
    n_atoms_list = [state.n_atoms for state in baseline_states]

    # Convert to tensors
    n_atoms_tensor = torch.tensor(n_atoms_list, dtype=torch.float)
    memory_tensor = torch.tensor(memory_list, dtype=torch.float)

    # Prepare design matrix (with column of ones for intercept)
    X = torch.stack([torch.ones_like(n_atoms_tensor), n_atoms_tensor], dim=1)

    # Solve normal equations
    beta = torch.linalg.lstsq(X, memory_tensor.unsqueeze(1)).solution.squeeze()

    # Extract intercept (b) and slope (m)
    intercept, _ = beta[0].item(), beta[1].item()

    return intercept


def calculate_scaling_metric(
    state_slice: BaseState,
    metric: Literal["n_atoms_x_density", "n_atoms"] = "n_atoms_x_density",
) -> float:
    """Calculate scaling metric for a state.

    Args:
        state_slice: The state to calculate metric for.
        metric: The type of metric to calculate.

    Returns:
        The calculated metric value.
    """
    if metric == "n_atoms":
        return state_slice.n_atoms
    if metric == "n_atoms_x_density":
        volume = torch.abs(torch.linalg.det(state_slice.cell[0])) / 1000
        number_density = state_slice.n_atoms / volume.item()
        return state_slice.n_atoms * number_density
    raise ValueError(f"Invalid metric: {metric}")


def estimate_max_metric(
    model: ModelInterface,
    state_list: list[BaseState],
    metric_values: list[float],
    max_atoms: int = 20000,
) -> float:
    """Estimate maximum metric value that fits in GPU memory.

    Args:
        model: The model to test with.
        state_list: List of states to test.
        metric_values: Corresponding metric values for each state.
        max_atoms: Maximum number of atoms to try.

    Returns:
        Maximum metric value that fits in GPU memory.
    """
    # all_metrics = torch.tensor(
    #     [calculate_scaling_metric(state_slice, metric) for state_slice in state_list]
    # )

    # select one state with the min n_atoms
    min_metric = metric_values.min()
    max_metric = metric_values.max()

    min_state = state_list[metric_values.argmin()]
    max_state = state_list[metric_values.argmax()]

    min_state_max_batches = determine_max_batch_size(model, min_state, max_atoms)
    max_state_max_batches = determine_max_batch_size(model, max_state, max_atoms)

    return min(min_state_max_batches * min_metric, max_state_max_batches * max_metric)


class ChunkingAutoBatcher:
    """Batcher that chunks states into bins of similar computational cost."""

    def __init__(
        self,
        model: ModelInterface,
        states: list[BaseState] | BaseState,
        metric: Literal["n_atoms", "n_atoms_x_density"] = "n_atoms_x_density",
        max_metric: float | None = None,
        max_atoms_to_try: int = 1_000_000,
    ) -> None:
        """Initialize the batcher.

        Args:
            model: The model to batch for.
            states: States to batch.
            metric: Metric to use for batching.
            max_metric: Maximum metric value per batch.
            max_atoms_to_try: Maximum number of atoms to try when estimating max_metric.
        """
        self.state_slices = (
            split_state(states) if isinstance(states, BaseState) else states
        )
        self.metrics = [
            calculate_scaling_metric(state_slice, metric)
            for state_slice in self.state_slices
        ]
        if not max_metric:
            self.max_metric = estimate_max_metric(
                model, self.state_slices, self.metrics, max_atoms_to_try
            )
        else:
            self.max_metric = max_metric

        # verify that no systems are too large
        max_metric_value = max(self.metrics)
        max_metric_idx = self.metrics.index(max_metric_value)
        if max_metric_value > self.max_metric:
            raise ValueError(
                f"Max metric of system with index {max_metric_idx} in states: "
                f"{max(self.metrics)} is greater than max_metric {self.max_metric}, "
                f"please set a larger max_metric or run smaller systems metric."
            )

        self.index_to_metric = dict(enumerate(self.metrics))
        self.index_bins = binpacking.to_constant_volume(
            self.index_to_metric, V_max=self.max_metric
        )
        self.state_bins = []
        for index_bin in self.index_bins:
            self.state_bins.append([self.state_slices[i] for i in index_bin])
        self.current_state_bin = 0

    def next_batch(
        self, *, return_indices: bool = False
    ) -> list[BaseState] | tuple[list[BaseState], list[int]] | None:
        """Get the next batch of states.

        Args:
            return_indices: Whether to return indices along with the batch.

        Returns:
            The next batch of states, optionally with indices, or None if no more batches.
        """
        # TODO: need to think about how this intersects with reporting too
        # TODO: definitely a clever treatment to be done with iterators here
        if self.current_state_bin < len(self.state_bins):
            state_bin = self.state_bins[self.current_state_bin]
            self.current_state_bin += 1
            if return_indices:
                return state_bin, self.index_bins[self.current_state_bin - 1]
            return state_bin
        return None

    def restore_original_order(
        self, state_bins: list[list[BaseState]]
    ) -> list[BaseState]:
        """Take the state bins and reorder them into a list.

        Args:
            state_bins: List of state batches to reorder.

        Returns:
            States in their original order.
        """
        # TODO: need to assert at some point that the input states list
        # are all batch size 1

        # Flatten lists
        all_states = list(chain.from_iterable(state_bins))
        original_indices = list(chain.from_iterable(self.index_bins))

        # sort states by original indices
        indexed_states = list(zip(original_indices, all_states, strict=False))
        return [state for _, state in sorted(indexed_states)]


class HotswappingAutoBatcher:
    """Batcher that dynamically swaps states in and out based on convergence."""

    def __init__(
        self,
        model: ModelInterface,
        states: list[BaseState] | Iterator[BaseState] | BaseState,
        metric: Literal["n_atoms", "n_atoms_x_density"] = "n_atoms_x_density",
        max_metric: float | None = None,
        max_atoms_to_try: int = 1_000_000,
    ) -> None:
        """Initialize the batcher.

        Args:
            model: The model to batch for.
            states: States to batch.
            metric: Metric to use for batching.
            max_metric: Maximum metric value per batch.
            max_atoms_to_try: Maximum number of atoms to try when estimating max_metric.
        """
        if isinstance(states, BaseState):
            states = split_state(states)
        if isinstance(states, list):
            states = iter(states)

        self.model = model
        self.states_iterator = states
        self.metric = metric
        self.max_metric = max_metric or None
        self.max_atoms_to_try = max_atoms_to_try

        self.total_metric = 0

        # TODO: could be smarter about making these all together
        self.current_states_list = []
        self.current_metrics_list = []
        self.current_idx_list = []

        self.completed_idx_og_order = []

    def _insert_next_states(self) -> None:
        """Insert states from the iterator until max_metric is reached."""
        for state in self.states_iterator:
            metric = calculate_scaling_metric(state, self.metric)
            if metric > self.max_metric:
                raise ValueError(
                    f"State metric {metric} is greater than max_metric "
                    f"{self.max_metric}, please set a larger max_metric "
                    f"or run smaller systems metric."
                )
            if self.total_metric + metric > self.max_metric:
                # put the state back in the iterator
                self.states_iterator = chain([state], self.states_iterator)
                break
            self.total_metric += metric

            # TODO: could be smarter about making these all together
            self.current_metrics_list += [metric]
            self.current_states_list += [state]
            self.current_idx_list += [self.iterator_idx]
            self.iterator_idx += 1

    def first_batch(self) -> BaseState:
        """Get the first batch of states.

        Returns:
            The first batch of states.
        """
        # we need to estimate the max metric for the first batch
        first_state = next(self.states_iterator)
        first_metric = calculate_scaling_metric(first_state, self.metric)

        # if max_metric is not set, estimate it
        has_max_metric = bool(self.max_metric)
        if not has_max_metric:
            self.max_metric = estimate_max_metric(
                self.model,
                [first_state],
                [first_metric],
                max_atoms_to_try=self.max_atoms_to_try,
            )
            self.max_metric *= 0.8

        self.total_metric = first_metric
        self.current_states_list = [first_state]
        self.current_metrics_list = [first_metric]
        self.current_idx_list = [0]
        self.iterator_idx = 1

        self._insert_next_states()

        # update estimate of max metric if it was not set
        if not has_max_metric:
            self.max_metric = estimate_max_metric(
                self.model,
                self.current_states_list,
                self.current_metrics_list,
                max_atoms_to_try=1_000_000,
            )
        return concatenate_states(self.current_states_list)

    def next_batch(
        self, convergence_tensor: torch.Tensor, *, return_indices: bool = False
    ) -> list[BaseState] | tuple[list[BaseState], list[int]] | None:
        """Get the next batch of states based on convergence.

        Args:
            convergence_tensor: Boolean tensor indicating which states have converged.
            return_indices: Whether to return indices along with the batch.

        Returns:
            The next batch of states.
        """
        assert len(convergence_tensor) == len(self.current_states_list)
        assert len(convergence_tensor.shape) == 1

        # find indices of all convergence_tensor elements that are True
        completed_idx = list(torch.where(convergence_tensor)[0])

        # Sort in descending order to avoid index shifting problems
        completed_idx.sort(reverse=True)

        # remove states at these indices
        for idx in completed_idx:
            self.current_states_list.pop(idx)
            self.total_metric -= self.current_metrics_list.pop(idx)
            self.completed_idx_og_order.append(idx + len(self.completed_idx_og_order))
            self.current_idx_list.pop(idx)

        # insert next states
        self._insert_next_states()

        if not self.current_states_list:
            return None

        if return_indices:
            return self.current_states_list, self.current_idx_list

        return self.current_states_list

    def restore_original_order(
        self, completed_states: list[BaseState]
    ) -> list[BaseState]:
        """Take the list of completed states and reconstruct the original order.

        Args:
            completed_states: List of completed states to reorder.

        Returns:
            States in their original order.

        Raises:
            ValueError: If the number of completed states doesn't match
            the number of indices.
        """
        if len(completed_states) != len(self.completed_idx_og_order):
            raise ValueError(
                f"Number of completed states ({len(completed_states)}) does not match "
                f"number of completed indices ({len(self.completed_idx_og_order)})"
            )

        # Create pairs of (original_index, state)
        indexed_states = list(
            zip(self.completed_idx_og_order, completed_states, strict=False)
        )

        # Sort by original index
        return [state for _, state in sorted(indexed_states)]
