"""Utilities for batching and memory management in torchsim."""

from collections.abc import Iterator
from itertools import chain
from typing import Literal

import binpacking
import torch

from torchsim.models.interface import ModelInterface
from torchsim.state import BaseState, concatenate_states, pop_states, split_state


def measure_model_memory_forward(state: BaseState, model: ModelInterface) -> float:
    """Measure peak GPU memory usage during model forward pass.

    Args:
        state: The input state to pass to the model.
        model: The model to measure memory usage for.

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
    state: BaseState, model: ModelInterface, max_atoms: int = 500_000
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
            measure_model_memory_forward(concat_state, model)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                return fib[i - 2]
            raise

    return fib[-2]


def calculate_memory_scaler(
    state_slice: BaseState,
    memory_scales_with: Literal["n_atoms_x_density", "n_atoms"] = "n_atoms_x_density",
) -> float:
    """Calculate scaling metric for a state.

    Args:
        state_slice: The state to calculate metric for.
        memory_scales_with: The type of metric to calculate.

    Returns:
        The calculated metric value.
    """
    if memory_scales_with == "n_atoms":
        return state_slice.n_atoms
    if memory_scales_with == "n_atoms_x_density":
        volume = torch.abs(torch.linalg.det(state_slice.cell[0])) / 1000
        number_density = state_slice.n_atoms / volume.item()
        return state_slice.n_atoms * number_density
    raise ValueError(f"Invalid metric: {memory_scales_with}")


def estimate_max_memory_scaler(
    model: ModelInterface,
    state_list: list[BaseState],
    metric_values: list[float],
    max_atoms: int = 500_000,
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
    metric_values = torch.tensor(metric_values)

    # select one state with the min n_atoms
    min_metric = metric_values.min()
    max_metric = metric_values.max()

    min_state = state_list[metric_values.argmin()]
    max_state = state_list[metric_values.argmax()]

    min_state_max_batches = determine_max_batch_size(min_state, model, max_atoms)
    max_state_max_batches = determine_max_batch_size(max_state, model, max_atoms)

    return min(min_state_max_batches * min_metric, max_state_max_batches * max_metric)


class ChunkingAutoBatcher:
    """Batcher that chunks states into bins of similar computational cost."""

    def __init__(
        self,
        states: list[BaseState] | BaseState,
        model: ModelInterface,
        *,
        memory_scales_with: Literal["n_atoms", "n_atoms_x_density"] = "n_atoms_x_density",
        max_memory_scaler: float | None = None,
        max_atoms_to_try: int = 500_000,
        return_indices: bool = False,
    ) -> None:
        """Initialize the batcher.

        Args:
            model: The model to batch for.
            states: States to batch.
            memory_scales_with: Metric to use for batching.
            max_memory_scaler: Maximum metric value per batch.
            max_atoms_to_try: Max number of atoms to try when estimating max_metric.
            return_indices: Whether to return indices along with the batch.
        """
        self.state_slices = (
            split_state(states) if isinstance(states, BaseState) else states
        )
        self.memory_scalers = [
            calculate_memory_scaler(state_slice, memory_scales_with)
            for state_slice in self.state_slices
        ]
        if not max_memory_scaler:
            self.max_memory_scaler = estimate_max_memory_scaler(
                model, self.state_slices, self.memory_scalers, max_atoms_to_try
            )
            print(f"Max metric calculated: {self.max_memory_scaler}")
        else:
            self.max_memory_scaler = max_memory_scaler

        self.return_indices = return_indices
        # verify that no systems are too large
        max_metric_value = max(self.memory_scalers)
        max_metric_idx = self.memory_scalers.index(max_metric_value)
        if max_metric_value > self.max_memory_scaler:
            raise ValueError(
                f"Max metric of system with index {max_metric_idx} in states: "
                f"{max(self.memory_scalers)} is greater than max_metric "
                f"{self.max_memory_scaler}, please set a larger max_metric "
                f"or run smaller systems metric."
            )

        self.index_to_scaler = dict(enumerate(self.memory_scalers))
        self.index_bins = binpacking.to_constant_volume(
            self.index_to_scaler, V_max=self.max_memory_scaler
        )
        self.batched_states = []
        for index_bin in self.index_bins:
            self.batched_states.append([self.state_slices[i] for i in index_bin])
        self.current_state_bin = 0

    def next_batch(
        self, *, return_indices: bool = False
    ) -> BaseState | tuple[list[BaseState], list[int]] | None:
        """Get the next batch of states.

        Args:
            return_indices: Whether to return indices along with the batch.

        Returns:
            The next batch of states, optionally with indices, or None if no more batches.
        """
        # TODO: need to think about how this intersects with reporting too
        # TODO: definitely a clever treatment to be done with iterators here
        if self.current_state_bin < len(self.batched_states):
            state_bin = self.batched_states[self.current_state_bin]
            state = concatenate_states(state_bin)
            self.current_state_bin += 1
            if return_indices:
                return state, self.index_bins[self.current_state_bin - 1]
            return state
        return None

    def __iter__(self) -> Iterator[BaseState]:
        """Iterate over the batches."""
        return self

    def __next__(self) -> BaseState:
        """Get the next batch."""
        next_batch = self.next_batch(return_indices=self.return_indices)
        if next_batch is None:
            raise StopIteration
        return next_batch

    def restore_original_order(self, batched_states: list[BaseState]) -> list[BaseState]:
        """Take the state bins and reorder them into a list.

        Args:
            batched_states: List of state batches to reorder.

        Returns:
            States in their original order.
        """
        state_bins = [split_state(state) for state in batched_states]

        # Flatten lists
        all_states = list(chain.from_iterable(state_bins))
        original_indices = list(chain.from_iterable(self.index_bins))

        if len(all_states) != len(original_indices):
            raise ValueError(
                f"Number of states ({len(all_states)}) does not match "
                f"number of original indices ({len(original_indices)})"
            )

        # sort states by original indices
        indexed_states = list(zip(original_indices, all_states, strict=False))
        return [state for _, state in sorted(indexed_states, key=lambda x: x[0])]


class HotswappingAutoBatcher:
    """Batcher that dynamically swaps states in and out based on convergence."""

    def __init__(
        self,
        states: list[BaseState] | Iterator[BaseState] | BaseState,
        model: ModelInterface,
        memory_scales_with: Literal["n_atoms", "n_atoms_x_density"] = "n_atoms_x_density",
        max_memory_scaler: float | None = None,
        max_atoms_to_try: int = 500_000,
    ) -> None:
        """Initialize the batcher.

        Args:
            model: The model to batch for.
            states: States to batch.
            memory_scales_with: Metric to use for batching.
            max_memory_scaler: Maximum metric value per batch.
            max_atoms_to_try: Maximum number of atoms to try when estimating max_metric.
        """
        if isinstance(states, BaseState):
            states = split_state(states)
        if isinstance(states, list):
            states = iter(states)

        self.model = model
        self.states_iterator = states
        self.memory_scales_with = memory_scales_with
        self.max_memory_scaler = max_memory_scaler or None
        self.max_atoms_to_try = max_atoms_to_try

        self.current_scalers = []
        self.current_idx = []
        self.iterator_idx = 0

        self.completed_idx_og_order = []

    def _get_next_states(self) -> None:
        """Insert states from the iterator until max_metric is reached."""
        new_metrics = []
        new_idx = []
        new_states = []
        for state in self.states_iterator:
            metric = calculate_memory_scaler(state, self.memory_scales_with)
            if metric > self.max_memory_scaler:
                raise ValueError(
                    f"State metric {metric} is greater than max_metric "
                    f"{self.max_memory_scaler}, please set a larger max_metric "
                    f"or run smaller systems metric."
                )
            # new_metric += sum(new_metrics)
            if (
                sum(self.current_scalers) + sum(new_metrics) + metric
                > self.max_memory_scaler
            ):
                # put the state back in the iterator
                self.states_iterator = chain([state], self.states_iterator)
                break

            new_metrics.append(metric)
            new_idx.append(self.iterator_idx)
            new_states.append(state)
            self.iterator_idx += 1

        self.current_scalers.extend(new_metrics)
        self.current_idx.extend(new_idx)

        return new_states

    def _delete_old_states(self, completed_idx: list[int]) -> None:
        # Sort in descending order to avoid index shifting problems
        completed_idx.sort(reverse=True)

        # update state tracking lists
        for idx in completed_idx:
            og_idx = self.current_idx.pop(idx)
            self.current_scalers.pop(idx)
            self.completed_idx_og_order.append(og_idx)

    def _first_batch(self) -> BaseState:
        """Get the first batch of states.

        Returns:
            The first batch of states.
        """
        # we need to sample a state and use it to estimate the max metric
        # for the first batch
        first_state = next(self.states_iterator)
        first_metric = calculate_memory_scaler(first_state, self.memory_scales_with)
        self.current_scalers += [first_metric]
        self.current_idx += [0]
        self.iterator_idx += 1
        # self.total_metric += first_metric

        # if max_metric is not set, estimate it
        has_max_metric = bool(self.max_memory_scaler)
        if not has_max_metric:
            self.max_memory_scaler = estimate_max_memory_scaler(
                self.model,
                [first_state],
                [first_metric],
                max_atoms=self.max_atoms_to_try,
            )
            self.max_memory_scaler *= 0.8

        states = self._get_next_states()

        if not has_max_metric:
            self.max_memory_scaler = estimate_max_memory_scaler(
                self.model,
                [first_state, *states],
                self.current_scalers,
                max_atoms=self.max_atoms_to_try,
            )
            print(f"Max metric calculated: {self.max_memory_scaler}")
        return concatenate_states([first_state, *states]), []

    def next_batch(
        self,
        updated_state: BaseState,
        convergence_tensor: torch.Tensor | None = None,
        *,
        return_indices: bool = False,
    ) -> tuple[BaseState, list[BaseState]] | tuple[BaseState, list[BaseState], list[int]]:
        """Get the next batch of states based on convergence.

        Args:
            updated_state: The updated state.
            convergence_tensor: Boolean tensor indicating which states have converged.
            return_indices: Whether to return indices along with the batch.

        Returns:
            The next batch of states.
        """
        # TODO: this needs to be refactored to avoid so
        # many split and concatenate operations, we should
        # take the updated_concat_state and pop off
        # the states that have converged. with the pop_states function

        if convergence_tensor is None:
            if self.iterator_idx > 0:
                raise ValueError(
                    "A convergence tensor must be provided after the "
                    "first batch has been run."
                )
            return self._first_batch()

        # assert statements helpful for debugging, should be moved to validate fn
        # the first two are most important
        assert len(convergence_tensor) == updated_state.n_batches
        assert len(self.current_idx) == len(self.current_scalers)
        assert len(convergence_tensor.shape) == 1
        assert updated_state.n_batches > 0

        completed_idx = torch.where(convergence_tensor)[0].tolist()
        completed_idx.sort(reverse=True)

        remaining_state, completed_states = pop_states(updated_state, completed_idx)

        self._delete_old_states(completed_idx)
        next_states = self._get_next_states()

        # there are no states left to run, return the completed states
        if not self.current_idx:
            return (
                (None, completed_states, [])
                if return_indices
                else (None, completed_states)
            )

        # concatenate remaining state with next states
        if remaining_state.n_batches > 0:
            next_states = [remaining_state, *next_states]
        next_batch = concatenate_states(next_states)

        if return_indices:
            return next_batch, completed_states, self.current_idx

        return next_batch, completed_states

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
        # TODO: should act on full states, not state slices

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
        return [state for _, state in sorted(indexed_states, key=lambda x: x[0])]
