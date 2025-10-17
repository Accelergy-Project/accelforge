from collections import defaultdict
from numbers import Number

from sympy import Symbol

from fastfusion.frontend.workload import Workload
from fastfusion.frontend.workload._symbolic import get_stride_and_halo
from fastfusion.frontend.mapping import (
    Iteration,
    Mapping,
)


class SymbolValueRelations:
    def __init__(self):
        self.what_tiles_symbol: list[tuple[Symbol | int, Symbol | int]] = []
        self.stride_and_initial: list[tuple[Symbol | int, Symbol | int]] = []
        self.delta_choices: list[tuple[Symbol, frozenset[int]]] = []

    def is_stride(self, symbol: Symbol) -> bool:
        """Check if `symbol` is a stride."""
        for stride, initial in self.stride_and_initial:
            if stride == symbol:
                return True
            if initial == symbol:
                return False
        return True

    def is_initial_tile_shape(self, symbol: Symbol) -> bool:
        """Check if `symbol` is a initial tile shape."""
        for stride, initial in self.stride_and_initial:
            if stride == symbol:
                return False
            if initial == symbol:
                return True
        return False

    def get_stride(self, symbol: Symbol) -> Symbol | int:
        """Get the stride corresponding to the initial tile shape `symbol`."""
        for stride, initial in self.stride_and_initial:
            if initial == symbol:
                return stride
        raise ValueError(f"Symbol {symbol} not found as initial in {self}")

    def get_initial(self, symbol: Symbol, none_if_fail: bool=False) -> Symbol | int:
        for stride, initial in self.stride_and_initial:
            if stride == symbol:
                return initial
        if not none_if_fail:
            raise ValueError(f"Symbol {symbol} not found as stride in {self}")
        else:
            return None

    def get_delta_choices(self, symbol: Symbol) -> frozenset[int]:
        """Get the possible initial deltas for the rank variable represented by `symbol`."""
        for initial, choices in self.delta_choices:
            if initial == symbol:
                return choices
        raise ValueError(f"Symbol {symbol} not found in {self}")

    def get_inner_tiles(
        self, symbol: Symbol, none_if_fail: bool = False
    ) -> Symbol | int | None:
        """Get tiles within the tile represented by `symbol`."""
        for tiled_by, what_tiles in self.what_tiles_symbol:
            if tiled_by == symbol:
                return what_tiles
        if none_if_fail:
            return None
        raise ValueError(f"Symbol {symbol} not found in {self}")

    def get_outer_tiles(
        self, symbol: Symbol, none_if_fail: bool = False
    ) -> Symbol | int | None:
        """Get the tile that contain the tile represented by `symbol`."""
        for tiled_by, what_tiles in self.what_tiles_symbol:
            if what_tiles == symbol:
                return tiled_by
        if none_if_fail:
            return None
        raise ValueError(f"Symbol {symbol} not found in {self}")

    def get_max_size(self, symbol: Symbol) -> Number:
        while not isinstance(symbol, Number):
            symbol = self.get_outer_tiles(symbol)
        return symbol

    @staticmethod
    def from_pmapping_and_shape(
        pmapping: Mapping, shape: dict[str, int], workload: Workload
    ) -> "SymbolValueRelations":
        initial_delta_choices = get_initial_delta_choices(
            pmapping.nodes[-1].einsum, workload
        )

        relation = SymbolValueRelations()
        last_seen_loop_per_rank_var: dict[str, Symbol | int] = dict(shape)
        for node in pmapping.nodes:
            if not isinstance(node, Iteration):
                continue
            prev = last_seen_loop_per_rank_var.get(node.rank_variable, None)
            # If we're a symbol and we've seen an outer loop with the same rank variable,
            # then we tile that one.
            if prev is not None:
                relation.what_tiles_symbol.append((prev, node.stride))
            last_seen_loop_per_rank_var[node.rank_variable] = node.stride

            if (
                isinstance(node.initial_tile_shape, Symbol)
                and node.initial_tile_shape != node.stride
            ):
                relation.stride_and_initial.append(
                    (node.stride, node.initial_tile_shape)
                )
                relation.delta_choices.append(
                    (
                        node.initial_tile_shape,
                        frozenset(initial_delta_choices[node.rank_variable]),
                    )
                )

        for r, s in last_seen_loop_per_rank_var.items():
            if isinstance(s, Symbol):
                relation.what_tiles_symbol.append((s, 1))
        return relation


def get_initial_delta_choices(einsum_name: str, workload: Workload):
    stride_and_halo = get_stride_and_halo(workload)
    einsum = workload.einsums[einsum_name]

    choices = defaultdict(lambda: set([0]))
    consumer_chains = []
    stack = [[(None, einsum)]]
    while stack:
        cur_chain = stack.pop()
        last_tensor, last_einsum = cur_chain[-1]
        for tensor in last_einsum.output_tensors():
            einsums_that_read_tensor = workload.einsums_that_read_tensor(tensor)

            if len(einsums_that_read_tensor) == 0:
                consumer_chains.append(cur_chain)

            for next_einsum in einsums_that_read_tensor:
                stack.append(cur_chain + [(tensor, next_einsum)])

    for chain in consumer_chains:
        for (_, producer), (tensor, consumer) in zip(
            list(reversed(chain))[1:], reversed(chain)
        ):
            rank_stride_and_halo = stride_and_halo[(consumer.name, tensor)]
            if tensor is None:
                break  # done

            for cons_rank_var in consumer.rank_variables:
                for prod_rank_var in producer.rank_variables:
                    prod_rank = prod_rank_var.upper()
                    for cons_choice in choices[cons_rank_var]:
                        key = (prod_rank, cons_rank_var)
                        if key not in rank_stride_and_halo:
                            continue
                        stride, halo = rank_stride_and_halo[key]
                        choices[prod_rank_var].add(int(cons_choice*stride + halo))

    return choices
