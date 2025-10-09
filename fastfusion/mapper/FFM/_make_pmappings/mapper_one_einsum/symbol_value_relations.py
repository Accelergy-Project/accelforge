from numbers import Number

from sympy import Symbol

from fastfusion.frontend.mapping import (
    Iteration,
    Mapping,
)


class SymbolValueRelations:
    def __init__(self):
        self.what_tiles_symbol: list[tuple[Symbol | int, Symbol | int]] = []

    def get_tiled_by(self, symbol: Symbol, none_if_fail: bool=False) -> Symbol | int | None:
        for tiled_by, what_tiles in self.what_tiles_symbol:
            if tiled_by == symbol:
                return what_tiles
        if none_if_fail:
            return None
        raise ValueError(f"Symbol {symbol} not found in {self}")

    def get_tiles(self, symbol: Symbol, none_if_fail: bool=False) -> Symbol | int | None:
        for tiled_by, what_tiles in self.what_tiles_symbol:
            if what_tiles == symbol:
                return tiled_by
        if none_if_fail:
            return None
        raise ValueError(f"Symbol {symbol} not found in {self}")

    def get_max_size(self, symbol: Symbol) -> Number:
        while not isinstance(symbol, Number):
            symbol = self.get_tiles(symbol)
        return symbol

    @staticmethod
    def from_pmapping_and_shape(pmapping: Mapping, shape: dict[str, int]) -> "SymbolValueRelations":
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

        for r, s in last_seen_loop_per_rank_var.items():
            if isinstance(s, Symbol):
                relation.what_tiles_symbol.append((s, 1))
        return relation
