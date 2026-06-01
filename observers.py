from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from elimination import EliminationResult


@dataclass(frozen=True)
class ReducedObserver:
    C_red: sp.Matrix
    d_red: sp.Expr
    external_nodes: list[str]

    def expression(self, voltage_symbols: Sequence[sp.Symbol]) -> sp.Expr:
        V_e = sp.Matrix(list(voltage_symbols))
        if V_e.shape != (self.C_red.shape[1], 1):
            raise ValueError(f"Expected {self.C_red.shape[1]} voltage symbols, got {V_e.shape[0]}")
        return sp.simplify((self.C_red * V_e)[0, 0] + self.d_red)


def reduce_observer(
    C_full: sp.Matrix,
    d: sp.Expr,
    all_nodes: Sequence[str],
    elimination: EliminationResult,
) -> ReducedObserver:
    """Reduce i_obs = C_full V_full + d after internal-node elimination."""

    all_nodes = list(all_nodes)
    C_full = sp.Matrix(C_full)
    if C_full.shape != (1, len(all_nodes)):
        raise ValueError(f"C_full must be a 1x{len(all_nodes)} row vector, got {C_full.shape}")

    order = elimination.external_nodes + elimination.internal_nodes
    permutation = [all_nodes.index(node) for node in order]
    C = C_full.extract([0], permutation)

    n_e = len(elimination.external_nodes)
    C_e = C.extract([0], list(range(n_e)))
    C_i = C.extract([0], list(range(n_e, C.shape[1])))

    C_red = sp.simplify(C_e + C_i * elimination.K_v)
    d_red = sp.simplify((C_i * elimination.K_h)[0, 0] + d)

    return ReducedObserver(
        C_red=C_red,
        d_red=d_red,
        external_nodes=list(elimination.external_nodes),
    )
