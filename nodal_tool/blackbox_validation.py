from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import sympy as sp


@dataclass(frozen=True)
class BlackBoxBranchObserver:
    name: str
    from_node: str
    to_node: str
    G: sp.Expr
    Ihis: sp.Expr = sp.Integer(0)
    description: str = ""


@dataclass(frozen=True)
class ObserverRow:
    name: str
    from_node: str
    to_node: str
    C: sp.Matrix
    d: sp.Expr
    expression: sp.Expr
    description: str = ""


@dataclass
class BlackBoxObserverValidationResult:
    status: str
    G_obs: sp.Matrix
    Ihis_obs: sp.Matrix
    Delta_G: sp.Matrix
    Delta_Ihis: sp.Matrix
    warnings: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    unknown_symbols: set[sp.Symbol] = field(default_factory=set)
    observer_rows: list[ObserverRow] = field(default_factory=list)


def simplify_matrix(M: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(M).applyfunc(sp.simplify)


def is_zero_matrix(M: sp.Matrix) -> bool:
    return all(sp.simplify(item) == 0 for item in sp.Matrix(M))


def collect_symbols_from_matrix(G: sp.Matrix, Ihis: sp.Matrix) -> set[sp.Symbol]:
    symbols: set[sp.Symbol] = set()
    for item in list(sp.Matrix(G)) + list(sp.Matrix(Ihis)):
        symbols.update(sp.sympify(item).free_symbols)
    return symbols


def _as_column_vector(vec: sp.Matrix, expected_rows: int, name: str) -> sp.Matrix:
    out = sp.Matrix(vec)
    if out.shape == (expected_rows,):
        out = sp.Matrix(expected_rows, 1, list(out))
    if out.shape != (expected_rows, 1):
        raise ValueError(f"{name} must be a {expected_rows}x1 column vector, got {out.shape}")
    return out


def _validate_node_list(nodes: Sequence[str]) -> list[str]:
    nodes = [str(node) for node in nodes]
    if len(set(nodes)) != len(nodes):
        raise ValueError("black-box node list must not contain duplicates")
    return nodes


def _observer_expression(row: sp.Matrix, nodes: Sequence[str], d: sp.Expr) -> sp.Expr:
    voltages = sp.Matrix([sp.Symbol(f"V_{node}") for node in nodes])
    return sp.simplify((row * voltages)[0, 0] + d)


def build_observer_stamp(
    nodes: Sequence[str],
    observers: Sequence[BlackBoxBranchObserver],
) -> tuple[sp.Matrix, sp.Matrix, list[ObserverRow]]:
    nodes = _validate_node_list(nodes)
    node_index = {node: index for index, node in enumerate(nodes)}
    n = len(nodes)
    G_obs = sp.zeros(n, n)
    Ihis_obs = sp.zeros(n, 1)
    observer_rows: list[ObserverRow] = []

    for observer in observers:
        missing = [node for node in (observer.from_node, observer.to_node) if node not in node_index]
        if missing:
            raise ValueError(
                f"Observer {observer.name} uses node {', '.join(missing)} not present in black-box node list."
            )

        g = sp.sympify(observer.G)
        ihis = sp.sympify(observer.Ihis)
        f = node_index[observer.from_node]
        t = node_index[observer.to_node]

        G_obs[f, f] += g
        G_obs[t, t] += g
        G_obs[f, t] -= g
        G_obs[t, f] -= g
        Ihis_obs[f, 0] += ihis
        Ihis_obs[t, 0] -= ihis

        C = sp.zeros(1, n)
        C[0, f] += g
        C[0, t] -= g
        observer_rows.append(
            ObserverRow(
                name=observer.name,
                from_node=observer.from_node,
                to_node=observer.to_node,
                C=simplify_matrix(C),
                d=sp.simplify(ihis),
                expression=_observer_expression(C, nodes, ihis),
                description=observer.description,
            )
        )

    return simplify_matrix(G_obs), simplify_matrix(Ihis_obs), observer_rows


def _observer_symbols(observers: Iterable[BlackBoxBranchObserver]) -> set[sp.Symbol]:
    symbols: set[sp.Symbol] = set()
    for observer in observers:
        symbols.update(sp.sympify(observer.G).free_symbols)
        symbols.update(sp.sympify(observer.Ihis).free_symbols)
    return symbols


def validate_blackbox_observers(
    G_bb: sp.Matrix,
    Ihis_bb: sp.Matrix,
    nodes: Sequence[str],
    observers: Sequence[BlackBoxBranchObserver],
) -> BlackBoxObserverValidationResult:
    nodes = _validate_node_list(nodes)
    n = len(nodes)
    G_bb = sp.Matrix(G_bb)
    if G_bb.shape != (n, n):
        raise ValueError(f"G_bb must be a {n}x{n} matrix, got {G_bb.shape}")
    Ihis_bb = _as_column_vector(sp.Matrix(Ihis_bb), n, "Ihis_bb")
    observers = list(observers)

    zero_G = sp.zeros(n, n)
    zero_I = sp.zeros(n, 1)
    if not observers:
        return BlackBoxObserverValidationResult(
            status="floating",
            G_obs=zero_G,
            Ihis_obs=zero_I,
            Delta_G=simplify_matrix(G_bb),
            Delta_Ihis=simplify_matrix(Ihis_bb),
            messages=["No branch current observers defined. Branch currents are unobservable/floating."],
            observer_rows=[],
        )

    G_obs, Ihis_obs, observer_rows = build_observer_stamp(nodes, observers)
    Delta_G = simplify_matrix(G_bb - G_obs)
    Delta_Ihis = simplify_matrix(Ihis_bb - Ihis_obs)

    known_symbols = collect_symbols_from_matrix(G_bb, Ihis_bb)
    unknown_symbols = _observer_symbols(observers) - known_symbols
    warnings: list[str] = []
    if unknown_symbols:
        formatted = ", ".join(sorted(str(symbol) for symbol in unknown_symbols))
        warnings.append(f"Observer uses symbols not present in the black-box G/Ihis: {{{formatted}}}")

    if is_zero_matrix(Delta_G) and is_zero_matrix(Delta_Ihis):
        status = "matched"
        messages = ["User-defined branch current observers fully reproduce the black-box G/Ihis."]
    else:
        status = "mismatch"
        messages = [
            "User-defined branch current observers do not fully reproduce the black-box G/Ihis. "
            "Nodal equations still use the provided G/Ihis, but branch current observers may be incomplete or inconsistent."
        ]

    return BlackBoxObserverValidationResult(
        status=status,
        G_obs=G_obs,
        Ihis_obs=Ihis_obs,
        Delta_G=Delta_G,
        Delta_Ihis=Delta_Ihis,
        warnings=warnings,
        messages=messages,
        unknown_symbols=unknown_symbols,
        observer_rows=observer_rows,
    )
