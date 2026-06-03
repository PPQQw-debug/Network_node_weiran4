from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp


@dataclass(frozen=True)
class EliminationResult:
    G_red: sp.Matrix
    Ihis_red: sp.Matrix
    external_nodes: list[str]
    internal_nodes: list[str]
    K_v: sp.Matrix
    K_h: sp.Matrix
    new_order: list[str]
    G_ee: sp.Matrix
    G_ei: sp.Matrix
    G_ie: sp.Matrix
    G_ii: sp.Matrix
    Ihis_e: sp.Matrix
    Ihis_i: sp.Matrix


def _as_column_vector(vec: sp.Matrix, expected_rows: int, name: str) -> sp.Matrix:
    out = sp.Matrix(vec)
    if out.shape == (expected_rows,):
        out = sp.Matrix(expected_rows, 1, list(out))
    if out.shape != (expected_rows, 1):
        raise ValueError(f"{name} must be a {expected_rows}x1 column vector, got {out.shape}")
    return out


def _validate_nodes(all_nodes: Sequence[str], external_nodes: Sequence[str]) -> None:
    if len(set(all_nodes)) != len(all_nodes):
        raise ValueError("all_nodes must not contain duplicates")
    if len(set(external_nodes)) != len(external_nodes):
        raise ValueError("external_nodes must not contain duplicates")

    missing = [node for node in external_nodes if node not in all_nodes]
    if missing:
        raise ValueError(f"external_nodes contains nodes not present in all_nodes: {missing}")


def _light_cancel(expr: sp.Expr) -> sp.Expr:
    return expr if expr != 0 else sp.Integer(0)


def _light_matrix(matrix: sp.Matrix) -> sp.Matrix:
    return matrix


def eliminate_internal_nodes(
    G_full: sp.Matrix,
    Ihis_full: sp.Matrix,
    all_nodes: Sequence[str],
    external_nodes: Sequence[str],
) -> EliminationResult:
    """Eliminate internal nodes from I = G V + Ihis using Schur complement.

    Internal nodes are all nodes in ``all_nodes`` that are not listed in
    ``external_nodes``. The order of external nodes is exactly the user-provided
    order. The order of internal nodes follows ``all_nodes``.
    """

    all_nodes = list(all_nodes)
    external_nodes = list(external_nodes)
    _validate_nodes(all_nodes, external_nodes)

    G_full = sp.Matrix(G_full)
    n = len(all_nodes)
    if G_full.shape != (n, n):
        raise ValueError(f"G_full must have shape {(n, n)}, got {G_full.shape}")

    Ihis_full = _as_column_vector(sp.Matrix(Ihis_full), n, "Ihis_full")

    internal_nodes = [node for node in all_nodes if node not in set(external_nodes)]
    new_order = external_nodes + internal_nodes
    permutation = [all_nodes.index(node) for node in new_order]

    G = G_full.extract(permutation, permutation)
    J = Ihis_full.extract(permutation, [0])

    n_e = len(external_nodes)
    n_i = len(internal_nodes)

    if n_i == 0:
        G_red = sp.simplify(G)
        Ihis_red = sp.simplify(J)
        return EliminationResult(
            G_red=G_red,
            Ihis_red=Ihis_red,
            external_nodes=external_nodes,
            internal_nodes=[],
            K_v=sp.zeros(0, n_e),
            K_h=sp.zeros(0, 1),
            new_order=new_order,
            G_ee=G_red,
            G_ei=sp.zeros(n_e, 0),
            G_ie=sp.zeros(0, n_e),
            G_ii=sp.zeros(0, 0),
            Ihis_e=Ihis_red,
            Ihis_i=sp.zeros(0, 1),
        )

    e_idx = list(range(n_e))
    i_idx = list(range(n_e, n_e + n_i))

    G_ee = G.extract(e_idx, e_idx)
    G_ei = G.extract(e_idx, i_idx)
    G_ie = G.extract(i_idx, e_idx)
    G_ii = G.extract(i_idx, i_idx)

    Ihis_e = J.extract(e_idx, [0])
    Ihis_i = J.extract(i_idx, [0])

    current_nodes = list(new_order)
    current_G = sp.Matrix(G)
    current_J = sp.Matrix(J)
    recovery_steps: list[tuple[str, list[str], list[sp.Expr], sp.Expr]] = []

    try:
        for node in internal_nodes:
            pivot_index = current_nodes.index(node)
            keep = [index for index in range(len(current_nodes)) if index != pivot_index]
            remaining_nodes = [current_nodes[index] for index in keep]
            pivot = current_G[pivot_index, pivot_index]
            if pivot == 0:
                raise ZeroDivisionError(f"zero pivot for internal node {node}")

            row = current_G.extract([pivot_index], keep)
            col = current_G.extract(keep, [pivot_index])
            kept_G = current_G.extract(keep, keep)
            kept_J = current_J.extract(keep, [0])
            node_source = current_J[pivot_index, 0]

            recovery_coeffs = [_light_cancel(-row[0, col_index] / pivot) for col_index in range(row.cols)]
            recovery_source = _light_cancel(-node_source / pivot)
            recovery_steps.append((node, remaining_nodes, recovery_coeffs, recovery_source))

            current_G = _light_matrix(kept_G - (col * row) / pivot)
            current_J = _light_matrix(kept_J - col * node_source / pivot)
            current_nodes = remaining_nodes
    except Exception as exc:
        raise ValueError(f"cannot eliminate internal nodes {internal_nodes}: {exc}") from exc

    G_red = current_G
    Ihis_red = current_J

    recovery_by_node: dict[str, tuple[sp.Matrix, sp.Expr]] = {
        node: (sp.eye(n_e).row(index).T, sp.Integer(0))
        for index, node in enumerate(external_nodes)
    }
    for node, remaining_nodes, coeffs, source in reversed(recovery_steps):
        coeff_vector = sp.zeros(n_e, 1)
        source_expr = source
        for remaining_node, coeff in zip(remaining_nodes, coeffs):
            remaining_coeffs, remaining_source = recovery_by_node[remaining_node]
            coeff_vector += coeff * remaining_coeffs
            source_expr += coeff * remaining_source
        recovery_by_node[node] = (_light_matrix(coeff_vector), _light_cancel(source_expr))

    K_v = sp.Matrix.vstack(*[recovery_by_node[node][0].T for node in internal_nodes])
    K_h = sp.Matrix([[_light_cancel(recovery_by_node[node][1])] for node in internal_nodes])

    return EliminationResult(
        G_red=G_red,
        Ihis_red=Ihis_red,
        external_nodes=external_nodes,
        internal_nodes=internal_nodes,
        K_v=K_v,
        K_h=K_h,
        new_order=new_order,
        G_ee=_light_matrix(G_ee),
        G_ei=_light_matrix(G_ei),
        G_ie=_light_matrix(G_ie),
        G_ii=_light_matrix(G_ii),
        Ihis_e=_light_matrix(Ihis_e),
        Ihis_i=_light_matrix(Ihis_i),
    )
