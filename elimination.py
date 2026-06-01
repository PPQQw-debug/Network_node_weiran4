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

    try:
        Gii_inv = G_ii.inv()
        Gii_inv_Gie = Gii_inv * G_ie
        Gii_inv_Ihis_i = Gii_inv * Ihis_i
    except Exception as exc:
        raise ValueError(f"G_ii is not invertible; cannot eliminate internal nodes {internal_nodes}") from exc

    K_v = sp.simplify(-Gii_inv_Gie)
    K_h = sp.simplify(-Gii_inv_Ihis_i)

    G_red = sp.simplify(G_ee - G_ei * Gii_inv_Gie)
    Ihis_red = sp.simplify(Ihis_e - G_ei * Gii_inv_Ihis_i)

    return EliminationResult(
        G_red=G_red,
        Ihis_red=Ihis_red,
        external_nodes=external_nodes,
        internal_nodes=internal_nodes,
        K_v=K_v,
        K_h=K_h,
        new_order=new_order,
        G_ee=sp.simplify(G_ee),
        G_ei=sp.simplify(G_ei),
        G_ie=sp.simplify(G_ie),
        G_ii=sp.simplify(G_ii),
        Ihis_e=sp.simplify(Ihis_e),
        Ihis_i=sp.simplify(Ihis_i),
    )
