from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import sympy as sp


@dataclass(frozen=True)
class GroundConstraintResult:
    G_ng: sp.Matrix
    Ihis_ng: sp.Matrix
    remaining_nodes: list[str]
    ground_voltage_map: dict[str, sp.Expr]


@dataclass(frozen=True)
class GroundPartitionValidation:
    external_nodes: list[str]
    internal_nodes: list[str]
    ground_nodes: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class GroundElement:
    """A 0 V known-voltage node declaration.

    Ground is the 0 V special case of a known-voltage node. For a future
    non-zero known voltage V_k, the reduced unknown-node equation would use
    Ihis_new = Ihis_u + G_uk * V_k. In this implementation V_k is always 0,
    so that additional term is zero and ground rows/columns are simply removed.
    """

    name: str
    node: str


def _as_column_vector(vec: sp.Matrix, expected_rows: int, name: str) -> sp.Matrix:
    out = sp.Matrix(vec)
    if out.shape == (expected_rows,):
        out = sp.Matrix(expected_rows, 1, list(out))
    if out.shape != (expected_rows, 1):
        raise ValueError(f"{name} must be a {expected_rows}x1 column vector, got {out.shape}")
    return out


def ordered_ground_nodes(node_order: Sequence[str], ground_nodes: Iterable[str]) -> list[str]:
    ground_set = set(ground_nodes)
    return [node for node in node_order if node in ground_set]


def apply_ground_constraint(
    G_full: sp.Matrix,
    Ihis_full: sp.Matrix,
    node_order: Sequence[str],
    ground_nodes: Iterable[str],
) -> GroundConstraintResult:
    """Remove 0 V ground nodes from I = G V + Ihis.

    Ground nodes are known-voltage nodes with V_ground = 0. They are not
    internal nodes and must not be eliminated with a KCL constraint.
    """

    node_order = list(node_order)
    if len(set(node_order)) != len(node_order):
        raise ValueError("node_order must not contain duplicates")

    G_full = sp.Matrix(G_full)
    n = len(node_order)
    if G_full.shape != (n, n):
        raise ValueError(f"G_full must have shape {(n, n)}, got {G_full.shape}")
    Ihis_full = _as_column_vector(sp.Matrix(Ihis_full), n, "Ihis_full")

    unknown_ground = [node for node in ground_nodes if node not in set(node_order)]
    if unknown_ground:
        raise ValueError(f"ground_nodes contains nodes not present in node_order: {unknown_ground}")

    ground_ordered = ordered_ground_nodes(node_order, ground_nodes)
    ground_set = set(ground_ordered)
    keep = [index for index, node in enumerate(node_order) if node not in ground_set]
    remaining_nodes = [node_order[index] for index in keep]

    return GroundConstraintResult(
        G_ng=G_full.extract(keep, keep),
        Ihis_ng=Ihis_full.extract(keep, [0]),
        remaining_nodes=remaining_nodes,
        ground_voltage_map={node: sp.Integer(0) for node in ground_ordered},
    )


def validate_ground_partition(
    all_nodes: Sequence[str],
    external_nodes: Sequence[str],
    internal_nodes: Sequence[str],
    ground_nodes: Iterable[str],
) -> GroundPartitionValidation:
    ground_ordered = ordered_ground_nodes(all_nodes, ground_nodes)
    ground_set = set(ground_ordered)
    external_clean = [node for node in external_nodes if node not in ground_set]
    internal_clean = [node for node in internal_nodes if node not in ground_set]
    warnings: list[str] = []
    if len(external_clean) != len(list(external_nodes)):
        warnings.append("Ground node is a fixed-voltage node and cannot be external.")
    if len(internal_clean) != len(list(internal_nodes)):
        warnings.append("Ground node is a fixed-voltage node and cannot be internal.")
    return GroundPartitionValidation(
        external_nodes=external_clean,
        internal_nodes=internal_clean,
        ground_nodes=ground_ordered,
        warnings=warnings,
    )


def restrict_observer_to_remaining_nodes(
    C_full: sp.Matrix,
    d: sp.Expr,
    node_order: Sequence[str],
    ground_voltage_map: dict[str, sp.Expr],
) -> tuple[sp.Matrix, sp.Expr, list[str]]:
    """Substitute known ground voltages in i_obs = C V + d.

    For 0 V ground this drops ground columns. The function is written for the
    general known-voltage case so a future V_known term can be added to d.
    """

    node_order = list(node_order)
    C_full = sp.Matrix(C_full)
    if C_full.shape != (1, len(node_order)):
        raise ValueError(f"C_full must be a 1x{len(node_order)} row vector, got {C_full.shape}")
    ground_set = set(ground_voltage_map)
    remaining_indices = [index for index, node in enumerate(node_order) if node not in ground_set]
    remaining_nodes = [node_order[index] for index in remaining_indices]
    known_offset = sp.Integer(0)
    for index, node in enumerate(node_order):
        if node in ground_voltage_map:
            known_offset += C_full[0, index] * ground_voltage_map[node]
    return C_full.extract([0], remaining_indices), sp.simplify(d + known_offset), remaining_nodes
