from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import sympy as sp

from .controlled_sources import parse_expr


GROUND_NODE_NAMES = {"0", "GND", "GROUND", "gnd", "ground"}


def _is_ground_node(node: str) -> bool:
    return str(node).strip() in GROUND_NODE_NAMES


def _is_nonpositive_number(value: sp.Expr) -> bool:
    value = sp.simplify(value)
    if value == 0:
        return True
    if value.is_number:
        return bool(value.evalf() <= 0)
    if value.is_nonpositive is True:
        return True
    return False


@dataclass(frozen=True)
class VoltageSourceSeriesR:
    """Voltage source with series conductance, stamped as a Norton equivalent.

    The branch current direction is node_p -> node_n and the nodal convention is
    current leaving the node is positive:

      i = Gs * (V_p - V_n - Vs)

    Therefore the stamp is a symmetric conductance Gs plus history
    terms -Gs*Vs at p and +Gs*Vs at n. This avoids MNA and introduces no extra
    current unknown.
    """

    name: str
    node_p: str
    node_n: str
    voltage: str | sp.Expr
    conductance: str | sp.Expr

    def parsed_voltage(self) -> sp.Expr:
        return parse_expr(self.voltage)

    def series_conductance(self) -> sp.Expr:
        if self.conductance is None or str(self.conductance).strip() == "":
            raise ValueError("VoltageSourceSeriesR requires a series conductance Gs.")
        conductance = parse_expr(self.conductance)
        if _is_nonpositive_number(conductance):
            raise ValueError("VoltageSourceSeriesR series conductance Gs must be positive.")
        return sp.simplify(conductance)

    def validate(self, nodes: Sequence[str]) -> list[str]:
        node_set = set(nodes)
        errors: list[str] = []
        if self.node_p not in node_set and not _is_ground_node(self.node_p):
            errors.append(f"VoltageSourceSeriesR node_p {self.node_p} not present in node list.")
        if self.node_n not in node_set and not _is_ground_node(self.node_n):
            errors.append(f"VoltageSourceSeriesR node_n {self.node_n} not present in node list.")
        try:
            self.parsed_voltage()
        except Exception as exc:
            errors.append(f"VoltageSourceSeriesR voltage could not be parsed: {exc}")
        try:
            self.series_conductance()
        except Exception as exc:
            errors.append(str(exc))
        return errors

    def stamp(self, G: sp.Matrix, Ihis: sp.Matrix, node_index: Mapping[str, int]) -> None:
        errors = self.validate(list(node_index.keys()))
        if errors:
            raise ValueError("; ".join(errors))

        p_present = self.node_p in node_index
        n_present = self.node_n in node_index
        p = node_index[self.node_p] if p_present else None
        n = node_index[self.node_n] if n_present else None
        gs = self.series_conductance()
        vs = self.parsed_voltage()

        if p_present:
            G[p, p] += gs
            Ihis[p, 0] += -gs * vs
        if n_present:
            G[n, n] += gs
            Ihis[n, 0] += gs * vs
        if p_present and n_present:
            G[p, n] += -gs
            G[n, p] += -gs

    def to_observer(self, node_order: Sequence[str]) -> tuple[sp.Matrix, sp.Expr]:
        errors = self.validate(node_order)
        if errors:
            raise ValueError("; ".join(errors))
        node_index = {node: index for index, node in enumerate(node_order)}
        c = sp.zeros(1, len(node_order))
        gs = self.series_conductance()
        if self.node_p in node_index:
            c[0, node_index[self.node_p]] += gs
        if self.node_n in node_index:
            c[0, node_index[self.node_n]] -= gs
        return c, -gs * self.parsed_voltage()


def stamp_voltage_source_series_r(
    G: sp.Matrix,
    Ihis: sp.Matrix,
    node_index: Mapping[str, int],
    element: VoltageSourceSeriesR,
) -> None:
    element.stamp(G, Ihis, node_index)


def voltage_source_series_r_to_observer(
    element: VoltageSourceSeriesR,
    node_order: Sequence[str],
) -> tuple[sp.Matrix, sp.Expr]:
    return element.to_observer(node_order)


def validate_voltage_source_series_r(
    element: VoltageSourceSeriesR,
    nodes: Sequence[str],
) -> list[str]:
    return element.validate(nodes)
