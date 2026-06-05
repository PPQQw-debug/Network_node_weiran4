from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import sympy as sp


def _symbol_locals(text: str) -> dict[str, sp.Symbol]:
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", text or ""))
    reserved = {"sin", "cos", "tan", "exp", "log", "sqrt", "Abs"}
    return {name: sp.Symbol(name) for name in names if name not in reserved}


def parse_expr(text: str | sp.Expr) -> sp.Expr:
    if isinstance(text, sp.Expr):
        return text
    value = str(text or "0").strip() or "0"
    return sp.sympify(value, locals=_symbol_locals(value))


def parse_control_terms(input_text: str | Mapping[str, str | sp.Expr] | None) -> dict[str, sp.Expr]:
    if input_text is None:
        return {}
    if isinstance(input_text, Mapping):
        return {str(node).strip(): parse_expr(value) for node, value in input_text.items() if str(node).strip()}

    text = str(input_text).strip()
    if not text:
        return {}
    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("control_terms JSON must be an object")
        return parse_control_terms(parsed)

    terms: dict[str, sp.Expr] = {}
    matches = list(re.finditer(r"([A-Za-z_]\w*)\s*:", text))
    if matches:
        raw_items = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            raw_items.append(f"{match.group(1)}:{text[match.end():end]}")
    else:
        raw_items = re.split(r"[\n,;]+", text)

    for raw_item in raw_items:
        item = raw_item.strip().strip(",;")
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Control term must use node: coefficient format: {item}")
        node, coefficient = item.split(":", 1)
        node = node.strip()
        if not node:
            raise ValueError("Control term node name cannot be empty")
        terms[node] = parse_expr(coefficient)
    return terms


@dataclass(frozen=True)
class ControlledCurrentSource:
    """Linear voltage-controlled current source.

    Ordinary two-terminal admittance branch:
      i = G * (V_from - V_to) + Ihis
      stamps symmetric columns at the branch endpoints.

    Controlled current source:
      i = sum(k_j * V_j) + Ihis
      stamps output rows and control-node columns, so the resulting matrix can
      be non-symmetric and can have positive off-diagonal entries.
    """

    name: str
    out_from: str
    out_to: str
    control_terms: Mapping[str, str | sp.Expr]
    Ihis: str | sp.Expr = sp.Integer(0)
    description: str = ""

    def parsed_terms(self) -> dict[str, sp.Expr]:
        return parse_control_terms(self.control_terms)

    def parsed_ihis(self) -> sp.Expr:
        return parse_expr(self.Ihis)

    def validate(self, nodes: Sequence[str]) -> list[str]:
        node_set = set(nodes)
        errors: list[str] = []
        if self.out_from not in node_set:
            errors.append(f"Controlled source output node {self.out_from} not present in node list.")
        if self.out_to not in node_set:
            errors.append(f"Controlled source output node {self.out_to} not present in node list.")
        try:
            terms = self.parsed_terms()
        except Exception as exc:
            errors.append(f"Controlled source control_terms could not be parsed: {exc}")
            terms = {}
        for node in terms:
            if node not in node_set:
                errors.append(f"Controlled source uses control node {node} not present in node list.")
        try:
            self.parsed_ihis()
        except Exception as exc:
            errors.append(f"Controlled source Ihis could not be parsed: {exc}")
        return errors

    def stamp(self, G: sp.Matrix, Ihis: sp.Matrix, node_index: Mapping[str, int]) -> None:
        errors = self.validate(list(node_index.keys()))
        if errors:
            raise ValueError("; ".join(errors))
        out_from = node_index[self.out_from]
        out_to = node_index[self.out_to]
        for node, coefficient in self.parsed_terms().items():
            col = node_index[node]
            G[out_from, col] += coefficient
            G[out_to, col] -= coefficient
        h = self.parsed_ihis()
        Ihis[out_from, 0] += h
        Ihis[out_to, 0] -= h

    def to_observer(self, node_order: Sequence[str]) -> tuple[sp.Matrix, sp.Expr]:
        errors = self.validate(node_order)
        if errors:
            raise ValueError("; ".join(errors))
        node_index = {node: index for index, node in enumerate(node_order)}
        C = sp.zeros(1, len(node_order))
        for node, coefficient in self.parsed_terms().items():
            C[0, node_index[node]] += coefficient
        return C, self.parsed_ihis()


def stamp_controlled_current_source(
    G: sp.Matrix,
    Ihis: sp.Matrix,
    node_index: Mapping[str, int],
    source: ControlledCurrentSource,
) -> None:
    source.stamp(G, Ihis, node_index)


def controlled_current_source_to_observer(
    source: ControlledCurrentSource,
    node_order: Sequence[str],
) -> tuple[sp.Matrix, sp.Expr]:
    return source.to_observer(node_order)


def validate_controlled_current_source(
    source: ControlledCurrentSource,
    nodes: Sequence[str],
) -> list[str]:
    return source.validate(nodes)
