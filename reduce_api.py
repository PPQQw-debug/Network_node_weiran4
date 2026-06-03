from __future__ import annotations

import json
import re
import sys

import sympy as sp

from elimination import eliminate_internal_nodes
from nodal_tool.blackbox_validation import BlackBoxBranchObserver, validate_blackbox_observers


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_SYMPY_FUNCTIONS = {
    "Abs",
    "acos",
    "asin",
    "atan",
    "cos",
    "cosh",
    "exp",
    "log",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
}


def _symbol_locals(text: str) -> dict[str, sp.Symbol]:
    names = set(_IDENTIFIER_RE.findall(text))
    return {name: sp.Symbol(name) for name in names if name not in _SYMPY_FUNCTIONS}


def _parse_expr(text: str) -> sp.Expr:
    cleaned = str(text or "0").strip()
    if not cleaned:
        return sp.Integer(0)
    return sp.sympify(cleaned, locals=_symbol_locals(cleaned))


def _clean_expr(expr: sp.Expr) -> str:
    return str(expr)


def _clean_observer_expr(expr: sp.Expr) -> str:
    return str(expr)


def _clean_matrix(matrix: sp.Matrix) -> list[list[str]]:
    return [[_clean_expr(matrix[r, c]) for c in range(matrix.cols)] for r in range(matrix.rows)]


def _clean_vector(matrix: sp.Matrix) -> list[str]:
    return [_clean_expr(matrix[r, 0]) for r in range(matrix.rows)]


def _voltage_symbol(name: str) -> sp.Symbol:
    return sp.Symbol(f"V_{name}")


def _blackbox_observer_from_payload(item: dict) -> BlackBoxBranchObserver:
    return BlackBoxBranchObserver(
        name=str(item.get("name") or "observer"),
        from_node=str(item.get("from_node") or item.get("from") or ""),
        to_node=str(item.get("to_node") or item.get("to") or ""),
        G=_parse_expr(item.get("G") or item.get("g") or "0"),
        Ihis=_parse_expr(item.get("Ihis") or item.get("ihis") or "0"),
        description=str(item.get("description") or ""),
    )


def _blackbox_validation_response(payload: dict) -> dict:
    nodes = [str(node) for node in payload["nodes"]]
    G_bb = sp.Matrix([[_parse_expr(item) for item in row] for row in payload["G_bb"]])
    Ihis_bb = sp.Matrix([[_parse_expr(item)] for item in payload["Ihis_bb"]])
    observers = [_blackbox_observer_from_payload(item) for item in payload.get("observers", [])]
    result = validate_blackbox_observers(G_bb, Ihis_bb, nodes, observers)
    return {
        "ok": True,
        "status": result.status,
        "G_obs": _clean_matrix(result.G_obs),
        "Ihis_obs": _clean_vector(result.Ihis_obs),
        "Delta_G": _clean_matrix(result.Delta_G),
        "Delta_Ihis": _clean_vector(result.Delta_Ihis),
        "warnings": result.warnings,
        "messages": result.messages,
        "unknown_symbols": sorted(str(symbol) for symbol in result.unknown_symbols),
        "observer_rows": [
            {
                "name": row.name,
                "from_node": row.from_node,
                "to_node": row.to_node,
                "C": _clean_matrix(row.C),
                "d": _clean_expr(row.d),
                "expression": _clean_observer_expr(row.expression),
                "description": row.description,
            }
            for row in result.observer_rows
        ],
    }


def main() -> None:
    payload = json.load(sys.stdin)
    if payload.get("mode") == "validate_blackbox_observers":
        json.dump(_blackbox_validation_response(payload), sys.stdout, ensure_ascii=False)
        return

    all_nodes = list(payload["all_nodes"])
    external_nodes = list(payload["external_nodes"])
    voltage_nodes = list(payload.get("voltage_nodes", all_nodes))

    G_full = sp.Matrix([[_parse_expr(item) for item in row] for row in payload["G_full"]])
    Ihis_full = sp.Matrix([[_parse_expr(item)] for item in payload["Ihis_full"]])

    result = eliminate_internal_nodes(G_full, Ihis_full, all_nodes, external_nodes)
    voltage_by_node = dict(zip(all_nodes, voltage_nodes))
    V_e = sp.Matrix([[_voltage_symbol(voltage_by_node[node])] for node in result.external_nodes])
    recovered = result.K_v * V_e + result.K_h
    substitutions = {
        _voltage_symbol(voltage_by_node[node]): recovered[index, 0]
        for index, node in enumerate(result.internal_nodes)
    }

    reduced_observers = []
    for observer in payload.get("observers", []):
        expr = _parse_expr(observer.get("rhs", "0"))
        reduced_observers.append(
            {
                "lhs": observer.get("lhs", ""),
                "rhs": _clean_observer_expr(expr.subs(substitutions)),
            }
        )

    json.dump(
        {
            "ok": True,
            "external_nodes": result.external_nodes,
            "internal_nodes": result.internal_nodes,
            "G_red": _clean_matrix(result.G_red),
            "Ihis_red": _clean_vector(result.Ihis_red),
            "K_v": _clean_matrix(result.K_v),
            "K_h": _clean_vector(result.K_h),
            "reduced_observers": reduced_observers,
        },
        sys.stdout,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        json.dump({"ok": False, "error": str(exc)}, sys.stdout, ensure_ascii=False)
        sys.exit(1)
