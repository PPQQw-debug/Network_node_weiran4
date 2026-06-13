from __future__ import annotations

from typing import Sequence

import sympy as sp


def _as_column_vector(vec: sp.Matrix, expected_rows: int, name: str) -> sp.Matrix:
    out = sp.Matrix(vec)
    if out.shape == (expected_rows,):
        out = sp.Matrix(expected_rows, 1, list(out))
    if out.shape != (expected_rows, 1):
        raise ValueError(f"{name} must be a {expected_rows}x1 column vector, got {out.shape}")
    return out


def _validate_partition(
    node_order: Sequence[str],
    external_nodes: Sequence[str],
    internal_nodes: Sequence[str],
) -> None:
    if len(set(node_order)) != len(node_order):
        raise ValueError("node_order must not contain duplicates")
    if len(set(external_nodes)) != len(external_nodes):
        raise ValueError("external_nodes must not contain duplicates")
    if len(set(internal_nodes)) != len(internal_nodes):
        raise ValueError("internal_nodes must not contain duplicates")

    missing_external = [node for node in external_nodes if node not in node_order]
    missing_internal = [node for node in internal_nodes if node not in node_order]
    if missing_external:
        raise ValueError(f"external_nodes contains nodes not present in node_order: {missing_external}")
    if missing_internal:
        raise ValueError(f"internal_nodes contains nodes not present in node_order: {missing_internal}")

    overlap = sorted(set(external_nodes) & set(internal_nodes))
    if overlap:
        raise ValueError(f"external_nodes and internal_nodes overlap: {overlap}")


def simplify_expr(expr: sp.Expr, simplify_level: str = "light") -> sp.Expr:
    level = (simplify_level or "light").lower()
    if level == "none":
        return expr
    if level == "full":
        return sp.simplify(expr)
    return sp.cancel(expr)


def light_simplify_matrix(M: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(M).applyfunc(sp.cancel)


def _simplify_matrix(M: sp.Matrix, simplify_level: str = "light") -> sp.Matrix:
    return sp.Matrix(M).applyfunc(lambda expr: simplify_expr(expr, simplify_level))


def check_symmetric(G: sp.Matrix) -> tuple[bool, list[tuple[int, int, sp.Expr, sp.Expr]]]:
    G = sp.Matrix(G)
    if G.rows != G.cols:
        return False, []
    mismatches: list[tuple[int, int, sp.Expr, sp.Expr]] = []
    for row in range(G.rows):
        for col in range(row + 1, G.cols):
            if sp.simplify(G[row, col] - G[col, row]) != 0:
                mismatches.append((row, col, G[row, col], G[col, row]))
    return not mismatches, mismatches


def _extract_partition(
    G: sp.Matrix,
    Ihis: sp.Matrix,
    node_order: Sequence[str],
    external_nodes: Sequence[str],
    internal_nodes: Sequence[str],
) -> tuple[sp.Matrix, sp.Matrix, list[str], list[int]]:
    node_order = list(node_order)
    external_nodes = list(external_nodes)
    internal_nodes = list(internal_nodes)
    _validate_partition(node_order, external_nodes, internal_nodes)

    G = sp.Matrix(G)
    n = len(node_order)
    if G.shape != (n, n):
        raise ValueError(f"G must have shape {(n, n)}, got {G.shape}")
    Ihis = _as_column_vector(sp.Matrix(Ihis), n, "Ihis")

    ordered_nodes = external_nodes + internal_nodes
    permutation = [node_order.index(node) for node in ordered_nodes]
    return G.extract(permutation, permutation), Ihis.extract(permutation, [0]), ordered_nodes, permutation


def sequential_symmetric_eliminate(
    G: sp.Matrix,
    Ihis: sp.Matrix,
    node_order: Sequence[str],
    external_nodes: Sequence[str],
    internal_nodes: Sequence[str],
    simplify_level: str = "light",
    reverse_internal_order: bool = True,
    assume_spd: bool = True,
) -> dict:
    """Eliminate internal nodes one pivot at a time without forming inv(Gii)."""

    external_nodes = list(external_nodes)
    internal_nodes = list(internal_nodes)
    current_G, current_Ihis, current_nodes, _ = _extract_partition(
        G, Ihis, node_order, external_nodes, internal_nodes
    )
    warnings: list[str] = []
    symmetric, mismatches = check_symmetric(current_G)
    if not symmetric:
        warnings.append("Matrix is not symmetric. SPD optimized elimination may be invalid.")

    elimination_order = list(reversed(internal_nodes)) if reverse_internal_order else list(internal_nodes)
    elimination_steps = []
    recovery_steps = []

    for node in elimination_order:
        if node not in current_nodes:
            continue
        pivot_index = current_nodes.index(node)
        keep = [index for index in range(len(current_nodes)) if index != pivot_index]
        rest_nodes = [current_nodes[index] for index in keep]
        pivot = current_G[pivot_index, pivot_index]
        if pivot == 0:
            raise ZeroDivisionError(f"zero pivot for internal node {node}")

        row = current_G.extract([pivot_index], keep)
        col = current_G.extract(keep, [pivot_index])
        kept_G = current_G.extract(keep, keep)
        kept_Ihis = current_Ihis.extract(keep, [0])
        node_Ihis = current_Ihis[pivot_index, 0]

        next_G = kept_G - (col * row) / pivot
        next_Ihis = kept_Ihis - col * node_Ihis / pivot
        if assume_spd and symmetric:
            for row_index in range(next_G.rows):
                for col_index in range(row_index + 1, next_G.cols):
                    value = simplify_expr(next_G[row_index, col_index], simplify_level)
                    next_G[row_index, col_index] = value
                    next_G[col_index, row_index] = value

        next_G = _simplify_matrix(next_G, simplify_level)
        next_Ihis = _simplify_matrix(next_Ihis, simplify_level)
        recovery_coeffs = [simplify_expr(-row[0, col_index] / pivot, simplify_level) for col_index in range(row.cols)]
        recovery_source = simplify_expr(-node_Ihis / pivot, simplify_level)

        elimination_steps.append(
            {
                "node": node,
                "pivot": pivot,
                "rest_nodes": rest_nodes,
                "G_row": [row[0, col_index] for col_index in range(row.cols)],
                "G_col": [col[row_index, 0] for row_index in range(col.rows)],
                "Ihis": node_Ihis,
            }
        )
        recovery_steps.append(
            {
                "node": node,
                "rest_nodes": rest_nodes,
                "coefficients": recovery_coeffs,
                "source": recovery_source,
            }
        )

        current_G = next_G
        current_Ihis = next_Ihis
        current_nodes = rest_nodes

    return {
        "G_red": current_G,
        "Ihis_red": current_Ihis,
        "remaining_nodes": current_nodes,
        "external_nodes": external_nodes,
        "internal_nodes": internal_nodes,
        "elimination_order": elimination_order,
        "elimination_steps": elimination_steps,
        "recovery_steps": recovery_steps,
        "warnings": warnings,
        "symmetric": symmetric,
        "asymmetric_entries": mismatches[:20],
    }


def _ccode(expr: sp.Expr) -> str:
    return sp.ccode(expr).replace("M_PI", "PI")


def analyze_internal_block_structure(Gii: sp.Matrix, internal_nodes: Sequence[str]) -> dict:
    Gii = sp.Matrix(Gii)
    internal_nodes = list(internal_nodes)
    if Gii.shape != (len(internal_nodes), len(internal_nodes)):
        raise ValueError("Gii shape must match internal_nodes")

    is_symmetric, mismatches = check_symmetric(Gii)
    warnings: list[str] = []
    if not is_symmetric:
        warnings.append("Matrix is not symmetric; symmetric block inverse display may be invalid.")

    edges: list[tuple[int, int]] = []
    for row in range(Gii.rows):
        for col in range(row + 1, Gii.cols):
            if sp.simplify(Gii[row, col]) != 0 or sp.simplify(Gii[col, row]) != 0:
                edges.append((row, col))

    if not edges:
        diagonal_nodes = list(internal_nodes)
        coupled_nodes: list[str] = []
    else:
        # Pick a small vertex cover as the coupled core S. The remaining nodes form
        # D and therefore have no mutual off-diagonal couplings inside D.
        best_cover: tuple[int, ...] | None = None
        n = len(internal_nodes)
        for mask in range(1, 1 << n):
            cover = tuple(index for index in range(n) if mask & (1 << index))
            if best_cover is not None and len(cover) >= len(best_cover):
                continue
            cover_set = set(cover)
            if all(row in cover_set or col in cover_set for row, col in edges):
                best_cover = cover
        coupled_index_set = set(best_cover or range(n))
        diagonal_nodes = [node for index, node in enumerate(internal_nodes) if index not in coupled_index_set]
        coupled_nodes = [node for index, node in enumerate(internal_nodes) if index in coupled_index_set]
        if len(diagonal_nodes) < 2:
            diagonal_nodes = []
            coupled_nodes = list(internal_nodes)

    if diagonal_nodes and coupled_nodes:
        block_type = "diagonal_plus_coupled"
    elif diagonal_nodes and not coupled_nodes:
        block_type = "pure_diagonal"
    else:
        block_type = "general"

    suggested_order = diagonal_nodes + coupled_nodes
    if suggested_order != internal_nodes:
        warnings.append("Suggested block order differs from user order.")

    return {
        "is_symmetric": is_symmetric,
        "diagonal_nodes": diagonal_nodes,
        "coupled_nodes": coupled_nodes,
        "suggested_order": suggested_order,
        "warnings": warnings,
        "block_type": block_type,
        "asymmetric_entries": mismatches[:20],
    }


def _inverse_diagonal(D: sp.Matrix) -> sp.Matrix:
    return sp.diag(*[1 / D[index, index] for index in range(D.rows)])


def _permute_internal_blocks(
    G: sp.Matrix,
    Ihis: sp.Matrix,
    node_order: Sequence[str],
    external_nodes: Sequence[str],
    internal_nodes: Sequence[str],
) -> dict:
    ordered_G, ordered_Ihis, ordered_nodes, _ = _extract_partition(G, Ihis, node_order, external_nodes, internal_nodes)
    n_e = len(external_nodes)
    n_i = len(internal_nodes)
    e_idx = list(range(n_e))
    i_idx = list(range(n_e, n_e + n_i))
    return {
        "ordered_nodes": ordered_nodes,
        "G_rr": ordered_G.extract(e_idx, e_idx),
        "G_ri": ordered_G.extract(e_idx, i_idx),
        "G_ir": ordered_G.extract(i_idx, e_idx),
        "G_ii": ordered_G.extract(i_idx, i_idx),
        "Ihis_r": ordered_Ihis.extract(e_idx, [0]),
        "Ihis_i": ordered_Ihis.extract(i_idx, [0]),
    }


def build_structured_formula(
    G: sp.Matrix,
    Ihis: sp.Matrix,
    node_order: Sequence[str],
    external_nodes: Sequence[str],
    internal_nodes: Sequence[str],
    use_suggested_order: bool = False,
    simplify_level: str = "light",
) -> dict:
    external_nodes = list(external_nodes)
    internal_nodes = list(internal_nodes)
    initial = _permute_internal_blocks(G, Ihis, node_order, external_nodes, internal_nodes)
    analysis = analyze_internal_block_structure(initial["G_ii"], internal_nodes)
    effective_internal_nodes = list(analysis["suggested_order"]) if use_suggested_order else internal_nodes
    blocks = _permute_internal_blocks(G, Ihis, node_order, external_nodes, effective_internal_nodes)
    effective_analysis = analyze_internal_block_structure(blocks["G_ii"], effective_internal_nodes)

    Gii = blocks["G_ii"]
    details: dict = {}
    block_type = effective_analysis["block_type"]
    diagonal_nodes = effective_analysis["diagonal_nodes"]
    coupled_nodes = effective_analysis["coupled_nodes"]
    node_pos = {node: index for index, node in enumerate(effective_internal_nodes)}

    if block_type == "pure_diagonal":
        D_values = [Gii[index, index] for index in range(Gii.rows)]
        W = sp.diag(*[1 / value for value in D_values])
        details = {
            "D_values": D_values,
            "W": W,
            "rank_update_terms": [
                {
                    "node": node,
                    "D": D_values[index],
                    "u": blocks["G_ri"].extract(list(range(blocks["G_ri"].rows)), [index]),
                    "H": blocks["Ihis_i"][index, 0],
                }
                for index, node in enumerate(effective_internal_nodes)
            ],
        }
    elif block_type == "diagonal_plus_coupled":
        d_idx = [node_pos[node] for node in diagonal_nodes]
        s_idx = [node_pos[node] for node in coupled_nodes]
        D = Gii.extract(d_idx, d_idx)
        U = Gii.extract(d_idx, s_idx)
        S = Gii.extract(s_idx, s_idx)
        D_inv = _inverse_diagonal(D)
        M = _simplify_matrix(S - U.T * D_inv * U, simplify_level)
        if M.shape == (2, 2):
            M11, M12, M22 = M[0, 0], M[0, 1], M[1, 1]
            detM = simplify_expr(M11 * M22 - M12**2, simplify_level)
            M_inv = sp.Matrix([[M22 / detM, -M12 / detM], [-M12 / detM, M11 / detM]])
        else:
            detM = None
            M_inv = sp.MatrixSymbol("M_inv", M.rows, M.cols)
        if isinstance(M_inv, sp.MatrixBase):
            W = sp.Matrix.vstack(
                sp.Matrix.hstack(D_inv + D_inv * U * M_inv * U.T * D_inv, -D_inv * U * M_inv),
                sp.Matrix.hstack(-M_inv * U.T * D_inv, M_inv),
            )
        else:
            W = None
        details = {
            "D": D,
            "U": U,
            "S": S,
            "D_inv": D_inv,
            "M": M,
            "M_inv": M_inv if isinstance(M_inv, sp.MatrixBase) else None,
            "detM": detM,
            "W": W,
            "diagonal_nodes": diagonal_nodes,
            "coupled_nodes": coupled_nodes,
        }
    else:
        details = {"W_symbol": "Gii^{-1}"}

    return {
        "external_nodes": external_nodes,
        "user_internal_nodes": internal_nodes,
        "effective_internal_nodes": effective_internal_nodes,
        "use_suggested_order": use_suggested_order,
        "analysis": analysis,
        "effective_analysis": effective_analysis,
        "blocks": blocks,
        "block_type": block_type,
        "details": details,
        "warnings": list(dict.fromkeys([*analysis["warnings"], *effective_analysis["warnings"]])),
    }


def cse_c_draft_for_formula_mode(G_red: sp.Matrix, Ihis_red: sp.Matrix) -> str:
    G_red = sp.Matrix(G_red)
    Ihis_red = sp.Matrix(Ihis_red)
    expressions = [G_red[row, col] for row in range(G_red.rows) for col in range(G_red.cols)]
    expressions.extend(Ihis_red[row, 0] for row in range(Ihis_red.rows))
    replacements, reduced = sp.cse(expressions, symbols=sp.numbered_symbols("t"))
    lines = ["// Formula mode C draft. Generated from explicit sequential Schur results."]
    for symbol, expr in replacements:
        lines.append(f"double {symbol} = {_ccode(expr)};")
    cursor = 0
    for row in range(G_red.rows):
        for col in range(G_red.cols):
            lines.append(f"Gred[{row}][{col}] = {_ccode(reduced[cursor])};")
            cursor += 1
    for row in range(Ihis_red.rows):
        lines.append(f"Ihis_red[{row}] = {_ccode(reduced[cursor])};")
        cursor += 1
    return "\n".join(lines)


def c_draft_for_structured_formula(structured: dict) -> str:
    block_type = structured.get("block_type")
    lines = []
    if block_type == "pure_diagonal":
        terms = structured.get("details", {}).get("rank_update_terms", [])
        for index, _term in enumerate(terms):
            lines.append(f"invD[{index}] = 1.0 / Gkk[{index}][{index}];")
        lines.extend(
            [
                "",
                "W = diag(invD);",
                "Gred = Grr - G_rk * W * G_kr;",
                "Ihis_red = Ihis_r - G_rk * W * Ihis_k;",
                "V_k = -W * G_kr * V_r - W * Ihis_k;",
            ]
        )
        return "\n".join(lines)

    if block_type == "diagonal_plus_coupled":
        details = structured.get("details", {})
        D = details.get("D")
        M = details.get("M")
        for index in range(D.rows if isinstance(D, sp.MatrixBase) else 0):
            lines.append(f"invD[{index}] = 1.0 / D[{index}][{index}];")
        if isinstance(M, sp.MatrixBase) and M.shape == (2, 2):
            lines.extend(
                [
                    "",
                    "Dinv = diag(invD);",
                    "M = S - transpose(U) * Dinv * U;",
                    "detM = M[0][0] * M[1][1] - M[0][1] * M[0][1];",
                    "Minv = (1.0 / detM) * [[M[1][1], -M[0][1]], [-M[0][1], M[0][0]]];",
                    "",
                    "W = [[Dinv + Dinv*U*Minv*transpose(U)*Dinv, -Dinv*U*Minv],",
                    "     [-Minv*transpose(U)*Dinv,              Minv]];",
                    "Gred = Grr - G_rk * W * G_kr;",
                    "Ihis_red = Ihis_r - G_rk * W * Ihis_k;",
                    "V_k = -W * G_kr * V_r - W * Ihis_k;",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Dinv = diag(invD);",
                    "M = S - transpose(U) * Dinv * U;",
                    "Minv = inverse(M);",
                    "W = block_inverse_from(Dinv, U, Minv);",
                    "Gred = Grr - G_rk * W * G_kr;",
                    "Ihis_red = Ihis_r - G_rk * W * Ihis_k;",
                    "V_k = -W * G_kr * V_r - W * Ihis_k;",
                ]
            )
        return "\n".join(lines)

    lines = [
        "",
        "W = inverse(Gkk);",
        "Gred = Grr - G_rk * W * G_kr;",
        "Ihis_red = Ihis_r - G_rk * W * Ihis_k;",
        "V_k = -W * G_kr * V_r - W * Ihis_k;",
    ]
    return "\n".join(lines)
