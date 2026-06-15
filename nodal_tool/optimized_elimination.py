from __future__ import annotations

import re
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
    code = sp.ccode(expr).replace("M_PI", "PI")
    return re.sub(r"(?<![eE][+-])(?<![\w.])(\d+)(?![\w.])", r"\1.0", code)


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


def _c_matrix_literal(matrix: sp.Matrix, name: str) -> str:
    matrix = sp.Matrix(matrix)
    if matrix.rows == 0 or matrix.cols == 0:
        return f"double {name}[1][1] = {{ {{0.0}} }};  /* Empty {matrix.rows}x{matrix.cols} block. */"
    rows = []
    for row in range(matrix.rows):
        items = ", ".join(_ccode(matrix[row, col]) for col in range(matrix.cols))
        rows.append(f"    {{{items}}}")
    return f"double {name}[{matrix.rows}][{matrix.cols}] = {{\n" + ",\n".join(rows) + "\n};"


def _c_vector_literal(vector: sp.Matrix, name: str) -> str:
    vector = _as_column_vector(sp.Matrix(vector), sp.Matrix(vector).rows, name)
    if vector.rows == 0:
        return f"double {name}[1][1] = {{ {{0.0}} }};  /* Empty 0x1 block. */"
    rows = [f"    {{{_ccode(vector[row, 0])}}}" for row in range(vector.rows)]
    return f"double {name}[{vector.rows}][1] = {{\n" + ",\n".join(rows) + "\n};"


def _c_zero_matrix(name: str, rows: int, cols: int) -> str:
    if rows == 0 or cols == 0:
        return f"double {name}[1][1] = {{ {{0.0}} }};  /* Empty {rows}x{cols} workspace. */"
    return f"double {name}[{rows}][{cols}] = {{0.0}};"


def _c_copy_subblock(dst: str, src: str, row_offset: int, col_offset: int, rows: int, cols: int) -> list[str]:
    if rows == 0 or cols == 0:
        return []
    lines = []
    for row in range(rows):
        for col in range(cols):
            lines.append(f"{dst}[{row + row_offset}][{col + col_offset}] = {src}[{row}][{col}];")
    return lines


def _c_sym_inverse_call(matrix_name: str, inverse_name: str, dim: int) -> list[str]:
    if dim <= 0:
        return []
    if dim == 1:
        return [f"{inverse_name}[0][0] = 1.0 / {matrix_name}[0][0];"]
    if dim == 2:
        return [
            f"mat_2x2_sym_inv_code({matrix_name}[0][0], {matrix_name}[0][1], {matrix_name}[1][1],",
            f"                     &{inverse_name}[0][0], &{inverse_name}[0][1], &{inverse_name}[1][1]);",
            f"{inverse_name}[1][0] = {inverse_name}[0][1];",
        ]
    if dim == 3:
        return [
            f"mat_3x3_sym_inv_code({matrix_name}[0][0], {matrix_name}[0][1], {matrix_name}[0][2],",
            f"                     {matrix_name}[1][1], {matrix_name}[1][2],",
            f"                     {matrix_name}[2][2],",
            f"                     &{inverse_name}[0][0], &{inverse_name}[0][1], &{inverse_name}[0][2],",
            f"                     &{inverse_name}[1][1], &{inverse_name}[1][2],",
            f"                     &{inverse_name}[2][2]);",
            f"{inverse_name}[1][0] = {inverse_name}[0][1];",
            f"{inverse_name}[2][0] = {inverse_name}[0][2];",
            f"{inverse_name}[2][1] = {inverse_name}[1][2];",
        ]
    return [
        f"/* WARNING: {matrix_name} is {dim}x{dim}; RTDS fast symmetric inverse helpers only cover 2x2 and 3x3. */",
        f"MATH_matx_invert({dim}, &({matrix_name}[0][0]), {dim}, &({inverse_name}[0][0]), {dim});",
    ]


def _c_manual_transpose_assignments(src: str, dst: str, rows: int, cols: int) -> list[str]:
    lines = []
    for row in range(cols):
        for col in range(rows):
            lines.append(f"{dst}[{row}][{col}] = {src}[{col}][{row}];")
    return lines


def _c_symbol_name(name: str) -> str:
    suffix = re.sub(r"\W", "_", str(name or "n"))
    if not suffix or suffix[0].isdigit():
        suffix = f"n_{suffix}"
    return suffix


def _c_voltage_variable_name(node: str, node_display_names: dict[str, str] | None = None) -> str:
    display = (node_display_names or {}).get(str(node), str(node))
    suffix = _c_symbol_name(display)
    return f"V{suffix}"


def _c_node_symbol_vector(name: str, nodes: Sequence[str], node_display_names: dict[str, str] | None = None) -> str:
    nodes = list(nodes)
    if not nodes:
        return f"double {name}[1][1] = {{ {{0.0}} }};  /* Empty symbolic node vector. */"
    rows = []
    for node in nodes:
        display = (node_display_names or {}).get(str(node), str(node))
        rows.append(f"    {{{_c_symbol_name(display)}}}")
    return f"double {name}[{len(nodes)}][1] = {{\n" + ",\n".join(rows) + "\n};"


def _c_emit_rtds_reduction_tail(
    nr: int,
    nk: int,
    external_nodes: Sequence[str],
    internal_nodes: Sequence[str],
    node_display_names: dict[str, str] | None = None,
) -> list[str]:
    return [
        "",
        "/* Final Schur complement:",
        "   Gred    = Grr - Grk * W * Gkr",
        "   Ihisred = Ihisr - Grk * W * Ihisk",
        "   Vk      = -W * Gkr * Vr - W * Ihisk */",
        _c_zero_matrix("tmp_Grk_W", nr, nk),
        _c_zero_matrix("tmp_Grk_W_Gkr", nr, nr),
        _c_zero_matrix("tmp_Grk_W_Ihisk", nr, 1),
        _c_zero_matrix("Gred", nr, nr),
        _c_zero_matrix("Ihisred", nr, 1),
        "matrix_Mul(NR, NK, NK, tmp_Grk_W, Grk, W);",
        "matrix_Mul(NR, NK, NR, tmp_Grk_W_Gkr, tmp_Grk_W, Gkr);",
        "matrix_Sub(NR, NR, Gred, Grr, tmp_Grk_W_Gkr);",
        "matrix_Mul(NR, NK, 1, tmp_Grk_W_Ihisk, tmp_Grk_W, Ihisk);",
        "matrix_Sub(NR, 1, Ihisred, Ihisr, tmp_Grk_W_Ihisk);",
        "",
        "/* Internal-node voltage recovery. Vr/Vk start as symbolic node-name placeholders. */",
        _c_node_symbol_vector("Vr", external_nodes, node_display_names),
        _c_zero_matrix("tmp_W_Gkr", nk, nr),
        _c_zero_matrix("tmp_W_Gkr_Vr", nk, 1),
        _c_zero_matrix("tmp_W_Ihisk", nk, 1),
        _c_zero_matrix("tmp_Vk_sum", nk, 1),
        _c_node_symbol_vector("Vk", internal_nodes, node_display_names),
        "matrix_Mul(NK, NK, NR, tmp_W_Gkr, W, Gkr);",
        "matrix_Mul(NK, NR, 1, tmp_W_Gkr_Vr, tmp_W_Gkr, Vr);",
        "matrix_Mul(NK, NK, 1, tmp_W_Ihisk, W, Ihisk);",
        "matrix_Add(NK, 1, tmp_Vk_sum, tmp_W_Gkr_Vr, tmp_W_Ihisk);",
        "matrix_Scale(NK, 1, -1.0, Vk, tmp_Vk_sum);",
        "",
        "/* One variable per eliminated node, in effective k order. */",
        *[
            f"double {_c_voltage_variable_name(node, node_display_names)} = Vk[{index}][0];"
            for index, node in enumerate(internal_nodes)
        ],
    ]


def c_draft_for_structured_formula(structured: dict, node_display_names: dict[str, str] | None = None) -> str:
    block_type = structured.get("block_type")
    blocks = structured.get("blocks", {})
    Grr = sp.Matrix(blocks.get("G_rr", []))
    Grk = sp.Matrix(blocks.get("G_ri", []))
    Gkr = sp.Matrix(blocks.get("G_ir", []))
    Gkk = sp.Matrix(blocks.get("G_ii", []))
    Ihisr = sp.Matrix(blocks.get("Ihis_r", []))
    Ihisk = sp.Matrix(blocks.get("Ihis_i", []))
    nr = Grr.rows
    nk = Gkk.rows
    external_nodes = list(structured.get("external_nodes", []))
    effective_internal_nodes = list(structured.get("effective_internal_nodes", []))
    node_display_names = {str(key): str(value) for key, value in (node_display_names or {}).items()}

    lines = [
        "/* RTDS-style C draft for structured node elimination.",
        "   Required math helpers: matrix_Add, matrix_Sub, matrix_Mul, matrix_Scale,",
        "   matrix_Copy, MATH_matx_invert, mat_2x2_sym_inv_code,",
        "   mat_3x3_sym_inv_code. See LOCAL_math_builtin_functions.md. */",
        f"enum {{ NR = {nr}, NK = {nk} }};",
        "",
        "/* Input blocks: I = G * V + Ihis, partitioned as r = retained, k = eliminated. */",
        _c_matrix_literal(Grr, "Grr"),
        _c_matrix_literal(Grk, "Grk"),
        _c_matrix_literal(Gkr, "Gkr"),
        _c_matrix_literal(Gkk, "Gkk"),
        _c_vector_literal(Ihisr, "Ihisr"),
        _c_vector_literal(Ihisk, "Ihisk"),
        "",
    ]

    if block_type == "pure_diagonal":
        lines.extend([
            "/* Gkk is diagonal here. Build inv_D directly; do not call a matrix inverse. */",
            _c_zero_matrix("D", nk, nk),
            _c_zero_matrix("inv_D", nk, nk),
            _c_zero_matrix("W", nk, nk),
            "matrix_Copy(NK, NK, D, Gkk);",
        ])
        for index in range(nk):
            lines.append(f"inv_D[{index}][{index}] = 1.0 / D[{index}][{index}];")
        lines.append("matrix_Copy(NK, NK, W, inv_D);")
        lines.extend(_c_emit_rtds_reduction_tail(nr, nk, external_nodes, effective_internal_nodes, node_display_names))
        return "\n".join(lines)

    if block_type == "diagonal_plus_coupled":
        details = structured.get("details", {})
        D = sp.Matrix(details.get("D", []))
        U = sp.Matrix(details.get("U", []))
        S = sp.Matrix(details.get("S", []))
        kd = D.rows
        ks = S.rows
        lines.extend([
            f"enum {{ KD = {kd}, KS = {ks} }};",
            "",
            "/* Gkk = [[D, U], [U^T, S]]. D is diagonal; build inv_D directly. */",
            _c_matrix_literal(D, "D"),
            _c_matrix_literal(U, "U"),
            _c_matrix_literal(S, "S"),
            _c_zero_matrix("inv_D", kd, kd),
        ])
        for index in range(kd):
            lines.append(f"inv_D[{index}][{index}] = 1.0 / D[{index}][{index}];")
        lines.extend([
            "",
            "/* Build M = S - U^T * inv_D * U using RTDS matrix helpers. */",
            _c_zero_matrix("U_T", ks, kd),
            *_c_manual_transpose_assignments("U", "U_T", kd, ks),
            _c_zero_matrix("tmp_UT_invD", ks, kd),
            _c_zero_matrix("tmp_UT_invD_U", ks, ks),
            _c_zero_matrix("M", ks, ks),
            "matrix_Mul(KS, KD, KD, tmp_UT_invD, U_T, inv_D);",
            "matrix_Mul(KS, KD, KS, tmp_UT_invD_U, tmp_UT_invD, U);",
            "matrix_Sub(KS, KS, M, S, tmp_UT_invD_U);",
            "",
            "/* Invert M. Use reciprocal for 1x1, symmetric fast inverse for 2x2/3x3. */",
            _c_zero_matrix("M_inv", ks, ks),
            *_c_sym_inverse_call("M", "M_inv", ks),
            "",
            "/* Build W = inverse(Gkk) from D, U, S block inverse terms. */",
            _c_zero_matrix("tmp_Dinv_U", kd, ks),
            _c_zero_matrix("tmp_Dinv_U_Minv", kd, ks),
            _c_zero_matrix("tmp_Dinv_U_Minv_UT", kd, kd),
            _c_zero_matrix("tmp_Dinv_U_Minv_UT_Dinv", kd, kd),
            _c_zero_matrix("W_DD", kd, kd),
            _c_zero_matrix("W_DS", kd, ks),
            _c_zero_matrix("tmp_Minv_UT", ks, kd),
            _c_zero_matrix("W_SD", ks, kd),
            _c_zero_matrix("W_SS", ks, ks),
            _c_zero_matrix("W", nk, nk),
            "matrix_Mul(KD, KD, KS, tmp_Dinv_U, inv_D, U);",
            "matrix_Mul(KD, KS, KS, tmp_Dinv_U_Minv, tmp_Dinv_U, M_inv);",
            "matrix_Mul(KD, KS, KD, tmp_Dinv_U_Minv_UT, tmp_Dinv_U_Minv, U_T);",
            "matrix_Mul(KD, KD, KD, tmp_Dinv_U_Minv_UT_Dinv, tmp_Dinv_U_Minv_UT, inv_D);",
            "matrix_Add(KD, KD, W_DD, inv_D, tmp_Dinv_U_Minv_UT_Dinv);",
            "matrix_Scale(KD, KS, -1.0, W_DS, tmp_Dinv_U_Minv);",
            "matrix_Mul(KS, KS, KD, tmp_Minv_UT, M_inv, U_T);",
            "matrix_Mul(KS, KD, KD, W_SD, tmp_Minv_UT, inv_D);",
            "matrix_Scale(KS, KD, -1.0, W_SD, W_SD);",
            "matrix_Copy(KS, KS, W_SS, M_inv);",
            *_c_copy_subblock("W", "W_DD", 0, 0, kd, kd),
            *_c_copy_subblock("W", "W_DS", 0, kd, kd, ks),
            *_c_copy_subblock("W", "W_SD", kd, 0, ks, kd),
            *_c_copy_subblock("W", "W_SS", kd, kd, ks, ks),
        ])
        lines.extend(_c_emit_rtds_reduction_tail(nr, nk, external_nodes, effective_internal_nodes, node_display_names))
        return "\n".join(lines)

    lines = [
        *lines,
        "/* General dense Gkk. Prefer the structured block modes above when possible. */",
        _c_zero_matrix("W", nk, nk),
        *(
            [f"/* WARNING: Gkk is {nk}x{nk}; this uses the general inverse routine. */"]
            if nk > 3
            else []
        ),
        f"MATH_matx_invert(NK, &(Gkk[0][0]), NK, &(W[0][0]), NK);",
        *_c_emit_rtds_reduction_tail(nr, nk, external_nodes, effective_internal_nodes, node_display_names),
    ]
    return "\n".join(lines)
