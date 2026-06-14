import unittest

import sympy as sp

from nodal_tool.optimized_elimination import (
    analyze_internal_block_structure,
    build_structured_formula,
    c_draft_for_structured_formula,
)


def assert_matrix_equal(testcase, actual, expected):
    actual = sp.Matrix(actual)
    expected = sp.Matrix(expected)
    testcase.assertEqual(actual.shape, expected.shape)
    for value, target in zip(actual, expected):
        testcase.assertEqual(sp.simplify(value - target), 0)


class StructuredFormulaEliminationTests(unittest.TestCase):
    def test_pure_diagonal_internal_block(self):
        Dx, Dy, Dz = sp.symbols("Dx Dy Dz")
        Gii = sp.diag(Dx, Dy, Dz)

        analysis = analyze_internal_block_structure(Gii, ["x", "y", "z"])
        structured = build_structured_formula(Gii, sp.zeros(3, 1), ["x", "y", "z"], [], ["x", "y", "z"])

        self.assertEqual(analysis["block_type"], "pure_diagonal")
        assert_matrix_equal(self, structured["details"]["W"], sp.diag(1 / Dx, 1 / Dy, 1 / Dz))
        self.assertEqual(len(structured["details"]["rank_update_terms"]), 3)

    def test_diagonal_plus_2x2_coupled_block(self):
        Dx, Dy, Dz, AP, AN, BP, BN, CP, CN, PP, PN, NN = sp.symbols("Dx Dy Dz AP AN BP BN CP CN PP PN NN")
        Gii = sp.Matrix(
            [
                [Dx, 0, 0, AP, AN],
                [0, Dy, 0, BP, BN],
                [0, 0, Dz, CP, CN],
                [AP, BP, CP, PP, PN],
                [AN, BN, CN, PN, NN],
            ]
        )

        analysis = analyze_internal_block_structure(Gii, ["x", "y", "z", "P", "N"])
        structured = build_structured_formula(Gii, sp.zeros(5, 1), ["x", "y", "z", "P", "N"], [], ["x", "y", "z", "P", "N"])
        details = structured["details"]

        self.assertEqual(analysis["block_type"], "diagonal_plus_coupled")
        self.assertEqual(analysis["diagonal_nodes"], ["x", "y", "z"])
        self.assertEqual(analysis["coupled_nodes"], ["P", "N"])

        assert_matrix_equal(self, details["D"], sp.diag(Dx, Dy, Dz))
        assert_matrix_equal(self, details["U"], sp.Matrix([[AP, AN], [BP, BN], [CP, CN]]))
        assert_matrix_equal(self, details["S"], sp.Matrix([[PP, PN], [PN, NN]]))

        M11 = PP - (AP**2 / Dx + BP**2 / Dy + CP**2 / Dz)
        M12 = PN - (AP * AN / Dx + BP * BN / Dy + CP * CN / Dz)
        M22 = NN - (AN**2 / Dx + BN**2 / Dy + CN**2 / Dz)
        assert_matrix_equal(self, details["M"], sp.Matrix([[M11, M12], [M12, M22]]))

        detM = M11 * M22 - M12**2
        assert_matrix_equal(self, details["M_inv"], sp.Matrix([[M22 / detM, -M12 / detM], [-M12 / detM, M11 / detM]]))
        assert_matrix_equal(self, details["W"], Gii.inv())

        draft = c_draft_for_structured_formula(structured)
        self.assertIn("double Grr", draft)
        self.assertIn("double Grk", draft)
        self.assertIn("double Gkr", draft)
        self.assertIn("double Gkk", draft)
        self.assertIn("double Ihisr", draft)
        self.assertIn("double Ihisk", draft)
        self.assertNotIn("matrix_transpose_CODE", draft)
        self.assertIn("U_T[0][0] = U[0][0];", draft)
        self.assertIn("U_T[0][1] = U[1][0];", draft)
        self.assertIn("U_T[1][2] = U[2][1];", draft)
        self.assertIn("matrix_Mul", draft)
        self.assertIn("matrix_Sub", draft)
        self.assertIn("mat_2x2_sym_inv_code", draft)
        self.assertIn("Gred", draft)
        self.assertIn("Ihisred", draft)
        self.assertIn("Vk", draft)
        self.assertIn("double Vx = Vk[0][0];", draft)
        self.assertIn("double Vy = Vk[1][0];", draft)
        self.assertIn("double Vz = Vk[2][0];", draft)
        self.assertIn("double VP = Vk[3][0];", draft)
        self.assertIn("double VN = Vk[4][0];", draft)
        self.assertIn("{0.0}", draft)

    def test_suggests_order_when_user_order_hides_blocks(self):
        Dx, Dy, Dz, AP, AN, BP, BN, CP, CN, PP, PN, NN = sp.symbols("Dx Dy Dz AP AN BP BN CP CN PP PN NN")
        natural = ["x", "y", "z", "P", "N"]
        user_order = ["P", "N", "x", "y", "z"]
        G_natural = sp.Matrix(
            [
                [Dx, 0, 0, AP, AN],
                [0, Dy, 0, BP, BN],
                [0, 0, Dz, CP, CN],
                [AP, BP, CP, PP, PN],
                [AN, BN, CN, PN, NN],
            ]
        )
        permutation = [natural.index(node) for node in user_order]
        G_user = G_natural.extract(permutation, permutation)

        analysis = analyze_internal_block_structure(G_user, user_order)

        self.assertEqual(analysis["suggested_order"], ["x", "y", "z", "P", "N"])
        self.assertIn("Suggested block order differs from user order.", analysis["warnings"])

    def test_non_symmetric_warning(self):
        Gii = sp.Matrix([[1, 2], [3, 4]])

        analysis = analyze_internal_block_structure(Gii, ["x", "y"])

        self.assertFalse(analysis["is_symmetric"])
        self.assertIn("Matrix is not symmetric; symmetric block inverse display may be invalid.", analysis["warnings"])

    def test_general_dense_block(self):
        a, b, c, d, e, f = sp.symbols("a b c d e f")
        Gii = sp.Matrix([[a, b, c], [b, d, e], [c, e, f]])

        analysis = analyze_internal_block_structure(Gii, ["x", "y", "z"])

        self.assertEqual(analysis["block_type"], "general")
        self.assertEqual(analysis["diagonal_nodes"], [])
        self.assertEqual(analysis["coupled_nodes"], ["x", "y", "z"])

    def test_c_draft_uses_1x1_m_reciprocal(self):
        Dx, Dy, AP, BP, PP = sp.symbols("Dx Dy AP BP PP")
        Gii = sp.Matrix(
            [
                [Dx, 0, AP],
                [0, Dy, BP],
                [AP, BP, PP],
            ]
        )
        structured = build_structured_formula(Gii, sp.zeros(3, 1), ["x", "y", "P"], [], ["x", "y", "P"])

        draft = c_draft_for_structured_formula(structured)

        self.assertIn("M_inv[0][0] = 1.0 / M[0][0];", draft)

    def test_c_draft_uses_display_names_for_voltage_recovery(self):
        G1, G2 = sp.symbols("G1 G2")
        nodes = ["gA", "gB", "gX"]
        G = sp.Matrix([[G1, 0, -G1], [0, G2, -G2], [-G1, -G2, G1 + G2]])
        structured = build_structured_formula(G, sp.zeros(3, 1), nodes, ["gA", "gB"], ["gX"])

        draft = c_draft_for_structured_formula(
            structured,
            node_display_names={"gA": "A", "gB": "B", "gX": "x"},
        )

        self.assertIn("double Vr[2][1] = {\n    {A},\n    {B}\n};", draft)
        self.assertIn("double Vk[1][1] = {\n    {x}\n};", draft)
        self.assertIn("double Vx = Vk[0][0];", draft)
        self.assertNotIn("VgX", draft)

    def test_c_draft_uses_3x3_symmetric_inverse(self):
        D1, D2 = sp.symbols("D1 D2")
        symbols = sp.symbols("u11 u12 u13 u21 u22 u23 s11 s12 s13 s22 s23 s33")
        u11, u12, u13, u21, u22, u23, s11, s12, s13, s22, s23, s33 = symbols
        Gii = sp.Matrix(
            [
                [D1, 0, u11, u12, u13],
                [0, D2, u21, u22, u23],
                [u11, u21, s11, s12, s13],
                [u12, u22, s12, s22, s23],
                [u13, u23, s13, s23, s33],
            ]
        )
        structured = build_structured_formula(Gii, sp.zeros(5, 1), ["d1", "d2", "a", "b", "c"], [], ["d1", "d2", "a", "b", "c"])

        draft = c_draft_for_structured_formula(structured)

        self.assertIn("mat_3x3_sym_inv_code", draft)

    def test_c_draft_warns_for_large_m_inverse(self):
        diagonal = sp.diag(*sp.symbols("D1 D2"))
        U = sp.Matrix(sp.symbols("u0:8")).reshape(2, 4)
        S_symbols = sp.symbols("s0:10")
        S = sp.Matrix(
            [
                [S_symbols[0], S_symbols[1], S_symbols[2], S_symbols[3]],
                [S_symbols[1], S_symbols[4], S_symbols[5], S_symbols[6]],
                [S_symbols[2], S_symbols[5], S_symbols[7], S_symbols[8]],
                [S_symbols[3], S_symbols[6], S_symbols[8], S_symbols[9]],
            ]
        )
        Gii = sp.Matrix.vstack(sp.Matrix.hstack(diagonal, U), sp.Matrix.hstack(U.T, S))
        nodes = ["d1", "d2", "a", "b", "c", "e"]
        structured = build_structured_formula(Gii, sp.zeros(6, 1), nodes, [], nodes)

        draft = c_draft_for_structured_formula(structured)

        self.assertIn("WARNING: M is 4x4", draft)
        self.assertIn("MATH_matx_invert(4", draft)


if __name__ == "__main__":
    unittest.main()
