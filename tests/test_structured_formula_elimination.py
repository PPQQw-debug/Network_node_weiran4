import unittest

import sympy as sp

from nodal_tool.optimized_elimination import analyze_internal_block_structure, build_structured_formula


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


if __name__ == "__main__":
    unittest.main()
