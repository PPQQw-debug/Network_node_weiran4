import unittest

import sympy as sp

from elimination import eliminate_internal_nodes
from nodal_tool.optimized_elimination import (
    check_symmetric,
    cse_c_draft_for_formula_mode,
    sequential_symmetric_eliminate,
)


def assert_matrix_equal(testcase, actual, expected):
    actual = sp.Matrix(actual)
    expected = sp.Matrix(expected)
    testcase.assertEqual(actual.shape, expected.shape)
    for value, target in zip(actual, expected):
        testcase.assertEqual(sp.simplify(value - target), 0)


class OptimizedEliminationTests(unittest.TestCase):
    def test_two_conductances_in_series(self):
        G1, G2 = sp.symbols("G1 G2")
        nodes = ["A", "X", "B"]
        G = sp.Matrix([[G1, -G1, 0], [-G1, G1 + G2, -G2], [0, -G2, G2]])
        Ihis = sp.zeros(3, 1)

        result = sequential_symmetric_eliminate(G, Ihis, nodes, ["A", "B"], ["X"])

        G_eq = G1 * G2 / (G1 + G2)
        assert_matrix_equal(self, result["G_red"], sp.Matrix([[G_eq, -G_eq], [-G_eq, G_eq]]))
        assert_matrix_equal(self, result["Ihis_red"], sp.zeros(2, 1))
        recovery = result["recovery_steps"][0]
        self.assertEqual(recovery["node"], "X")
        assert_matrix_equal(self, sp.Matrix([recovery["coefficients"]]), sp.Matrix([[G1 / (G1 + G2), G2 / (G1 + G2)]]))

    def test_two_internal_nodes_in_reverse_order(self):
        G1, G2, G3 = sp.symbols("G1 G2 G3")
        nodes = ["A", "X", "Y", "B"]
        G = sp.Matrix(
            [
                [G1, -G1, 0, 0],
                [-G1, G1 + G2, -G2, 0],
                [0, -G2, G2 + G3, -G3],
                [0, 0, -G3, G3],
            ]
        )
        result = sequential_symmetric_eliminate(G, sp.zeros(4, 1), nodes, ["A", "B"], ["X", "Y"])

        self.assertEqual(result["elimination_order"], ["Y", "X"])
        G_eq = G1 * G2 * G3 / (G1 * G2 + G1 * G3 + G2 * G3)
        assert_matrix_equal(self, result["G_red"], sp.Matrix([[G_eq, -G_eq], [-G_eq, G_eq]]))

    def test_matches_existing_schur_result_on_small_symmetric_network(self):
        G1, G2, G3, h = sp.symbols("G1 G2 G3 h")
        nodes = ["A", "X", "B"]
        G = sp.Matrix([[G1 + G3, -G1, -G3], [-G1, G1 + G2, -G2], [-G3, -G2, G2 + G3]])
        Ihis = sp.Matrix([[h], [0], [-h]])

        existing = eliminate_internal_nodes(G, Ihis, nodes, ["A", "B"])
        optimized = sequential_symmetric_eliminate(G, Ihis, nodes, ["A", "B"], ["X"])

        assert_matrix_equal(self, optimized["G_red"], existing.G_red)
        assert_matrix_equal(self, optimized["Ihis_red"], existing.Ihis_red)

    def test_non_symmetric_warning(self):
        G = sp.Matrix([[1, 2], [3, 4]])

        symmetric, mismatches = check_symmetric(G)

        self.assertFalse(symmetric)
        self.assertEqual(mismatches[0][:2], (0, 1))
        result = sequential_symmetric_eliminate(G, sp.zeros(2, 1), ["A", "X"], ["A"], ["X"])
        self.assertIn("Matrix is not symmetric. SPD optimized elimination may be invalid.", result["warnings"])

    def test_formula_c_draft_uses_cse_temporaries(self):
        G1, G2 = sp.symbols("G1 G2")
        common = G1 * G2 / (G1 + G2)
        draft = cse_c_draft_for_formula_mode(sp.Matrix([[common, -common], [-common, common]]), sp.zeros(2, 1))

        self.assertIn("double t", draft)
        self.assertIn("Gred[0][0]", draft)
        self.assertLess(draft.count("G1 + G2"), 2)


if __name__ == "__main__":
    unittest.main()
