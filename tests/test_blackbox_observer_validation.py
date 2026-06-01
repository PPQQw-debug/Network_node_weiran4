import unittest

import sympy as sp

from elimination import eliminate_internal_nodes
from nodal_tool.blackbox_validation import (
    BlackBoxBranchObserver,
    build_observer_stamp,
    validate_blackbox_observers,
)
from observers import reduce_observer


def assert_matrix_equal(testcase, actual, expected):
    actual = sp.Matrix(actual)
    expected = sp.Matrix(expected)
    testcase.assertEqual(actual.shape, expected.shape)
    for r in range(actual.rows):
        for c in range(actual.cols):
            testcase.assertEqual(sp.simplify(actual[r, c] - expected[r, c]), 0)


class BlackBoxObserverValidationTests(unittest.TestCase):
    def test_no_observers_is_floating(self):
        G1, G2 = sp.symbols("G1 G2")
        nodes = ["A", "B"]
        G_bb = sp.Matrix([[G1 + G2, -(G1 + G2)], [-(G1 + G2), G1 + G2]])
        Ihis_bb = sp.Matrix([0, 0])

        result = validate_blackbox_observers(G_bb, Ihis_bb, nodes, [])

        self.assertEqual(result.status, "floating")
        self.assertEqual(result.observer_rows, [])
        self.assertIn("unobservable/floating", result.messages[0])

    def test_parallel_partial_observer_is_mismatch(self):
        G1, G2, h1, h2 = sp.symbols("G1 G2 h1 h2")
        nodes = ["A", "B"]
        G_bb = sp.Matrix([[G1 + G2, -(G1 + G2)], [-(G1 + G2), G1 + G2]])
        Ihis_bb = sp.Matrix([h1 + h2, -(h1 + h2)])
        observers = [BlackBoxBranchObserver("B1", "A", "B", G1, h1)]

        result = validate_blackbox_observers(G_bb, Ihis_bb, nodes, observers)

        self.assertEqual(result.status, "mismatch")
        assert_matrix_equal(self, result.G_obs, sp.Matrix([[G1, -G1], [-G1, G1]]))
        assert_matrix_equal(self, result.Ihis_obs, sp.Matrix([h1, -h1]))
        assert_matrix_equal(self, result.Delta_G, sp.Matrix([[G2, -G2], [-G2, G2]]))
        assert_matrix_equal(self, result.Delta_Ihis, sp.Matrix([h2, -h2]))
        self.assertEqual(sp.simplify(result.observer_rows[0].expression - (G1 * (sp.Symbol("V_A") - sp.Symbol("V_B")) + h1)), 0)

    def test_parallel_complete_observers_are_matched(self):
        G1, G2, h1, h2 = sp.symbols("G1 G2 h1 h2")
        nodes = ["A", "B"]
        G_bb = sp.Matrix([[G1 + G2, -(G1 + G2)], [-(G1 + G2), G1 + G2]])
        Ihis_bb = sp.Matrix([h1 + h2, -(h1 + h2)])
        observers = [
            BlackBoxBranchObserver("B1", "A", "B", G1, h1),
            BlackBoxBranchObserver("B2", "A", "B", G2, h2),
        ]

        result = validate_blackbox_observers(G_bb, Ihis_bb, nodes, observers)

        self.assertEqual(result.status, "matched")
        assert_matrix_equal(self, result.G_obs, G_bb)
        assert_matrix_equal(self, result.Ihis_obs, Ihis_bb)
        assert_matrix_equal(self, result.Delta_G, sp.zeros(2, 2))
        assert_matrix_equal(self, result.Delta_Ihis, sp.zeros(2, 1))

    def test_unknown_symbol_warns(self):
        G1, G2, h1 = sp.symbols("G1 G2 h1")
        nodes = ["A", "B"]
        G_bb = sp.Matrix([[G1, -G1], [-G1, G1]])
        Ihis_bb = sp.Matrix([h1, -h1])
        observers = [BlackBoxBranchObserver("B1", "A", "B", G2, h1)]

        result = validate_blackbox_observers(G_bb, Ihis_bb, nodes, observers)

        self.assertEqual(result.status, "mismatch")
        self.assertIn(G2, result.unknown_symbols)
        self.assertTrue(any("{G2}" in warning for warning in result.warnings))

    def test_missing_node_is_not_silent(self):
        G1 = sp.Symbol("G1")
        nodes = ["A", "B"]
        observer = BlackBoxBranchObserver("B1", "A", "C", G1, 0)

        with self.assertRaisesRegex(ValueError, "Observer B1 uses node C not present"):
            build_observer_stamp(nodes, [observer])

    def test_observer_reduction_uses_standard_formula(self):
        G1, G2 = sp.symbols("G1 G2")
        nodes = ["A", "X", "B"]
        G_bb = sp.Matrix([[G1, -G1, 0], [-G1, G1 + G2, -G2], [0, -G2, G2]])
        Ihis_bb = sp.Matrix([0, 0, 0])
        observer = BlackBoxBranchObserver("B1", "A", "X", G1, 0)
        validation = validate_blackbox_observers(G_bb, Ihis_bb, nodes, [observer])

        elimination = eliminate_internal_nodes(G_bb, Ihis_bb, nodes, ["A", "B"])
        reduced = reduce_observer(validation.observer_rows[0].C, validation.observer_rows[0].d, nodes, elimination)
        V_A, V_B = sp.symbols("V_A V_B")
        i_red = reduced.expression([V_A, V_B])
        expected = G1 * G2 / (G1 + G2) * (V_A - V_B)

        self.assertEqual(sp.simplify(elimination.K_v[0, 0] - G1 / (G1 + G2)), 0)
        self.assertEqual(sp.simplify(elimination.K_v[0, 1] - G2 / (G1 + G2)), 0)
        self.assertEqual(sp.simplify(i_red - expected), 0)


if __name__ == "__main__":
    unittest.main()
