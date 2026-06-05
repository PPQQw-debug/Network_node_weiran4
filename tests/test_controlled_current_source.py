import unittest

import sympy as sp

from elimination import eliminate_internal_nodes
from nodal_tool.controlled_sources import (
    ControlledCurrentSource,
    controlled_current_source_to_observer,
    parse_control_terms,
    stamp_controlled_current_source,
    validate_controlled_current_source,
)
from observers import reduce_observer


def assert_matrix_equal(testcase, actual, expected):
    actual = sp.Matrix(actual)
    expected = sp.Matrix(expected)
    testcase.assertEqual(actual.shape, expected.shape)
    testcase.assertEqual(sp.simplify(actual - expected), sp.zeros(*actual.shape))


class ControlledCurrentSourceTests(unittest.TestCase):
    def _stamp(self, nodes, source):
        G = sp.zeros(len(nodes), len(nodes))
        Ihis = sp.zeros(len(nodes), 1)
        stamp_controlled_current_source(G, Ihis, {node: index for index, node in enumerate(nodes)}, source)
        return G, Ihis

    def test_single_control_voltage_difference(self):
        K, h = sp.symbols("K h")
        nodes = ["A", "B", "C", "D"]
        source = ControlledCurrentSource("Gm1", "C", "D", {"A": K, "B": -K}, h)

        G, Ihis = self._stamp(nodes, source)

        assert_matrix_equal(self, G, sp.Matrix([[0, 0, 0, 0], [0, 0, 0, 0], [K, -K, 0, 0], [-K, K, 0, 0]]))
        assert_matrix_equal(self, Ihis, sp.Matrix([[0], [0], [h], [-h]]))

        C, d = controlled_current_source_to_observer(source, nodes)
        V_A, V_B, V_C, V_D = sp.symbols("V_A V_B V_C V_D")
        self.assertEqual(sp.simplify((C * sp.Matrix([V_A, V_B, V_C, V_D]))[0] + d - (K * V_A - K * V_B + h)), 0)

    def test_multiple_control_voltage_differences(self):
        K1, K2, h = sp.symbols("K1 K2 h")
        nodes = ["A", "B", "E", "F", "C", "D"]
        source = ControlledCurrentSource(
            "Gm2",
            "C",
            "D",
            {"A": K1, "B": -K1, "E": K2, "F": -K2},
            h,
        )

        G, Ihis = self._stamp(nodes, source)

        expected = sp.zeros(6, 6)
        expected[4, 0] = K1
        expected[4, 1] = -K1
        expected[4, 2] = K2
        expected[4, 3] = -K2
        expected[5, 0] = -K1
        expected[5, 1] = K1
        expected[5, 2] = -K2
        expected[5, 3] = K2
        assert_matrix_equal(self, G, expected)
        assert_matrix_equal(self, Ihis, sp.Matrix([[0], [0], [0], [0], [h], [-h]]))

    def test_control_nodes_overlap_output_nodes_matches_passive_branch(self):
        K, h = sp.symbols("K h")
        nodes = ["A", "B"]
        source = ControlledCurrentSource("Gm3", "A", "B", {"A": K, "B": -K}, h)

        G, Ihis = self._stamp(nodes, source)

        assert_matrix_equal(self, G, sp.Matrix([[K, -K], [-K, K]]))
        assert_matrix_equal(self, Ihis, sp.Matrix([[h], [-h]]))

    def test_asymmetric_matrix_is_allowed(self):
        K = sp.symbols("K")
        nodes = ["A", "B", "C", "D"]
        source = ControlledCurrentSource("Gm4", "C", "D", {"A": K}, 0)

        G, Ihis = self._stamp(nodes, source)

        expected = sp.zeros(4, 4)
        expected[2, 0] = K
        expected[3, 0] = -K
        assert_matrix_equal(self, G, expected)
        assert_matrix_equal(self, Ihis, sp.zeros(4, 1))
        self.assertNotEqual(G, G.T)

    def test_internal_node_elimination_and_observer_reduction(self):
        G1, G2, K, h = sp.symbols("G1 G2 K h")
        V_A, V_B, V_C, V_D = sp.symbols("V_A V_B V_C V_D")
        nodes = ["A", "X", "B", "C", "D"]
        G = sp.Matrix(
            [
                [G1, -G1, 0, 0, 0],
                [-G1, G1 + G2, -G2, 0, 0],
                [0, -G2, G2, 0, 0],
                [0, K, 0, 0, 0],
                [0, -K, 0, 0, 0],
            ]
        )
        Ihis = sp.Matrix([[0], [0], [0], [h], [-h]])

        result = eliminate_internal_nodes(G, Ihis, nodes, ["A", "B", "C", "D"])

        self.assertEqual(result.internal_nodes, ["X"])
        self.assertEqual(sp.simplify(result.G_red[2, 0] - K * G1 / (G1 + G2)), 0)
        self.assertEqual(sp.simplify(result.G_red[2, 1] - K * G2 / (G1 + G2)), 0)
        self.assertEqual(sp.simplify(result.G_red[3, 0] + K * G1 / (G1 + G2)), 0)
        self.assertEqual(sp.simplify(result.G_red[3, 1] + K * G2 / (G1 + G2)), 0)

        source = ControlledCurrentSource("Gm5", "C", "D", {"X": K}, h)
        C, d = controlled_current_source_to_observer(source, nodes)
        reduced = reduce_observer(C, d, nodes, result)
        expected_i = K * (G1 * V_A + G2 * V_B) / (G1 + G2) + h
        self.assertEqual(sp.simplify(reduced.expression([V_A, V_B, V_C, V_D]) - expected_i), 0)

    def test_control_node_error_is_reported(self):
        K = sp.symbols("K")
        source = ControlledCurrentSource("Gm6", "C", "D", {"Z": K}, 0)

        errors = validate_controlled_current_source(source, ["A", "B", "C", "D"])

        self.assertIn("Controlled source uses control node Z not present in node list.", errors)

    def test_output_node_error_is_reported(self):
        K = sp.symbols("K")
        source = ControlledCurrentSource("Gm7", "C", "D", {"A": K}, 0)

        errors = validate_controlled_current_source(source, ["A", "B", "C"])

        self.assertIn("Controlled source output node D not present in node list.", errors)

    def test_parse_control_terms_text_and_json(self):
        K1, K2 = sp.symbols("K1 K2")

        self.assertEqual(parse_control_terms("A: K1\nB: -K1"), {"A": K1, "B": -K1})
        self.assertEqual(parse_control_terms("A: K1 B: -K1"), {"A": K1, "B": -K1})
        self.assertEqual(parse_control_terms('{"E": "K2", "F": "-K2"}'), {"E": K2, "F": -K2})


if __name__ == "__main__":
    unittest.main()
