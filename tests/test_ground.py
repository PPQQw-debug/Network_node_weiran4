import unittest

import sympy as sp

from elimination import eliminate_internal_nodes
from nodal_tool.controlled_sources import ControlledCurrentSource, stamp_controlled_current_source
from nodal_tool.ground import (
    apply_ground_constraint,
    restrict_observer_to_remaining_nodes,
    validate_ground_partition,
)
from observers import reduce_observer


def assert_matrix_equal(testcase, actual, expected):
    actual = sp.Matrix(actual)
    expected = sp.Matrix(expected)
    testcase.assertEqual(actual.shape, expected.shape)
    testcase.assertEqual(sp.simplify(actual - expected), sp.zeros(*actual.shape))


class GroundConstraintTests(unittest.TestCase):
    def test_single_branch_to_ground(self):
        Gr, h = sp.symbols("Gr h")
        result = apply_ground_constraint(
            sp.Matrix([[Gr, -Gr], [-Gr, Gr]]),
            sp.Matrix([[h], [-h]]),
            ["A", "GND"],
            ["GND"],
        )

        self.assertEqual(result.remaining_nodes, ["A"])
        assert_matrix_equal(self, result.G_ng, sp.Matrix([[Gr]]))
        assert_matrix_equal(self, result.Ihis_ng, sp.Matrix([[h]]))
        self.assertEqual(result.ground_voltage_map, {"GND": sp.Integer(0)})

    def test_reverse_branch_to_ground(self):
        Gr, h = sp.symbols("Gr h")
        result = apply_ground_constraint(
            sp.Matrix([[Gr, -Gr], [-Gr, Gr]]),
            sp.Matrix([[-h], [h]]),
            ["A", "GND"],
            ["GND"],
        )

        self.assertEqual(result.remaining_nodes, ["A"])
        assert_matrix_equal(self, result.G_ng, sp.Matrix([[Gr]]))
        assert_matrix_equal(self, result.Ihis_ng, sp.Matrix([[-h]]))

    def test_two_nodes_with_ground_branches(self):
        G1, G2, G3, h2, h3 = sp.symbols("G1 G2 G3 h2 h3")
        G = sp.Matrix(
            [
                [G1 + G2, -G1, -G2],
                [-G1, G1 + G3, -G3],
                [-G2, -G3, G2 + G3],
            ]
        )
        Ihis = sp.Matrix([[h2], [h3], [-h2 - h3]])

        result = apply_ground_constraint(G, Ihis, ["A", "B", "GND"], ["GND"])

        self.assertEqual(result.remaining_nodes, ["A", "B"])
        assert_matrix_equal(self, result.G_ng, sp.Matrix([[G1 + G2, -G1], [-G1, G1 + G3]]))
        assert_matrix_equal(self, result.Ihis_ng, sp.Matrix([[h2], [h3]]))

    def test_ground_before_internal_elimination(self):
        G1, G2 = sp.symbols("G1 G2")
        G_full = sp.Matrix(
            [
                [G1, -G1, 0],
                [-G1, G1 + G2, -G2],
                [0, -G2, G2],
            ]
        )
        Ihis_full = sp.zeros(3, 1)

        grounded = apply_ground_constraint(G_full, Ihis_full, ["A", "X", "GND"], ["GND"])
        result = eliminate_internal_nodes(grounded.G_ng, grounded.Ihis_ng, grounded.remaining_nodes, ["A"])

        assert_matrix_equal(self, result.G_red, sp.Matrix([[G1 * G2 / (G1 + G2)]]))
        V_A = sp.symbols("V_A")
        V_X = sp.simplify((result.K_v * sp.Matrix([[V_A]]) + result.K_h)[0, 0])
        self.assertEqual(sp.simplify(V_X - G1 * V_A / (G1 + G2)), 0)

    def test_ground_cannot_be_external_or_internal(self):
        result = validate_ground_partition(["A", "GND"], ["A", "GND"], ["GND"], ["GND"])

        self.assertEqual(result.external_nodes, ["A"])
        self.assertEqual(result.internal_nodes, [])
        self.assertEqual(result.ground_nodes, ["GND"])
        self.assertTrue(any("cannot be external" in item for item in result.warnings))
        self.assertTrue(any("cannot be internal" in item for item in result.warnings))

    def test_observer_substitutes_ground(self):
        Gr, h, V_A = sp.symbols("Gr h V_A")
        nodes = ["A", "GND"]

        C_forward, d_forward, remaining = restrict_observer_to_remaining_nodes(
            sp.Matrix([[Gr, -Gr]]),
            h,
            nodes,
            {"GND": sp.Integer(0)},
        )
        self.assertEqual(remaining, ["A"])
        self.assertEqual(sp.simplify((C_forward * sp.Matrix([[V_A]]))[0] + d_forward - (Gr * V_A + h)), 0)

        C_reverse, d_reverse, _ = restrict_observer_to_remaining_nodes(
            sp.Matrix([[-Gr, Gr]]),
            h,
            nodes,
            {"GND": sp.Integer(0)},
        )
        self.assertEqual(sp.simplify((C_reverse * sp.Matrix([[V_A]]))[0] + d_reverse - (-Gr * V_A + h)), 0)

    def test_controlled_current_source_control_term_can_be_grounded(self):
        K, h = sp.symbols("K h")
        nodes = ["A", "GND", "C", "D"]
        source = ControlledCurrentSource("GmGnd", "C", "D", {"A": K, "GND": -K}, h)
        G = sp.zeros(len(nodes), len(nodes))
        Ihis = sp.zeros(len(nodes), 1)
        stamp_controlled_current_source(G, Ihis, {node: index for index, node in enumerate(nodes)}, source)

        grounded = apply_ground_constraint(G, Ihis, nodes, ["GND"])

        self.assertEqual(grounded.remaining_nodes, ["A", "C", "D"])
        expected = sp.zeros(3, 3)
        expected[1, 0] = K
        expected[2, 0] = -K
        assert_matrix_equal(self, grounded.G_ng, expected)
        assert_matrix_equal(self, grounded.Ihis_ng, sp.Matrix([[0], [h], [-h]]))


if __name__ == "__main__":
    unittest.main()
