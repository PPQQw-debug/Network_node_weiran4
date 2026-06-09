import unittest

import sympy as sp

from nodal_tool.ground import apply_ground_constraint
from nodal_tool.voltage_sources import (
    VoltageSourceSeriesR,
    stamp_voltage_source_series_r,
    validate_voltage_source_series_r,
)


def assert_matrix_equal(testcase, actual, expected):
    actual = sp.Matrix(actual)
    expected = sp.Matrix(expected)
    testcase.assertEqual(actual.shape, expected.shape)
    for value, target in zip(actual, expected):
        testcase.assertEqual(sp.simplify(value - target), 0)


class VoltageSourceSeriesRTests(unittest.TestCase):
    def _stamp(self, nodes, element):
        G = sp.zeros(len(nodes), len(nodes))
        Ihis = sp.zeros(len(nodes), 1)
        stamp_voltage_source_series_r(G, Ihis, {node: index for index, node in enumerate(nodes)}, element)
        return G, Ihis

    def test_two_node_voltage_source(self):
        Vs, Gs = sp.symbols("Vs Gs")
        nodes = ["A", "B"]
        source = VoltageSourceSeriesR("V1", "A", "B", Vs, Gs)

        G, Ihis = self._stamp(nodes, source)

        assert_matrix_equal(self, G, sp.Matrix([[Gs, -Gs], [-Gs, Gs]]))
        assert_matrix_equal(self, Ihis, sp.Matrix([[-Gs * Vs], [Gs * Vs]]))

    def test_voltage_source_to_ground_full_and_ground_eliminated(self):
        Vs, Gs = sp.symbols("Vs Gs")
        source = VoltageSourceSeriesR("V1", "A", "GND", Vs, Gs)

        G_full, Ihis_full = self._stamp(["A", "GND"], source)

        assert_matrix_equal(self, G_full, sp.Matrix([[Gs, -Gs], [-Gs, Gs]]))
        assert_matrix_equal(self, Ihis_full, sp.Matrix([[-Gs * Vs], [Gs * Vs]]))

        grounded = apply_ground_constraint(G_full, Ihis_full, ["A", "GND"], ["GND"])
        assert_matrix_equal(self, grounded.G_ng, sp.Matrix([[Gs]]))
        assert_matrix_equal(self, grounded.Ihis_ng, sp.Matrix([[-Gs * Vs]]))

        G_ng, Ihis_ng = self._stamp(["A"], source)
        assert_matrix_equal(self, G_ng, sp.Matrix([[Gs]]))
        assert_matrix_equal(self, Ihis_ng, sp.Matrix([[-Gs * Vs]]))

    def test_parallel_with_passive_resistor(self):
        Vs, Gs, R = sp.symbols("Vs Gs R")
        nodes = ["A", "B"]
        G, Ihis = self._stamp(nodes, VoltageSourceSeriesR("V1", "A", "B", Vs, Gs))
        g_resistor = 1 / R
        G[0, 0] += g_resistor
        G[0, 1] += -g_resistor
        G[1, 0] += -g_resistor
        G[1, 1] += g_resistor

        total_g = Gs + 1 / R
        assert_matrix_equal(self, G, sp.Matrix([[total_g, -total_g], [-total_g, total_g]]))
        assert_matrix_equal(self, Ihis, sp.Matrix([[-Gs * Vs], [Gs * Vs]]))

    def test_rejects_zero_negative_or_missing_series_conductance(self):
        for conductance in ["0", "-1", ""]:
            source = VoltageSourceSeriesR("Vbad", "A", "B", "Vs", conductance)
            errors = validate_voltage_source_series_r(source, ["A", "B"])
            self.assertTrue(errors)
            with self.assertRaises(ValueError):
                self._stamp(["A", "B"], source)


if __name__ == "__main__":
    unittest.main()
