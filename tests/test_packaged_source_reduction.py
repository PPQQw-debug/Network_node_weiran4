import unittest

import sympy as sp

from elimination import eliminate_internal_nodes


def assert_matrix_equal(testcase, actual, expected):
    actual = sp.Matrix(actual)
    expected = sp.Matrix(expected)
    testcase.assertEqual(actual.shape, expected.shape)
    for value, target in zip(actual, expected):
        testcase.assertEqual(sp.simplify(value - target), 0)


class PackagedSourceReductionTests(unittest.TestCase):
    def test_packaged_independent_current_source_preserves_ihis(self):
        I = sp.Symbol("I10")
        nodes = ["P1", "P2"]
        G_full = sp.zeros(2, 2)
        Ihis_full = sp.Matrix([[I], [-I]])

        result = eliminate_internal_nodes(G_full, Ihis_full, nodes, nodes)

        assert_matrix_equal(self, result.G_red, sp.zeros(2, 2))
        assert_matrix_equal(self, result.Ihis_red, sp.Matrix([[I], [-I]]))

    def test_packaged_voltage_source_series_g_preserves_stamp(self):
        Gs, Vs = sp.symbols("G Vs")
        nodes = ["P1", "P2"]
        G_full = sp.Matrix([[Gs, -Gs], [-Gs, Gs]])
        Ihis_full = sp.Matrix([[-Gs * Vs], [Gs * Vs]])

        result = eliminate_internal_nodes(G_full, Ihis_full, nodes, nodes)

        assert_matrix_equal(self, result.G_red, G_full)
        assert_matrix_equal(self, result.Ihis_red, Ihis_full)


if __name__ == "__main__":
    unittest.main()
