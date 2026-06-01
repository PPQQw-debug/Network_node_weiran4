import unittest

import sympy as sp

from elimination import eliminate_internal_nodes
from observers import reduce_observer


def assert_matrix_equal(testcase, actual, expected):
    actual = sp.Matrix(actual)
    expected = sp.Matrix(expected)
    testcase.assertEqual(actual.shape, expected.shape)
    diff = sp.simplify(actual - expected)
    testcase.assertEqual(diff, sp.zeros(*actual.shape))


class EliminationRlTests(unittest.TestCase):
    def test_rl_series_elimination_and_observers(self):
        Gr, Gl, ILhis = sp.symbols("Gr Gl ILhis")
        V_A, V_B = sp.symbols("V_A V_B")

        all_nodes = ["A", "X", "B"]
        external_nodes = ["A", "B"]

        G_full = sp.Matrix(
            [
                [Gr, -Gr, 0],
                [-Gr, Gr + Gl, -Gl],
                [0, -Gl, Gl],
            ]
        )

        Ihis_full = sp.Matrix(
            [
                [0],
                [ILhis],
                [-ILhis],
            ]
        )

        result = eliminate_internal_nodes(
            G_full=G_full,
            Ihis_full=Ihis_full,
            all_nodes=all_nodes,
            external_nodes=external_nodes,
        )

        G_expected = sp.Matrix(
            [
                [Gr * Gl / (Gr + Gl), -Gr * Gl / (Gr + Gl)],
                [-Gr * Gl / (Gr + Gl), Gr * Gl / (Gr + Gl)],
            ]
        )

        Ihis_expected = sp.Matrix(
            [
                [Gr * ILhis / (Gr + Gl)],
                [-Gr * ILhis / (Gr + Gl)],
            ]
        )

        self.assertEqual(result.external_nodes, ["A", "B"])
        self.assertEqual(result.internal_nodes, ["X"])
        assert_matrix_equal(self, result.G_red, G_expected)
        assert_matrix_equal(self, result.Ihis_red, Ihis_expected)

        V_e = sp.Matrix([[V_A], [V_B]])
        V_X_recovered = sp.simplify((result.K_v * V_e + result.K_h)[0, 0])
        V_X_expected = (Gr * V_A + Gl * V_B - ILhis) / (Gr + Gl)
        self.assertEqual(sp.simplify(V_X_recovered - V_X_expected), 0)

        # R branch current: i_R = Gr * (V_A - V_X)
        C_R_full = sp.Matrix([[Gr, -Gr, 0]])
        i_R_observer = reduce_observer(C_R_full, sp.Integer(0), all_nodes, result)
        i_R_red = i_R_observer.expression([V_A, V_B])

        # L branch current: i_L = Gl * (V_X - V_B) + ILhis
        C_L_full = sp.Matrix([[0, Gl, -Gl]])
        i_L_observer = reduce_observer(C_L_full, ILhis, all_nodes, result)
        i_L_red = i_L_observer.expression([V_A, V_B])

        i_expected = Gr * Gl / (Gr + Gl) * (V_A - V_B) + Gr / (Gr + Gl) * ILhis

        self.assertEqual(sp.simplify(i_R_red - i_expected), 0)
        self.assertEqual(sp.simplify(i_L_red - i_expected), 0)
        self.assertEqual(sp.simplify(i_R_red - i_L_red), 0)


    def test_no_internal_nodes_reorders_to_external_order(self):
        a, b, c = sp.symbols("a b c")
        G_full = sp.Matrix([[a, 0], [0, b]])
        Ihis_full = sp.Matrix([[c], [-c]])

        result = eliminate_internal_nodes(
            G_full=G_full,
            Ihis_full=Ihis_full,
            all_nodes=["A", "B"],
            external_nodes=["B", "A"],
        )

        self.assertEqual(result.internal_nodes, [])
        self.assertEqual(result.external_nodes, ["B", "A"])
        assert_matrix_equal(self, result.G_red, sp.Matrix([[b, 0], [0, a]]))
        assert_matrix_equal(self, result.Ihis_red, sp.Matrix([[-c], [c]]))


if __name__ == "__main__":
    unittest.main()
