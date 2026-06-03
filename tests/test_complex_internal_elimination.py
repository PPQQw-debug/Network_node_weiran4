import time
import unittest

import sympy as sp

from elimination import eliminate_internal_nodes


class ComplexInternalEliminationTests(unittest.TestCase):
    def test_ybox_ucm_with_pn_internal_finishes_quickly(self):
        symbols = sp.symbols(
            "AA AN AP BB BN BP CC CN CP G_rc G11 G12 G22 gC "
            "IAhis_rc IBhis_rc IC_his1 IChis_rc Ihis_a Ihis_A Ihis_b Ihis_B "
            "Ihis_c Ihis_C Ihis_exclude_c0 Ihis_exclude_c1 Ihis_exclude_c2 "
            "Ihis_exclude_c3 Ihis_exclude_c4 NN PN PP w1 w2"
        )
        (
            AA,
            AN,
            AP,
            BB,
            BN,
            BP,
            CC,
            CN,
            CP,
            G_rc,
            G11,
            G12,
            G22,
            gC,
            IAhis_rc,
            IBhis_rc,
            IC_his1,
            IChis_rc,
            Ihis_a,
            Ihis_A,
            Ihis_b,
            Ihis_B,
            Ihis_c,
            Ihis_C,
            Ihis_exclude_c0,
            Ihis_exclude_c1,
            Ihis_exclude_c2,
            Ihis_exclude_c3,
            Ihis_exclude_c4,
            NN,
            PN,
            PP,
            w1,
            w2,
        ) = symbols

        all_nodes = ["A", "B", "C", "G", "RC_a", "RC_b", "RC_c", "x", "y", "z", "P", "N"]
        external_nodes = ["A", "B", "C", "G", "RC_a", "RC_b", "RC_c"]
        G_full = sp.Matrix(
            [
                [2 * G11 + 2 * w1, -G11 - w1, -G11 - w1, 0, 0, 0, 0, G12, 0, -G12, 0, 0],
                [-G11 - w1, 2 * G11 + 2 * w1, -G11 - w1, 0, 0, 0, 0, -G12, G12, 0, 0, 0],
                [-G11 - w1, -G11 - w1, 2 * G11 + 2 * w1, 0, 0, 0, 0, 0, -G12, G12, 0, 0],
                [0, 0, 0, 3 * G22 + 3 * w2, 0, 0, 0, -G22 - w2, -G22 - w2, -G22 - w2, 0, 0],
                [0, 0, 0, 0, G_rc, 0, 0, -G_rc, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, G_rc, 0, 0, -G_rc, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, G_rc, 0, 0, -G_rc, 0, 0],
                [G12, -G12, 0, -G22 - w2, -G_rc, 0, 0, AA + G22 + G_rc + w2, 0, 0, AP, AN],
                [0, G12, -G12, -G22 - w2, 0, -G_rc, 0, 0, BB + G22 + G_rc + w2, 0, BP, BN],
                [-G12, 0, G12, -G22 - w2, 0, 0, -G_rc, 0, 0, CC + G22 + G_rc + w2, CP, CN],
                [0, 0, 0, 0, 0, 0, 0, AP, BP, CP, PP + gC, PN - gC],
                [0, 0, 0, 0, 0, 0, 0, AN, BN, CN, PN - gC, NN + gC],
            ]
        )
        Ihis_full = sp.Matrix(
            [
                [Ihis_A - Ihis_C],
                [-Ihis_A + Ihis_B],
                [-Ihis_B + Ihis_C],
                [-Ihis_a - Ihis_b - Ihis_c],
                [-IAhis_rc],
                [-IBhis_rc],
                [-IChis_rc],
                [Ihis_exclude_c0 + Ihis_a + IAhis_rc],
                [Ihis_exclude_c1 + Ihis_b + IBhis_rc],
                [Ihis_exclude_c2 + Ihis_c + IChis_rc],
                [Ihis_exclude_c3 - IC_his1],
                [Ihis_exclude_c4 + IC_his1],
            ]
        )

        start = time.monotonic()
        result = eliminate_internal_nodes(G_full, Ihis_full, all_nodes, external_nodes)

        self.assertLess(time.monotonic() - start, 10)
        self.assertEqual(result.external_nodes, external_nodes)
        self.assertEqual(result.internal_nodes, ["x", "y", "z", "P", "N"])
        self.assertEqual(result.G_red.shape, (7, 7))
        self.assertEqual(result.Ihis_red.shape, (7, 1))
        self.assertEqual(result.K_v.shape, (5, 7))
        self.assertEqual(result.K_h.shape, (5, 1))


if __name__ == "__main__":
    unittest.main()
