import unittest

import sympy as sp

from blackbox_validation_api import _parse_expr as parse_blackbox_expr
from reduce_api import _parse_expr as parse_reduction_expr


class ApiSymbolParsingTests(unittest.TestCase):
    def test_common_circuit_names_override_sympy_reserved_names(self):
        for parse_expr in (parse_blackbox_expr, parse_reduction_expr):
            expr = parse_expr("AA + BB + CC + NN")
            AA, BB, CC, NN = sp.symbols("AA BB CC NN")

            self.assertEqual(sp.simplify(expr - (AA + BB + CC + NN)), 0)
            self.assertIn(CC, expr.free_symbols)


if __name__ == "__main__":
    unittest.main()
