import math
import unittest

import numpy as np

from market_condition_report import forward_curve, forward_volatility


class ForwardVolatilityTests(unittest.TestCase):
    def test_flat_curve_produces_flat_forward_curve(self):
        result = forward_curve(np.array([20.0, 20.0, 20.0, 20.0]))
        np.testing.assert_allclose(result, [20.0, 20.0, 20.0, 20.0])

    def test_forward_volatility_recomposes_total_variance(self):
        forward = forward_volatility(15.0, 9, 18.0, 30)
        recomposed = (9 * 15.0**2 + 21 * forward**2) / 30
        self.assertAlmostEqual(recomposed, 18.0**2)

    def test_negative_forward_variance_is_unavailable(self):
        self.assertTrue(math.isnan(forward_volatility(40.0, 9, 15.0, 30)))


if __name__ == "__main__":
    unittest.main()
