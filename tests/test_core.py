import math
import unittest

from spherical_bessel_functions_kit import spherical_jn, spherical_yn


class TestSphericalJn(unittest.TestCase):
    def test_j0_at_zero(self):
        self.assertEqual(spherical_jn(0, 0.0), 1.0)

    def test_jn_at_zero_for_n_positive(self):
        for n in range(1, 6):
            self.assertEqual(spherical_jn(n, 0.0), 0.0)

    def test_j0_known_value(self):
        # j_0(1) = sin(1)/1
        self.assertAlmostEqual(spherical_jn(0, 1.0), math.sin(1.0), places=14)

    def test_j1_known_value(self):
        # j_1(1) = sin(1)/1^2 - cos(1)/1
        expected = math.sin(1.0) - math.cos(1.0)
        self.assertAlmostEqual(spherical_jn(1, 1.0), expected, places=14)

    def test_j2_known_value(self):
        # j_2(x) = (3/x^3 - 1/x) sin x - 3 cos x / x^2
        x = 2.0
        expected = (3.0 / x**3 - 1.0 / x) * math.sin(x) - 3.0 * math.cos(x) / x**2
        self.assertAlmostEqual(spherical_jn(2, x), expected, places=14)

    def test_negative_x_parity(self):
        # j_n(-x) = (-1)^n j_n(x)
        for n in range(0, 5):
            sign = -1.0 if (n % 2) else 1.0
            self.assertAlmostEqual(spherical_jn(n, -2.0), sign * spherical_jn(n, 2.0), places=14)

    def test_small_x_series(self):
        # For tiny x the series path should agree with the recurrence limit.
        x = 1e-12
        self.assertAlmostEqual(spherical_jn(0, x), 1.0, places=14)
        self.assertAlmostEqual(spherical_jn(1, x), x / 3.0, places=14)
        self.assertAlmostEqual(spherical_jn(2, x), x * x / 15.0, places=14)

    def test_negative_order_raises(self):
        with self.assertRaises(ValueError):
            spherical_jn(-1, 1.0)


class TestSphericalYn(unittest.TestCase):
    def test_y0_known_value(self):
        # y_0(x) = -cos(x)/x
        self.assertAlmostEqual(spherical_yn(0, 1.0), -math.cos(1.0), places=14)

    def test_y1_known_value(self):
        # y_1(x) = -cos(x)/x^2 - sin(x)/x
        x = 1.5
        expected = -math.cos(x) / (x * x) - math.sin(x) / x
        self.assertAlmostEqual(spherical_yn(1, x), expected, places=14)

    def test_y2_known_value(self):
        # y_2(x) = (-3/x^3 + 1/x) cos x - 3 sin x / x^2
        x = 2.0
        expected = (-3.0 / x**3 + 1.0 / x) * math.cos(x) - 3.0 * math.sin(x) / x**2
        self.assertAlmostEqual(spherical_yn(2, x), expected, places=14)

    def test_non_positive_x_raises(self):
        with self.assertRaises(ValueError):
            spherical_yn(0, 0.0)
        with self.assertRaises(ValueError):
            spherical_yn(0, -1.0)

    def test_negative_order_raises(self):
        with self.assertRaises(ValueError):
            spherical_yn(-1, 1.0)

    def test_recurrence_consistency(self):
        # Check the recurrence relation f_{n+1} = (2n+1)/x f_n - f_{n-1}
        x = 3.0
        for n in range(1, 6):
            lhs = spherical_yn(n + 1, x)
            rhs = (2 * n + 1) / x * spherical_yn(n, x) - spherical_yn(n - 1, x)
            self.assertAlmostEqual(lhs, rhs, places=12)


if __name__ == "__main__":
    unittest.main()
