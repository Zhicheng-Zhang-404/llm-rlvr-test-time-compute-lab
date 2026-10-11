import unittest

from ttc_rlvr.pass_at_k import estimate_pass_at_k, mean_pass_at_k


class PassAtKTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(estimate_pass_at_k(16, 0, 1), 0.0)
        self.assertEqual(estimate_pass_at_k(16, 1, 16), 1.0)
        self.assertAlmostEqual(estimate_pass_at_k(16, 1, 1), 1 / 16)
        self.assertAlmostEqual(estimate_pass_at_k(4, 2, 2), 5 / 6)

    def test_monotonicity(self):
        values = [estimate_pass_at_k(16, 3, k) for k in [1, 2, 4, 8, 16]]
        self.assertEqual(values, sorted(values))

    def test_average(self):
        result = mean_pass_at_k([[True, False], [False, False]], [1, 2])
        self.assertAlmostEqual(result[1], 0.25)
        self.assertAlmostEqual(result[2], 0.5)

    def test_invalid_inputs(self):
        for args in [(0, 0, 1), (4, 5, 1), (4, -1, 1), (4, 1, 0), (4, 1, 5)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                estimate_pass_at_k(*args)


if __name__ == "__main__":
    unittest.main()

