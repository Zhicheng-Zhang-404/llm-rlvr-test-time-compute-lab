import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REWARD_PATH = Path(__file__).parents[1] / "verl_reward" / "math_reward.py"
SPEC = importlib.util.spec_from_file_location("math_reward", REWARD_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DummyLatexConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class RewardContractTests(unittest.TestCase):
    def _math_verify_module(self):
        fake = types.ModuleType("math_verify")
        fake.LatexExtractionConfig = DummyLatexConfig
        fake.parse = lambda value, **kwargs: [value] if value else []
        fake.verify = lambda gold, answer: gold == answer
        return fake

    def test_correct_and_incorrect(self):
        with patch.dict(sys.modules, {"math_verify": self._math_verify_module()}):
            self.assertEqual(MODULE.compute_score("math500", "42", "42"), 1.0)
            self.assertEqual(MODULE.compute_score("math500", "41", "42"), 0.0)

    def test_parse_failure_is_zero(self):
        fake = self._math_verify_module()
        fake.parse = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad latex"))
        with patch.dict(sys.modules, {"math_verify": fake}):
            self.assertEqual(MODULE.compute_score("math500", "bad", "42"), 0.0)


if __name__ == "__main__":
    unittest.main()
