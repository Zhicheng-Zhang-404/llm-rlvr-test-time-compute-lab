import json
import tempfile
import unittest
from pathlib import Path

from ttc_rlvr.evaluate import aggregate, write_metrics
from ttc_rlvr.io_utils import append_jsonl, read_jsonl


class EvaluateTests(unittest.TestCase):
    def test_aggregate_and_json_round_trip(self):
        records = [
            {"id": 0, "level": "Level 1", "correct": [True, False, False, False]},
            {"id": 1, "level": "Level 2", "correct": [False, False, False, False]},
        ]
        result = aggregate(records, [1, 2, 4])
        self.assertEqual(result["num_problems"], 2)
        self.assertAlmostEqual(result["pass_at_k"][1], 0.125)
        self.assertAlmostEqual(result["pass_at_k"][4], 0.5)
        with tempfile.TemporaryDirectory() as directory:
            jsonl = Path(directory) / "data.jsonl"
            metrics = Path(directory) / "metrics.json"
            append_jsonl(jsonl, records)
            self.assertEqual(read_jsonl(jsonl), records)
            write_metrics(result, metrics)
            self.assertEqual(json.loads(metrics.read_text())["num_problems"], 2)


if __name__ == "__main__":
    unittest.main()
