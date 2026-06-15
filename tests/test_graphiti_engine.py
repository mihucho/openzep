import json
import os
import unittest
from datetime import datetime, timezone

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11111/v1")
os.environ.setdefault("LLM_MODEL", "test-model")

from engine.graphiti_engine import sanitize_graph_attributes


class GraphitiEngineTests(unittest.TestCase):
    def test_sanitize_graph_attributes_stringifies_nested_structures(self):
        now = datetime(2026, 3, 16, 12, 30, tzinfo=timezone.utc)
        attributes = {
            "flat_text": "ok",
            "flat_list": ["a", "b"],
            "nested_dict": {
                "project": "OpenClaw",
                "history": ["Old Name", "New Name"],
            },
            "nested_list": [
                {"name": "A"},
                {"name": "B"},
            ],
            "when": now,
        }

        sanitized = sanitize_graph_attributes(attributes)

        self.assertEqual(sanitized["flat_text"], "ok")
        self.assertEqual(sanitized["flat_list"], ["a", "b"])
        self.assertEqual(sanitized["when"], now.isoformat())
        self.assertIsInstance(sanitized["nested_dict"], str)
        self.assertEqual(json.loads(sanitized["nested_dict"])["project"], "OpenClaw")
        self.assertIsInstance(sanitized["nested_list"], str)
        self.assertEqual(len(json.loads(sanitized["nested_list"])), 2)

    def test_sanitize_graph_attributes_stringifies_mixed_lists(self):
        sanitized = sanitize_graph_attributes(
            {
                "mixed": ["a", 1],
                "complex": [{"a": 1}, "b"],
            }
        )

        self.assertIsInstance(sanitized["mixed"], str)
        self.assertEqual(json.loads(sanitized["mixed"]), ["a", 1])
        self.assertIsInstance(sanitized["complex"], str)
        self.assertEqual(json.loads(sanitized["complex"])[0]["a"], 1)


if __name__ == "__main__":
    unittest.main()
