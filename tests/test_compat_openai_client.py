import copy
import unittest

from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.prompts.extract_nodes import ExtractedEntities

from engine.compat_openai_client import CompatOpenAIGenericClient


class CompatOpenAIClientTests(unittest.TestCase):
    def test_wraps_single_edge_payload_for_extracted_edges(self):
        payload = {
            "source_entity_name": "Alice",
            "target_entity_name": "Bob",
            "relation_type": "KNOWS",
            "fact": "Alice knows Bob",
            "valid_at": None,
            "invalid_at": None,
        }

        normalized = CompatOpenAIGenericClient._normalize_payload(
            payload,
            ExtractedEdges,
            [],
        )

        self.assertEqual(list(normalized.keys()), ["edges"])
        self.assertEqual(len(normalized["edges"]), 1)
        self.assertEqual(normalized["edges"][0]["relation_type"], "KNOWS")

    def test_wraps_nested_single_edge_payload_for_extracted_edges(self):
        payload = {
            "edges": {
                "source_entity_name": "Alice",
                "target_entity_name": "Bob",
                "relation_type": "KNOWS",
                "fact": "Alice knows Bob",
                "valid_at": None,
                "invalid_at": None,
            }
        }

        normalized = CompatOpenAIGenericClient._normalize_payload(
            payload,
            ExtractedEdges,
            [],
        )

        self.assertEqual(len(normalized["edges"]), 1)
        self.assertEqual(normalized["edges"][0]["fact"], "Alice knows Bob")

    def test_wraps_entity_list_and_maps_entity_type_names(self):
        payload = [
            {
                "entity_name": "Alice",
                "entity_type_name": "Person",
            }
        ]
        messages = [
            type(
                "Message",
                (),
                {
                    "content": """
<ENTITY TYPES>
[{"entity_type_name":"Person","entity_type_id":7}]
</ENTITY TYPES>
""",
                },
            )()
        ]

        normalized = CompatOpenAIGenericClient._normalize_payload(
            payload,
            ExtractedEntities,
            messages,
        )

        self.assertEqual(len(normalized["extracted_entities"]), 1)
        self.assertEqual(normalized["extracted_entities"][0]["name"], "Alice")
        self.assertEqual(normalized["extracted_entities"][0]["entity_type_id"], 7)


class StrictSchemaTests(unittest.TestCase):
    def test_normalize_strict_schema_sets_additional_properties_false_and_required(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {
                    "type": "object",
                    "properties": {"c": {"type": "string"}},
                },
            },
        }

        normalized = CompatOpenAIGenericClient._normalize_strict_schema(schema)

        self.assertFalse(normalized["additionalProperties"])
        self.assertEqual(normalized["required"], ["a", "b"])
        nested = normalized["properties"]["b"]
        self.assertFalse(nested["additionalProperties"])
        self.assertEqual(nested["required"], ["c"])

    def test_normalize_strict_schema_from_real_model(self):
        from graphiti_core.prompts.extract_nodes import ExtractedEntities

        normalized = CompatOpenAIGenericClient._normalize_strict_schema(
            ExtractedEntities.model_json_schema()
        )

        item = normalized["$defs"]["ExtractedEntity"]
        self.assertFalse(item["additionalProperties"])
        self.assertIn("name", item["required"])
        self.assertIn("entity_type_id", item["required"])

    def test_normalize_strict_schema_is_idempotent_and_non_destructive(self):
        original = {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "object", "properties": {"z": {"type": "string"}}},
            },
        }
        snapshot = copy.deepcopy(original)

        first = CompatOpenAIGenericClient._normalize_strict_schema(original)
        second = CompatOpenAIGenericClient._normalize_strict_schema(first)

        self.assertEqual(first, second)
        # Input must not be mutated
        self.assertEqual(original, snapshot)

    def test_normalize_strict_schema_preserves_map_type_values(self):
        """A dict[str, X] field emits `additionalProperties` as a schema object
        (the value type). That must survive normalization — overwriting it with
        False would change the field's meaning from a map to a closed object."""
        schema = {
            "type": "object",
            "properties": {
                "counts": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                },
            },
        }

        normalized = CompatOpenAIGenericClient._normalize_strict_schema(schema)

        counts = normalized["properties"]["counts"]
        self.assertEqual(counts["additionalProperties"], {"type": "integer"})
        # The fixed-property root object still gets the closure
        self.assertFalse(normalized["additionalProperties"])


class SchemaErrorClassificationTests(unittest.TestCase):
    def test_strong_token_triggers_schema_error(self):
        from engine.compat_openai_client import _looks_like_schema_error

        class _E(Exception):
            def __init__(self, msg):
                super().__init__(msg)

        self.assertTrue(_looks_like_schema_error(_E("unsupported json_schema")))
        self.assertTrue(_looks_like_schema_error(_E("invalid response_format")))
        self.assertTrue(_looks_like_schema_error(_E("additional_properties not allowed")))

    def test_bare_strict_does_not_trigger(self):
        """Bare 'strict' is too generic — must not cause a fallback on its own."""
        from engine.compat_openai_client import _looks_like_schema_error

        class _E(Exception):
            def __init__(self, msg):
                super().__init__(msg)

        self.assertFalse(_looks_like_schema_error(_E("strict mode violation in policy")))
        self.assertFalse(_looks_like_schema_error(_E("content too strict")))

    def test_strict_with_schema_context_triggers(self):
        from engine.compat_openai_client import _looks_like_schema_error

        class _E(Exception):
            def __init__(self, msg):
                super().__init__(msg)

        self.assertTrue(_looks_like_schema_error(_E("strict structured output rejected")))
        self.assertTrue(_looks_like_schema_error(_E("json strict mode error")))


if __name__ == "__main__":
    unittest.main()
