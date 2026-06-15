import unittest
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11111/v1")
os.environ.setdefault("LLM_MODEL", "test-model")

from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode

from models.graph import GraphAddBatchRequest
from routers.graph import _add_episode_bulk_resilient, _list_all_graphs


def _raw_episode(name: str) -> RawEpisode:
    return RawEpisode(
        name=name,
        content=f"{name} content",
        source_description="test",
        source=EpisodeType.text,
        reference_time=datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc),
    )


class GraphRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_bulk_fallback_splits_batch_after_rate_limit(self):
        graphiti = AsyncMock()

        async def bulk_side_effect(raw_episodes, **kwargs):
            if len(raw_episodes) > 1:
                raise RuntimeError("429 rate limit exceeded")
            return None

        graphiti.add_episode_bulk.side_effect = bulk_side_effect
        body = GraphAddBatchRequest(graph_id="graph-test", episodes=[])
        raw_episodes = [_raw_episode("ep1"), _raw_episode("ep2")]

        with patch("routers.graph.asyncio.sleep", new=AsyncMock()):
            await _add_episode_bulk_resilient(graphiti, body, raw_episodes, ontology=None)

        self.assertEqual(graphiti.add_episode_bulk.await_count, 4)
        graphiti.add_episode.assert_not_awaited()

    async def test_bulk_fallback_uses_single_episode_write_for_last_item(self):
        graphiti = AsyncMock()
        graphiti.add_episode_bulk.side_effect = RuntimeError("429 rate limit exceeded")
        body = GraphAddBatchRequest(graph_id="graph-test", episodes=[])
        raw_episodes = [_raw_episode("ep1")]

        with patch("routers.graph.asyncio.sleep", new=AsyncMock()):
            await _add_episode_bulk_resilient(graphiti, body, raw_episodes, ontology=None)

        self.assertEqual(graphiti.add_episode_bulk.await_count, 2)
        graphiti.add_episode.assert_awaited_once()
        self.assertEqual(graphiti.add_episode.await_args.kwargs["name"], "ep1")

    async def test_list_all_enumerates_databases_excluding_system(self):
        """#12: graphs are Neo4j databases; neo4j/system must be filtered out."""
        db_rows = [
            {"name": "graphA"},
            {"name": "neo4j"},
            {"name": "system"},
            {"name": "graphB"},
        ]

        def _records_with(key):
            class _Result:
                def __init__(self, value):
                    self.records = [SimpleNamespace(**{key: value})]
            return _Result

        async def client_execute(query, **kwargs):
            # EagerResult.records are indexable via attribute on SimpleNamespace
            class _R:
                def __init__(self, rows):
                    self.records = [SimpleNamespace(**row) for row in rows]
            return _R(db_rows)

        clone_calls = {"n": 0}

        async def clone_execute(query, **kwargs):
            # alternate node/edge counts deterministically
            clone_calls["n"] += 1
            value = 5 if clone_calls["n"] % 2 == 1 else 2

            class _R:
                def __init__(self, v):
                    self.records = [SimpleNamespace(c=v, t=None)]
            return _R(value)

        clone = SimpleNamespace(execute_query=AsyncMock(side_effect=clone_execute))

        driver = SimpleNamespace(
            client=SimpleNamespace(execute_query=AsyncMock(side_effect=client_execute)),
            clone=Mock(return_value=clone),
            execute_query=AsyncMock(),
        )

        response = await _list_all_graphs(driver, limit=10, offset=0)

        self.assertEqual(response.total_count, 2)
        names = {item.name for item in response.graphs}
        self.assertEqual(names, {"graphA", "graphB"})
        self.assertNotIn("neo4j", names)
        self.assertNotIn("system", names)
        self.assertEqual(response.limit, 10)
        self.assertEqual(response.offset, 0)


if __name__ == "__main__":
    unittest.main()
