import unittest
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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

    async def test_list_all_aggregates_by_group_id_in_single_db(self):
        """#12: all graphs share one Neo4j DB partitioned by group_id, so
        list-all aggregates via GROUP BY n.group_id — not SHOW DATABASES."""
        # node aggregation: two groups with counts + earliest created_at
        node_rows = [
            {"gid": "graphA", "c": 5, "t": datetime(2026, 6, 1, tzinfo=timezone.utc)},
            {"gid": "graphB", "c": 2, "t": datetime(2026, 5, 1, tzinfo=timezone.utc)},
        ]
        # edge aggregation: graphA has 3 edges; graphC has edges but no Entity nodes yet
        edge_rows = [
            {"gid": "graphA", "c": 3},
            {"gid": "graphC", "c": 4},
        ]

        def _result(rows):
            return SimpleNamespace(records=[SimpleNamespace(**row) for row in rows])

        async def execute_query(query, **kwargs):
            if "MATCH (n:Entity)" in query:
                return _result(node_rows)
            return _result(edge_rows)

        driver = SimpleNamespace(execute_query=AsyncMock(side_effect=execute_query))

        response = await _list_all_graphs(driver, limit=10, offset=0)

        # union of node groups + edge-only groups
        self.assertEqual(response.total_count, 3)
        by_id = {g.graph_id: g for g in response.graphs}

        self.assertEqual(by_id["graphA"].node_count, 5)
        self.assertEqual(by_id["graphA"].edge_count, 3)
        self.assertEqual(by_id["graphB"].node_count, 2)
        self.assertEqual(by_id["graphB"].edge_count, 0)
        # graphC appears via edges only, zero nodes
        self.assertEqual(by_id["graphC"].node_count, 0)
        self.assertEqual(by_id["graphC"].edge_count, 4)

        # newest first (graphA 2026-06 > graphB 2026-05 > graphC no created_at)
        order = [g.graph_id for g in response.graphs]
        self.assertEqual(order, ["graphA", "graphB", "graphC"])
        self.assertEqual(response.limit, 10)
        self.assertEqual(response.offset, 0)
        # driver.execute_query called exactly twice (nodes + edges), no N+1
        self.assertEqual(driver.execute_query.await_count, 2)

    async def test_list_all_pagination_applies_after_global_sort(self):
        node_rows = [
            {"gid": f"g{i}", "c": 1, "t": datetime(2026, 1, i + 1, tzinfo=timezone.utc)}
            for i in range(5)
        ]

        async def execute_query(query, **kwargs):
            if "MATCH (n:Entity)" in query:
                return SimpleNamespace(records=[SimpleNamespace(**r) for r in node_rows])
            return SimpleNamespace(records=[])

        driver = SimpleNamespace(execute_query=AsyncMock(side_effect=execute_query))

        page1 = await _list_all_graphs(driver, limit=2, offset=0)
        page2 = await _list_all_graphs(driver, limit=2, offset=2)

        # total is full set, but each page is sliced
        self.assertEqual(page1.total_count, 5)
        self.assertEqual(len(page1.graphs), 2)
        # newest (g4, 2026-01-05) first
        self.assertEqual([g.graph_id for g in page1.graphs], ["g4", "g3"])
        self.assertEqual([g.graph_id for g in page2.graphs], ["g2", "g1"])


if __name__ == "__main__":
    unittest.main()
