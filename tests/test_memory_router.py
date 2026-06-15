import unittest
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11111/v1")
os.environ.setdefault("LLM_MODEL", "test-model")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import memory as memory_router


class _FakeRow(dict):
    def keys(self):
        return super().keys()


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    async def fetchone(self):
        return self._row


class _FakeDB:
    async def execute(self, query, params):
        return _FakeCursor(_FakeRow({"session_id": params[0], "user_id": "user-1"}))


class _FakeLLMClient:
    async def generate_response(self, messages, response_model=None):
        return {"summary": "User-1 prefers robotics projects and is currently troubleshooting access issues."}


class _FakeGraphiti:
    def __init__(self):
        self.driver = object()
        self.llm_client = _FakeLLMClient()

    async def search(self, query, group_ids, num_results):
        now = datetime.now(timezone.utc)
        return [
            SimpleNamespace(
                uuid="edge-1",
                fact="User-1 is working on a robotics project",
                created_at=now,
                valid_at=now,
                invalid_at=None,
                expired_at=None,
                score=0.88,
            )
        ]


async def _override_get_db():
    yield _FakeDB()


class MemoryRouterTests(unittest.TestCase):
    @patch("engine.context_assembly.EntityNode.get_by_group_ids", new_callable=AsyncMock)
    def test_get_memory_returns_structured_context(self, mock_get_nodes):
        mock_get_nodes.return_value = [SimpleNamespace(name="User-1", summary="Robotics student")]

        app = FastAPI()
        app.state.graphiti = _FakeGraphiti()
        app.include_router(memory_router.router)
        app.dependency_overrides[memory_router.get_db] = _override_get_db
        app.dependency_overrides[memory_router.verify_api_key] = lambda: None

        client = TestClient(app)
        response = client.get(
            "/api/v2/sessions/session-1/memory",
            params={"lastn": 5, "max_tokens": 200},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("<USER_SUMMARY>", payload["context"])
        self.assertIn("<FACTS>", payload["context"])
        self.assertEqual(payload["user_summary"], "User-1 prefers robotics projects and is currently troubleshooting access issues.")
        self.assertEqual(len(payload["facts"]), 1)
        self.assertEqual(payload["facts"][0]["uuid"], "edge-1")
        self.assertIsNotNone(payload["facts"][0]["valid_at"])


if __name__ == "__main__":
    unittest.main()
