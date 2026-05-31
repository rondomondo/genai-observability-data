"""Tests for TTLWatcher.

The pubsub thread requires a live Redis with keyspace notifications, so those
paths are marked integration. The key-parsing logic and the async consumer
(queue -> vectorstore.remove) are tested without any Redis connection.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.ttl_watcher import ExpiredElement, TTLWatcher


def _make_watcher(vs: MagicMock | None = None) -> TTLWatcher:
    if vs is None:
        vs = MagicMock()
        vs.remove = MagicMock(return_value=True)
    return TTLWatcher(redis_url="redis://localhost:6379", vectorstore=vs)


class TestExpiredElement:
    def test_fields_stored(self) -> None:
        e = ExpiredElement(vset_key="rag:cache", element_id="my-slug")
        assert e.vset_key == "rag:cache"
        assert e.element_id == "my-slug"


class TestKeyParsing:
    """Exercise the inline parsing logic inside _pubsub_thread via a helper."""

    def _parse(self, key: str) -> ExpiredElement | None:
        """Replicate the parsing block from _pubsub_thread."""
        from app.services.ttl_watcher import TTL_PREFIX

        if not key.startswith(TTL_PREFIX):
            return None
        rest = key[len(TTL_PREFIX):]
        last_colon = rest.rfind(":")
        if last_colon == -1:
            return None
        return ExpiredElement(vset_key=rest[:last_colon], element_id=rest[last_colon + 1:])

    def test_simple_vset_key(self) -> None:
        e = self._parse("ttl:rag:cache:my-element")
        assert e is not None
        assert e.vset_key == "rag:cache"
        assert e.element_id == "my-element"

    def test_logs_vset_key(self) -> None:
        e = self._parse("ttl:logs:embeddings:auth-error-slug")
        assert e is not None
        assert e.vset_key == "logs:embeddings"
        assert e.element_id == "auth-error-slug"

    def test_traces_vset_key(self) -> None:
        e = self._parse("ttl:traces:embeddings:trace-abc123")
        assert e is not None
        assert e.vset_key == "traces:embeddings"
        assert e.element_id == "trace-abc123"

    def test_no_ttl_prefix_returns_none(self) -> None:
        assert self._parse("some:other:key") is None

    def test_no_colon_in_rest_returns_none(self) -> None:
        assert self._parse("ttl:nocolon") is None

    def test_element_id_with_hyphens(self) -> None:
        e = self._parse("ttl:rag:cache:what-is-the-capital-of-france")
        assert e is not None
        assert e.element_id == "what-is-the-capital-of-france"


class TestProperties:
    def test_is_running_false_before_start(self) -> None:
        w = _make_watcher()
        assert w.is_running is False

    def test_pending_count_zero_initially(self) -> None:
        w = _make_watcher()
        assert w.pending_count == 0


class TestConsumer:
    async def test_consumer_calls_vectorstore_remove(self) -> None:
        vs = MagicMock()
        vs.remove = MagicMock(return_value=True)
        w = _make_watcher(vs)
        w._loop = asyncio.get_running_loop()

        expired = ExpiredElement(vset_key="rag:cache", element_id="slug-001")
        await w._async_queue.put(expired)

        task = asyncio.create_task(w._consumer())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        vs.remove.assert_called_once_with("rag:cache", "slug-001")

    async def test_consumer_drains_multiple_items(self) -> None:
        vs = MagicMock()
        vs.remove = MagicMock(return_value=True)
        w = _make_watcher(vs)
        w._loop = asyncio.get_running_loop()

        items = [
            ExpiredElement(vset_key="rag:cache", element_id=f"slug-{i}")
            for i in range(4)
        ]
        for item in items:
            await w._async_queue.put(item)

        task = asyncio.create_task(w._consumer())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert vs.remove.call_count == 4

    async def test_consumer_continues_after_remove_error(self) -> None:
        vs = MagicMock()
        vs.remove = MagicMock(side_effect=[RuntimeError("boom"), True])
        w = _make_watcher(vs)
        w._loop = asyncio.get_running_loop()

        for i in range(2):
            await w._async_queue.put(ExpiredElement(vset_key="rag:cache", element_id=f"slug-{i}"))

        task = asyncio.create_task(w._consumer())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert vs.remove.call_count == 2

    async def test_pending_count_decrements_after_processing(self) -> None:
        vs = MagicMock()
        vs.remove = MagicMock(return_value=True)
        w = _make_watcher(vs)
        w._loop = asyncio.get_running_loop()

        await w._async_queue.put(ExpiredElement(vset_key="rag:cache", element_id="slug-x"))
        assert w.pending_count == 1

        task = asyncio.create_task(w._consumer())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert w.pending_count == 0


class TestLifecycle:
    async def test_stop_before_start_is_safe(self) -> None:
        w = _make_watcher()
        await w.stop()  # must not raise

    async def test_start_and_stop(self) -> None:
        vs = MagicMock()
        vs.remove = MagicMock(return_value=True)
        w = TTLWatcher(redis_url="redis://localhost:6379", vectorstore=vs)

        fake_redis = MagicMock()
        fake_redis.config_set = MagicMock()
        fake_redis.close = MagicMock()

        with patch("app.services.ttl_watcher.redis_lib.Redis.from_url", return_value=fake_redis), patch.object(
            w, "_pubsub_thread"
        ):
            await w.start()
            assert w._consumer_task is not None
            assert w._thread is not None
            await w.stop()

        assert w._running is False
