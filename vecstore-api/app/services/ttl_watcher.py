"""TTL watcher — keyspace notification listener.

Redis vector sets don't support per-element TTL natively.
Pattern: store a shadow key `ttl:{vset}:{element}` with a real Redis TTL.
This watcher subscribes to keyspace expiry events, parses the shadow key,
and removes the corresponding vector set element.

Architecture:
  - Runs in a daemon thread (blocking redis pubsub)
  - Thread puts expired element info onto an asyncio.Queue
  - An asyncio consumer task drains the queue and calls vectorstore.remove()

This keeps all async/redis vset logic in the async world and the thread dumb.
"""

import asyncio
import threading
from dataclasses import dataclass

import redis as redis_lib

from app.logging_config import get_logger
from app.services.vectorstore import VectorStore

log = get_logger(__name__)

TTL_PREFIX = "ttl:"


@dataclass
class ExpiredElement:
    vset_key: str
    element_id: str


class TTLWatcher:
    def __init__(
        self,
        redis_url: str,
        vectorstore: VectorStore,
        db: int = 0,
    ) -> None:
        self._redis_url = redis_url
        self._vs = vectorstore
        self._db = db
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_queue: asyncio.Queue[ExpiredElement] = asyncio.Queue()
        self._thread: threading.Thread | None = None
        self._consumer_task: asyncio.Task | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._pending_count = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()

        # Enable keyspace notifications for expired events (if not already set)
        r = redis_lib.Redis.from_url(self._redis_url, decode_responses=True)
        try:
            r.config_set("notify-keyspace-events", "Ex")
            log.info("ttl_watcher_keyspace_notifications_enabled")
        except redis_lib.ResponseError as exc:
            log.warning("ttl_watcher_config_set_failed", error=str(exc))
        finally:
            r.close()

        self._running = True
        self._consumer_task = asyncio.create_task(self._consumer(), name="ttl-consumer")

        self._thread = threading.Thread(
            target=self._pubsub_thread,
            name="ttl-watcher",
            daemon=True,
        )
        self._thread.start()
        log.info("ttl_watcher_started")

    async def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        log.info("ttl_watcher_stopped")

    @property
    def is_running(self) -> bool:
        return self._running and (self._thread is not None and self._thread.is_alive())

    @property
    def pending_count(self) -> int:
        return self._async_queue.qsize()

    # ── Pubsub thread (blocking) ──────────────────────────────────────────────

    def _pubsub_thread(self) -> None:
        log.debug("ttl_watcher_thread_started")
        r = redis_lib.Redis.from_url(self._redis_url, decode_responses=True)
        pubsub = r.pubsub()
        channel = f"__keyevent@{self._db}__:expired"
        pubsub.subscribe(channel)

        try:
            for message in pubsub.listen():
                if self._stop_event.is_set():
                    break
                if message["type"] != "message":
                    continue
                key: str = message["data"]
                if not key.startswith(TTL_PREFIX):
                    continue

                # Parse `ttl:{vset_key}:{element_id}`
                # vset_key itself may contain colons (e.g. rag:cache)
                # Convention: everything after the second colon segment is element_id
                # Format stored by vectorstore.set_ttl: ttl:{vset}:{element}
                # We split on first two colons only
                rest = key[len(TTL_PREFIX) :]  # "{vset_key}:{element_id}"
                # vset keys are like "rag:cache", "logs:embeddings" — 2 parts
                # element IDs don't contain colons (slugified)
                # Split from the right: last segment is element_id
                last_colon = rest.rfind(":")
                if last_colon == -1:
                    log.warning("ttl_watcher_unparseable_key", key=key)
                    continue

                vset_key = rest[:last_colon]
                element_id = rest[last_colon + 1 :]

                expired = ExpiredElement(vset_key=vset_key, element_id=element_id)
                if self._loop and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(self._async_queue.put_nowait, expired)

        except Exception as exc:
            log.error("ttl_watcher_thread_error", error=str(exc))
        finally:
            pubsub.unsubscribe()
            r.close()
            log.debug("ttl_watcher_thread_stopped")

    # ── Async consumer ────────────────────────────────────────────────────────

    async def _consumer(self) -> None:
        log.debug("ttl_consumer_started")
        loop = asyncio.get_running_loop()
        while True:
            expired = await self._async_queue.get()
            try:
                removed = await loop.run_in_executor(
                    None,
                    lambda e=expired: self._vs.remove(e.vset_key, e.element_id),
                )
                log.info(
                    "ttl_expired",
                    vset_key=expired.vset_key,
                    element_id=expired.element_id,
                    removed=removed,
                )
            except Exception as exc:
                log.error(
                    "ttl_consumer_error",
                    vset_key=expired.vset_key,
                    element_id=expired.element_id,
                    error=str(exc),
                )
            finally:
                self._async_queue.task_done()
