"""任务事件总线（SSE 实时推送 + SQLite 历史回放）。"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("ams.gateway.events")

QUEUE_MAXSIZE = 1000


class EventBus:
    """按 job_id 管理 SSE 订阅队列。

    事件先写 SQLite（历史回放），再推送到所有订阅者（实时）。
    队列满时丢弃最旧事件，避免慢客户端拖垮 Gateway。
    """

    def __init__(self) -> None:
        self._queues: dict[str, set[asyncio.Queue]] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self._queues.setdefault(job_id, set()).add(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        subscribers = self._queues.get(job_id)
        if not subscribers:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._queues.pop(job_id, None)

    def subscriber_count(self, job_id: str) -> int:
        return len(self._queues.get(job_id, ()))

    def publish(self, job_id: str, event: dict) -> None:
        subscribers = list(self._queues.get(job_id, ()))
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning("SSE 队列持续满，丢弃事件 job=%s", job_id)
