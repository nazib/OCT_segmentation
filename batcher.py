"""Generic asyncio micro-batcher that coalesces concurrent calls into one batched call."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, List, Optional, TypeVar

ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")

BatchFn = Callable[[List[ItemT]], Awaitable[List[ResultT]]]
OnBatchFn = Callable[[int, float], None]


@dataclass
class _PendingRequest(Generic[ItemT, ResultT]):
    item: ItemT
    future: "asyncio.Future[ResultT]"


class AsyncBatcher(Generic[ItemT, ResultT]):
    """Coalesces concurrent `submit()` calls into batches.

    A caller `await`s `submit(item)`. Behind the scenes, a single background
    worker collects items arriving within `max_wait_seconds` of the first item
    in a batch (or until `max_batch_size` is reached, whichever comes first),
    then invokes `batch_fn` once on the whole batch and fans the results back
    out to each caller's future. This turns N concurrent single-item requests
    into one model forward pass.
    """

    def __init__(
        self,
        batch_fn: BatchFn,
        max_batch_size: int = 8,
        max_wait_seconds: float = 0.01,
        on_batch: Optional[OnBatchFn] = None,
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must be >= 0")
        self._batch_fn = batch_fn
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_seconds
        self._on_batch = on_batch
        self._queue: "asyncio.Queue[_PendingRequest]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    @property
    def running(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._run(), name="async-batcher")

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    async def submit(self, item: ItemT) -> ResultT:
        if not self.running:
            raise RuntimeError("AsyncBatcher is not running; call start() first")
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[ResultT]" = loop.create_future()
        await self._queue.put(_PendingRequest(item=item, future=future))
        return await future

    async def _run(self) -> None:
        while True:
            pending = [await self._queue.get()]
            deadline = time.monotonic() + self._max_wait_seconds
            while len(pending) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    pending.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break

            batch_start = time.monotonic()
            try:
                results = await self._batch_fn([p.item for p in pending])
                if len(results) != len(pending):
                    raise RuntimeError(
                        f"batch_fn returned {len(results)} results for {len(pending)} items"
                    )
                for p, result in zip(pending, results):
                    if not p.future.done():
                        p.future.set_result(result)
            except Exception as exc:  # propagate the failure to every waiter in the batch
                for p in pending:
                    if not p.future.done():
                        p.future.set_exception(exc)
            finally:
                if self._on_batch is not None:
                    self._on_batch(len(pending), time.monotonic() - batch_start)
