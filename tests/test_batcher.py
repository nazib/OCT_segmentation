import asyncio

import pytest

from batcher import AsyncBatcher

pytestmark = pytest.mark.asyncio


async def test_coalesces_concurrent_submissions_into_one_batch():
    calls = []

    async def batch_fn(items):
        calls.append(list(items))
        return [item * 2 for item in items]

    batcher = AsyncBatcher(batch_fn=batch_fn, max_batch_size=10, max_wait_seconds=0.05)
    batcher.start()
    try:
        results = await asyncio.gather(*(batcher.submit(i) for i in range(5)))
    finally:
        await batcher.stop()

    assert sorted(results) == [0, 2, 4, 6, 8]
    assert len(calls) == 1, "all five concurrent submissions should coalesce into one call"
    assert sorted(calls[0]) == [0, 1, 2, 3, 4]


async def test_respects_max_batch_size():
    calls = []

    async def batch_fn(items):
        calls.append(list(items))
        return items

    batcher = AsyncBatcher(batch_fn=batch_fn, max_batch_size=2, max_wait_seconds=0.05)
    batcher.start()
    try:
        await asyncio.gather(*(batcher.submit(i) for i in range(5)))
    finally:
        await batcher.stop()

    assert all(len(c) <= 2 for c in calls)
    assert len(calls) >= 3


async def test_exception_in_batch_fn_propagates_to_all_waiters():
    async def batch_fn(items):
        raise ValueError("boom")

    batcher = AsyncBatcher(batch_fn=batch_fn, max_batch_size=4, max_wait_seconds=0.01)
    batcher.start()
    try:
        with pytest.raises(ValueError):
            await asyncio.gather(*(batcher.submit(i) for i in range(3)))
    finally:
        await batcher.stop()


async def test_submit_before_start_raises():
    async def batch_fn(items):
        return items

    batcher = AsyncBatcher(batch_fn=batch_fn)
    with pytest.raises(RuntimeError):
        await batcher.submit(1)
