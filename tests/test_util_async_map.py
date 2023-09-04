
import asyncio

import pytest

from introspect.util import AsyncMap

@pytest.mark.asyncio
async def test_async_map_jobs_run_in_parallel():
    started: set[int] = set()
    finished: set[int] = set()

    async def worker(job_id):
        started.add(job_id)
        await asyncio.sleep(0.01)
        finished.add(job_id)
        return job_id

    iterable = AsyncMap(worker, [1, 2, 3], max_tasks=3)
    await asyncio.sleep(0)  # allow time for tasks to start the coro
    assert started == set([])
    assert finished == set([])

    iterator = aiter(iterable)
    await asyncio.sleep(0)  # allow time for tasks to start the coro
    assert started == {1, 2, 3}
    assert finished == set([])

    async for job_id in iterator:
        assert job_id in finished

    assert finished == {1, 2, 3}


@pytest.mark.asyncio
async def test_async_map_jobs_throttle():
    started: set[int] = set()
    finished: set[int] = set()

    async def worker(job_id):
        started.add(job_id)
        await asyncio.sleep(0.01)
        finished.add(job_id)
        return job_id

    iterable = AsyncMap(worker, [1, 2, 3], max_tasks=1)
    await asyncio.sleep(0)  # allow time for tasks to start the coro
    assert started == set([])
    assert finished == set([])

    iterator = aiter(iterable)
    await asyncio.sleep(0)  # allow time for tasks to start the coro
    assert started == {1}
    assert finished == set([])

    completed = 0
    async for job_id in iterator:
        completed += 1
        assert job_id in finished
        await asyncio.sleep(0)  # allow time for tasks to start the coro
        assert len(started) == min(completed + 1, 3)

    assert started == {1, 2, 3}
    assert finished == {1, 2, 3}
