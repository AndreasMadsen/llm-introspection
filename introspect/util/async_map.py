
import asyncio
from typing import Any, TypeVar, Callable
from asyncio import Task
from collections.abc import AsyncIterable, AsyncIterator, Iterator, Iterable, Sized, Coroutine

YieldInType = TypeVar('YieldInType')
YieldOutType = TypeVar('YieldOutType')

class AsyncMapIterable(AsyncIterable[YieldOutType]):
    def __init__(self,
                 worker: Callable[[YieldInType], Coroutine[Any, Any, YieldOutType]],
                 queue: Iterable[YieldInType],
                 max_tasks: int=8) -> None:
        """Maps over the queue using the async worker. It runs `max_tasks` workers in parallel.

        Example:
        async for delay in AsyncMap(asyncio.sleep, range(10)):
            print(delay)

        Args:
            worker (Callable[[YieldInType], Awaitable[YieldOutType]): An async map function,
                this maps from YieldInType to YieldOutType
            queue (Iterable[YieldInType]): An regular iterable which describes the jobs.
            max_tasks (int, optional): The maximum number of async workers to run in parallel.
                Defaults to 8.
        """
        self._queue = queue
        self._worker = worker
        self._max_tasks = max_tasks

    def __aiter__(self):
        return AsyncMapIterator(self._worker, iter(self._queue), max_tasks=self._max_tasks)

    def __len__(self):
        if isinstance(self._queue, Sized):
            return len(self._queue)
        else:
            raise NotImplementedError

class AsyncMapIterator(AsyncIterator[YieldOutType]):
    _tasks: set[Task[YieldOutType]]

    def __init__(self,
                 worker: Callable[[YieldInType], Coroutine[Any, Any, YieldOutType]],
                 queue: Iterator[YieldInType],
                 max_tasks: int=8) -> None:
        self._queue = queue
        self._worker = worker
        self._max_tasks = max_tasks
        self._tasks = set()

        for _ in range(self._max_tasks):
            self._start_next_task()

    def _start_next_task(self) -> None:
            try:
                job = next(self._queue)
            except StopIteration:
                return

            self._tasks.add(asyncio.create_task(self._worker(job)))

    async def __anext__(self) -> YieldOutType:
        if len(self._tasks) == 0:
            raise StopAsyncIteration

        done, _ = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
        finished_task = done.pop()

        # Add tasks if neccesary
        self._tasks.remove(finished_task)
        self._start_next_task()

        # return result
        return await finished_task
