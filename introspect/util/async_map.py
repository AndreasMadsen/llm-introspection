
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
        self._queue: Iterator[YieldInType] = queue
        self._worker = worker
        self._max_tasks = max_tasks
        self._tasks = set()
        self._is_canceled = False

        for _ in range(self._max_tasks):
            self._start_next_task()

    def _cancel(self) -> None:
        self._is_canceled = True
        # dereference queue to prevent memory leaks
        self._queue = [] # type: ignore

        for task in self._tasks:
            task.cancel()
        self._tasks = set()

    def _collect_exceptions(self) -> Exception|None:
        all_exceptions = []
        for task in self._tasks:
            if not task.done() or task.cancelled():
                continue

            exception = task.exception()
            if exception is not None:
                all_exceptions.append(exception)

        match len(all_exceptions):
            case 0:
                return None
            case 1:
                return all_exceptions[0]
            case _:
                return ExceptionGroup('AsyncMap detected multiple exceptions', all_exceptions)

    def _start_next_task(self) -> None:
            try:
                job = next(self._queue)
            except StopIteration:
                return

            self._tasks.add(asyncio.create_task(self._worker(job)))

    async def __anext__(self) -> YieldOutType:
        if len(self._tasks) == 0 or self._is_canceled:
            raise StopAsyncIteration

        done, _ = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
        finished_task = done.pop()
        has_exception = finished_task.exception() is not None

        # An true exception happend or the task is cancelled (CancelledError)
        if has_exception:
            exception = self._collect_exceptions()
            self._cancel()

            # There are no exceptions from any tasks, so just raise the CancelledError
            if exception is None:
                return await finished_task

            # There is at least one exception, so raise it/them
            raise exception

        # Not cancelled, no exception from task. Continue.
        # Note, other tasks may still have exceptions, but we will learn about
        # those in the next iteration.
        self._tasks.remove(finished_task)
        self._start_next_task()

        return await finished_task
