
import asyncio
import signal

def cancel_eventloop_on_signal(sig: signal.Signals, fullstop=False):
    """Cancel all running async tasks on signal.

    By catching asyncio.CancelledError, any running task can perform
    any necessary cleanup when it's cancelled.

    Args:
        sig (signal.Signals): Signal to listen for.
        fullstop (bool, optional): If True, the eventloop is stopped. Defaults to False.
    """
    loop = asyncio.get_event_loop()

    async def shutdown(sig: signal.Signals) -> None:
        tasks = []
        for task in asyncio.all_tasks(loop):
            if task is not asyncio.current_task(loop):
                task.cancel()
                tasks.append(task)

        _ = await asyncio.gather(*tasks, return_exceptions=True)
        if fullstop:
            loop.stop()

    loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(sig)))
