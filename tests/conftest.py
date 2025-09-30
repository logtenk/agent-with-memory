import asyncio
import inspect


def _wrap_async(func):
    async_marker = getattr(func, "__pytest_async_marked__", False)
    if async_marker:
        return func

    async def _async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    def sync_wrapper(*args, **kwargs):
        return asyncio.run(_async_wrapper(*args, **kwargs))

    sync_wrapper.__pytest_async_marked__ = True
    sync_wrapper.__name__ = getattr(func, "__name__", "wrapped_async")
    sync_wrapper.__doc__ = func.__doc__
    return sync_wrapper


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        marker = item.get_closest_marker("asyncio")
        if marker is None:
            continue
        test_func = item.obj
        if not inspect.iscoroutinefunction(test_func):
            continue
        item.obj = _wrap_async(test_func)


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test to run with asyncio.run")
