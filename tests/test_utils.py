import asyncio
import logging
import time

import pytest

from src.utils.logging_config import get_file_logger
from src.utils.utils import UniversalTimer

# Setup logging to verify output in tests if needed
logger = get_file_logger(__name__, "tests")


# --- Test 1: Recursion Safety (Sync) ---
def test_recursion_safety(caplog):
    """
    Verifies that a decorated function calling itself doesn't break
    the timing of the outer call.
    """
    caplog.set_level(logging.INFO)

    @UniversalTimer("Recursive")
    def countdown(n):
        if n > 0:
            time.sleep(0.1)
            countdown(n - 1)

    # Run recursion 3 times (depth 3)
    # Total time should be approx 0.3s (0.1s * 3)
    # If buggy (shared state), the outer timer might report ~0.1s (only the last segment)
    countdown(3)

    # We expect 4 log entries:
    # 1. Inner-most (n=0): ~0.0s
    # 2. n=1: ~0.1s
    # 3. n=2: ~0.2s
    # 4. Outer-most (n=3): ~0.3s

    logs = [r.message for r in caplog.records if "Recursive" in r.message and "END   [" in r.message]
    assert len(logs) == 4

    # Check the longest duration (the outer call)
    # It should be at least 0.3s. If state was shared, this would likely fail.
    outer_duration = float(logs[-1].split(": ")[1].replace("s", ""))
    assert outer_duration >= 0.3


# --- Test 2: Async Concurrency Safety ---
@pytest.mark.asyncio
async def test_async_concurrency(caplog):
    """
    Verifies that two async tasks running in parallel do not conflict.
    """
    caplog.set_level(logging.INFO)

    @UniversalTimer("AsyncWorker")
    async def worker(delay):
        await asyncio.sleep(delay)

    # Launch two workers effectively at the same time
    # Worker A takes 0.2s
    # Worker B takes 0.1s
    task1 = asyncio.create_task(worker(0.2))
    task2 = asyncio.create_task(worker(0.1))

    await asyncio.gather(task1, task2)

    logs = [r.message for r in caplog.records if "AsyncWorker" in r.message and "END   [" in r.message]
    assert len(logs) == 2

    # Parse durations
    durations = sorted([float(msg.split(": ")[1].replace("s", "")) for msg in logs])

    # We expect one ~0.1s and one ~0.2s
    # If buggy (shared state), one might overwrite the other or both end up wrong.
    assert 0.09 <= durations[0] <= 0.15
    assert 0.19 <= durations[1] <= 0.25


# --- Test 3: Correct Wrapper Selection ---
def test_wrapper_type():
    """
    Verifies that the decorator returns the correct type of function.
    """

    @UniversalTimer("Sync")
    def sync_func():
        pass

    @UniversalTimer("Async")
    async def async_func():
        pass

    assert not asyncio.iscoroutinefunction(sync_func)
    assert asyncio.iscoroutinefunction(async_func)
