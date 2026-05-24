import os

import pytest

from src.utils.logging_config import setup_central_logging


@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    """
    This fixture runs once per test session (scope="session").
    It sets up the logging configuration automatically.
    """
    # 1. Use a separate log file for tests so we don't mix with real app logs
    test_log_file = "reports/tests/test_run.json.log"

    # 2. Clean up old test logs (Optional, keeps tests clean)
    if os.path.exists(test_log_file):
        os.remove(test_log_file)

    # 3. Initialize the logging system
    # We pass empty global labels or test-specific labels
    setup_central_logging(log_file_path=test_log_file, global_labels={"env": "testing", "runner": "pytest"})

    print("\n--- Logging Initialized for Tests ---")

    # Yield control to the tests
    yield

    # Teardown: Properly flush and close CloudLoggingHandler to avoid threading issues
    import logging

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if hasattr(handler, "flush"):
            try:
                handler.flush()  # Try to send pending logs
            except Exception:
                pass  # Ignore flush errors
        if hasattr(handler, "close"):
            try:
                handler.close()
            except Exception:
                pass  # Ignore close errors
