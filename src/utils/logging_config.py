import logging
import sys
import json
import re
from datetime import datetime, timezone
from typing import Any, override

from logging.handlers import TimedRotatingFileHandler

try:
    import google.cloud.logging
    from google.cloud.logging.handlers import CloudLoggingHandler

    GCP_AVAILABLE = True
except (ImportError, Exception):
    GCP_AVAILABLE = False


class FileContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        if "labels" not in extra:
            extra["labels"] = {}
        extra["labels"].update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for structured logging.
    Supports colored output for terminal readability.
    """

    # ANSI Color Codes
    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    BLUE = "\x1b[34;20m"
    CYAN = "\x1b[36;20m"
    RESET = "\x1b[0m"

    LEVEL_COLORS = {"DEBUG": GREY, "INFO": GREEN, "WARNING": YELLOW, "ERROR": RED, "CRITICAL": BOLD_RED}

    def __init__(self, use_color: bool = False, global_labels: dict = None, **kwargs):
        """
        Args:
            use_color (bool): If True, adds ANSI colors and indentation for terminal.
                              If False, outputs compact valid JSON for files.
        """
        super().__init__(**kwargs)
        self.use_color = use_color
        self.global_labels = global_labels if global_labels else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        # Build the dictionary
        log_record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "line_no": record.lineno,
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        # HANDLE LABELS (Global + Local)
        # Start with global labels (e.g. env=prod)
        final_labels = self.global_labels.copy()

        # Merge in record-specific labels (from FileContextAdapter or extra={'labels':...})
        # Check if 'labels' exists in the record (added via extra=...)
        if hasattr(record, "labels") and isinstance(record.labels, dict):
            final_labels.update(record.labels)

        # Add to record if we have any labels
        if final_labels:
            log_record["labels"] = final_labels

        # Add extra fields
        standard_attrs = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())
        for key, value in record.__dict__.items():
            if key == "labels":
                continue  # Skip, already handled
            if key not in standard_attrs and key not in log_record:
                log_record[key] = value

        # Serialize to JSON
        # If coloring (Terminal), we use indent=2 for readability.
        # If not (File), we use default (compact) to save space.
        if self.use_color:
            json_str = json.dumps(log_record, default=str, indent=2)
            return self._colorize_json(json_str, record.levelname)
        else:
            return json.dumps(log_record, default=str)

    def _colorize_json(self, json_str: str, levelname: str) -> str:
        """
        Uses Regex to inject colors into the JSON string.
        We colorize the Keys and the Level Value.
        """

        # Color the "keys" (e.g., "timestamp":) in CYAN
        def replace_key(match):
            return f"{self.CYAN}{match.group(1)}{self.RESET}:"

        json_str = re.sub(r"\"(.*?)\":", replace_key, json_str)

        # Color the specific value of the "level" field
        level_color = self.LEVEL_COLORS.get(levelname, self.RESET)
        json_str = json_str.replace(f': "{levelname}"', f': "{level_color}{levelname}{self.RESET}"')

        return json_str


def setup_central_logging(log_file_path="app.json.log", global_labels=None):
    if global_labels is None:
        global_labels = {}

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # if root_logger.hasHandlers():
    #     root_logger.handlers.clear()

    # 1. Terminal Handler (Colored + Indented JSON)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(JsonFormatter(use_color=True, global_labels=global_labels))
    root_logger.addHandler(console_handler)

    # 2. File Handler (Compact + Valid JSON)
    file_handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=356, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    # ❌ Disable Color for File (keeps it machine-readable)
    file_handler.setFormatter(JsonFormatter(use_color=False, global_labels=global_labels))
    file_handler.suffix = "%Y-%m-%d"
    root_logger.addHandler(file_handler)

    # 3. Google Cloud Handler
    if GCP_AVAILABLE:
        try:
            client = google.cloud.logging.Client()
            cloud_handler = CloudLoggingHandler(client, labels=global_labels)
            cloud_handler.setLevel(logging.INFO)
            root_logger.addHandler(cloud_handler)
            print("✅ Google Cloud Logging attached successfully.")
        except Exception as e:
            # We use a fallback print here
            print(f"⚠️ Could not attach Google Cloud Logging: {e}")


def get_file_logger(module_name, file_specific_label):
    logger = logging.getLogger(module_name)
    context = {"file_id": file_specific_label}
    return FileContextAdapter(logger, context)
