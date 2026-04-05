import logging
import sys
import json
import re
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, override
from logging.handlers import TimedRotatingFileHandler


try:
    import google.cloud.logging
    from google.cloud.logging.handlers import CloudLoggingHandler
    GCP_LIB_AVAILABLE = True
except (ImportError, Exception):
    GCP_LIB_AVAILABLE = False

# --- CONFIGURATION & DETECTION ---
IS_CLOUD_RUN = os.environ.get("K_SERVICE") is not None
# Detect if we are running in an interactive terminal (Local)
IS_INTERACTIVE = sys.stdout.isatty()

class FileContextAdapter(logging.LoggerAdapter):
    """
    Adapter to inject file-specific context/labels into logs.
    """
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

    # ANSI Color Codes for Local Dev
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
        log_record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname, # GCP prefers 'severity' over 'levelname' for filtering
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "line_no": record.lineno,
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # HANDLE LABELS (Global + Local)
        final_labels = self.global_labels.copy()
        # Merge in record-specific labels (from FileContextAdapter or extra={'labels':...})
        # Check if 'labels' exists in the record (added via extra=...)
        if hasattr(record, "labels") and isinstance(record.labels, dict):
            final_labels.update(record.labels)
        
        # In GCP, 'logging.googleapis.com/labels' is the standard key for labels 
        # when logging JSON to stdout, but a top-level "labels" key often works too.
        if final_labels:
            log_record["logging.googleapis.com/labels"] = final_labels

        # Add extra fields that aren't standard LogRecord attributes
        standard_attrs = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())
        for key, value in record.__dict__.items():
            if key in ["labels", "message"]: continue
            if key not in standard_attrs and key not in log_record:
                log_record[key] = value

        # Serialize
        if self.use_color:
            # Local Development: Pretty print
            json_str = json.dumps(log_record, default=str, indent=2)
            return self._colorize_json(json_str, record.levelname)
        else:
            # Cloud/File: Compact JSON
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
        level_color = self.LEVEL_COLORS.get(levelname, self.RESET)
        # Regex to find the severity value and color it
        json_str = re.sub(f': "{levelname}"', f': "{level_color}{levelname}{self.RESET}"', json_str)
        return json_str


def setup_central_logging(log_file_path="app.json.log", global_labels=None):
    if global_labels is None:
        global_labels = {}

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates during reloads
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # --- SCENARIO 1: GOOGLE CLOUD RUN ---
    if IS_CLOUD_RUN:
        # GCR Requirement: Log JSON to Stdout/Stderr.
        # Do NOT use CloudLoggingHandler (API) as it blocks requests and isn't needed.
        # We output to stdout (INFO+) and stderr (ERROR) or just stdout for all.
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO) # Usually INFO in Prod
        stream_handler.setFormatter(JsonFormatter(use_color=False, global_labels=global_labels))
        root_logger.addHandler(stream_handler)
        
        print(f"🚀 Logging setup for Cloud Run (JSON to Stdout).")
        return

    # --- SCENARIO 2 & 3: LOCAL OR COMPUTE ENGINE (VM) ---
    
    # 2a. Terminal Handler
    # Use colors ONLY if we are in an interactive terminal (Local)
    use_color = IS_INTERACTIVE
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(JsonFormatter(use_color=use_color, global_labels=global_labels))
    root_logger.addHandler(console_handler)

    # 2b. File Handler
    full_log_file_path = get_safe_path(log_file_path)
    full_log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = TimedRotatingFileHandler(full_log_file_path, when="midnight", interval=1, backupCount=356, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(JsonFormatter(use_color=False, global_labels=global_labels))
    root_logger.addHandler(file_handler)

    # 2c. Google Cloud Handler (Direct API Push)
    # Best for Local (to see logs in GCP console) or Compute Engine.
    if GCP_LIB_AVAILABLE:
        try:
            client = google.cloud.logging.Client()
            cloud_handler = CloudLoggingHandler(client, labels=global_labels) 
            cloud_handler.setLevel(logging.INFO)
            root_logger.addHandler(cloud_handler)
            print("✅ Google Cloud Logging API Handler attached.")
        except Exception as e:
            print(f"⚠️ Could not attach Google Cloud Logging API: {e}")
    
    mode = "Local Development" if IS_INTERACTIVE else "Headless/VM"
    print(f"💻 Logging setup for {mode}. Colors: {use_color}")

def get_file_logger(module_name, file_specific_label):
    logger = logging.getLogger(module_name)
    context = {"file_id": file_specific_label}
    return FileContextAdapter(logger, context)

def get_project_root() -> Path:
    """
    Traverses upwards from the current file to find the project root.
    Looks for common root markers.
    """
    # Start from the folder containing THIS script
    current_path = Path(__file__).resolve().parent
    
    # Root markers to look for
    root_markers = ("pyproject.toml", ".git", "requirements.txt", "Makefile")
    
    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in root_markers):
            project_root = parent.resolve()
            break
    if not project_root:
        project_root = current_path.resolve()
    # Fallback to the script's directory if no marker found
    return project_root

def get_safe_path(user_provided_path: str) -> Path:
    """
    Guards a path against traversal by verifying it stays inside the root.
    """
    root = get_project_root()
    
    # 2. RESOLVE: Convert to absolute path and strip all '../'
    # .resolve() is the key. It turns 'root/logs/../../etc/passwd' into '/etc/passwd'
    target_path = (root / user_provided_path).resolve()

    # 3. VALIDATE: Check if the final destination is still under root
    if not target_path.is_relative_to(root):
        raise PermissionError(f"Traversal Attempt Blocked: {target_path} is outside {root}")
        
    return target_path