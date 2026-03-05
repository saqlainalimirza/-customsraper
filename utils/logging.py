import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class TokenLogAdapter(logging.LoggerAdapter):
    """Logger adapter that adds token usage context."""
    
    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra = kwargs.get("extra", {})
        if self.extra:
            extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with both console and structured JSON output.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    logger.propagate = False
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger


def log_tokens(
    logger: logging.Logger,
    operation: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Log token usage for an AI operation.
    
    Args:
        logger: Logger instance
        operation: Name of the operation (e.g., "filter_urls", "extract_answer")
        model: AI model used
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        extra: Additional data to log
    """
    total = input_tokens + output_tokens
    
    message = (
        f"TOKEN_USAGE | operation={operation} | model={model} | "
        f"input={input_tokens} | output={output_tokens} | total={total}"
    )
    
    if extra:
        extra_str = " | ".join(f"{k}={v}" for k, v in extra.items())
        message += f" | {extra_str}"
    
    logger.info(message)


def log_request(
    logger: logging.Logger,
    method: str,
    url: str,
    status_code: int | None = None,
    duration_ms: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Log an HTTP request.
    
    Args:
        logger: Logger instance
        method: HTTP method
        url: Request URL
        status_code: Response status code (if available)
        duration_ms: Request duration in milliseconds
        extra: Additional data to log
    """
    parts = [f"HTTP_REQUEST | {method} {url}"]
    
    if status_code is not None:
        parts.append(f"status={status_code}")
    
    if duration_ms is not None:
        parts.append(f"duration={duration_ms:.2f}ms")
    
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v}")
    
    logger.info(" | ".join(parts))


def log_pipeline_step(
    logger: logging.Logger,
    step: str,
    row_id: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> None:
    """
    Log a pipeline processing step.
    
    Args:
        logger: Logger instance
        step: Pipeline step name
        row_id: Database row ID being processed
        status: Step status (started, completed, failed)
        details: Additional details
    """
    message = f"PIPELINE | step={step} | row_id={row_id} | status={status}"
    
    if details:
        details_str = " | ".join(f"{k}={v}" for k, v in details.items())
        message += f" | {details_str}"
    
    if status == "failed":
        logger.error(message)
    else:
        logger.info(message)


def log_summary(
    logger: logging.Logger,
    dataset_id: str,
    total_rows: int,
    successful: int,
    failed: int,
    total_tokens: int,
) -> None:
    """
    Log a processing summary.
    
    Args:
        logger: Logger instance
        dataset_id: Dataset ID processed
        total_rows: Total rows processed
        successful: Number of successful rows
        failed: Number of failed rows
        total_tokens: Total tokens used
    """
    logger.info(
        f"SUMMARY | dataset_id={dataset_id} | "
        f"total={total_rows} | successful={successful} | failed={failed} | "
        f"total_tokens={total_tokens}"
    )
