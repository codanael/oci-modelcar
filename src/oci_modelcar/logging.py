"""Pipeline logging: text + Azure DevOps formatters, output_variable."""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from typing import IO


def fmt_bytes(n: int) -> str:
    """Human-readable byte count using GB/MB/KB scaling."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


class AzureFormatter(logging.Formatter):
    """Azure DevOps logging commands. WARNING/ERROR get task.logissue prefix."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f"##[error]{msg}"
        if record.levelno >= logging.WARNING:
            return f"##[warning]{msg}"
        return msg


class PipelineLogger:
    def __init__(
        self,
        stream: IO[str] | None = None,
        log_style: str = "text",
        verbose: bool = False,
        quiet: bool = False,
    ) -> None:
        self.stream = stream or sys.stdout
        self.log_style = log_style
        self.verbose = verbose
        self.quiet = quiet
        self._fmt: logging.Formatter = AzureFormatter() if log_style == "azure" else TextFormatter()
        self._lock = threading.Lock()

    def _emit(self, level: int, msg: str) -> None:
        if self.quiet and level < logging.WARNING:
            return
        if level == logging.DEBUG and not self.verbose:
            return
        rec = logging.LogRecord(
            name="oci-modelcar",
            level=level,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        with self._lock:
            print(self._fmt.format(rec), file=self.stream, flush=True)

    def section(self, title: str) -> None:
        with self._lock:
            if self.log_style == "azure":
                print(f"##[section]{title}", file=self.stream, flush=True)
            else:
                print(f"\n== {title} ==", file=self.stream, flush=True)

    def debug(self, msg: str) -> None:
        self._emit(logging.DEBUG, msg)

    def info(self, msg: str) -> None:
        self._emit(logging.INFO, msg)

    def warning(self, msg: str) -> None:
        self._emit(logging.WARNING, msg)

    def error(self, msg: str) -> None:
        self._emit(logging.ERROR, msg)

    def output_variable(self, name: str, value: str) -> None:
        with self._lock:
            if self.log_style == "azure":
                print(
                    f"##vso[task.setvariable variable={name};isOutput=true]{value}",
                    file=self.stream,
                    flush=True,
                )
            else:
                print(f"{name}={value}", file=self.stream, flush=True)


class ProgressEmitter:
    """Throttled progress reporter: emits at most once per `interval` seconds.

    Designed for wiring into HfDownloader's `progress_cb`. The first .update()
    emits immediately so a fast download (< interval) still logs at least one
    line; subsequent calls are throttled to one emit per `interval`.
    """

    def __init__(
        self,
        emit: Callable[[str], None],
        path: str,
        total: int,
        interval: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._emit = emit
        self._path = path
        self._total = total
        self._interval = interval
        self._clock = clock
        # Sentinel: -inf forces the first update() past the throttle.
        self._last = float("-inf")

    def update(self, transferred: int) -> None:
        now = self._clock()
        if now - self._last < self._interval:
            return
        self._last = now
        pct = int(100 * transferred / self._total) if self._total > 0 else 0
        self._emit(f"{self._path}: {pct}% ({fmt_bytes(transferred)} / {fmt_bytes(self._total)})")
