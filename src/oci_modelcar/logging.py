"""Pipeline logger with text and Azure DevOps formatters."""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import IO, Literal

LogStyle = Literal["text", "azure"]


def detect_log_style(explicit: str | None) -> LogStyle:
    if explicit in ("text", "azure"):
        return explicit  # type: ignore[return-value]
    if os.environ.get("TF_BUILD", "").strip().lower() == "true":
        return "azure"
    return "text"


def _supports_color(stream: IO[str]) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return stream.isatty()
    except (AttributeError, ValueError):  # fmt: skip
        return False


class _Formatter:
    def section(self, title: str) -> str:
        raise NotImplementedError

    def group_start(self, title: str) -> str:
        raise NotImplementedError

    def group_end(self, summary: str) -> str:
        raise NotImplementedError

    def info(self, msg: str) -> str:
        return msg + "\n"

    def warning(self, msg: str) -> str:
        raise NotImplementedError

    def error(self, msg: str) -> str:
        raise NotImplementedError

    def debug(self, msg: str) -> str:
        raise NotImplementedError

    def progress(self, percent: int, msg: str) -> str:
        raise NotImplementedError

    def output_variable(self, name: str, value: str) -> str:
        raise NotImplementedError


class AzureFormatter(_Formatter):
    def section(self, title: str) -> str:
        return f"##[section]{title}\n"

    def group_start(self, title: str) -> str:
        return f"##[group]{title}\n"

    def group_end(self, summary: str) -> str:
        return (summary + "\n" if summary else "") + "##[endgroup]\n"

    def warning(self, msg: str) -> str:
        return f"##[warning]{msg}\n"

    def error(self, msg: str) -> str:
        return f"##[error]{msg}\n"

    def debug(self, msg: str) -> str:
        return f"##[debug]{msg}\n"

    def progress(self, percent: int, msg: str) -> str:
        return f"##vso[task.setprogress value={percent}]{msg}\n"

    def output_variable(self, name: str, value: str) -> str:
        vso = f"##vso[task.setvariable variable={name};isOutput=true]{value}\n"
        kv = f"{name.upper()}={value}\n"
        return vso + kv


_RESET = "\033[0m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_BOLD = "\033[1m"


class TextFormatter(_Formatter):
    def __init__(self, use_color: bool) -> None:
        self.use_color = use_color

    def _c(self, code: str, msg: str) -> str:
        return f"{code}{msg}{_RESET}" if self.use_color else msg

    def section(self, title: str) -> str:
        bar = "─" * max(2, 60 - len(title))
        line = f"── {title} {bar}"
        return self._c(_BOLD, line) + "\n"

    def group_start(self, title: str) -> str:
        return f"{title}\n"

    def group_end(self, summary: str) -> str:
        return (summary + "\n") if summary else ""

    def warning(self, msg: str) -> str:
        return self._c(_YELLOW, f"WARN  {msg}") + "\n"

    def error(self, msg: str) -> str:
        return self._c(_RED, f"ERROR {msg}") + "\n"

    def debug(self, msg: str) -> str:
        return self._c(_DIM, f"DEBUG {msg}") + "\n"

    def progress(self, percent: int, msg: str) -> str:
        return f"[{percent:>3}%] {msg}\n"

    def output_variable(self, name: str, value: str) -> str:
        return f"{name.upper()}={value}\n"


class PipelineLogger:
    def __init__(
        self,
        stream: IO[str] | None = None,
        style: LogStyle = "text",
        use_color: bool | None = None,
        verbose: bool = False,
        quiet: bool = False,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.verbose = verbose
        self.quiet = quiet
        if use_color is None:
            use_color = _supports_color(self.stream)
        self.formatter: _Formatter = (
            AzureFormatter() if style == "azure" else TextFormatter(use_color)
        )
        self._lock = threading.Lock()

    def _emit(self, text: str) -> None:
        with self._lock:
            self.stream.write(text)
            self.stream.flush()

    def section(self, title: str) -> None:
        if self.quiet:
            return
        self._emit(self.formatter.section(title))

    def group_start(self, title: str) -> None:
        self._emit(self.formatter.group_start(title))

    def group_end(self, summary: str = "") -> None:
        self._emit(self.formatter.group_end(summary))

    def info(self, msg: str) -> None:
        if self.quiet:
            return
        self._emit(self.formatter.info(msg))

    def warning(self, msg: str) -> None:
        self._emit(self.formatter.warning(msg))

    def error(self, msg: str) -> None:
        self._emit(self.formatter.error(msg))

    def debug(self, msg: str) -> None:
        if not self.verbose:
            return
        self._emit(self.formatter.debug(msg))

    def progress(self, percent: int, msg: str) -> None:
        if self.quiet:
            return
        self._emit(self.formatter.progress(percent, msg))

    def output_variable(self, name: str, value: str) -> None:
        self._emit(self.formatter.output_variable(name, value))

    def heartbeat(self, line: str) -> None:
        # Heartbeats are plain text in both styles, no tags
        self._emit(f"[HB] {line}\n")

    @contextmanager
    def file_scope(self, title: str) -> Iterator[FileScopedLogger]:
        """Buffer per-file logs (for parallel mode); flush atomically."""
        scoped = FileScopedLogger(title, self)
        try:
            yield scoped
        finally:
            scoped.flush()


def _fmt_bytes(n: int) -> str:
    """Human-readable byte count using GB/MB/KB scaling."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


class ProgressEmitter:
    """Throttled progress reporter: emits at most once per `interval` seconds."""

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
        self._last = clock()

    def update(self, transferred: int) -> None:
        now = self._clock()
        if now - self._last < self._interval:
            return
        self._last = now
        pct = 100.0 * transferred / self._total if self._total > 0 else 0.0
        self._emit(
            f"{self._path}: {pct:.0f}% ({_fmt_bytes(transferred)} / {_fmt_bytes(self._total)})"
        )


class FileScopedLogger:
    """Buffer all logs for one file; flush atomically at close."""

    def __init__(self, title: str, parent: PipelineLogger) -> None:
        self._buf: list[str] = []
        self._parent = parent
        self._summary = ""
        self._title = title
        self._buf.append(parent.formatter.group_start(title))

    def info(self, msg: str) -> None:
        if not self._parent.quiet:
            self._buf.append(self._parent.formatter.info(msg))

    def warning(self, msg: str) -> None:
        self._buf.append(self._parent.formatter.warning(msg))

    def error(self, msg: str) -> None:
        self._buf.append(self._parent.formatter.error(msg))

    def debug(self, msg: str) -> None:
        if self._parent.verbose:
            self._buf.append(self._parent.formatter.debug(msg))

    def set_summary(self, summary: str) -> None:
        self._summary = summary

    def flush(self) -> None:
        self._buf.append(self._parent.formatter.group_end(self._summary))
        self._parent._emit("".join(self._buf))
