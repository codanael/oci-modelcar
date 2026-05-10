"""Pipeline logging: text + Azure DevOps formatters, output_variable."""

from __future__ import annotations

import logging
import sys
from typing import IO


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
        print(self._fmt.format(rec), file=self.stream, flush=True)

    def section(self, title: str) -> None:
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
        if self.log_style == "azure":
            print(
                f"##vso[task.setvariable variable={name};isOutput=true]{value}",
                file=self.stream,
                flush=True,
            )
        else:
            print(f"{name}={value}", file=self.stream, flush=True)
