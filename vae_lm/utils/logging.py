from typing import Union
import torch
import logging
from pathlib import Path
from loguru import logger
from .filters import Filter
from datetime import datetime
from rich.console import Console


class SaveBatchHandler(logging.Handler):
    """Handler to save batches with error during training."""
    def __init__(self, level: Union[int, str] = logging.NOTSET) -> None:
        super().__init__(level=level)
        self._directory = Path.cwd() / "error-batches"
        if not self._directory.exists():
            self._directory.mkdir(exist_ok=False)

    def emit(self, record: logging.LogRecord) -> None:
        time = datetime.now()
        file_suffix = time.strftime("%d-%m-%Y_%H-%M")
        torch.save(record["batch"], self._directory / f"error-batch-{file_suffix}.pt")


class RichExceptionHandler(logging.Handler):
    """Much better Rich handler which works with Loguru."""
    def __init__(self, level: Union[int, str] = logging.NOTSET) -> None:
        super().__init__(level=level)
        self._console = Console()

    def emit(self, record: logging.LogRecord) -> None:
        self._console.print_exception()


def setup_logging() -> None:
    logger.add(RichExceptionHandler(), level=logging.ERROR, format="{message}")
    logger.add(
        SaveBatchHandler(),
        filter=Filter(type="has_attr", condition="batch"),
        format="{message}",
    )
