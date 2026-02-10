import logging
from pathlib import Path
import sys


def setup_logger(
    level: str = "DEBUG",
    log_dir: str = "logs",
    service_name: str = "Project-RAG",
):
    log_level = getattr(logging, level.upper(), logging.INFO)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Prevent duplicate handlers
    if root_logger.handlers:
        return root_logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # ---------- Single file handler ----------
    file_handler = logging.FileHandler(
        filename=log_path / f"{service_name}.log",
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # ---------- Stdout handler ----------
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stdout_handler)

    return root_logger
