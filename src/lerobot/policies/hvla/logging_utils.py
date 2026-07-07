"""Logging setup for HVLA processes.

Spawned processes (via multiprocessing 'spawn' context) start with a fresh
Python interpreter — logging config from the parent is NOT inherited.
Call setup_process_logging() at the top of any spawned process entry point.
"""

import logging
import sys


def setup_process_logging(level: int = logging.INFO):
    """Configure logging for a spawned process. Must be called before any logger use."""
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers (avoid duplicates on re-init)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    # Force flush on every write so logs appear immediately
    handler.flush = sys.stdout.flush

    root.addHandler(handler)
