"""Bounded executor for ingestion jobs."""
from concurrent.futures import ThreadPoolExecutor

_executor: ThreadPoolExecutor | None = None


def start() -> None:
    """Start the ingestion executor during application startup."""
    global _executor
    _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="book-ingest")


def submit(fn, *args) -> None:
    """Submit an ingestion job without blocking the HTTP response."""
    if _executor is None:
        raise RuntimeError("Ingestion executor is not running")
    _executor.submit(fn, *args)


def shutdown() -> None:
    """Wait for running jobs to stop when the service shuts down."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True, cancel_futures=True)
        _executor = None
