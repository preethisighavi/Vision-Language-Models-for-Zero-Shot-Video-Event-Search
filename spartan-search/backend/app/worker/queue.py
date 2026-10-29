"""Enqueues work on RQ when REDIS_URL is set, otherwise on FastAPI BackgroundTasks."""

import logging
from typing import Callable

from fastapi import BackgroundTasks

from app.config import get_settings

log = logging.getLogger(__name__)


def _swallow(fn: Callable, *args) -> None:
    try:
        fn(*args)
    except Exception:
        log.exception("background task %s failed", fn.__name__)


def enqueue(background: BackgroundTasks, fn: Callable, *args) -> str:
    """Returns "rq" or "background" so callers can report how the job runs."""
    redis_url = get_settings().redis_url
    if redis_url:
        try:
            from redis import Redis
            from rq import Queue

            Queue("spartan", connection=Redis.from_url(redis_url)).enqueue(fn, *args, job_timeout=3600)
            return "rq"
        except Exception:
            log.exception("RQ enqueue failed; falling back to BackgroundTasks")
    background.add_task(_swallow, fn, *args)
    return "background"
