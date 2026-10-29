import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, Request

from app.config import get_settings
from app.errors import ApiError


def require_admin(x_admin_key: str = Header(default="")) -> None:
    expected = get_settings().admin_key
    if not expected or not hmac.compare_digest(x_admin_key.encode(), expected.encode()):
        raise ApiError(401, "unauthorized", "A valid X-Admin-Key header is required.")


def require_upload_key(x_upload_key: str = Header(default="")) -> None:
    expected = get_settings().demo_upload_key
    if expected and not hmac.compare_digest(x_upload_key.encode(), expected.encode()):
        raise ApiError(401, "unauthorized", "A valid X-Upload-Key header is required for uploads.")


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Sliding-window limit per key, kept in memory (one API process)."""

    def __init__(self, limit: int, window_seconds: float):
        self.limit, self.window = limit, window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                retry = int(self.window - (now - hits[0])) + 1
                raise ApiError(429, "rate_limited", f"Too many requests. Try again in {retry} s.")
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


settings = get_settings()
search_limiter = RateLimiter(settings.search_rate_per_min, 60)
upload_limiter = RateLimiter(settings.upload_rate_per_hour, 3600)
