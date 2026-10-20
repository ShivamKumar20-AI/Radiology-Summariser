"""Request guards for the public API: size cap, rate limits and a cap on simultaneous work.

Kept free of torch/transformers so it can be tested quickly and in CI.
"""
import threading
import time
from collections import deque

MAX_FINDINGS_CHARS = 5000      # same limit the web page shows; now enforced by the server too
MAX_REQUESTS_PER_IP_PER_MIN = 6
MAX_REQUESTS_ALL_PER_MIN = 20
MAX_CONCURRENT_SUMMARIES = 2   # 1 CPU / 1 GB: more at once just makes everyone slower


class SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            while self._times and now - self._times[0] >= self.window_seconds:
                self._times.popleft()
            if len(self._times) >= self.max_requests:
                return False
            self._times.append(now)
            return True

    def idle(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            return not self._times or now - self._times[-1] >= self.window_seconds


class PerKeyLimiter:
    """One sliding window per key (here: per client address)."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0, max_keys: int = 1000):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._limiters: dict[str, SlidingWindowLimiter] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        with self._lock:
            if key not in self._limiters and len(self._limiters) >= self.max_keys:
                # forget addresses that have been quiet for a full window, so memory stays bounded
                self._limiters = {k: v for k, v in self._limiters.items() if not v.idle(now)}
            limiter = self._limiters.setdefault(key, SlidingWindowLimiter(self.max_requests, self.window_seconds))
        return limiter.allow(now)


def client_address(headers, fallback: str = "unknown") -> str:
    """Fly.io puts the real visitor address in Fly-Client-IP; the socket address is Fly's proxy."""
    return headers.get("fly-client-ip") or fallback
