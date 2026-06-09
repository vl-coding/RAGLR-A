import threading
import time
from collections import defaultdict, deque
from typing import Optional


class RateLimiter:
    """
    Sliding-window rate limiter keyed by an arbitrary string (e.g. IP address).
    Thread-safe. No external dependencies.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._windows: dict = defaultdict(deque)

    def _evict(self, key: str, now: float) -> None:
        cutoff = now - self.window_seconds
        q = self._windows[key]
        while q and q[0] < cutoff:
            q.popleft()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            self._evict(key, now)
            if len(self._windows[key]) >= self.max_requests:
                return False
            self._windows[key].append(now)
            return True

    def remaining(self, key: str) -> int:
        now = time.time()
        with self._lock:
            self._evict(key, now)
            return max(0, self.max_requests - len(self._windows[key]))

    def retry_after(self, key: str) -> Optional[int]:
        """Seconds until the oldest request in the window expires."""
        now = time.time()
        with self._lock:
            self._evict(key, now)
            q = self._windows[key]
            if not q or len(q) < self.max_requests:
                return None
            return max(1, int(q[0] + self.window_seconds - now))
