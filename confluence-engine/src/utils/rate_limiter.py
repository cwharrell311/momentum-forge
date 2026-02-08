"""
Token bucket rate limiter for external API calls.

Why this exists: APIs like FMP and Unusual Whales have rate limits.
If you exceed them, you get temporarily banned. This module ensures
we never send requests faster than allowed.

How it works (token bucket algorithm):
- You start with a bucket of N tokens
- Each API call costs 1 token
- Tokens refill at a fixed rate (e.g., 5 per second)
- If the bucket is empty, you wait until a token refills
- The bucket has a max size so tokens don't accumulate forever

Usage:
    limiter = RateLimiter(rate=5.0, max_tokens=10)  # 5 requests/sec, burst of 10
    await limiter.acquire()  # Waits if necessary
    response = await client.get(url)
"""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Async token bucket rate limiter."""

    def __init__(self, rate: float = 5.0, max_tokens: int = 10):
        """
        Args:
            rate: Tokens added per second (i.e., max sustained requests/sec)
            max_tokens: Maximum burst size (bucket capacity)
        """
        self.rate = rate
        self.max_tokens = max_tokens
        self._tokens = float(max_tokens)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Wait until a token is available, then consume one.

        This is the only method you call. It blocks (awaits) if
        the bucket is empty, ensuring you never exceed the rate limit.
        """
        async with self._lock:
            self._refill()

            if self._tokens < 1.0:
                # Calculate how long to wait for 1 token
                wait_time = (1.0 - self._tokens) / self.rate
                await asyncio.sleep(wait_time)
                self._refill()

            self._tokens -= 1.0

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.max_tokens,
            self._tokens + elapsed * self.rate,
        )
        self._last_refill = now
