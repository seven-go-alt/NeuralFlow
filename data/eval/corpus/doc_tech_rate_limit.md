# Meridian Analytics — Token Bucket Rate Limiting for GraphQL Resolvers

**Document ID:** doc_tech_rate_limit
**Owner:** API Platform Team
**Last updated:** 2026-07-05

## Overview

Meridian Analytics exposes customer-facing analytics data through a GraphQL API that aggregates information from multiple downstream services. Several **GraphQL resolver** functions call a third-party **downstream REST API** for enrichment data, including market trend signals and competitive benchmarks. This downstream API enforces a hard limit of 100 **requests per minute** per API key. This document describes the **rate limiter** implementation using the **token bucket** algorithm at the GraphQL resolver level to enforce compliance with these downstream constraints while maximizing throughput for Meridian's customers.

## Token Bucket Algorithm Implementation

Meridian implements the token bucket algorithm as a standalone rate limiting library (`meridian-ratelimit`) deployed alongside the GraphQL resolver service. The algorithm works as follows for the 100 requests per minute constraint:

```python
import time
import threading

class TokenBucket:
    def __init__(self, capacity, refill_rate, refill_interval=1.0):
        self.capacity = capacity          # maximum token count (100)
        self.tokens = capacity            # start full
        self.refill_rate = refill_rate    # tokens per second (100/60 ≈ 1.667)
        self.refill_interval = refill_interval
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        if new_tokens > 0:
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now

    def try_acquire(self, tokens=1):
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
```

The bucket has a **capacity** of 100 tokens (representing the per-minute limit). Tokens are **refilled** at a rate of 100/60 = 1.667 tokens per second, checked on a 1-second granularity. When a resolver tries to acquire a token and none are available, the request is queued or rejected based on the resolver's priority level. This approach smooths out request bursts while keeping the average within the 100 requests per minute limit.

## GraphQL Resolver Integration

Each GraphQL resolver that calls the downstream REST API is wrapped with the rate limiter middleware:

```python
from meridian_ratelimit import TokenBucket, RateLimitExceeded

# One bucket per API key
rate_limit_buckets: dict[str, TokenBucket] = {}

def get_bucket(api_key: str) -> TokenBucket:
    if api_key not in rate_limit_buckets:
        rate_limit_buckets[api_key] = TokenBucket(
            capacity=100,
            refill_rate=100 / 60  # 1.667 tokens/sec
        )
    return rate_limit_buckets[api_key]

@RateLimited(bucket_provider=get_bucket, max_wait_ms=500)
async def market_trends_resolver(parent, args, context, info):
    """GraphQL resolver for market trends, rate limited per API key."""
    api_key = context.headers.get("x-api-key")
    bucket = get_bucket(api_key)

    if not bucket.try_acquire():
        raise RateLimitExceeded(
            "Market trends downstream API rate limit reached. "
            f"Retry after {bucket.seconds_until_token_available():.1f} seconds.",
            retry_after_seconds=bucket.seconds_until_token_available()
        )

    response = await call_downstream_market_api(api_key, args)
    return normalize_market_data(response)
```

The middleware checks the bucket before making the downstream call. If empty, it returns a `RateLimitExceeded` error with a `retry_after_seconds` value derived from the time until the next token. Meridian's Apollo Gateway translates this into an HTTP 429 status.

## Burst Handling and Priority Queuing

The token bucket capacity of 100 tokens naturally accommodates bursts: an idle client accumulates up to 100 tokens and can send 100 requests rapidly. After exhausting the bucket, requests are limited to 1.667/sec. For high-priority Tier 1 customers, Meridian configures capacity of 200 with the same refill rate, allowing larger bursts while averaging 100 req/min.

## Monitoring and Token Utilization

Meridian tracks token bucket metrics through CloudWatch: `rate_limit_bucket_level` (current token count), `rate_limit_accepted`, and `rate_limit_rejected`. The `token_utilization_pct` metric (100 - (available_tokens / capacity * 100)) determines whether the 100 requests per minute limit is sufficient. When average utilization exceeds 85%, an upstream limit increase is requested.
