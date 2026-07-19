# Meridian Analytics — AWS SDK Retry Logic with Exponential Backoff

**Document ID:** doc_tech_retry
**Owner:** Platform Engineering
**Last updated:** 2026-07-12

## Overview

Meridian Analytics integrates with several AWS services (S3, DynamoDB, SQS) through the AWS SDK for Python (boto3). Many of these callers operate within a total **deadline** of 30 seconds. This document defines the recommended **retry logic** with **exponential backoff** and **jitter** that Meridian's engineering teams use to maximize success rates within the time budget.

## Retry Configuration

Meridian's standard **retry logic** configuration for AWS SDK callers with a 30-second **deadline** uses a capped **exponential backoff** strategy with full **jitter**:

```python
import time
import random
from botocore.config import Config
from botocore.exceptions import ClientError

RETRYABLE_ERRORS = [
    "ThrottlingException",
    "ProvisionedThroughputExceededException",
    "RequestLimitExceeded",
    "ServiceUnavailable",
]

def call_with_retry(client, operation, **kwargs):
    max_attempts = 5
    base_delay = 0.1  # 100ms initial backoff
    max_delay = 5.0   # Cap at 5 seconds per attempt

    for attempt in range(1, max_attempts + 1):
        try:
            return getattr(client, operation)(**kwargs)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code not in RETRYABLE_ERRORS:
                raise  # Non-retryable — propagate immediately

            if attempt == max_attempts:
                raise  # Last attempt exhausted — propagate

            # Exponential backoff with full jitter
            sleep_time = min(base_delay * (2 ** (attempt - 1)), max_delay)
            sleep_time = random.uniform(0, sleep_time)  # Full jitter
            time.sleep(sleep_time)
```

The **exponential backoff** starts at 100ms and doubles with each attempt: 100ms, 200ms, 400ms, 800ms, 1600ms. The **jitter** randomizes each delay between 0 and the computed backoff value to prevent thundering herd problems.

## Deadline Budget Allocation

With a 30-second **deadline** for the overall operation, Meridian allocates the time budget across **retry logic** attempts:

| Attempt | Backoff (with jitter) | Cumulative time (worst case) |
|---------|----------------------|------------------------------|
| 1 (initial) | 0 ms | 0–100ms (request time) |
| 2 | 0–200ms | ~300ms |
| 3 | 0–400ms | ~700ms |
| 4 | 0–800ms | ~1.5s |
| 5 | 0–1600ms | ~3.1s |

The total worst case for 1 initial request plus 4 retries is approximately 3.1 seconds of backoff + API latency, which fits comfortably within the 30-second **deadline**. This leaves over 26 seconds for the actual API call processing time.

## Jitter Implementation

Meridian uses **full jitter** for its **retry logic**:

```python
sleep = random.uniform(0, min(base * 2**attempt, cap))  # Full jitter
```

Full **jitter** spreads retry traffic uniformly across the recovery window, minimizing synchronized retries across multiple callers — critical for Meridian's container-based deployments where dozens of replicas may fail simultaneously during throttling events.

## AWS SDK Integration

Meridian configures the AWS SDK's built-in **retry logic** through the botocore Config object, choosing the adaptive retry mode which automatically detects throttling patterns:

```python
config = Config(
    retries={
        "max_attempts": 5,
        "mode": "adaptive"
    },
    connect_timeout=5,
    read_timeout=10,
)
```

The adaptive mode uses a client-side rate limiter that adjusts the maximum send rate based on observed throttling rates, which supplements the **exponential backoff** logic.

## Monitoring Retry Effectiveness

Meridian tracks the following metrics for each AWS SDK caller:
- `aws_sdk_retry_attempt_count`: Retry attempts per operation.
- `aws_sdk_retry_success`: Operations that succeeded after retry.
- `aws_sdk_retry_deadline_exceeded`: Operations that exhausted the **deadline** or max attempts.
- `aws_sdk_retry_jitter_delay_ms`: Actual delay applied by the **jitter** function.

An alert fires if `aws_sdk_retry_deadline_exceeded` exceeds 1% of total calls, indicating the **exponential backoff** configuration needs tuning or the AWS service limit must be increased.

## Revision History

This document was last updated on 12 July 2026 following the transition from equal jitter to full jitter across all Meridian AWS SDK callers.
