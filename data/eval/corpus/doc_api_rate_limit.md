# Meridian Analytics — Search API Rate Limiting Reference

**Document ID:** doc_api_rate_limit
**Owner:** Platform Engineering
**Last updated:** 2026-03-01

## Overview

This document explains how rate limiting works on the Search API endpoint at Meridian Analytics and what the default thresholds are for the free tier versus the enterprise tier.

## Search API endpoint

`POST https://api.meridian-analytics.com/v3/search`

The Search API endpoint accepts a query vector and returns the nearest neighbors from the indexed document collection. Rate limiting on the Search API endpoint is applied per API key and is measured in requests per minute (RPM).

## How rate limiting works on the Search API endpoint

Rate limiting on the Search API endpoint uses a sliding window algorithm with a 60-second window. Each API key has a configurable request budget that resets on a rolling basis. The rate limiter evaluates the number of requests made in the preceding 60 seconds and compares it to the configured threshold for the customer's tier.

When the rate limit is exceeded, the API returns a `429 Too Many Requests` response with the following headers:

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | The maximum number of requests per minute for the tier |
| `X-RateLimit-Remaining` | The number of requests remaining in the current window |
| `X-RateLimit-Reset` | Unix timestamp when the rate limit window resets |
| `Retry-After` | Seconds to wait before retrying |

The rate limiter also tracks concurrency limits in addition to request rate. The concurrency limit is the maximum number of simultaneous in-flight requests allowed for a given API key. If the concurrency limit is exceeded, the endpoint returns a `429` response immediately.

## Default thresholds for free tier versus enterprise tier

The default thresholds for the Search API endpoint differ significantly between the free tier and enterprise tier:

| Threshold | Free tier | Enterprise tier |
|-----------|-----------|-----------------|
| Requests per minute | 60 RPM | 10,000 RPM |
| Requests per second (burst) | 5 RPS | 500 RPS |
| Concurrent requests | 10 | 500 |
| Monthly request quota | 100,000 | Unlimited (fair use) |

These default thresholds for the free tier are designed to support evaluation and small-scale prototyping. The enterprise tier thresholds accommodate production workloads with high query volumes. Customers on the free tier who exceed the monthly quota of 100,000 requests receive a notification and can either upgrade to the enterprise tier or wait until the quota resets at the start of the next billing cycle.

## Request prioritization

When the Search API endpoint approaches its rate limiting threshold, enterprise tier requests are prioritized over free tier requests. This priority-based queuing ensures that enterprise customers experience consistent latency even during traffic spikes. The prioritization is transparent to the client and does not require any configuration changes.

## Rate limit increase requests

Enterprise tier customers can request a rate limit increase by contacting their account manager. Rate limit increase requests are evaluated based on historical usage patterns and infrastructure capacity. Increases are typically provisioned within 5 business days. Temporary rate limit increases for special events (e.g., product launches, marketing campaigns) can be arranged with 2 weeks notice.

## Best practices

Clients should implement exponential backoff with jitter when handling 429 responses. The `Retry-After` header provides a suggested wait time. Meridian recommends a base delay of 1 second with a multiplier of 2 and maximum delay of 60 seconds.

## Revision history

This reference was last updated on 1 March 2026. Rate limiting thresholds are reviewed quarterly based on platform capacity planning.
