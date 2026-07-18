# Meridian Analytics — API Throughput Limits Reference

**Document ID:** doc_api_throughput
**Owner:** Platform Engineering
**Last updated:** 2026-03-08

## Overview

This document describes the throughput limits for Meridian Analytics API endpoints, focusing on the maximum number of API requests per second for the batch processing endpoint and how it compares to the real-time endpoint limit. Understanding these limits is essential for designing efficient data pipelines and avoiding throttling.

## Endpoints

- **Real-time endpoint:** `POST https://api.meridian-analytics.com/v3/process/realtime`
- **Batch processing endpoint:** `POST https://api.meridian-analytics.com/v3/process/batch`

## Maximum API requests per second for the batch processing endpoint

The maximum number of API requests per second for the batch processing endpoint is 5,000 requests per second (RPS) for enterprise tier customers. This throughput limit applies to the total number of batch job submissions, not to the individual document processing rate within each batch. Each batch job can contain up to 10,000 documents, and the batch processing endpoint processes documents asynchronously with results delivered to a configured webhook or retrieved via the job status endpoint.

The batch processing endpoint is designed for high-throughput, non-real-time workloads such as nightly re-indexing jobs, bulk metadata updates, and large-scale document ingestion. The 5,000 RPS limit allows customers to submit large volumes of work quickly while the platform manages the underlying processing capacity.

## Real-time endpoint limit comparison

The real-time endpoint has a significantly lower limit by design. The maximum number of API requests per second for the real-time endpoint is 500 RPS for enterprise tier customers. This throughput limit ensures consistent low-latency responses for interactive use cases such as search queries, single-document processing, and embedding generation.

The following table compares the throughput limits across both endpoints:

| Metric | Real-time endpoint | Batch processing endpoint |
|--------|--------------------|--------------------------|
| Maximum requests per second | 500 RPS | 5,000 RPS |
| Maximum documents per request | 1 | 10,000 |
| Processing mode | Synchronous | Asynchronous |
| Typical latency | < 500 ms | 1–30 minutes per batch |
| Free tier limit | 5 RPS | 50 RPS |

## Throughput factors

The actual throughput a customer experiences depends on several factors beyond the API requests per second limits. Document complexity, file size, and the selected embedding dimensions affect processing time. For the batch processing endpoint, the total throughput is also influenced by the number of concurrent batch jobs. Enterprise customers can run up to 25 concurrent batch jobs. Free tier customers are limited to 2 concurrent batch jobs.

## Burst behavior

Both the real-time endpoint and the batch processing endpoint allow short bursts above the stated maximum API requests per second limits. The burst allowance is 20% above the limit for up to 30 seconds. After a burst period, the client must reduce the request rate to stay within the sustained limit for at least 60 seconds before another burst is allowed.

## Monitoring throughput

Customers can monitor their current throughput usage through the Meridian Dashboard Metrics API. The endpoint `/v3/metrics/throughput` returns real-time and historical request rates for each endpoint, including the number of API requests per second and the number of concurrent jobs.

## Revision history

This reference was last updated on 8 March 2026. Throughput limits are subject to change with 30 days notice for contract customers.
