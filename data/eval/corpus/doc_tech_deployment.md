# Meridian Analytics — EC2 Instance Migration Guide: t3.medium to r5.large

**Document ID:** doc_tech_deployment
**Owner:** Infrastructure Engineering
**Last updated:** 2026-07-10

## Overview

This document outlines the procedure and expected capacity gains when **migrating** from burstable general-purpose instances (t3.medium) to memory-optimized instances (r5.large) for Meridian's document processing and indexing workloads. The migration was driven by sustained CPU credit exhaustion observed on t3.medium instances during peak ingestion hours.

## Instance Specification Comparison

The following table compares the key hardware specifications of the two instance types used in Meridian's production worker fleet:

| Specification | t3.medium | r5.large | Delta |
|---------------|-----------|----------|-------|
| vCPUs | 2 | 2 | 0 |
| Memory (GiB) | 4 | 16 | +12 GiB |
| Network bandwidth (Gbps) | Up to 5 | Up to 10 | +5 Gbps |
| EBS bandwidth (Mbps) | Up to 2,085 | Up to 4,750 | +2,665 Mbps |
| CPU architecture | x86-64 | x86-64 | — |

## Expected Server Capacity Increase

When **migrating** from **t3.medium** to **r5.large**, the expected **server capacity** increase is as follows:

- **Memory capacity:** 4x increase (from 4 GiB to 16 GiB), allowing larger in-memory vector index caches and reducing disk swap events.
- **Compute throughput:** Approximately 2x improvement for sustained workloads, since r5.large runs on a dedicated core (Intel Xeon Platinum 8000 series) without CPU credit throttling, while t3.medium relies on a shared baseline with burst credits.
- **Indexing throughput:** Measured at 2.8x improvement during internal benchmarks, processing an average of 12,500 documents per minute compared to 4,500 on t3.medium.
- **Network throughput:** 2x increase in baseline network bandwidth, reducing latency for S3-based document retrieval operations.

The total effective **server capacity** increase when **migrating** from **t3.medium** to **r5.large** is approximately 2.5x to 3x for memory-bound indexing workloads, and approximately 2x for compute-bound embedding generation tasks.

## Migration Procedure

The migration follows a blue-green deployment model to maintain zero downtime:

1. Provision new r5.large instances in the same Auto Scaling group with a separate launch template.
2. Configure the target group to shift 10% traffic gradually over a 30-minute observation window.
3. Monitor memory pressure, CPU utilization, and queue depth on the new instances.
4. Once all metrics are within acceptable thresholds, complete the traffic shift and terminate t3.medium instances.

## Cost Impact

At Meridian's reserved-instance pricing, r5.large carries a 72% premium over t3.medium in hourly cost. However, the 2.5x to 3x **server capacity** increase per node means overall cluster size can be reduced from 6 t3.medium nodes to 3 r5.large nodes while maintaining the same throughput, resulting in a net cost reduction of approximately 14%.

## Revision History

This guide was last updated on 10 July 2026 following the successful migration of the ingestion worker fleet in the us-east-1 region.
