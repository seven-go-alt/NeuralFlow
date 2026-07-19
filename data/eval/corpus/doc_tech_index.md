# Meridian Analytics — B-tree vs LSM-tree Index Performance for Write-Heavy Workloads

**Document ID:** doc_tech_index
**Owner:** Data Engineering
**Last updated:** 2026-07-15

## Overview

Meridian Analytics ingests time-series financial data at high velocity, producing a workload with approximately 90% **inserts** and 10% **point lookups** against the primary index. This document compares the **performance** characteristics of the **B-tree** index and the **LSM-tree** index for this **write-heavy** workload.

## Methodology

The comparison was conducted using Meridian's index benchmarking framework against a dataset of 50 million financial ticks. The workload mix was 90% **inserts** and 10% **point lookups** (random access by instrument ID + timestamp). Both indexes were configured with a 4 GB buffer pool and run on identical r5.large instances with NVMe storage.

## Insert Performance

For a **write-heavy** workload dominated by **inserts**, the **LSM-tree** index outperforms the **B-tree** index significantly:

| Metric | B-tree | LSM-tree | Improvement |
|--------|--------|----------|-------------|
| Sustained insert throughput | 85,000 ops/s | 340,000 ops/s | 4.0x |
| p99 insert latency | 12 ms | 3.1 ms | 3.9x |
| Write amplification | 8.5x | 3.2x | 2.7x |
| Disk space overhead (steady state) | 1.3x | 1.8x | -1.4x |

The **LSM-tree** achieves superior **insert** performance because it buffers incoming writes in a memory-resident memtable and flushes them to disk in sequential SSTable files, avoiding the random page writes that burden the **B-tree** index. For Meridian's 90% **inserts** workload, this difference is decisive.

## Point Lookup Performance

For the 10% **point lookups**, the **B-tree** index performs better due to its in-place update structure:

| Metric | B-tree | LSM-tree |
|--------|--------|----------|
| Average point lookup latency | 0.4 ms | 1.8 ms |
| p99 point lookup latency | 2.1 ms | 14.3 ms |

The **B-tree** index can locate a key in O(log n) with at most 3–4 disk seeks for a 50 million record index. The **LSM-tree** index must search multiple SSTable levels, resulting in higher tail latency for **point lookups**. However, this is acceptable for Meridian's workload given that **point lookups** represent only 10% of operations.

## Write Amplification Considerations

Write amplification is a critical **performance** factor for **write-heavy** workloads on SSDs. The **B-tree** index exhibits 8.5x write amplification because each insert triggers random page writes and B-tree rebalancing operations. The **LSM-tree** index achieves only 3.2x write amplification through compaction in the background, though this comes at the cost of periodic compaction spikes that can temporarily degrade **point lookups** latency.

## Recommendation

For Meridian's 90% **inserts** / 10% **point lookups** workload, the **LSM-tree** index is the recommended choice because:

1. Insert throughput is the primary bottleneck, and the **LSM-tree** index provides 4x improvement over the **B-tree** index.
2. The 10% **point lookups** penalty (1.8 ms vs 0.4 ms average) is within Meridian's service-level objective of 20 ms.
3. Compaction management can be tuned with rate limiters and leveled compaction strategies to avoid latency spikes during peak ingestion hours.
4. The additional disk space overhead (1.8x vs 1.3x) is acceptable given Meridian's provisioned storage capacity.

The **B-tree** index remains the appropriate choice for workloads where **point lookups** dominate or where low tail latency for reads is the primary requirement.

## Revision History

This comparison was last updated on 15 July 2026 following the benchmarking of Meridian's tick ingestion pipeline against the new LSM-tree storage engine.
