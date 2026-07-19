# Meridian Analytics — Embedding Model Performance Benchmarking

**Document ID:** doc_tech_embedding
**Owner:** ML Engineering
**Last updated:** 2026-07-02

## Benchmark Setup

This document presents the **performance benchmarks** for the Meridian **embedding model** (v3, 1536 dimensions) when processing **10,000 documents** of **512 tokens** each. Benchmarks were conducted on a dedicated inference cluster comprising 4 nodes with NVIDIA A10G GPUs (24 GB VRAM each), using ONNX Runtime with CUDA execution provider. Each document was padded or truncated to exactly **512 tokens** before inference.

## Latency Metrics

The **performance benchmarks** for the **embedding model** processing **10,000 documents** of **512 tokens** each yielded the following latency results:

| Metric | Value |
|--------|-------|
| P50 latency per batch (32 docs) | 185 ms |
| P95 latency per batch (32 docs) | 312 ms |
| P99 latency per batch (32 docs) | 480 ms |
| Mean document latency | 7.2 ms |
| Total batch processing time | 5,780 ms |

The **embedding model** achieves a mean per-document latency of 7.2 ms when processing **10,000 documents** of **512 tokens** each, making it suitable for real-time indexing workflows.

## Throughput Metrics

Throughput **performance benchmarks** for the **embedding model** across **10,000 documents** at **512 tokens** per document:

| Metric | Value |
|--------|-------|
| Documents per second | 132 docs/s |
| Tokens per second | 67,584 tokens/s |
| Total end-to-end time | 76 seconds |
| GPU utilization (avg) | 87% |
| GPU memory utilization (avg) | 14.2 GB / 24 GB |

The **embedding model** sustains a throughput of 132 documents per second under the **10,000 documents** / **512 tokens** workload, which exceeds the production requirement of 100 docs/s at peak ingestion.

## Batch Size Optimization

The **performance benchmarks** identified the optimal batch size for the **embedding model** to be 32 documents per batch when processing inputs of **512 tokens**. Smaller batch sizes (8 or 16) increase throughput variance due to underutilized GPU cores, while larger batches (64) cause VRAM pressure and occasional out-of-memory errors. The following table shows throughput at different batch sizes:

| Batch size | Docs/sec | GPU utilization |
|------------|----------|-----------------|
| 8 | 89 docs/s | 62% |
| 16 | 114 docs/s | 74% |
| 32 | 132 docs/s | 87% |
| 64 | 108 docs/s | 93% (OOM at p99) |

## Cost per 10,000 Documents

At Meridian's reserved A10G pricing ($1.80 per GPU-hour), processing **10,000 documents** of **512 tokens** costs approximately $0.026 in compute resources when accounting for the 76-second end-to-end time across 4 nodes. This is well within the budget target of $0.05 per 10K documents set by the finance team for the indexing pipeline.

## Revision History

These **performance benchmarks** were last updated on 2 July 2026 following the v3 **embedding model** deployment and regression testing.
