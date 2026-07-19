# Meridian Analytics — RAG Pipeline Performance Analysis

**Document ID:** doc_tech_perf
**Owner:** ML Engineering
**Last updated:** 2026-07-14

## End-to-End Latency Breakdown

This document provides a **performance analysis** of the AskMeridian RAG pipeline, with a focus on identifying the **bottleneck** among the pipeline stages. The **end-to-end latency** is the sum of all sequential stage latencies plus network overhead.

## Latency Calculation

Given the following stage latencies measured under production load:

| Stage | Measured latency |
|-------|-----------------|
| **Retrieval step** (vector search + BM25 keyword merge) | 200 ms |
| Context assembly (re-ranking and deduplication) | 150 ms |
| **LLM generation** (prompt execution on 70B model) | 1,500 ms |
| Response post-processing (citation formatting) | 50 ms |

The total **end-to-end latency** is:

```
End-to-end latency = 200 ms (retrieval step)
                   + 150 ms (context assembly)
                   + 1,500 ms (LLM generation)
                   + 50 ms (post-processing)
                   = 1,900 ms
```

If the **retrieval step** takes 200 ms and the **LLM generation** step takes 1,500 ms, the total **end-to-end latency** is 1,700 ms (excluding context assembly and post-processing), and the clear **bottleneck** is the **LLM generation** stage.

## Bottleneck Identification

The **bottleneck** in the current pipeline is the **LLM generation** stage, which accounts for approximately 1,500 ms of the total **end-to-end latency**. This represents roughly 79% of the pipeline's total processing time when all four stages are considered, or 88% when considering only the **retrieval step** and **LLM generation** stages together.

The **LLM generation** stage is the **bottleneck** because:

1. It has the highest absolute latency (1,500 ms) of any single stage.
2. It is computationally intensive, requiring a full forward pass through a 70B-parameter model.
3. It cannot be parallelized with other stages since it depends on the output of the **retrieval step**.

## Optimization Recommendations

To reduce the **end-to-end latency** and alleviate the **LLM generation** **bottleneck**, the following optimizations are recommended:

- **Speculative decoding:** Deploy a draft model (125M parameters) that generates candidate tokens in parallel with the main 70B model, reducing generation time by an estimated 30–40%.
- **KV-cache optimization:** Implement prefix caching for frequently asked queries to avoid recomputing attention keys and values.
- **Output streaming:** Begin streaming tokens to the client as they are generated rather than waiting for full completion, improving perceived latency.

Even after these optimizations, the **LLM generation** stage will remain the primary **bottleneck** in the RAG pipeline due to the fundamental compute cost of autoregressive decoding.

## Monitoring the Bottleneck

Meridian's observability platform tracks the following metrics for **bottleneck** detection:

- `rag_stage_duration_ms{stage="llm_generation"}`
- `rag_pipeline_end_to_end_latency_ms`
- `rag_stage_ratio` — proportion of total **end-to-end latency** contributed by each stage

An alert is raised when the **LLM generation** stage exceeds 2,000 ms for more than 1% of requests in a 5-minute window.

## Revision History

This **performance analysis** was last updated on 14 July 2026 following the v3 model deployment.
