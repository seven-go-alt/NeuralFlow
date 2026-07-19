# Meridian Analytics — RAG Pipeline Reference Architecture

**Document ID:** doc_tech_rag_arch
**Owner:** ML Engineering
**Last updated:** 2026-07-12

## Overview

This document describes the **reference architecture** for the Retrieval-Augmented Generation (RAG) pipeline used by Meridian Analytics' AskMeridian product. The pipeline is decomposed into a series of **RAG pipeline stages**, each with a defined **latency budget** to guarantee an end-to-end response time of under 3 seconds for the p95 percentile.

## RAG Pipeline Stages and Latency Budgets

The **reference architecture** defines 5 **RAG pipeline stages**. Their **latency budgets** are allocated as follows:

| # | Stage | Component | **Latency budget** | % of total budget |
|---|-------|-----------|--------------------|-------------------|
| 1 | Query understanding | LLM-based query rewriting and decomposition | 300 ms | 11.5% |
| 2 | Document retrieval | Vector search + keyword (hybrid) retrieval from the document store | 400 ms | 15.4% |
| 3 | Context assembly | Chunk merging, deduplication, re-ranking, and context window packing | 200 ms | 7.7% |
| 4 | LLM generation | Augmented prompt execution against the generation model | 1,500 ms | 57.7% |
| 5 | Response post-processing | Citation formatting, filtering, safety checks | 200 ms | 7.7% |

The total allocated **latency budget** across all 5 **RAG pipeline stages** is 2,600 ms, leaving 400 ms headroom for network overhead, authentication, and API gateway processing to meet the 3-second p95 SLA.

## Stage Details

### Stage 1: Query Understanding (300 ms budget)

The input query is rewritten by a smaller LLM (Meridian's Lightweight Rewriter model) to improve retrieval quality. The rewriter handles query expansion, synonym insertion, and decomposition of multi-part questions.

### Stage 2: Document Retrieval (400 ms budget)

Hybrid retrieval combines a vector index search (cosine similarity on 1536-dimension embeddings) with a BM25 keyword index. Results are merged using a weighted score (0.7 vector / 0.3 keyword) and the top 20 chunks are passed to the next stage.

### Stage 3: Context Assembly (200 ms budget)

Retrieved chunks are deduplicated, re-ranked using a cross-encoder model, and packed into the generation context window within the 200 ms **latency budget**.

### Stage 4: LLM Generation (1,500 ms budget)

This is the largest **latency budget** consumer. The augmented prompt is sent to Meridian's generation model (a 70B-parameter LLM hosted on dedicated A100 nodes). The model generates the answer with inline citations.

### Stage 5: Response Post-Processing (200 ms budget)

The generated response passes through a safety filter, citation links are verified against the source documents, and the final JSON response is formatted.

## Revision History

This **reference architecture** was last updated on 12 July 2026 to reflect the addition of the query understanding stage and updated **latency budgets**.
