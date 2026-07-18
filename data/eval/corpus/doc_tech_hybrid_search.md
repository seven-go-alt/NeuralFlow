# Meridian Analytics — Hybrid Search Architecture with Dense and Sparse Retrieval Fusion

**Document ID:** doc_tech_hybrid_search
**Owner:** Search Platform Team
**Last updated:** 2026-07-01

## Overview

Meridian Analytics' search platform serves millions of queries daily across our customer analytics dashboards. To deliver relevant results for both semantic similarity queries and exact term matches, we implemented a hybrid search architecture that fuses dense and sparse retrieval signals. This document describes the architectural decisions, algorithm design, and weighting strategy that power our production hybrid search system.

## Dense and Sparse Retrieval Components

Our hybrid search pipeline combines two complementary retrieval paths. The **dense** path uses a transformer-based **embedding** model (Meridian-Encoder-v2) that maps queries and documents into a 768-dimensional vector space for semantic matching. The **sparse** path relies on **keyword** **retrieval** using BM25, which excels at exact term matching for product codes and identifiers that dense models may overlook. Both paths return the top K candidates (K=200) before the fusion stage.

## Fusion Algorithm Design

The fusion stage combines ranked lists from both paths using Reciprocal Rank Fusion (RRF) with dynamic **weighting**:

```
score(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))
```

Where `k` is 60, `rank_dense(d)` is the position in dense results, `rank_sparse(d)` is the position in sparse results, and weights sum to 1.0. Documents appearing in only one path receive their RRF contribution from that path's weight only.

## Dynamic Weighting Strategy

The weighting between dense and sparse signals is not static. Meridian's hybrid search employs a query-classification-based dynamic weighting mechanism. A lightweight classifier (DistilBERT-based, <10ms inference) categories each query into one of three types:

- **Semantic queries** (e.g., "show me churn trends for Q3"): `w_dense = 0.8`, `w_sparse = 0.2`
- **Keyword queries** (e.g., "order_id = INV-44932"): `w_dense = 0.2`, `w_sparse = 0.8`
- **Ambiguous queries** (e.g., "Q3 revenue report"): `w_dense = 0.5`, `w_sparse = 0.5`

The classifier is trained on labeled query logs with manual relevance judgments and is re-evaluated quarterly. For unclassified queries, a default equal weighting of 0.5/0.5 is applied. This adaptive approach ensures that the hybrid search system delivers optimal relevance across the diverse query patterns our customers generate.

## Production Deployment and Monitoring

The hybrid search pipeline is deployed as a dedicated microservice (search-fusion-service) with horizontal autoscaling based on query latency. Each request passes through a thin orchestration layer that fans out to the dense and sparse backends in parallel, then performs fusion on the aggregated results.

We track three key metrics for fusion quality: Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG@10), and the fusion gain metric -- the improvement in recall@20 compared to running either path alone. Our current production measurements show a 34% improvement in recall@20 with hybrid fusion compared to dense-only retrieval, and a 28% improvement compared to sparse-only retrieval.
