# Meridian Analytics — Graceful Degradation for Partial Response When Downstream Services Are Degraded

**Document ID:** doc_tech_degradation
**Owner:** Service Architecture Engineering
**Last updated:** 2026-07-18

## Overview

Meridian Analytics' document enrichment pipeline aggregates data from five **dependent downstream** services: entity extraction, sentiment analysis, language detection, document classification, and metadata tagging. When one of these services experiences **high latency** while the remaining four remain healthy, the pipeline must continue serving responses with available data rather than failing entirely. This document describes the **partial response** **degradation** strategy that implements **fallback** logic and **graceful degradation** for these scenarios.

## Degradation Detection Strategy

The enrichment pipeline monitors each **dependent downstream** service's response latency. **Graceful degradation** is triggered when a **dependent downstream** service exceeds its latency budget:

| Downstream Service | Normal Latency (p99) | Degradation Threshold | Timeout |
|---------------------|----------------------|-----------------------|---------|
| Entity Extraction | 200ms | 1000ms | 1500ms |
| Sentiment Analysis | 150ms | 800ms | 1200ms |
| Language Detection | 100ms | 500ms | 800ms |
| Document Classification | 300ms | 1500ms | 2000ms |
| Metadata Tagging | 150ms | 800ms | 1200ms |

When **high latency** is detected on any single service, the pipeline applies a **partial response** strategy: it collects results from the four healthy services, substitutes a **fallback** value for the degraded service, and returns the aggregate response with a degradation indicator.

## Partial Response Implementation

The implementation uses a fan-out pattern with per-service timeouts. If one of the **dependent downstream** services fails to respond within its timeout due to **high latency**, the pipeline assembles a **partial response**:

```python
import asyncio
from dataclasses import dataclass

@dataclass
class EnrichmentResult:
    entities: list
    sentiment: dict
    language: str
    classification: str
    metadata: dict
    degraded_services: list

async def enrich_document(doc_id: str) -> EnrichmentResult:
    services = {
        "entity_extraction": call_entity_extraction(doc_id),
        "sentiment_analysis": call_sentiment_analysis(doc_id),
        "language_detection": call_language_detection(doc_id),
        "classification": call_classification(doc_id),
        "metadata_tagging": call_metadata_tagging(doc_id),
    }

    results = {}
    degraded = []

    for name, coro in services.items():
        try:
            result = await asyncio.wait_for(coro, timeout=get_timeout(name))
            results[name] = result
        except asyncio.TimeoutError:
            logger.warning(f"{name} timeout due to high latency")
            results[name] = get_fallback(name)
            degraded.append(name)

    return EnrichmentResult(
        entities=results.get("entity_extraction", []),
        sentiment=results.get("sentiment_analysis", {}),
        language=results.get("language_detection", "unknown"),
        classification=results.get("classification", "unclassified"),
        metadata=results.get("metadata_tagging", {}),
        degraded_services=degraded,
    )
```

## Fallback Values for Degraded Services

Each **dependent downstream** service has a defined **fallback** that preserves the structure of the **partial response**:

- **Entity extraction**: Returns an empty list.
- **Sentiment analysis**: Returns `{"score": 0.0, "label": "neutral"}`.
- **Language detection**: Returns `"unknown"`.
- **Document classification**: Returns `"unclassified"`.
- **Metadata tagging**: Returns an empty dictionary.

The **fallback** values ensure a single degraded service never blocks the entire enrichment pipeline. This **graceful degradation** approach means Meridian's document pipeline continues processing at full throughput with slightly reduced enrichment fidelity during an incident.

## Client Communication and SLAs

When the pipeline returns a **partial response**, the `X-Meridian-Degraded-Services` HTTP header lists the affected services. The API response body also includes a `degraded_services` field. Meridian's SLA allows **partial response** delivery as long as no more than one of the five services is degraded. If two or more experience **high latency**, the endpoint returns 503 with a retry-after header.

## Revision History

This document was last updated on 18 July 2026 following the deployment of timeout-based fallback logic in the enrichment pipeline.
