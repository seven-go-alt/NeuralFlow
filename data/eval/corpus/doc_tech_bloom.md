# Meridian Analytics — Bloom Filter Implementation for Document Deduplication

**Document ID:** doc_tech_bloom
**Owner:** Data Pipeline Engineering
**Last updated:** 2026-07-17

## Overview

Meridian Analytics ingests approximately 1 million documents per hour from various client uploads and web crawlers. To prevent duplicate documents from entering the processing pipeline, the Data Pipeline Engineering team implemented a **Bloom filter** for **document uniqueness** checking in the **deduplication** pipeline. This document describes the implementation details and parameters selected to achieve the target **false positive rate**.

## Bloom Filter Parameter Selection

A **Bloom filter** is a probabilistic data structure that checks set membership efficiently. For the **deduplication** pipeline processing 1 million documents per hour, the team calculated optimal parameters using the standard **Bloom filter** formulas:

- **n** (expected elements): 1,000,000 documents per hour
- **p** (target **false positive rate**): 0.001 (0.1%)
- **m** (optimal bits): `m = -n * ln(p) / ln(2)^2` ≈ 14.38 million bits ≈ 1.72 MB
- **k** (optimal **hash functions**): `k = (m/n) * ln(2)` ≈ 10 **hash functions**

The filter is allocated with 14.4 million bits (approximately 1.8 MB) and configured with 10 independent **hash functions**. This configuration achieves the target 0.1% **false positive rate** while consuming minimal memory for the throughput volume.

## Implementation Details

The **Bloom filter** is implemented as a reusable component in Meridian's document ingestion pipeline:

```python
import mmh3
import bitarray
import math

class DocumentBloomFilter:
    def __init__(self, capacity=1_000_000, error_rate=0.001):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_count = self._optimal_bits(capacity, error_rate)
        self.hash_count = self._optimal_hashes(self.bit_count, capacity)
        self.bits = bitarray.bitarray(self.bit_count)
        self.bits.setall(0)

    def _optimal_bits(self, n, p):
        return int(-n * math.log(p) / (math.log(2) ** 2))

    def _optimal_hashes(self, m, n):
        return int((m / n) * math.log(2))

    def add(self, item: str):
        for seed in range(self.hash_count):
            idx = mmh3.hash128(item, seed) % self.bit_count
            self.bits[idx] = 1

    def contains(self, item: str) -> bool:
        for seed in range(self.hash_count):
            idx = mmh3.hash128(item, seed) % self.bit_count
            if not self.bits[idx]:
                return False
        return True

    def check_and_add(self, document_hash: str) -> bool:
        """Returns True if document already exists (probabilistic)."""
        exists = self.contains(document_hash)
        if not exists:
            self.add(document_hash)
        return exists
```

## False Positive Rate Management

The target **false positive rate** of 0.1% means that approximately 1,000 out of every 1 million unique documents may be incorrectly flagged as duplicates and skipped. To mitigate the impact of these false positives:

1. **Secondary exact check**: Documents flagged as duplicates by the **Bloom filter** are rechecked against a Redis-backed exact index. This verification step catches false positives and reinserts documents that were incorrectly skipped.
2. **Periodic filter reset**: The **Bloom filter** is reset every hour at the top of the hour, corresponding to the 1 million document throughput window. This prevents saturation from degrading the **false positive rate** over time.
3. **False positive rate monitoring**: Meridian's observability system tracks the observed **false positive rate** in production and alerts when it exceeds 0.15%, triggering an early filter rotation.

## Hash Functions and Performance

The implementation uses MurmurHash3 (mmh3) with variable seeds to produce 10 independent **hash functions**. This approach avoids the overhead of implementing multiple hash algorithms. Each **document uniqueness** check completes in under 2 microseconds on Meridian's production worker instances, making the Bloom filter negligible in the overall per-document processing latency.

## Revision History

This document was last updated on 17 July 2026 following the optimization of hash function count and bit array sizing for the current document throughput volume.
