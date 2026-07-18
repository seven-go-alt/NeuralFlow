# Meridian Analytics — Content Delivery API Caching Strategy

**Document ID:** doc_tech_caching
**Owner:** Platform Engineering
**Last updated:** 2026-07-12

## Overview

Meridian Analytics serves a content delivery API that exposes documents ranging from 1 KB reports to 50 MB export archives, with highly variable access frequency. This document defines the multi-tier caching strategy employed to serve content efficiently while minimizing origin load and latency.

## Multi-Tier Cache Hierarchy

Meridian uses a three-tier caching strategy built on a **cache hierarchy** that places content at the appropriate level based on size and popularity:

1. **L1 — In-memory object cache (Redis):** Documents under 100 KB with high **access frequency** are cached in Redis clusters colocated with application instances. This tier serves the majority of API requests with sub-millisecond latency.

2. **L2 — Local SSD cache (NVMe):** Documents between 100 KB and 10 MB are stored on instance-local NVMe SSDs, managed by a least-frequently-used eviction policy. This tier handles documents with moderate **access frequency** and provides single-digit millisecond read times.

3. **L3 — CDN edge cache:** Documents over 10 MB and all publicly accessible exports are served through a **CDN** with edge locations in all major AWS regions. The **CDN** cache is populated on first request and uses origin-pull with signed URLs for private content.

## Handling Varying Content Sizes

The content delivery API must handle documents with **varying sizes** efficiently. Meridian's approach differs by size band:

| Size Range | Cache Tier | Cache TTL | Compressed |
|------------|-----------|-----------|------------|
| 1–100 KB | L1 (Redis) | 10 minutes | No (already small) |
| 100 KB – 10 MB | L2 (SSD) | 30 minutes | Gzip |
| 10–50 MB | L3 (CDN) | 60 minutes | Brotli |

For documents with **varying sizes** that cross size thresholds during updates, the cache is invalidated at all tiers simultaneously using a publish–subscribe invalidation channel.

## Access Frequency Based Promotion

The **caching strategy** also considers **access frequency** to promote or demote content between tiers dynamically:

- **Hot content** (thousands of requests per minute): Automatically promoted to L1 cache regardless of size, up to a configurable memory budget of 4 GB per cluster. Larger hot documents are chunked into 512 KB blocks and distributed across Redis shards.
- **Warm content** (hundreds of requests per hour): Retained in L2 or L3 depending on size.
- **Cold content** (fewer than 10 requests per day): Served from origin (S3) and cached at the **CDN** tier only if requested again within a 24-hour window.

Promotion decisions are computed by a background process that analyzes request logs every 60 seconds and updates cache routing rules in a shared configuration store.

## Cache Invalidation

Meridian uses tag-based cache invalidation across all three tiers. Each document carries one or more cache tags (for example, `report:Q3`, `client:acme-corp`). When a document is updated, the API publishes invalidation events for its tags:

1. Redis keys matching the tag pattern are evicted from L1.
2. A local SSD scan removes all matching entries from L2.
3. The **CDN** provider's API is called with the tag for edge purge.

This invalidation strategy supports selective purging and avoids flushing the entire cache for a single document update, which is critical given the **varying sizes** and overlapping access patterns in the corpus.

## Revision History

This strategy was last updated on 12 July 2026 following the addition of Brotli compression for CDN-tier content.
