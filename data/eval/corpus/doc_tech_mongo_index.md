# Meridian Analytics — MongoDB Index Strategy for Geospatial and Text Search

**Document ID:** doc_tech_mongo_index
**Owner:** Data Platform Team
**Last updated:** 2026-07-04

## Overview

Meridian Analytics ingests and indexes millions of geographically-tagged records daily, including customer locations, asset deployments, field service events, and regional sales data. Our primary data store for this workload is **MongoDB**, chosen for its native support for both **geospatial queries** and **text search** on the same documents. Selecting the correct **index strategy** for a **MongoDB** collection that must support both query patterns simultaneously is critical for query performance at scale. This document describes Meridian's recommended **compound index** approach for collections requiring geospatial and text search capabilities.

## Collection Schema and Query Patterns

The primary collection is `field_assets`, which tracks deployed hardware assets. Each document has fields including `location` (GeoJSON Point), `site_name`, `description`, `tags`, and `asset_type`. The two primary query patterns are:
1. Geospatial: Find all assets within a 10km radius of a given coordinate, ordered by proximity.
2. Text search: Find all assets with descriptions or tags matching "temperature sensor".

## Compound Index Strategy

MongoDB does not support a single compound index that directly combines a 2dsphere index key with a text index key in one index definition. Instead, Meridian uses a dual-index strategy with query optimization using index intersection:

```javascript
// Index 1: Geospatial index
db.field_assets.createIndex(
  { location: "2dsphere" },
  { name: "idx_location_2dsphere" }
)

// Index 2: Text index on searchable fields
db.field_assets.createIndex(
  {
    site_name: "text",
    description: "text",
    tags: "text",
    asset_type: "text"
  },
  {
    name: "idx_fulltext_search",
    weights: {
      site_name: 10,
      asset_type: 5,
      tags: 3,
      description: 1
    },
    default_language: "english"
  }
)
```

For queries that combine geospatial and text search filters, MongoDB can use index intersection to satisfy both predicates efficiently:

```javascript
db.field_assets.find({
  $text: { $search: "temperature sensor" },
  location: {
    $nearSphere: {
      $geometry: {
        type: "Point",
        coordinates: [-73.9857, 40.7484]
      },
      $maxDistance: 10000
    }
  }
})
```

This query uses the `idx_fulltext_search` index for the `$text` predicate and the `idx_location_2dsphere` index for the `$nearSphere` predicate. MongoDB's query planner, at version 7.0+, efficiently intersects the results from both indexes when the query predicates are selective on both dimensions.

## Index Strategy Trade-offs and Optimization

The dual-index approach was chosen after evaluating several alternatives:

- **Single compound index with 2dsphere + text**: Not supported by MongoDB. A compound index can contain either a 2dsphere key or a text key, but not both in the same index.

- **Compound index with 2dsphere + other filters**: If queries include a non-text filter, a compound index like `{ location: "2dsphere", status: 1 }` can cover geospatial plus equality filters. This is used where text search is optional.

## Query Performance and Monitoring

In production with 50 million documents, the query completes in under 200ms for typical radius thresholds (5-50km). The index intersection plan is verified using `explain("executionStats")` and monitored through the MongoDB Atlas Performance Advisor. Key metrics include `IXSCAN` stage `totalKeysExamined` and `executionTimeMillis`. When either index scan examines more than 50,000 keys for a single query, the team reviews whether the query selectivity can be improved by adding additional equality filters or reducing the search radius.
