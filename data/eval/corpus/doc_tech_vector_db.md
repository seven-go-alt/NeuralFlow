# Meridian Analytics — Vector Database Capacity Planning Guide

**Document ID:** doc_tech_vector_db
**Owner:** Platform Engineering
**Last updated:** 2026-07-05

## Overview

Meridian Analytics uses a purpose-built **vector database** cluster for storing and querying dense vector embeddings produced by the Embedding API. This guide covers the default **connection pool** configuration and its impact on **concurrent connections** throughput.

## Default Connection Pool Configuration

The **vector database** client is configured with a **connection pool** that governs how many simultaneous requests the application layer can make to the database cluster. The default configuration for production workloads is specified in the application configuration file `config/vector_db.yaml`:

```yaml
connection_pool:
  min_size: 5
  max_size: 50
  max_idle_time_seconds: 300
  acquire_timeout_seconds: 10
```

The pool uses a fixed-size FIFO strategy. When all connections in the pool are in use, subsequent requests are queued up to the `acquire_timeout_seconds` threshold before returning a timeout error.

## Concurrent Connections Support

With the default **connection pool** configuration, the **vector database** supports a maximum of **50 concurrent connections** per application instance. This limit is enforced at the client side by the pool's `max_size` parameter. The server-side maximum is set to 500 **concurrent connections** per cluster node, meaning the bottleneck is the application-layer **connection pool** rather than the database server.

In practice, each application instance maintains its own **connection pool**, so total cluster **concurrent connections** scale linearly with the number of application replicas. A deployment of 6 replicas, each with a default pool of 50, can sustain up to 300 simultaneous query operations against the **vector database**.

## Scaling Guidance

When utilization of the **connection pool** consistently exceeds 80% for more than 5 minutes:

1. Scale out application replicas horizontally rather than increasing `max_size` beyond 50 per instance.
2. Verify that average query latency remains under 50ms at the p95 percentile.
3. If the cluster node is approaching the server-side limit of 500 **concurrent connections**, add a read replica.

## Monitoring and Alerting

The Platform Engineering team monitors pool utilization through the following metrics exposed by the **vector database** client:

- `vector_db_pool_active_connections`: Currently borrowed connections from the pool.
- `vector_db_pool_idle_connections`: Connections available for immediate use.
- `vector_db_pool_waiting_requests`: Requests waiting for a connection slot.
- `vector_db_pool_timeout_total`: Requests that timed out waiting for a connection.

An alert is triggered when `waiting_requests` exceeds 10 for more than 30 seconds, which typically indicates the **connection pool** is undersized for the current query load.

## Revision History

This guide was last updated on 5 July 2026 following the capacity audit of the vector database cluster in production.
