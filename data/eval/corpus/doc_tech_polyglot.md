# Meridian Analytics — Polyglot Persistence with CQRS for Cross-Service Joins

**Document ID:** doc_tech_polyglot
**Owner:** Data Engineering
**Last updated:** 2026-07-14

## Overview

Meridian Analytics operates a **polyglot persistence** architecture that spans a **PostgreSQL** relational store for transactional financial data and an **Elasticsearch** search index for full-text queries and aggregations. This document describes how **cross-service joins** between these stores are handled when **data consistency** is critical, using **CQRS**.

## Architecture

Meridian's **polyglot persistence** approach uses **CQRS** (Command Query Responsibility Segregation):

- **Command side:** All writes go to **PostgreSQL**, the system of record for financial transactions and account balances, enforcing referential integrity with SERIALIZABLE isolation.
- **Query side:** All search queries go to **Elasticsearch**, providing full-text search and faceted aggregation over the same data.

The challenge is executing **cross-service joins** — such as finding all transactions in **PostgreSQL** matching an **Elasticsearch** full-text query.

## Event-Driven Synchronization

To maintain **data consistency** across the **polyglot persistence** stores, Meridian uses an event-driven synchronization layer:

```python
@app.on_event("transaction.created")
def sync_transaction_to_search(transaction: Transaction):
    # Write to PostgreSQL (source of truth)
    db.session.add(transaction)
    db.session.commit()

    # Publish event for Elasticsearch sync
    event_bus.publish("search.sync.transaction", {
        "id": transaction.id,
        "client_id": transaction.client_id,
        "description": transaction.description,
        "amount": transaction.amount,
        "timestamp": transaction.created_at.isoformat()
    })
```

The **Elasticsearch** index is updated asynchronously via a consumer that reads from the event bus. This means writes are eventually consistent — the trade-off Meridian accepts for **polyglot persistence**.

## Cross-Service Join Strategy

Meridian performs **cross-service joins** between **PostgreSQL** and **Elasticsearch** by executing the join in the application layer using **CQRS** principles:

1. **Filter in Elasticsearch first:** Submit the full-text search query to **Elasticsearch** and retrieve the set of matching document IDs. For a client transaction query, this yields a list of transaction IDs matching the search terms.

2. **Fetch from PostgreSQL by ID:** Use the IDs from step 1 to query **PostgreSQL** with a `WHERE id IN (...)` clause. Since **PostgreSQL** clusters the primary key index, this returns the full relational records with all associated joins (client details, account hierarchy, audit trail) at minimal latency.

3. **Merge and return:** The application layer merges the search metadata from **Elasticsearch** with the relational data from **PostgreSQL** into the response DTO.

This **cross-service join** approach works within **data consistency** requirements because the **PostgreSQL** transaction is always authoritative for the latest state, and **Elasticsearch** is treated as a derived index that may lag by up to 500 milliseconds under normal conditions.

## Consistency Guarantees

For critical **cross-service joins** where **data consistency** is paramount, Meridian applies these patterns:

- **Read-your-writes consistency:** When a user creates a transaction and immediately searches for it, the application waits for the **CQRS** synchronization to complete by polling the **Elasticsearch** refresh endpoint for up to 2 seconds before falling back to a direct **PostgreSQL** query.
- **Compensating transactions:** If the **Elasticsearch** sync fails (for example, indexer down), the **PostgreSQL** transaction is not rolled back. Instead, a background reconciler periodically compares **PostgreSQL** and **Elasticsearch** counts and emits repair events for any discrepancies.
- **Consistency window:** The **CQRS** synchronization targets a p99 eventual consistency window of 2 seconds. This is tracked via a `search_sync_lag_seconds` metric.

## Monitoring

Meridian monitors **polyglot persistence** health through:
- `search_sync_lag_seconds`: Time between **PostgreSQL** commit and **Elasticsearch** availability.
- `search_sync_failed_total`: Count of failed synchronization events.
- `cross_service_join_latency_ms`: Duration of application-layer **cross-service joins**.

An alert fires if `search_sync_lag_seconds` exceeds 5 seconds for more than one minute, indicating a systemic issue with the **CQRS** pipeline.

## Revision History

This document was last updated on 14 July 2026 following the deployment of the read-your-writes consistency enhancement for client-facing search.
