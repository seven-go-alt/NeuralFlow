# Meridian Analytics — Disaster Recovery Runbook for Primary Database Cluster

**Document ID:** doc_tech_dr
**Owner:** Database Engineering
**Last updated:** 2026-07-01

## Overview

This runbook describes the **disaster recovery** procedure for the primary **database cluster** in the **us-east-1** region. The primary **database cluster** is a PostgreSQL 16 multi-AZ deployment that hosts customer document metadata, ingestion tracking, and tenant configuration. In the event of a regional failure, the **disaster recovery** plan activates a **failover** to the standby **database cluster** in **us-west-2**.

## Disaster Recovery Procedure

When a **disaster recovery** event is declared for the primary **database cluster** in **us-east-1**, the following procedure must be executed in order:

### Phase 1: Assessment (0–5 minutes)

1. Confirm the outage via AWS Health Dashboard and PagerDuty alerts.
2. Verify that the **us-east-1** region is indeed unavailable and the issue is not limited to a single Availability Zone.
3. Notify the Database Engineering team lead and the VP of Engineering.
4. Open an incident in PagerDuty with severity SEV-1 and tag `dr-failover`.

### Phase 2: Failover Initiation (5–15 minutes)

1. Connect to the **disaster recovery** management host in **us-west-2** via the bastion.
2. Verify the standby **database cluster** in **us-west-2** is healthy: check replication lag, connection count, and storage utilization.
3. Promote the standby **database cluster** using the command:

```sql
SELECT pg_promote();
```

4. Update DNS CNAME record `db-primary.meridian-analytics.com` to point to the **us-west-2** cluster endpoint (TTL is pre-configured to 60 seconds).

### Phase 3: Validation (15–30 minutes)

1. Run the health check query on the promoted **database cluster**:

```sql
SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';
```

2. Verify that data integrity checks pass: compare row counts on critical tables against the latest **disaster recovery** snapshot.
3. Validate that the application layer connects to the new cluster by checking application logs for successful connection events.
4. Update the status page to indicate active **disaster recovery** operations.

### Phase 4: Recovery (30 minutes to 24 hours)

1. Once **us-east-1** is restored, set up a new read replica in **us-east-1** from the promoted **us-west-2** cluster.
2. After re-replication lag reaches zero, perform a controlled **failover** back to **us-east-1** during a maintenance window.
3. Run full integrity checks on all tables.
4. Declare the **disaster recovery** event resolved and update the status page.

## Failover Prerequisites

The **failover** procedure requires the following to be verified quarterly:

| Prerequisite | Verification method | Frequency |
|-------------|-------------------|-----------|
| Standby cluster is running same PostgreSQL version | `SELECT version()` | Quarterly DR drill |
| Replication slot is active and lag under 100 MB | `pg_stat_replication` | Continuous monitoring |
| DNS TTL is configured to 60 seconds | Route53 record inspection | Post-deployment check |
| Application connection string supports multi-region | Code review | Per release |

## Revision History

This **disaster recovery** runbook was last updated on 1 July 2026 following the Q2 DR drill, which confirmed a **failover** time of 14 minutes 38 seconds.
