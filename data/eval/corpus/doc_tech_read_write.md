# Meridian Analytics — ProxySQL Read-Write Split with Automatic Failover Configuration

**Document ID:** doc_tech_read_write
**Owner:** Database Infrastructure Engineering
**Last updated:** 2026-07-17

## Overview

Meridian Analytics operates a MySQL-based document metadata store that handles high-volume read traffic alongside transactional write operations. To balance this workload, the Database Infrastructure Engineering team configured **ProxySQL** for **read-write split** with **query routing** that directs write queries to the **primary node** and read queries to replicas. This document describes the configuration, including **automatic failover** support when the **primary node** becomes unreachable.

## ProxySQL Read-Write Split Configuration

The **read-write split** is implemented using ProxySQL's query rules. Write queries (INSERT, UPDATE, DELETE) are routed to the **primary node**, while SELECT queries are distributed across read replicas. The configuration below defines the host groups and query rules:

```sql
-- Define host groups
-- 0 = primary (write), 1 = replicas (read)
INSERT INTO mysql_servers (hostgroup_id, hostname, port, weight)
VALUES
    (0, 'primary-db.meridian.internal', 3306, 100),
    (1, 'replica-1.meridian.internal', 3306, 50),
    (1, 'replica-2.meridian.internal', 3306, 50),
    (1, 'replica-3.meridian.internal', 3306, 50);

-- Define query rules for read-write split
INSERT INTO mysql_query_rules (rule_id, active, match_pattern, destination_hostgroup, apply)
VALUES
    (1, 1, '^SELECT.*', 1, 1),   -- Read queries go to replicas
    (2, 1, '^INSERT|^UPDATE|^DELETE', 0, 1); -- Write queries go to primary
```

This **query routing** pattern ensures that Meridian's document search queries (95% of total database operations) are directed to replicas, while transactional metadata writes land on the **primary node**.

## Automatic Failover Configuration

When the **primary node** becomes unreachable, **ProxySQL** must detect the failure and redirect write traffic to a promoted replica. Meridian uses ProxySQL's native **automatic failover** mechanism combined with an external orchestrator:

### ProxySQL Monitoring

ProxySQL monitors the health of all configured MySQL servers using its built-in monitor module:

```sql
-- Configure monitor intervals
SET mysql-monitor_enabled='true';
SET mysql-monitor_connect_interval='2000';
SET mysql-monitor_ping_interval='10000';
SET mysql-monitor_read_only_interval='1500';
SET mysql-monitor_read_only_timeout='500';

-- Define writer hostgroup with backup promotions
INSERT INTO mysql_replication_hostgroups
    (writer_hostgroup, reader_hostgroup, comment)
VALUES (0, 1, 'Primary-Replica hostgroup mapping');
```

When the **primary node**'s monitor check fails for three consecutive attempts, **ProxySQL** marks it as SHUNNED. At this point, the **automatic failover** process triggers:

1. The external failover orchestrator (a Kubernetes CronJob running at 10-second intervals) detects the primary node is shunned.
2. The orchestrator promotes the most advanced replica to become the new **primary node**.
3. It updates `mysql_servers` in ProxySQL: removing the old **primary node** from hostgroup 0 and adding the promoted replica.
4. **ProxySQL** immediately begins **query routing** write traffic to the new **primary node**.

### Orchestrator Integration

The orchestrator communicates with ProxySQL via its admin interface:

```sql
-- Update ProxySQL after failover
UPDATE mysql_servers SET status='OFFLINE_HARD' WHERE hostgroup_id=0;
INSERT INTO mysql_servers (hostgroup_id, hostname, port) VALUES (0, 'new-primary.meridian.internal', 3306);
LOAD MYSQL SERVERS TO RUNTIME;
SAVE MYSQL SERVERS TO DISK;
```

The entire failover sequence completes in under 15 seconds, well within Meridian's 30-second recovery SLA for the document metadata store.

## Query Routing Verification

After failover, the Database Infrastructure Engineering team verifies **query routing** through ProxySQL's stats:

```sql
SELECT hostgroup, srv_host, Queries, Latency_us
FROM stats_mysql_connection_pool ORDER BY srv_host;
```

This query confirms that write traffic has shifted to the new **primary node** and read traffic continues to be distributed across the remaining replicas. The team also validates that the split maintains transactional consistency by running a test insert followed by a select against the same session.

## Revision History

This document was last updated on 17 July 2026 following the successful failover drill validating automatic promotion of a read replica to primary through ProxySQL.
