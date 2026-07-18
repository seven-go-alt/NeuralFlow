# Meridian Analytics — Time-Series Database Sharding Strategy for IoT Sensor Readings

**Document ID:** doc_tech_sharding
**Owner:** Data Infrastructure Engineering
**Last updated:** 2026-07-14

## Overview

Meridian Analytics manages sensor data from over 10,000 IoT devices deployed across industrial client sites. Each device reports a reading every 5 seconds, generating approximately 172.8 million writes per day. This document defines the optimal **sharding strategy** for the **time-series** database cluster that ingests and stores these **sensor readings**.

## Data Volume Characterization

Before designing the **sharding strategy**, the Data Infrastructure Engineering team characterized the write workload:

| Metric | Value |
|--------|-------|
| IoT devices | 10,000 |
| Reporting interval | 5 seconds |
| Writes per second | 2,000 |
| Writes per day | ~172.8 million |
| Raw data per day | ~120 GB (uncompressed) |
| Retention period | 90 days hot, 365 days warm |

The workload is append-heavy with rare updates and predictable write distribution across devices, making it an ideal candidate for hash-based **partitioning** combined with **time-series** aware placement.

## Recommended Sharding Strategy

The optimal **sharding strategy** for this workload uses a two-level approach combining **time-series** range **partitioning** with hash-based sub-partitioning:

### Level 1: Time-Based Range Partitioning

Data is first partitioned by time window. Each partition covers a 24-hour period. This aligns with the natural **time-series** access pattern where queries nearly always carry a timestamp filter. A new partition is created daily by a scheduled job, and hot partitions are stored on high-performance NVMe storage while cold partitions are migrated to cheaper SSDs.

### Level 2: Hash-Based Device Partitioning

Within each daily partition, data is further divided into 64 shards using a consistent hash of the device ID. This ensures:

- Even distribution of **sensor readings** across shards regardless of device reporting frequency.
- Predictable shard capacity (approximately 2.7 million **sensor readings** per shard per day).
- Query isolation: queries for a single device's **sensor readings** hit exactly one shard, minimizing fan-out.

```
Partition: 2026-07-14
  ├── Shard 0: devices hash(device_id) % 64 == 0
  ├── Shard 1: devices hash(device_id) % 64 == 1
  ├── ...
  └── Shard 63: devices hash(device_id) % 64 == 63
```

## Partitioning Implementation

The **partitioning** is implemented in the ingestion layer using a custom routing function:

```sql
-- Time-range partition key: ingestion_date (DATE type)
-- Hash sub-partition key: device_id mod 64
CREATE TABLE sensor_readings (
    ingestion_date DATE NOT NULL,
    device_id VARCHAR(64) NOT NULL,
    reading_time TIMESTAMP NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    tags JSONB,
    PRIMARY KEY (ingestion_date, device_id, reading_time)
) PARTITION BY RANGE (ingestion_date);

-- Sub-partition template per device hash group
CREATE TABLE sensor_readings_20260714_shard_0
    PARTITION OF sensor_readings
    FOR VALUES FROM ('2026-07-14') TO ('2026-07-15')
    PARTITION BY HASH (device_id);
```

## Capacity and Scaling

With 64 shards per daily partition, each shard handles approximately 31 **IoT** device data streams. At 5-second intervals, each shard receives roughly 31 writes per second — well within the capacity of a single database node. As the device fleet grows beyond 10,000 devices, new shards can be added by increasing the hash modulus during a maintenance window, with data redistribution handled by background rebalancing jobs.

## Revision History

This document was last updated on 14 July 2026 following the successful deployment of the 64-shard partitioning scheme across the time-series database cluster.
