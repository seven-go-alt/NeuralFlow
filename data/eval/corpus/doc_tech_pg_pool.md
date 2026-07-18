# Meridian Analytics — PostgreSQL Connection Pooling for Serverless Lambda

**Document ID:** doc_tech_pg_pool
**Owner:** Data Platform Team
**Last updated:** 2026-07-03

## Overview

Meridian Analytics uses AWS Lambda for event-driven data processing including real-time ingestion, report generation, and pipeline orchestration. Several **Lambda** functions require access to **PostgreSQL** databases that power our operational reporting layer. Managing **connection pooling** for **serverless** functions presents unique challenges: Lambda scales to thousands of **concurrent** invocations, each potentially opening a database connection, quickly exhausting PostgreSQL's connection limit. This document describes Meridian's approach using **RDS Proxy** to handle up to 1,000 concurrent Lambda invocations.

## Architecture Overview

Meridian deploys RDS Proxy in front of all PostgreSQL databases accessed by Lambda functions. RDS Proxy maintains a warm connection pool and multiplexes connections across Lambda invocations. For 1,000 concurrent invocations, careful sizing is required.

The architecture places RDS Proxy in the same VPC as the Lambda functions and the RDS instance. Lambda functions connect to the RDS Proxy endpoint rather than directly to the database. RDS Proxy uses IAM database authentication, eliminating credential storage in Lambda environment variables.

## RDS Proxy Configuration for 1,000 Concurrent Invocations

Each RDS Proxy is configured with these parameters:

| Parameter | Value | Rationale |
|---|---|---|
| Max connections to database | 150 | Leaves headroom from PostgreSQL's 200 default |
| Proxy max connections to database | 150 | Matches the database-side limit |
| Connection pool max connections | 500 | Multiplexing headroom beyond 1,000 Lambda concurrency |
| Idle connection timeout | 300 seconds | Keeps connections warm for cold starts |
| IAM authentication | Enabled | Eliminates credential management |
| TLS enforcement | Required | All connections use TLS 1.2+ |

The key design principle: 1,000 concurrent Lambda invocations share 150 database connections through RDS Proxy -- a 6.7:1 multiplexing ratio. This works because each invocation holds a connection for 100-500ms, and RDS Proxy provides sub-millisecond connection borrowing.

## Lambda Connection Handling

Each Lambda uses an RDS Proxy-compatible driver. In Python, this uses `psycopg2`:

```python
DB_CONFIG = {
    "host": "meridian-analytics.proxy-xxxxx.us-east-1.rds.amazonaws.com",
    "port": 5432,
    "dbname": "analytics_reports",
    "sslmode": "require",
    "connect_timeout": 5,
}

def lambda_handler(event, context):
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            return {"statusCode": 200, "body": str(cur.fetchone())}
    finally:
        conn.close()
```

Each invocation creates a new connection and closes it after use. RDS Proxy returns the connection to its pool on close.

## Monitoring and Connection Utilization

Meridian monitors CloudWatch metrics: `DatabaseConnections` (proxy to database), `ClientConnections` (Lambda to proxy), and `MaxClientConnections`. Target utilization is 100-130 database connections out of 150. If `DatabaseConnections` exceeds 140 regularly, we increase the proxy pool and adjust `max_connections`. If `ClientConnections` approaches 500, we add a second RDS Proxy and split functions across database shards. Lambda reserved concurrency is capped at 1,000 for functions accessing this database.
