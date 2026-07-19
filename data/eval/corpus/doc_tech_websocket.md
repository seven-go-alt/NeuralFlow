# Meridian Analytics — WebSocket Server Graceful Shutdown Procedure

**Document ID:** doc_tech_websocket
**Owner:** Platform Engineering
**Last updated:** 2026-07-13

## Overview

Meridian Analytics operates a real-time event streaming service that maintains thousands of **WebSocket** connections for delivering live financial data to client dashboards. This document defines the **graceful shutdown** procedure for a **WebSocket** server node when there are 500 **active connections** with **in-flight messages** that must not be lost.

## Shutdown Sequence Overview

When a **WebSocket** server instance receives a SIGTERM signal (from Kubernetes pod termination or autoscaling scale-in), it begins the following **graceful shutdown** sequence designed to preserve all **in-flight messages** across the 500 **active connections**:

1. **Health check removal (0s):** The server deregisters itself from the service registry so that load balancers stop routing new connections to this node. Existing connections remain open.

2. **Drain initiation (0–2s):** The server broadcasts a `GOING_AWAY` frame with a 5-second **drain** window to all 500 **active connections**. Clients are expected to stop sending new requests and prepare for disconnection.

3. **In-flight message drain (2–25s):** The server enters a **drain** phase where it processes all buffered **in-flight messages** in priority order, flushing them to the output buffer of each **WebSocket** connection. This phase has a hard timeout of 25 seconds.

4. **Final flush (25–28s):** Any remaining **in-flight messages** that could not be delivered are persisted to a durable Redis stream and replayed when the client reconnects to a different server node.

5. **Connection close (28–30s):** The server sends close frames to all 500 **active connections** and waits up to 2 seconds for acknowledgment before forcibly closing.

## Handling In-Flight Messages

The critical requirement during **graceful shutdown** is that **in-flight messages** already sent to the **WebSocket** output buffer must be delivered before the connection closes. Meridian implements this with a two-phase **drain**:

```python
async def graceful_shutdown(server, active_connections: list[Connection]):
    # Phase 1: Stop accepting new messages from producer queues
    for conn in active_connections:
        conn.producer_queue.close()

    # Phase 2: Wait for all output buffers to drain
    drain_tasks = [drain_connection(conn) for conn in active_connections]
    done, pending = await asyncio.wait(
        drain_tasks, timeout=25.0, return_when=ALL_COMPLETED
    )

    # Phase 3: Persist any undelivered messages
    for task in pending:
        conn = task.conn
        await persist_undelivered(conn.id, conn.buffer_unflushed)
```

Each connection maintains a sequence number, and clients acknowledge receipt. During the **drain** phase, the server replays unacknowledged messages from the send buffer before closing.

## Monitoring the Drain Process

Meridian exposes the following metrics during **graceful shutdown**:

- `ws_drain_remaining_connections`: Connections still pending **drain** completion.
- `ws_drain_undelivered_messages`: **In-flight messages** that required persistence to Redis.
- `ws_drain_duration_seconds`: Total **drain** duration, tracked at p50, p95, and p99.

An alert fires if **drain** duration exceeds 20 seconds, as this indicates that one or more **active connections** are not consuming messages quickly enough.

## Recovery After Shutdown

Clients disconnected during **graceful shutdown** reconnect to a surviving **WebSocket** server using the cluster's load balancer. Upon reconnection, they send the last received sequence number, and the new server replays any persisted **in-flight messages** from the Redis stream. This guarantees at-least-once delivery for all messages that were in flight at the time of shutdown.

## Revision History

This guide was last updated on 13 July 2026 following a production incident where the drain timeout was insufficient for large message batches.
