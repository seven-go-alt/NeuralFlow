# Meridian Analytics — Java G1GC Memory Limit Configuration for Real-Time Trade Data Processing

**Document ID:** doc_tech_gc
**Owner:** Trading Platforms Engineering
**Last updated:** 2026-07-16

## Overview

Meridian Analytics operates a **latency-sensitive** Java service that processes **real-time** trade data from multiple exchange feeds. The application requires predictable pause times under 50 milliseconds to meet trading infrastructure SLAs. This document defines the recommended **memory limit** and **Java garbage collection** configuration using the G1 garbage collector (**G1GC**) with specific **heap size** parameters.

## Heap Size Recommendations

The **latency-sensitive** trading application is deployed on containers with a 4 vCPU and 8 GB memory allocation. The recommended **heap size** configuration minimizes full GC pauses while ensuring the JVM has sufficient headroom for **real-time** processing:

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| `-Xms` (initial heap) | 4 GB | Matches the maximum to avoid heap resize costs |
| `-Xmx` (max heap) | 4 GB | Fixed at 50% of container **memory limit** to leave room for OS, off-heap, and Metaspace |
| `-XX:MaxMetaspaceSize` | 256 MB | Prevents class metadata from consuming GC heap |

The **memory limit** enforced at the container level is 8 GB. Allocating 4 GB to the Java heap leaves 4 GB for the JVM's off-heap memory, native buffers, and the Metaspace, preventing OOM kills during GC pressure spikes.

## G1GC Configuration

**Java garbage collection** is configured with **G1GC** as the default collector. The following JVM flags tune the collector for **latency-sensitive** **real-time** trade processing:

```
-XX:+UseG1GC
-XX:MaxGCPauseMillis=50
-XX:G1HeapRegionSize=4m
-XX:G1NewSizePercent=10
-XX:G1MaxNewSizePercent=30
-XX:G1ReservePercent=15
-XX:G1HeapWastePercent=5
-XX:+UnlockExperimentalVMOptions
-XX:G1MixedGCLiveThresholdPercent=85
```

Key tuning decisions:

- **MaxGCPauseMillis=50**: Sets the pause time target to 50ms, matching Meridian's trading SLA.
- **G1HeapRegionSize=4m**: With a 4 GB **heap size**, 4 MB regions produce approximately 1024 regions, providing fine-grained collection granularity for the mixed GC phase.
- **G1NewSizePercent=10** and **G1MaxNewSizePercent=30**: The young generation starts at 400 MB and can grow to 1.2 GB. This accommodates the bursty allocation pattern of trade data processing while preventing excessive young GC frequency during low-traffic periods.
- **G1ReservePercent=15**: Reserves 15% of heap for "to-space" during evacuation failures, critical for a **latency-sensitive** service that cannot tolerate a full GC triggered by promotion failure.

## Real-Time Trade Data Processing Behavior

During peak market hours, Meridian's service processes approximately 50,000 trades per second. The allocation pattern is dominated by short-lived objects (trade events, price ticks) that are collected in young GC cycles:

- Young GC occurs approximately every 2-3 seconds and completes within 15-25ms.
- Mixed GC cycles run during off-peak windows (weekends and overnight) when trade volume is below 10% of peak.
- Full GC has not been observed in production since adopting this configuration in Q2 2026.

The combination of a 4 GB **heap size** and tuned **G1GC** parameters keeps pause times within the 50ms target for 99.9% of all GC events.

## Monitoring GC Performance

Meridian tracks the following **Java garbage collection** metrics through JMX export to Prometheus:

- `jvm_gc_pause_seconds`: Histogram of all GC pause durations.
- `jvm_gc_concurrent_phase_time`: Time spent in concurrent marking.
- `jvm_memory_heap_used`: Current heap utilization relative to the 4 GB **memory limit**.

An alert is raised if p99 GC pause time exceeds 40ms for any 5-minute window during trading hours.

## Revision History

This document was last updated on 16 July 2026 following the successful tuning of G1GC parameters for the trade data processing service in the us-east-1 region.
