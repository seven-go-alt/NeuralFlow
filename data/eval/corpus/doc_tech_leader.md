# Meridian Analytics — Leader Election for Distributed Cron Scheduler on Kubernetes

**Document ID:** doc_tech_leader
**Owner:** Platform Engineering
**Last updated:** 2026-07-17

## Overview

Meridian Analytics runs a **distributed cron** scheduler that manages recurring tasks such as document reindexing, report generation, and model retraining. The scheduler is deployed on **Kubernetes** with three **replicas** to ensure high availability. Only one replica must be active at any time to prevent duplicate task execution. This document describes the **leader election** mechanism that ensures single-active coordination using **Kubernetes** **lease** objects.

## Leader Election Design

The **leader election** mechanism is implemented using **Kubernetes** coordination.k8s.io **Lease** API. Each scheduler pod attempts to acquire and renew a **lease** with a configurable duration. The pod that holds the **lease** acts as the leader and executes cron jobs; the remaining **replicas** stand by as hot spares.

```
Pod A ───→ Holds Lease (Leader, runs cron tasks)
Pod B ───→ Watches Lease (Standby, ready to take over)
Pod C ───→ Watches Lease (Standby, ready to take over)
```

Key parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Lease duration | 30 seconds | Long enough to avoid flapping, short enough for fast failover |
| Renew deadline | 20 seconds | Leader must renew before this or the lease is considered expired |
| Retry period | 4 seconds | Interval between renewal attempts |
| Release on cancel | True | Leader releases lease gracefully on shutdown |

## Implementation Using Kubernetes Coordination API

The **distributed cron** scheduler uses the `coordination.k8s.io/v1` API to manage **leader election**. The implementation is based on the standard `client-go` leader election library:

```go
import (
    "os"
    "time"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/tools/leaderelection"
    "k8s.io/client-go/tools/leaderelection/resourcelock"
)

func startLeaderElection(client kubernetes.Interface, namespace, name string) {
    lock := &resourcelock.LeaseLock{
        LeaseMeta:  metav1.ObjectMeta{Name: name, Namespace: namespace},
        Client:     client.CoordinationV1(),
        LockConfig: resourcelock.ResourceLockConfig{Identity: os.Getenv("HOSTNAME")},
    }

    leaderelection.RunOrDie(context.Background(), leaderelection.LeaderElectionConfig{
        Lock:            lock,
        LeaseDuration:   30 * time.Second,
        RenewDeadline:   20 * time.Second,
        RetryPeriod:     4 * time.Second,
        Callbacks: leaderelection.LeaderCallbacks{
            OnStartedLeading: func(ctx context.Context) {
                startCronScheduler(ctx)
            },
            OnStoppedLeading: func() {
                // This replica is no longer the leader
                stopCronScheduler()
            },
        },
    })
}
```

## Lease-Based Coordination in Detail

The **lease** object is persisted in the same namespace as the scheduler deployment. When a leader pod holds the **lease**, it periodically renews the lease's `renewTime` field. The standby **replicas** watch the **lease** for modifications:

1. If the lease's `renewTime` is within the `LeaseDuration` window, the standby pods remain passive.
2. If the lease expires (no renewal within 30 seconds), all standby pods attempt to acquire the lease using a compare-and-swap operation mediated by the **Kubernetes** API server's resource version.
3. Exactly one standby pod succeeds in acquiring the **lease** and becomes the new leader, ensuring continuous **coordination** without split-brain scenarios.

## Operational Considerations for Three Replicas

With three **replicas**, the system tolerates one pod failure without losing cron scheduling capability. If the leader pod crashes:

- The lease expires after 30 seconds (or immediately if `ReleaseOnCancel` fires).
- The two remaining **replicas** contend for the **lease**.
- The new leader assumes control typically within 5-10 seconds of the failure.

The metric `cron_scheduler_leader_identity` is exported to Prometheus, allowing the Platform Engineering team to verify which pod is currently leading at any time. A `cron_scheduler_leader_missing` alert fires if no pod holds the **lease** for more than 45 seconds.

## Revision History

This document was last updated on 17 July 2026 following the migration of the cron scheduler to the coordination.k8s.io/v1 Lease API.
