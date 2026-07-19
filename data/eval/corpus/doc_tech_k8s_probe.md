# Meridian Analytics — Kubernetes Readiness Probe Configuration

**Document ID:** doc_tech_k8s_probe
**Owner:** Platform Infrastructure Team
**Last updated:** 2026-07-03

## Overview

Meridian Analytics runs its microservice fleet on **Kubernetes** across multiple AWS EKS clusters. Health probes are a critical component of our deployment strategy, ensuring that traffic reaches only containers that are ready to serve requests. This document specifies the standard **readiness probe** configuration for **HTTP endpoint** health checking, including the exact **annotation** syntax and the parameter values for `periodSeconds` and `timeoutSeconds`.

## Readiness Probe Annotation Syntax

Meridian defines readiness probes at the container spec level within the pod template, not as pod annotations. The following YAML shows the standard readiness probe configuration for Meridian's API gateway service, which checks an HTTP health endpoint every 10 seconds with a 5-second timeout:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: ingress
  labels:
    app: api-gateway
    team: platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
        - name: api-gateway
          image: meridian/api-gateway:2.5.1
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8080
              httpHeaders:
                - name: X-Health-Check
                  value: readiness
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 5
            successThreshold: 1
            failureThreshold: 3
```

The key parameters are:
- **periodSeconds**: 10 (probe every 10 seconds)
- **timeoutSeconds**: 5 (probe must respond within 5 seconds)

The probe hits the `/health/ready` endpoint with status 200 indicating the pod is ready to receive traffic. Three consecutive failures (`failureThreshold: 3`) mark the pod as not ready, removing it from the service endpoints. One success (`successThreshold: 1`) restores it to the ready state.

## Comparison with Liveness and Startup Probes

Meridian distinguishes three probe types. The readiness probe controls service endpoint membership and is the most critical for zero-downtime deployments. For comparison, here is the liveness probe configuration also used by the API gateway:

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 15
  periodSeconds: 20
  timeoutSeconds: 3
  failureThreshold: 6
```

The liveness probe uses a longer period (20s) and a higher failure threshold (6) to avoid unnecessary container restarts during brief transient failures that do not warrant recycling the pod. The startup probe, used for services with slow initialization (model loading, cache warming), allows up to 120 seconds for initial readiness without modifying the liveness probe sensitivity.

## Probe Implementation Best Practices

The `/health/ready` endpoint must check that the service can handle traffic by verifying downstream dependency connectivity: database connection pools have available connections, cache clusters are reachable, and any required initialization (schema migration, warmup) has completed. However, it must not depend on cross-service health to avoid cascading readiness failures -- a degraded upstream service should not cause all pods to appear not ready.

Meridian's readiness probe endpoint aggregates dependency status into a boolean ready state using a dependency health registry. A service is marked not ready if:
- Its database connection pool is exhausted (all connections in use)
- A required cache cluster is unreachable
- Persistent volume mounts are unavailable
- The service has not completed its warmup phase

Transient upstream HTTP 503s or brief latency spikes do not affect readiness, as these are handled by circuit breakers and retry logic at the client side rather than by pod removal.

## Deployment Strategy and Rollout Safety

During rolling updates, Meridian uses `maxSurge: 1` and `maxUnavailable: 0` to ensure zero downtime. The readiness probe with periodSeconds=10 and timeoutSeconds=5 means a healthy pod joins the service endpoints approximately 15 seconds after container start.
