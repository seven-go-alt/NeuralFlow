# Meridian Analytics — ArgoCD Canary Release Deployment Configuration

**Document ID:** doc_tech_canary
**Owner:** Platform Engineering
**Last updated:** 2026-07-16

## Overview

Meridian Analytics uses ArgoCD with the Argo Rollouts controller to manage **Canary release** deployments for all microservices. This document describes the standard configuration for rolling out a new version of the document processing service with an initial 5% **traffic routing** weight, monitoring **p99 latency** and **error rate** over a 15-minute **observation period**.

## Canary Release Configuration

The **Canary release** is defined using an Argo Rollouts `Rollout` resource with a canary strategy. The configuration below routes 5% of traffic to the new version while the remaining 95% continues to serve from the stable version:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: document-processor
spec:
  replicas: 10
  strategy:
    canary:
      steps:
      - setWeight: 5
      - pause:
          duration: 15m
      - setWeight: 25
      - pause:
          duration: 15m
      - setWeight: 50
      - pause:
          duration: 15m
      - setWeight: 75
      - pause:
          duration: 15m
      - setWeight: 100
  template:
    # Pod template with the new image version
```

The first step sets the **traffic routing** weight to 5% and pauses for 15 minutes. During this **observation period**, the Platform Engineering team monitors application health before proceeding to the next step.

## Observation Period Monitoring

During the 15-minute **observation period** at each canary step, Meridian's observability stack evaluates two key health signals:

### p99 Latency

The **p99 latency** of the canary version must remain within 120% of the stable version's baseline **p99 latency**. For the document processing service, the current baseline is 850ms. The canary version's **p99 latency** must stay below 1020ms. This threshold is evaluated as a rolling 5-minute window to avoid reacting to transient spikes.

### Error Rate

The **error rate** of the canary version must not exceed 0.1% of all requests. This is measured as the ratio of 5xx responses to total requests served by the canary instances. If the **error rate** exceeds 0.1% at any point during the **observation period**, Argo Rollouts automatically aborts the **Canary release** and routes all traffic back to the stable version.

## Automatic Rollback

Argo Rollouts is configured with metric-based analysis templates that automate the decision to abort or proceed:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: canary-metrics
spec:
  metrics:
  - name: p99-latency
    interval: 1m
    successCondition: result < 1020
    provider:
      prometheus:
        query: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket{
              app="document-processor",version="canary"}[5m])) by (le))
  - name: error-rate
    interval: 1m
    successCondition: result < 0.001
    provider:
      prometheus:
        query: |
          sum(rate(http_requests_total{
            app="document-processor",version="canary",status=~"5.."}[1m]))
          /
          sum(rate(http_requests_total{
            app="document-processor",version="canary"}[1m]))
```

## Operational Notes

The Canary release strategy is mandatory for all production services at Meridian. Key operational guidelines:

- The initial 5% weight ensures minimal blast radius if the new version introduces a regression.
- The 15-minute observation period is calibrated for the document processing service's warm-up characteristics; latency-sensitive services may require a longer window.
- When ArgoCD detects a failed analysis run, it automatically scales the canary ReplicaSet to zero and restores full **traffic routing** to the stable version.

## Revision History

This document was last updated on 16 July 2026 following the standardization of Canary release templates across all microservice deployments.
