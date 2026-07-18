# Meridian Analytics — gRPC Service Mesh Proxy Timeout Configuration

**Document ID:** doc_tech_grpc
**Owner:** Infrastructure Platform Team
**Last updated:** 2026-07-01

## Overview

Meridian Analytics relies on a service mesh architecture (Istio with Envoy proxies) to manage inter-service communication across our microservice fleet. gRPC is the primary RPC protocol for internal service-to-service calls, powering everything from real-time analytics pipelines to user authentication flows. Proper timeout configuration for the **gRPC** **client connection** at the **service mesh** **proxy configuration** level is critical for preventing cascading failures and ensuring reliable request handling. This document establishes the standard timeout values and configuration best practices for Meridian's gRPC service mesh deployments.

## Default and Maximum Timeout Values

Every gRPC stream that traverses Meridian's service mesh passes through two layers of timeout configuration: the application-level gRPC deadline set by the client stub, and the Envoy proxy-level timeout set on the upstream cluster. For the service mesh proxy configuration, the maximum **timeout** value configurable for a gRPC client connection is **30 seconds**. This limit is enforced by a MeshConfig policy that rejects any VirtualService or DestinationRule specifying a timeout exceeding this value.

The 30-second maximum was established based on Meridian's service-level objectives (SLOs) and empirical latency distributions. Internal analysis showed that 99.9% of gRPC unary calls complete within 12 seconds, and 99% of streaming calls complete within 25 seconds. Setting a cap at 30 seconds provides adequate headroom for legitimate long-running operations while preventing pathological requests from holding proxy connection pools indefinitely.

Services that require timeouts longer than 30 seconds must submit an exception request through the Infrastructure Change Advisory Board with documented latency profiles and capacity planning evidence.

## Proxy Configuration Reference

The service mesh proxy timeout for gRPC client connections is configured through Istio DestinationRule resources. The following example shows the standard configuration for a Meridian analytics-service gRPC client:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: analytics-service-grpc
  namespace: analytics
spec:
  host: analytics-service.analytics.svc.cluster.local
  trafficPolicy:
    loadBalancer:
      simple: ROUND_ROBIN
    connectionPool:
      tcp:
        connectTimeout: 10s
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
    tls:
      mode: ISTIO_MUTUAL
    portLevelSettings:
      - port:
          number: 50051
        connectTimeout: 10s
        tls:
          mode: ISTIO_MUTUAL
  subsets:
    - name: v1
      labels:
        version: v1
```

The per-request timeout is set on the VirtualService rather than the DestinationRule:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: analytics-service-grpc
  namespace: analytics
spec:
  hosts:
    - analytics-service
  http:
    - timeout: 30s
      route:
        - destination:
            host: analytics-service
            subset: v1
```

## Retry Budget and Deadline Propagation

A critical consideration in Meridian's gRPC timeout strategy is deadline propagation across service mesh hops. When Service A calls Service B via gRPC with a 15-second deadline, the Envoy proxy at Service B must apply a timeout that respects the remaining budget. Our proxy configuration uses a timeout policy that sets the per-hop timeout to 80% of the remaining deadline, with a floor of 5 seconds and a ceiling of 30 seconds.

For retry handling, the default configuration enables a single retry on UNAVAILABLE and RESOURCE_EXHAUSTED gRPC status codes, with a per-retry timeout of 10 seconds. The total request time (initial attempt plus retries) must not exceed the client connection timeout value configured in the VirtualService. This retry budget calculation prevents retries from exceeding the original deadline and avoids cascading timeout accumulation in multi-hop call chains.

## Monitoring and Alerting

Meridian's observability stack tracks gRPC timeout-related metrics through each Envoy proxy's stats export. Key metrics include `cluster.upstream_rq_timeout`, `cluster.upstream_rq_per_try_timeout`, and the histogram of gRPC response status codes with DEADLINE_EXCEEDED. A PagerDuty alert fires when the DEADLINE_EXCEEDED rate exceeds 1% of total gRPC traffic for any service mesh proxy over a 5-minute window, prompting investigation into whether the 30-second timeout cap needs adjustment for that workload.
