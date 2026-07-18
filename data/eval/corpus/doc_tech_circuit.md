# Meridian Analytics — Circuit Breaker Configuration for Authentication Service

**Document ID:** doc_tech_circuit
**Owner:** Platform Reliability Team
**Last updated:** 2026-07-02

## Overview

Meridian Analytics' user authentication service (auth-service) handles login, token refresh, and session validation for all customer-facing products. As the front door to the platform, auth-service must remain resilient under adverse conditions including upstream identity provider degradation, credential stuffing attacks, and database connection exhaustion. This document specifies the **circuit breaker** configuration for the **authentication service** using **Resilience4j**, with a focus on the **error rate** threshold and **sliding window** parameters that govern when the circuit opens.

## Resilience4j Circuit Breaker Configuration

Meridian standardizes on Resilience4j for circuit breaker implementation across all Java-based microservices. The auth-service circuit breaker is configured through the `application.yml` file:

```yaml
resilience4j.circuitbreaker:
  configs:
    default:
      registerHealthIndicator: true
      slidingWindowSize: 60
      slidingWindowType: TIME_BASED
      minimumNumberOfCalls: 10
      permittedNumberOfCallsInHalfOpenState: 5
      automaticTransitionFromOpenToHalfOpenEnabled: true
      waitDurationInOpenState: 30s
      failureRateThreshold: 50
      eventConsumerBufferSize: 10
      recordExceptions:
        - java.net.ConnectException
        - java.net.SocketTimeoutException
        - io.grpc.StatusRuntimeException
        - org.springframework.web.client.HttpServerErrorException
```

The key parameters for the authentication service are:
- **slidingWindowSize**: 60 (seconds)
- **slidingWindowType**: TIME_BASED (60-second sliding window)
- **failureRateThreshold**: 50 (50% error rate triggers the circuit)
- **waitDurationInOpenState**: 30 seconds before transitioning to half-open

When the error rate exceeds 50% over the 60-second sliding window, the circuit breaker transitions from CLOSED to OPEN state. In OPEN state, all calls to the authentication service downstream dependencies are rejected immediately with a `CallNotPermittedException` without waiting for a timeout or connection failure. This prevents resource exhaustion and allows the downstream system to recover.

## Sliding Window Mechanics

The sliding window operates on a time-based granularity. Resilience4j divides the 60-second window into 10 buckets of 6 seconds each. Each bucket records the outcome (success or failure) of every call made during that 6-second interval. As time progresses, the oldest bucket slides out and a new bucket enters.

At the end of each 6-second bucket, the circuit breaker recalculates the aggregate error rate across all 10 buckets. If the error rate exceeds the configured threshold of 50%, and the minimum number of calls (10) has been reached within the window, the circuit transitions to OPEN. This bucketing approach smooths out transient spikes while still responding to sustained degradation within one minute.

## Half-Open Recovery and Backoff Strategy

After the 30-second wait duration in OPEN state expires, the circuit breaker automatically transitions to HALF_OPEN. In this state, the auth-service permits a limited number of probe calls (permittedNumberOfCallsInHalfOpenState = 5) to the downstream identity provider. If at least 3 of the 5 probe calls succeed (maintaining the 50% threshold), the circuit transitions back to CLOSED. Otherwise, it returns to OPEN for another 30-second wait cycle.

The auth-service also implements an exponential backoff strategy for the wait duration in open state. If the circuit breaker cycles through OPEN-HALF_OPEN-OPEN three times within a 10-minute window, the wait duration increases to 60 seconds, then 120 seconds on the fourth cycle. This progressive backoff prevents rapid cycling during extended outages.

## Monitoring and Circuit State Visibility

Meridian exposes circuit breaker state through Spring Boot Actuator health endpoints. The auth-service health check includes circuit breaker states for each downstream dependency: `auth-service-circuit-breaker-state` reports CLOSED, OPEN, or HALF_OPEN with the current error rate and call count. Prometheus metrics track `resilience4j_circuitbreaker_state` and `resilience4j_circuitbreaker_calls` with labels for the circuit name and service. A Datadog dashboard alerts the Platform Reliability team whenever the auth-service circuit breaker spends more than 5 consecutive minutes in OPEN state.
