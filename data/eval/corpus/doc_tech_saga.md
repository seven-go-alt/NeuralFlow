# Meridian Analytics — Saga Orchestration Pattern for Distributed Transactions

**Document ID:** doc_tech_saga
**Owner:** Payments Engineering Team
**Last updated:** 2026-07-04

## Overview

Meridian Analytics processes multi-step workflows across multiple microservices, including customer onboarding, subscription provisioning, and billing reconciliation. Each workflow must maintain data consistency across service boundaries without distributed ACID transactions. This document describes Meridian's approach to the **Saga** **orchestration** pattern for managing **distributed transaction** flows, with specific guidance on handling **partial failures** and **compensation** logic when a step fails after compensation has already been attempted.

## Saga Orchestrator Architecture

Meridian implements orchestration-based Saga using a dedicated service built on Temporal.io. Each workflow is modeled as a sequence of transactional steps with corresponding compensation actions. The orchestrator maintains Saga execution state in Temporal's workflow history, enabling durable execution across restarts and failures.

For a five-step distributed transaction, the Saga is defined as follows:

```java
@Saga(name = "customer-provisioning")
public class CustomerProvisioningSaga {
    @SagaStep(step = 1, compensation = "revertAccountCreation")
    public Account createAccount(CreateAccountRequest request) { ... }
    @SagaStep(step = 2, compensation = "revertSubscriptionSetup")
    public Subscription setupSubscription(SubscriptionConfig config) { ... }
    @SagaStep(step = 3, compensation = "revertPaymentMethod")
    public PaymentMethod registerPaymentMethod(PaymentMethodDetails details) { ... }
    @SagaStep(step = 4, compensation = "revertDataPipeline")
    public PipelineConfig provisionDataPipeline(DataPipelineSpec spec) { ... }
    @SagaStep(step = 5, compensation = "revertNotifications")
    public void sendWelcomeNotifications(NotificationPreferences prefs) { ... }
}
```

Each step's compensation must be semantically idempotent.

## Handling Partial Failures After Compensation Attempt

The most complex failure scenario occurs when a step fails AND its compensation also fails. Consider Step 3 (`registerPaymentMethod`) succeeding in the external gateway but failing during local database commit. The orchestrator initiates compensation via `revertPaymentMethod`. If the external gateway is unreachable and the compensation also fails, the Saga enters a partially compensated state: Steps 1 and 2 are compensated, Step 3 is ambiguous, and Steps 4-5 never started.

Meridian's approach follows three principles:

**1. Retry with Exponential Backoff**: The orchestrator retries `revertPaymentMethod` up to 5 times with backoff (5s, 15s, 45s, 135s, 405s). If compensation succeeds on any retry, the Saga continues rolling back.

**2. Dead Letter Queue**: If all 5 retries fail, the pending compensations are recorded in a Saga Dead Letter Queue. The orchestrator persists the state and halts execution. An alert fires to the Payments Engineering team with the Saga ID.

**3. Idempotent Compensation Recovery**: A reconciliation job runs every 30 minutes and replays dead letter entries. It checks whether the compensation was applied by querying the gateway's GET endpoint. If the resource was already deleted, the compensation is marked complete without calling the API again.

## Forward Recovery vs. Backward Recovery

When a step fails, the orchestrator chooses between forward recovery (retry the failed step) and backward recovery (compensate completed steps):

- **Transient failures** (network timeout, 503) on Step 3: Retry up to 3 times with exponential backoff. If all fail, begin backward recovery.
- **Business logic failures** (validation error) on Step 3: Immediate backward recovery, no retry.
- **Compensation failure**: Always retries up to 5 times before escalating to dead letter queue. Failed compensations must never leave data inconsistent.

## Monitoring and Saga State Management

Temporal exposes per-workflow metrics including `saga_execution_status` and `saga_compensation_attempts`. Any Saga in compensating state for more than 30 minutes triggers a critical alert requiring manual reconciliation by the on-call engineer.
