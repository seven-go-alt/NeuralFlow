# Meridian Analytics — DynamoDB Optimistic Concurrency for Financial Transactions

**Document ID:** doc_tech_dynamo
**Owner:** Platform Engineering
**Last updated:** 2026-07-17

## Overview

Meridian Analytics processes financial transactions through a **DynamoDB** table that requires strict consistency guarantees to prevent double-spending and balance corruption. This document describes the **optimistic concurrency** control strategy using **conditional updates**, a **version number** attribute, and **conditional expression** evaluation.

## Data Model

The financial transactions table in **DynamoDB** uses a **version number** attribute as the concurrency token:

```json
{
  "account_id": "ACC-2026-07-17-12345",
  "balance": 1500000,
  "version_number": 42,
  "last_updated": "2026-07-17T14:30:00Z",
  "pending_transactions": 3
}
```

Every write operation must include the current **version number** in a **conditional expression** to ensure that no other process has modified the record before the write completes. This is the foundation of Meridian's **optimistic concurrency** strategy.

## Conditional Update Pattern

Meridian implements **optimistic concurrency** using **DynamoDB**'s `ConditionExpression` parameter:

```python
def update_balance(account_id: str, delta: int, current_version: int) -> bool:
    try:
        response = table.update_item(
            Key={"account_id": account_id},
            UpdateExpression="SET balance = balance + :delta, "
                            "version_number = version_number + :one, "
                            "last_updated = :now",
            ConditionExpression="version_number = :expected_version",
            ExpressionAttributeValues={
                ":delta": delta,
                ":one": 1,
                ":now": int(time.time()),
                ":expected_version": current_version
            },
            ReturnValues="UPDATED_NEW"
        )
        return True
    except client.exceptions.ConditionalCheckFailedException:
        logger.warning("Optimistic concurrency conflict on account %s", account_id)
        return False
```

The **conditional expression** `version_number = :expected_version` ensures that the update only proceeds if the **version number** matches the value read by the caller. If another process has incremented the **version number** in the meantime, the **conditional update** fails and the caller must retry with the refreshed **version number**.

## Handling Financial Transactions

Financial transactions at Meridian involve multiple **conditional updates** in sequence:

1. Debit the source account using a **conditional update** with the current **version number**.
2. If step 1 succeeds, credit the destination account using a **conditional update** with its **version number**.
3. If either step fails due to an **optimistic concurrency** conflict, the entire transaction is retried from the beginning with refreshed **version number** values.

This approach guarantees that no financial transaction is lost or double-counted, even under high contention. Meridian's internal benchmarks show that the conflict rate is under 0.3% for most accounts, with the exception of high-frequency trading accounts where conflict rates reach 4.5%.

## Retry with Backoff

When a **conditional expression** fails due to an **optimistic concurrency** conflict, Meridian's client retries with exponential backoff:

```python
def retry_update(account_id, delta, max_retries=5):
    for attempt in range(max_retries):
        item = table.get_item(Key={"account_id": account_id})["Item"]
        if update_balance(account_id, delta, item["version_number"]):
            return True
        time.sleep(0.01 * (2 ** attempt))  # Exponential backoff
    raise MaxRetriesExceededError(f"Failed to update {account_id}")
```

The retry re-reads the current item to obtain the latest **version number** before attempting the **conditional update** again. This ensures that the **conditional expression** has the freshest possible value.

## Monitoring and Alerting

Meridian tracks **optimistic concurrency** conflicts through CloudWatch metrics:

- `DynamoDB_ConditionalCheckFailed`: Captures the rate of **conditional expression** failures as a count per minute.
- Per-account conflict rates are logged and exposed through Meridian's internal dashboard.

An alert is triggered when the conflict rate for any single account exceeds 10%, as this pattern is often precursor to a hot-key issue in the **DynamoDB** table partition.

## Revision History

This document was last updated on 17 July 2026 following the deployment of improved retry logic for high-frequency trading accounts.
