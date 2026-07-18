# Meridian Analytics — Kafka Deserialization Error Handling

**Document ID:** doc_tech_error_handling
**Owner:** Data Engineering
**Last updated:** 2026-07-14

## Overview

Meridian Analytics operates an **event-driven** architecture built on Apache **Kafka** for processing financial data streams. This document defines the proper **error handling** pattern when a consumer encounters a **deserialization error** for a message in the middle of a **batch**.

## The Problem

In Meridian's **event-driven** ingestion pipeline, **Kafka** consumers read messages in configurable **batch** sizes (default 500 records per poll). When a **deserialization error** occurs for one message in the middle of a **batch**, the default **Kafka** client behavior is to throw an exception and retry the entire **batch** from the beginning, causing duplicate processing and potential ordering violations for messages that were already processed successfully.

## Error Handling Pattern

Meridian uses a per-message **error handling** pattern that isolates the problematic record and routes it to a **dead letter queue** without aborting the **batch**:

```java
@KafkaListener(topics = "meridian.events.financial", batch = "true")
public void processBatch(List<ConsumerRecord<String, byte[]>> records, Acknowledgment ack) {
    List<RecordMetadata> failedRecords = new ArrayList<>();

    for (ConsumerRecord<String, byte[]> record : records) {
        try {
            FinancialEvent event = deserializer.deserialize(record.key(), record.value());
            processEvent(event);
        } catch (DeserializationException e) {
            // Isolate the failed record — do not abort the batch
            failedRecords.add(new RecordMetadata(record, e));
            log.warn("Deserialization error at offset {}: {}", record.offset(), e.getMessage());
        }
    }

    // Route failures to dead letter queue after processing the batch
    if (!failedRecords.isEmpty()) {
        deadLetterQueue.send(failedRecords);
    }

    ack.acknowledge(); // Commit offsets for the entire batch
}
```

This approach ensures that a single **deserialization error** does not block processing of the remaining 499 messages in the **batch**.

## Dead Letter Queue Design

The **dead letter queue** (DLQ) for **Kafka** deserialization failures follows these design rules:

- **Topic:** `meridian.dlq.financial` with 12 partitions, keyed by the original message key.
- **Schema:** Each DLQ record stores the original topic, partition, offset, error type, error message, raw payload (base64), and a timestamp.
- **Retention:** Messages in the **dead letter queue** are retained for 14 days, allowing engineers to replay them after fixing the schema or producer issue.
- **Alerting:** A CloudWatch alarm triggers when the DLQ partition lag exceeds 100 messages, indicating a systemic serialization issue rather than a one-off corruption.

## Batch Processing Guarantees

Meridian's **event-driven** system requires strict ordering guarantees for financial events within a partition. The per-message **error handling** pattern preserves ordering because:

1. The successful messages in the **batch** are committed at the normal offset.
2. The failed message's offset is skipped, and the consumer resumes reading from the next offset.
3. The **dead letter queue** has its own offset tracking, independent of the source partition.

This means that a **deserialization error** at offset 5,000 in a partition does not prevent the consumer from processing offsets 5,001 onward, and the original message is preserved in the **dead letter queue** for forensic analysis.

## Operational Runbook

When a **deserialization error** is detected in the **dead letter queue**:

1. Identify whether the error is schema-related (field renamed, removed, type changed) or data corruption (bit rot, truncated payload).
2. For schema errors, update the Avro schema registry and replay the DLQ messages into the original topic with a new producer version.
3. For corruption errors, attempt recovery from the source system (for example, Meridian's S3 event archive) and publish corrected messages.

## Revision History

This document was last updated on 14 July 2026 following the migration to per-message error isolation in the financial events consumer group.
