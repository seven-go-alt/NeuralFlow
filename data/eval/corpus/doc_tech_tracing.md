# Meridian Analytics — Distributed Tracing Across Asynchronous Message Boundaries with RabbitMQ

**Document ID:** doc_tech_tracing
**Owner:** Observability Engineering
**Last updated:** 2026-07-15

## Overview

Meridian Analytics processes millions of asynchronous document analysis events daily through a RabbitMQ message broker. When a single consumer processes messages out of order, correlating spans across asynchronous message boundaries becomes challenging. This document describes the distributed tracing strategy adopted by the Observability Engineering team to maintain end-to-end trace visibility using the W3C trace context standard.

## W3C Trace Context Propagation

The foundation of Meridian's approach is the **W3C trace context** standard (Trace-Context HTTP headers and corresponding AMQP headers). Every producer publishes messages with two standard headers:

- **traceparent**: Carries the trace ID, span ID, and trace flags.
- **tracestate**: Carries vendor-specific tracing metadata.

When a producer publishes a message to RabbitMQ, it injects the current **W3C trace context** into the message headers before publishing. The consumer, upon receiving the message, extracts these headers and creates a new child span that references the parent span carried in the trace context. This creates a causally connected chain of spans even when message consumption order diverges from publication order.

```
Producer Span ──→ RabbitMQ ──→ Consumer Span (child of producer)
                                   ↓
                            Out-of-order processing
                                   ↓
                        Consumer Span B (sibling link)
```

## Handling Out-of-Order Consumption

Meridian's data processing pipeline features a single consumer that processes messages from multiple document analysis queues. Because latency varies per document type, messages are frequently processed out of order. To handle this, the **distributed tracing** system uses the following mechanism:

1. **Span linking via tracestate**: Each consumer span records the trace ID and parent span ID from the message's W3C trace context.
2. **Buffered span export**: Spans are not exported until the consumer acknowledges the message, ensuring the trace is complete before being reported to the tracing backend.
3. **Out-of-order correlation**: The tracing backend uses trace IDs to reassemble spans into the correct causal order, even though consumer span start times may not be sequential. The span hierarchy is derived from parent-child relationships in the W3C trace context, not from wall-clock timestamps.

This design ensures that a trace covering a "Publish → Consume → Process → Acknowledge" flow is accurately reconstructed regardless of how RabbitMQ delivers or reorders messages within a queue.

## RabbitMQ Consumer Implementation

The consumer application at Meridian implements the following sequence for each message:

```python
def handle_message(ch, method, properties, body):
    # Extract W3C trace context from RabbitMQ message headers
    traceparent = properties.headers.get("traceparent")
    tracestate = properties.headers.get("tracestate")

    # Create a child span in the existing trace
    with tracer.start_span("rabbitmq.consume", child_of=traceparent) as span:
        span.set_attribute("messaging.system", "rabbitmq")
        span.set_attribute("messaging.destination", method.routing_key)

        # Process the message body
        result = process_document(body)

        # Link spans for out-of-order processing
        span.set_attribute("process.order.sequence", result.sequence_number)
        ch.basic_ack(delivery_tag=method.delivery_tag)
```

## Monitoring and Alerting

The Observability Engineering team monitors **distributed tracing** health through:

- **Trace completion rate**: Percentage of traces containing all expected spans. An alert fires if completion drops below 95%.
- **Spans per trace**: A sudden increase may indicate runaway **asynchronous** processing loops.
- **W3C trace context injection rate**: Measures how many messages carry valid traceparent headers into RabbitMQ.

These metrics are visualized in Grafana dashboards that overlay trace data with queue depth and consumer lag, giving engineers a unified view of system health.

## Revision History

This document was last updated on 15 July 2026 following the rollout of W3C trace context headers across all RabbitMQ exchange bindings in production.
