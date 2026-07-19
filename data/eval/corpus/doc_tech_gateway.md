# Meridian Analytics — API Gateway Version Negotiation for REST API with JSON and Protocol Buffers

**Document ID:** doc_tech_gateway
**Owner:** API Platform Engineering
**Last updated:** 2026-07-15

## Overview

Meridian Analytics exposes a **REST API** for document ingestion and query operations that supports both JSON and **Protocol Buffers** serialization. The **API gateway** layer handles **version negotiation** and **content negotiation** by inspecting the **Accept header** on incoming requests. This document describes the gateway's routing and serialization selection logic.

## Content Negotiation via Accept Header

The **API gateway** implements **content negotiation** by parsing the **Accept header** of each incoming request. Clients specify their desired serialization format using standard MIME types:

| **Accept header** Value | Serialization Format |
|---------------------------|----------------------|
| `application/json` | JSON |
| `application/x-protobuf` | **Protocol Buffers** (binary) |
| `application/vnd.meridian.v1+json` | JSON, API version 1 |
| `application/vnd.meridian.v1+protobuf` | **Protocol Buffers**, API version 1 |
| `application/vnd.meridian.v2+json` | JSON, API version 2 |
| `application/vnd.meridian.v2+protobuf` | **Protocol Buffers**, API version 2 |

When a request arrives without an explicit **Accept header**, the gateway defaults to `application/json` with the latest stable **version negotiation** result.

## Version Negotiation Strategy

**Version negotiation** is handled through a combination of the **Accept header** and URI prefix. The gateway evaluates **version negotiation** in the following order of precedence:

1. **Vendor-specific media type**: If the **Accept header** contains `application/vnd.meridian.v2+json`, the gateway routes the request to the v2 handler and returns JSON.
2. **URI path prefix**: If no vendor media type is present, the gateway checks for a version prefix like `/v2/` in the request path. This is the fallback for clients that cannot control their **Accept header**.
3. **Default version**: If neither vendor media type nor path prefix is present, the gateway routes to the latest stable version (currently v2).

This dual approach supports both modern clients that use content-type-based **version negotiation** and legacy clients that rely on URI-based versioning.

## API Gateway Routing Implementation

The **API gateway** routing logic is implemented as a middleware pipeline in the gateway service:

```python
def negotiate_content(request):
    accept_header = request.headers.get("Accept", "application/json")

    # Parse the Accept header to determine version and format
    if "vnd.meridian.v2" in accept_header:
        version = "v2"
    elif "vnd.meridian.v1" in accept_header:
        version = "v1"
    elif request.path.startswith("/v2/"):
        version = "v2"
    elif request.path.startswith("/v1/"):
        version = "v1"
    else:
        version = get_latest_version()

    # Determine serialization format
    if "protobuf" in accept_header:
        format = "protobuf"
    else:
        format = "json"

    # Select serializer and route to the appropriate handler
    serializer = get_serializer(version, format)
    handler = get_handler(version, request.method, request.path)
    return handler(request, serializer)
```

## Protocol Buffers Integration

For clients that request **Protocol Buffers** serialization via the **Accept header**, the gateway performs automatic transcoding:

1. The gateway deserializes the incoming request using the negotiated format.
2. The internal handler processes the request using a common domain model (protobuf-native).
3. The response is serialized back to the client's requested format.

This approach means that all handlers operate on protobuf messages internally, and JSON support is provided by a transcoding layer. Clients using **Protocol Buffers** benefit from approximately 40% smaller payloads and 30% faster serialization compared to JSON, which is critical for Meridian's high-volume document ingestion endpoints.

## Revision History

This document was last updated on 15 July 2026 following the deployment of Protocol Buffers support in the API gateway for all v2 endpoints.
