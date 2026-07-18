# Meridian Analytics — Document Ingestion API v2.1 Reference

**Document ID:** doc_api_ingestion
**Owner:** Platform Engineering
**Last updated:** 2026-02-18

## Overview

The Meridian Document Ingestion API enables programmatic upload and processing of documents into the Meridian analytics platform. Version 2.1 introduces support for new content types, improved error handling, and updated payload size constraints. This reference covers the ingestion endpoint, request parameters, and the maximum payload size constraints that apply.

## Endpoint

`POST https://api.meridian-analytics.com/v2.1/ingestion/documents`

The Document Ingestion API version 2.1 endpoint accepts document payloads in JSON format. Each request must include the document content, metadata, and processing configuration.

## Authentication and headers

All requests to the API version 2.1 endpoint require the following headers:

| Header | Value | Required |
|--------|-------|----------|
| `Authorization` | `Bearer <api_key>` | Yes |
| `Content-Type` | `application/json` | Yes |
| `X-Meridian-Version` | `2.1` | Yes |
| `X-Request-ID` | UUID v4 string | Recommended |

## Maximum payload size

The maximum payload size for the Document Ingestion API version 2.1 is 50 MB per request. This payload size limit applies to the entire JSON request body, including the document content (encoded as base64), metadata fields, and processing configuration. Requests exceeding 50 MB will receive a `413 Payload Too Large` response.

The payload size limit was increased from 25 MB in version 2.0 to 50 MB in version 2.1 to accommodate larger documents and richer metadata. For documents larger than 50 MB, use the Document Upload endpoint with chunked transfer encoding (see doc_api_upload). The ingestion API supports the following document formats: PDF, DOCX, HTML, Markdown, TXT, CSV, and JSON.

## Request body parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `document_id` | string | Yes | Unique identifier for the document |
| `content` | string | Yes | Base64-encoded document content |
| `content_type` | string | Yes | MIME type of the document (e.g., `application/pdf`) |
| `metadata` | object | No | Key-value pairs for document metadata |
| `processing_config` | object | No | Processing options (chunking strategy, embedding model) |

## Rate limits

The Document Ingestion API version 2.1 is subject to the following rate limits:

- **Free tier:** 10 requests per minute, 1 GB total payload per hour
- **Enterprise tier:** 250 requests per minute, 25 GB total payload per hour

For higher throughput requirements, consider using the batch processing endpoint (see doc_api_throughput).

## Error codes

| HTTP status | Error code | Description |
|-------------|------------|-------------|
| 400 | `INVALID_PAYLOAD` | Malformed request body |
| 413 | `PAYLOAD_TOO_LARGE` | Request exceeds the 50 MB maximum payload size |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests, retry after the specified interval |
| 500 | `INTERNAL_ERROR` | Server-side processing failure |

## Example request

```
POST /v2.1/ingestion/documents
Authorization: Bearer mdn_abc123def456
Content-Type: application/json
X-Meridian-Version: 2.1

{
  "document_id": "doc-2026-001",
  "content": "<base64_encoded_content>",
  "content_type": "application/pdf",
  "metadata": {
    "department": "Finance",
    "author": "jdoe@meridian-analytics.com"
  }
}
```

## Revision history

This reference was last updated on 18 February 2026. API version 2.1 is scheduled for deprecation on 31 December 2027.
