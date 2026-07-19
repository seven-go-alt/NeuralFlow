# Meridian Analytics — Document Upload Endpoint Reference

**Document ID:** doc_api_upload
**Owner:** Platform Engineering
**Last updated:** 2026-02-22

## Overview

The Meridian Document Upload endpoint provides a chunked transfer encoding mechanism for uploading large documents to the Meridian analytics platform. This endpoint is designed for documents that exceed the 50 MB payload size limit of the Document Ingestion API v2.1. Chunked transfer encoding allows clients to send large files in manageable segments, with each segment acknowledged before the next is sent.

## Endpoint

`PUT https://api.meridian-analytics.com/v2.1/upload/documents`

## Maximum file size when using chunked transfer encoding

The maximum file size supported by the document upload endpoint when using chunked transfer encoding is 2 GB. This limit applies to the total assembled file after all chunks have been received and merged. Files exceeding 2 GB must be split into multiple documents and submitted through separate upload sessions.

The upload process works as follows:

1. The client initiates an upload session by sending a `POST` request to the session creation endpoint.
2. The server responds with a `session_id` and a list of `upload_urls`, each valid for 15 minutes.
3. The client uploads each chunk using a `PUT` request to the corresponding `upload_url`. Each chunk must be between 5 MB and 256 MB in size. The final chunk may be smaller than 5 MB.
4. After all chunks are uploaded, the client sends a `POST` request to the finalization endpoint with the `session_id`.
5. The server assembles the chunks into the complete file and validates the integrity using MD5 checksums provided in each chunk upload.

## Chunk upload headers

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Range` | Yes | Byte range of the chunk, e.g., `bytes 0-4194303/104857600` |
| `Content-Length` | Yes | Size of the chunk in bytes |
| `Content-MD5` | Yes | Base64-encoded MD5 hash of the chunk content |
| `X-Upload-Session-ID` | Yes | Session identifier returned by the session creation call |

## Supported file types

The document upload endpoint supports the same file types as the ingestion API, plus the following additional formats suitable for large documents: MP4 (for video-based analytics), ZIP archives, and Parquet files. Each upload session can handle only one file type, which must be declared in the session creation request.

## Error handling

If a chunk upload fails, the client may retry the same chunk using the same `upload_url` as long as the URL is still valid (within the 15-minute window). If the URL has expired, the client must obtain new `upload_urls` by calling the session status endpoint. After 3 consecutive failed chunk uploads, the session is automatically terminated and all uploaded chunks are discarded.

## Throttling

The document upload endpoint applies throughput-based throttling rather than request-based rate limiting. The maximum aggregate upload throughput per API key is 500 MB per minute for all concurrent upload sessions combined. Enterprise tier customers have a throughput limit of 2 GB per minute.

## Revision history

This reference was last updated on 22 February 2026. The chunked transfer encoding feature was introduced in API version 2.1.
