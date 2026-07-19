# Meridian Analytics — Document Chunking Strategy for RAG

**Document ID:** doc_tech_chunking
**Owner:** ML Engineering
**Last updated:** 2026-07-06

## Overview

This document defines the **text splitter** configuration used by Meridian Analytics to split ingested documents into chunks before embedding and indexing. Proper **chunk size** and **overlap** configuration is critical for retrieval quality in the AskMeridian RAG pipeline. Different document formats require different **text splitter** parameters.

## Recommended Configuration for PDF Documents

For **PDF** documents, the recommended **text splitter** configuration is as follows:

| Parameter | Value |
|-----------|-------|
| **Chunk size** | 512 tokens |
| **Overlap** | 64 tokens |
| Splitting strategy | Recursive character splitter |
| Separators | `\n\n`, `\n`, `.`, ` ` |
| Chunking target | Semantic paragraph boundaries preferred |

The **chunk size** of 512 tokens ensures that each chunk fits comfortably within the embedding model's optimal input length while capturing enough context for meaningful semantic representation. The **overlap** of 64 tokens (12.5% of the **chunk size**) ensures that sentences and concepts that span chunk boundaries are preserved in at least one complete chunk.

## Text Splitter Implementation

The **text splitter** is implemented as a middleware component in the document ingestion pipeline. For **PDF** documents specifically, the pipeline first extracts raw text using Meridian's PDF parser (based on PyMuPDF), then passes the text through the recursive character **text splitter** with the parameters shown above.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ".", " "],
    length_function=token_count,
)
```

The `token_count` function uses Meridian's tokenizer, which is compatible with the Embedding API v3 model, ensuring that **chunk size** is measured in model tokens rather than raw characters.

## Chunk Size and Overlap Tuning

The recommended **chunk size** of 512 tokens and **overlap** of 64 tokens for **PDF** documents were determined through an ablation study on Meridian's internal QA dataset. The study compared retrieval accuracy (Recall@5) across different **text splitter** configurations:

| Chunk size | Overlap | Recall@5 | Notes |
|------------|---------|----------|-------|
| 256 | 32 | 0.842 | Good for short documents |
| 384 | 48 | 0.871 | Balanced, general purpose |
| **512** | **64** | **0.913** | **Recommended for PDF** |
| 768 | 96 | 0.889 | Higher storage cost, marginal gain |
| 1024 | 128 | 0.862 | Context window pressure |

The **chunk size** of 512 and **overlap** of 64 tokens achieved the highest Recall@5 score of 0.913 for **PDF** documents in the study.

## Format-Specific Configurations

While this document focuses on **PDF** documents, other formats use different **text splitter** configurations:

| Format | Chunk size | Overlap | Separators |
|--------|-----------|---------|------------|
| **PDF** | 512 | 64 | `\n\n`, `\n`, `.`, ` ` |
| HTML | 384 | 48 | HTML tags, `\n\n`, `\n` |
| Markdown | 384 | 48 | Markdown headings, `\n\n` |
| Plaintext | 512 | 64 | `\n\n`, `\n`, `.` |

## Revision History

This document was last updated on 6 July 2026 following the completion of the chunking ablation study on the production QA dataset.
