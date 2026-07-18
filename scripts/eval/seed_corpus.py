"""Seed the synthetic eval corpus through the ingestion pipeline.

Each data/eval/corpus/<doc_id>.md becomes one document whose canonical
doc_id (= file stem) is carried into chunk metadata for citation
matching during evals. Idempotent: docs already ingested successfully
(READY) are skipped; failed or partial docs are retried on re-run.
Use --force to re-ingest everything.

Usage (from repo root):
    uv run python -m scripts.eval.seed_corpus
    uv run python -m scripts.eval.seed_corpus --force
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from pathlib import Path

from app.config import get_settings
from app.db.session import SessionLocal, init_db
from app.documents.enums import DocumentFileType, DocumentStatus
from app.documents.repository import DocumentRepository
from app.documents.schemas import DocumentCreate
from app.ingestion.pipeline import IngestionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed eval corpus via ingestion pipeline")
    parser.add_argument(
        "--corpus-dir",
        default="data/eval/corpus",
        help="Directory containing corpus .md files (default: data/eval/corpus)",
    )
    parser.add_argument(
        "--tenant", default=None, help="Tenant id (default: settings.tenant_default_id)"
    )
    parser.add_argument("--force", action="store_true", help="re-ingest existing documents")
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    settings = get_settings()
    tenant_id = args.tenant or settings.tenant_default_id
    corpus_dir = Path(args.corpus_dir).resolve()
    files = sorted(corpus_dir.glob("*.md"))
    if not files:
        print(f"no corpus files found in {corpus_dir}")
        return 1

    init_db()
    pipeline = IngestionPipeline()
    seeded = skipped = failed = 0
    for path in files:
        canonical_id = path.stem
        document_id = f"eval_{canonical_id}"
        db = SessionLocal()
        try:
            repo = DocumentRepository(db)
            existing = repo.get_document(tenant_id=tenant_id, document_id=document_id)
            if (
                existing is not None
                and not args.force
                and existing.status == DocumentStatus.READY
            ):
                skipped += 1
                continue
            if existing is None:
                content = path.read_bytes()
                repo.create_document(
                    DocumentCreate(
                        tenant_id=tenant_id,
                        owner_user_id="eval-seeder",
                        title=canonical_id,
                        filename=path.name,
                        original_filename=path.name,
                        file_type=DocumentFileType.MARKDOWN.value,
                        mime_type="text/markdown",
                        size_bytes=len(content),
                        storage_path=str(path),
                        checksum_sha256=hashlib.sha256(content).hexdigest(),
                        metadata_json={"canonical_doc_id": canonical_id, "eval_corpus": True},
                        source_info_json={"seeded_by": "seed_corpus.py"},
                    ),
                    document_id=document_id,
                )
            repo.update_status(tenant_id, document_id, DocumentStatus.QUEUED)
        finally:
            db.close()
        try:
            result = await pipeline.run(
                tenant_id=tenant_id,
                document_id=document_id,
                embedding_model=settings.embedding_model,
            )
            seeded += 1
            print(f"seeded {canonical_id}: {result['chunk_count']} chunks")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAILED {canonical_id}: {exc}")
    print(f"done: seeded={seeded} skipped={skipped} failed={failed} total={len(files)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
