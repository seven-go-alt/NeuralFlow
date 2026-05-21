from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

import aiofiles
from fastapi import UploadFile

from app.config import get_settings


@runtime_checkable
class DocumentStorage(Protocol):
    """Protocol for document storage backends."""

    async def save_upload(
        self, tenant_id: str, document_id: str, upload: UploadFile
    ) -> tuple[str, int, str]:
        """Save an uploaded file. Returns (storage_path, size_bytes, sha256_hex)."""
        ...

    async def read(self, storage_path: str) -> bytes:
        """Read file contents by storage path."""
        ...

    async def delete(self, storage_path: str) -> None:
        """Delete a file by storage path."""
        ...

    async def get_local_path(self, storage_path: str) -> str:
        """Return a local filesystem path readable by sync libraries (fitz, Pillow, etc.)."""
        ...


class LocalDocumentStorage:
    """Local filesystem document storage."""

    def __init__(self, base_dir: str | None = None) -> None:
        settings = get_settings()
        default_dir = os.path.join("data", "uploads")
        storage_dir = getattr(settings, "documents_storage_dir", default_dir)
        if storage_dir is None:
            storage_dir = default_dir
        self.base_dir = Path(base_dir or str(storage_dir))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(
        self, tenant_id: str, document_id: str, upload: UploadFile
    ) -> tuple[str, int, str]:
        tenant_dir = self.base_dir / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(upload.filename or "upload.bin").suffix
        file_path = tenant_dir / f"{document_id}{suffix}"

        sha256 = hashlib.sha256()
        size = 0
        async with aiofiles.open(file_path, "wb") as out:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                sha256.update(chunk)
                await out.write(chunk)
        await upload.close()
        return str(file_path), size, sha256.hexdigest()

    async def read(self, storage_path: str) -> bytes:
        async with aiofiles.open(storage_path, "rb") as f:
            return await f.read()

    async def delete(self, storage_path: str) -> None:
        path = Path(storage_path)
        if path.exists():
            path.unlink()

    async def get_local_path(self, storage_path: str) -> str:
        return storage_path


class S3DocumentStorage:
    """S3-compatible object storage backend."""

    def __init__(
        self,
        bucket: str | None = None,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self.bucket = bucket or settings.s3_bucket
        self.region = region or settings.s3_region
        self.endpoint_url = endpoint_url or settings.s3_endpoint_url
        self._s3 = self._get_client(access_key_id, secret_access_key)

    def _get_client(self, access_key_id: str | None, secret_access_key: str | None):
        import boto3

        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=self.region,
        )
        return session.client("s3", endpoint_url=self.endpoint_url or None)

    async def _ensure_bucket(self) -> None:
        try:
            self._s3.head_bucket(Bucket=self.bucket)
        except Exception:
            self._s3.create_bucket(Bucket=self.bucket)

    def _key(self, storage_path: str) -> str:
        return storage_path

    async def save_upload(
        self, tenant_id: str, document_id: str, upload: UploadFile
    ) -> tuple[str, int, str]:
        await self._ensure_bucket()
        suffix = Path(upload.filename or "upload.bin").suffix
        key = f"{tenant_id}/{document_id}{suffix}"

        sha256 = hashlib.sha256()
        size = 0
        import io

        buffer = io.BytesIO()
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            sha256.update(chunk)
            buffer.write(chunk)
        await upload.close()

        buffer.seek(0)
        self._s3.upload_fileobj(buffer, self.bucket, key)
        return key, size, sha256.hexdigest()

    async def read(self, storage_path: str) -> bytes:
        import io

        buffer = io.BytesIO()
        self._s3.download_fileobj(self.bucket, storage_path, buffer)
        buffer.seek(0)
        return buffer.read()

    async def delete(self, storage_path: str) -> None:
        self._s3.delete_object(Bucket=self.bucket, Key=storage_path)

    async def get_local_path(self, storage_path: str) -> str:
        """Download S3 object to a temp file for sync libraries."""
        suffix = Path(storage_path).suffix or ".tmp"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        self._s3.download_file(self.bucket, storage_path, tmp_path)
        return tmp_path


def create_storage(backend: str | None = None) -> DocumentStorage:
    """Factory: create a DocumentStorage from settings."""
    from app.config import get_settings

    s = get_settings()
    backend = (backend or s.document_storage_backend or "local").lower()

    if backend == "s3":
        return S3DocumentStorage()
    return LocalDocumentStorage()
