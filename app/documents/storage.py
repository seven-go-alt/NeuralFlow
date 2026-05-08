from __future__ import annotations

import hashlib
import os
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from app.config import get_settings


class LocalDocumentStorage:
    def __init__(self, base_dir: str | None = None) -> None:
        settings = get_settings()
        default_dir = os.path.join("data", "uploads")
        self.base_dir = Path(base_dir or getattr(settings, "documents_storage_dir", default_dir))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, tenant_id: str, document_id: str, upload: UploadFile) -> tuple[str, int, str]:
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
