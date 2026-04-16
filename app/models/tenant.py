from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TenantContext(BaseModel):
    tenant_id: str
    scope: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    subject: str = "anonymous"
    source: Literal["header", "default"] = "default"

    def can_access(self, target_tenant_id: str) -> bool:
        return "*" in self.scope or target_tenant_id in self.scope
