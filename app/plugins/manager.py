from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from app.utils.observability import configure_structured_logging

logger = configure_structured_logging(
    logger_name="neuralflow.plugins",
    audit_log_path=os.getenv("NEURALFLOW_AUDIT_LOG_PATH", "/tmp/neuralflow_audit.log"),
)


class PluginManager:
    def __init__(self, plugins: list[ModuleType] | None = None) -> None:
        self._plugins = plugins or []

    @classmethod
    def from_env(cls) -> "PluginManager":
        plugin_dir = os.getenv("NEURALFLOW_PLUGIN_DIR", "plugins")
        return cls.load_from_dir(Path(plugin_dir))

    @classmethod
    def load_from_dir(cls, plugin_dir: Path) -> "PluginManager":
        if not plugin_dir.exists() or not plugin_dir.is_dir():
            return cls([])

        plugins: list[ModuleType] = []
        for file_path in sorted(plugin_dir.glob("*.py")):
            if file_path.name.startswith("_"):
                continue
            module = _load_module(file_path)
            if module is not None:
                plugins.append(module)
        return cls(plugins)

    def emit(self, hook_name: str, payload: dict[str, Any]) -> None:
        for plugin in self._plugins:
            hook = getattr(plugin, hook_name, None)
            if not callable(hook):
                continue
            try:
                hook(payload)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "plugin_hook_failed",
                    extra={
                        "plugin": getattr(plugin, "__name__", "unknown"),
                        "hook": hook_name,
                        "error": str(exc),
                    },
                )



def _load_module(file_path: Path) -> ModuleType | None:
    spec = importlib.util.spec_from_file_location(f"neuralflow_plugin_{file_path.stem}", file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
