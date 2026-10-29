"""Loads models.yaml and instantiates adapters lazily."""

import importlib
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.models.base import ModelAdapter

log = logging.getLogger(__name__)

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass
class ModelEntry:
    id: str
    name: str
    description: str
    class_path: str
    params: dict = field(default_factory=dict)
    enabled: bool = True
    adapter: ModelAdapter | None = None
    status: str = "ready"  # ready | loading | error
    error: str | None = None
    loaded: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class ModelRegistry:
    def __init__(self, yaml_path: Path):
        with open(yaml_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        self.default_id: str = config.get("default", "")
        self._entries: dict[str, ModelEntry] = {}
        for raw in config.get("models", []):
            entry = ModelEntry(
                id=raw["id"],
                name=raw.get("name", raw["id"]),
                description=raw.get("description", ""),
                class_path=raw["class"],
                params=_expand_env(raw.get("params", {}) or {}),
                enabled=bool(raw.get("enabled", True)),
            )
            self._entries[entry.id] = entry
        if self.default_id not in self.enabled_ids():
            enabled = self.enabled_ids()
            self.default_id = enabled[0] if enabled else ""

    def enabled_ids(self) -> list[str]:
        return [e.id for e in self._entries.values() if e.enabled]

    def entries(self) -> list[ModelEntry]:
        return [e for e in self._entries.values() if e.enabled]

    def has(self, model_id: str) -> bool:
        entry = self._entries.get(model_id)
        return entry is not None and entry.enabled

    def entry(self, model_id: str) -> ModelEntry:
        entry = self._entries.get(model_id)
        if entry is None or not entry.enabled:
            raise KeyError(model_id)
        return entry

    def _instantiate(self, entry: ModelEntry) -> ModelAdapter:
        module_name, _, class_name = entry.class_path.rpartition(".")
        cls = getattr(importlib.import_module(module_name), class_name)
        adapter = cls(model_id=entry.id, **entry.params)
        if not isinstance(adapter, ModelAdapter):
            raise TypeError(f"{entry.class_path} does not implement ModelAdapter")
        return adapter

    def get(self, model_id: str) -> ModelAdapter:
        """Returns a loaded adapter, loading it on first use."""
        entry = self.entry(model_id)
        if entry.loaded and entry.adapter is not None:
            return entry.adapter
        with entry.lock:
            if entry.loaded and entry.adapter is not None:
                return entry.adapter
            entry.status = "loading"
            try:
                adapter = entry.adapter or self._instantiate(entry)
                entry.adapter = adapter
                adapter.load()
                entry.loaded = True
                entry.status = "ready"
                entry.error = None
            except Exception as exc:  # surfaced through /models
                entry.status = "error"
                entry.error = str(exc)
                log.exception("failed to load model %s", model_id)
                raise
        return entry.adapter

    def peek(self, model_id: str) -> ModelAdapter | None:
        """Instantiates without loading weights, for reading static attributes."""
        entry = self.entry(model_id)
        if entry.adapter is None:
            try:
                entry.adapter = self._instantiate(entry)
            except Exception as exc:
                entry.status = "error"
                entry.error = str(exc)
                return None
        return entry.adapter

    def loaded_ids(self) -> list[str]:
        return [e.id for e in self.entries() if e.loaded]

    def preload(self) -> None:
        for model_id in self.enabled_ids():
            try:
                self.get(model_id)
            except Exception:
                pass


_registry: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                from app.config import get_settings

                _registry = ModelRegistry(get_settings().models_yaml)
    return _registry


def reset_registry() -> None:
    global _registry
    _registry = None
