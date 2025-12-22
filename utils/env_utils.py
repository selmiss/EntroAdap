"""
Environment variable expansion utilities.

These helpers let you use values like `${DATA_DIR}` in YAML configs and CLI args
and have them expanded before training starts.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any


def _expand_env_vars(obj: Any, _seen: set[int] | None = None) -> Any:
    """Recursively expand environment variables in any strings inside obj."""
    if obj is None:
        return None

    if _seen is None:
        _seen = set()

    oid = id(obj)
    if oid in _seen:
        return obj
    _seen.add(oid)

    if isinstance(obj, str):
        return os.path.expandvars(obj)

    # Dataclasses (e.g., ScriptArguments / SFTConfig / ModelConfig / DatasetMixtureConfig)
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            try:
                current = getattr(obj, f.name)
            except Exception:
                continue
            updated = _expand_env_vars(current, _seen)
            if updated is not current:
                try:
                    setattr(obj, f.name, updated)
                except Exception:
                    # Some configs may have read-only properties; skip safely.
                    pass
        return obj

    # Common containers (mutate in-place when possible)
    if isinstance(obj, dict):
        new_dict: dict[Any, Any] = {}
        for k, v in obj.items():
            new_k = _expand_env_vars(k, _seen)
            new_v = _expand_env_vars(v, _seen)
            new_dict[new_k] = new_v
        obj.clear()
        obj.update(new_dict)
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = _expand_env_vars(obj[i], _seen)
        return obj
    if isinstance(obj, tuple):
        # Tuples are immutable; return a rebuilt tuple for callers that can reassign.
        return tuple(_expand_env_vars(x, _seen) for x in obj)
    if isinstance(obj, set):
        new_set = {_expand_env_vars(x, _seen) for x in obj}
        obj.clear()
        obj.update(new_set)
        return obj

    return obj


def expand_env_vars(*args: Any) -> None:
    """Expand environment variables in any strings inside provided arg objects."""
    for a in args:
        _expand_env_vars(a)


