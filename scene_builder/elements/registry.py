"""Helpers for registering Mitsuba scene elements."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Tuple

from utils.registry import ELEMENT_REGISTRY


def get_scene_element(name: str) -> Callable[..., Dict[str, Any]]:
    return ELEMENT_REGISTRY.get(name)


def list_scene_elements() -> Iterable[str]:
    return sorted(ELEMENT_REGISTRY.keys())


def iter_scene_elements() -> Iterable[Tuple[str, Callable]]:
    for key in ELEMENT_REGISTRY.keys():
        yield key, ELEMENT_REGISTRY.get(key)
