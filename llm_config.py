"""Shared LLM configuration for every agent and VLM baseline."""

import os
from typing import Optional


DEFAULT_LLM_MODEL = "gpt-5.6-terra"
API_KEY_ENV = "FURNITURE_API_KEY"
BASE_URL_ENV = "FURNITURE_BASE_URL"


def resolve_api_key(explicit: Optional[str] = None,
                    required: bool = True) -> Optional[str]:
    """Resolve the single API credential used by all project agents."""
    api_key = explicit or os.environ.get(API_KEY_ENV)
    if required and not api_key:
        raise ValueError(f"环境变量 {API_KEY_ENV} 未设置")
    return api_key


def resolve_base_url(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve the endpoint paired with ``FURNITURE_API_KEY``."""
    return explicit or os.environ.get(BASE_URL_ENV)


def require_shared_model(model: Optional[str] = None) -> str:
    """Reject per-agent model drift in experiments."""
    resolved = model or DEFAULT_LLM_MODEL
    if resolved != DEFAULT_LLM_MODEL:
        raise ValueError(
            f"所有 Agent 必须统一使用 {DEFAULT_LLM_MODEL}，收到: {resolved}")
    return resolved
