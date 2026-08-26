"""Structured, sanitized events for the analysis execution console."""

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field


ExecutionPhase = Literal[
    "started",
    "completed",
    "llm_completed",
    "degraded",
    "failed",
]
ExecutionLevel = Literal["info", "warning", "error"]


class ExecutionEvent(BaseModel):
    """A safe event that may be forwarded outside the Agent service."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node: str
    phase: ExecutionPhase
    level: ExecutionLevel
    message: str
    progress: int = Field(ge=0, le=100)
    details: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")


def classify_safe_error(error: Exception) -> Dict[str, str]:
    """Classify an exception without copying its potentially sensitive text."""

    text = str(error).lower()
    if any(marker in text for marker in ("未配置", "not configured", "missing key")):
        category, summary = "configuration", "模型服务配置缺失"
    elif isinstance(error, (TimeoutError, httpx.TimeoutException)) or "timeout" in text or "timed out" in text:
        category, summary = "timeout", "模型服务请求超时"
    elif any(marker in text for marker in ("401", "403", "unauthorized", "forbidden", "api key", "api_key")):
        category, summary = "authentication", "模型服务鉴权失败"
    elif isinstance(error, (ConnectionError, httpx.ConnectError)) or any(
        marker in text for marker in ("connection", "connect error", "offline", "unreachable")
    ):
        category, summary = "connection", "模型服务连接失败"
    elif isinstance(error, (ValueError, TypeError)) or any(
        marker in text for marker in ("json", "parse", "format", "字段", "响应")
    ):
        category, summary = "response_format", "模型响应格式错误"
    else:
        category, summary = "unknown", "模型调用失败"

    return {"error_category": category, "summary": summary}


def build_execution_event(
    *,
    node: str,
    phase: ExecutionPhase,
    level: ExecutionLevel,
    message: str,
    progress: int,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a JSON-compatible execution event dictionary."""

    return ExecutionEvent(
        node=node,
        phase=phase,
        level=level,
        message=message,
        progress=progress,
        details=details,
    ).model_dump(mode="json")
