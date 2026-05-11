"""
KML Agent Service - LLM 客户端工厂

统一管理 LLM 客户端的初始化，支持 DeepSeek / OpenAI 等兼容接口。
各 Agent 通过 get_llm() 获取已配置好的客户端实例。
"""

from functools import lru_cache
from langchain_openai import ChatOpenAI
from app.core.config import settings


@lru_cache()
def get_llm() -> ChatOpenAI:
    """
    获取 LLM 客户端单例

    根据 .env 中的配置自动连接对应服务：
    - OPENAI_BASE_URL=https://api.deepseek.com → DeepSeek
    - OPENAI_BASE_URL 留空 → OpenAI 官方

    Returns:
        ChatOpenAI 实例（langchain 封装，兼容 DeepSeek API）
    """
    kwargs = {
        "model": settings.openai_model,
        "temperature": settings.openai_temperature,
        "max_tokens": settings.openai_max_tokens,
        "api_key": settings.openai_api_key,
    }

    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url

    return ChatOpenAI(**kwargs)
