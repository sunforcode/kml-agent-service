"""
KML Agent Service - 配置管理
使用pydantic-settings管理环境变量配置
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """服务配置类"""

    # ========================================
    # 服务基础配置
    # ========================================
    service_name: str = "kml-agent-service"
    service_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"

    # ========================================
    # FastAPI 配置
    # ========================================
    host: str = "0.0.0.0"
    port: int = 8001

    # ========================================
    # LLM 配置
    # ========================================
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "deepseek-chat"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 4000

    # ========================================
    # 外部 API 配置
    # ========================================
    nominatim_url: str = "https://nominatim.openstreetmap.org"
    nominatim_timeout: int = 10
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    overpass_timeout: int = 30

    # ========================================
    # WalkBG 回调配置
    # ========================================
    walkbg_base_url: str = "http://localhost:8080"
    walkbg_callback_endpoint: str = "/walkbg/api/v1/route-analysis/callback"
    walkbg_api_timeout: int = 30
    walkbg_callback_enabled: bool = True

    # ========================================
    # 分析配置
    # ========================================
    poi_search_radius: int = 500
    enable_content_generation: bool = True
    enable_poi_query: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
