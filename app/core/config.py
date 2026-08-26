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
    # 跨域与接口暴露配置
    # ========================================
    # 逗号分隔的允许来源。本服务不面向浏览器，正常调用方是 walkbg 后端，
    # 因此生产环境应收敛（默认留空表示不允许任何跨域浏览器请求）。
    cors_allowed_origins: str = ""

    # 是否暴露 OpenAPI 文档（/docs、/redoc）。
    # 文档会完整列出内部接口与数据结构，生产环境默认关闭。
    enable_api_docs: bool = False

    # ========================================
    # WalkBG 回调配置
    # ========================================
    # 回调地址：walkbg 与本服务是两个独立部署的进程，
    # 单机 Compose 下为服务名，拆分主机后为内网地址，必须由环境注入。
    walkbg_base_url: str = "http://localhost:8080"
    walkbg_callback_endpoint: str = "/walkbg/api/v1/route-analysis/callback"
    walkbg_execution_event_endpoint: str = "/walkbg/api/v1/route-analysis/tasks/{task_id}/events"
    walkbg_api_timeout: int = 30
    walkbg_callback_enabled: bool = True
    # 回调重试次数（含首次尝试）与指数退避的基准秒数。
    # 分析结果只存在于本进程内存中，回调失败即永久丢失，因此需要重试。
    walkbg_callback_max_attempts: int = 4
    walkbg_callback_retry_base_delay: float = 2.0

    # ========================================
    # 分析配置
    # ========================================
    poi_search_radius: int = 500
    enable_content_generation: bool = True
    enable_poi_query: bool = True

    def cors_origins_list(self) -> list[str]:
        """将逗号分隔的来源配置解析为列表；留空则返回空列表（不允许跨域）。"""
        return [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ]

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
