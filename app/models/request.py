"""
KML Agent Service - 请求数据模型
定义API请求的Pydantic模型
"""

from pydantic import BaseModel
from typing import Optional
from enum import IntEnum


class RouteDifficulty(IntEnum):
    """路线难度等级"""
    EASY = 1
    MEDIUM = 2
    HARD = 3
    VERY_HARD = 4
    EXTREME = 5


class KmlAnalysisRequest(BaseModel):
    """
    KML分析任务请求模型
    
    后端(walkbg)调用POST /api/v1/analyze时使用此模型
    """
    
    route_id: Optional[str] = None
    """关联的walkbg路线ID（可选）"""
    
    kml_source: str
    """KML文件URL（必填）"""
    
    kml_content: Optional[str] = None
    """直接传入KML内容（优先级高于URL）"""
    
    enable_content_generation: bool = True
    """是否启用内容生成（LLM调用）"""
    
    enable_poi_query: bool = True
    """是否启用OSM POI查询"""
    
    poi_search_radius: int = 500
    """POI搜索半径（米）"""
    
    region_name: Optional[str] = None
    """区域名称提示（提高分析质量）"""
    
    estimated_difficulty: Optional[RouteDifficulty] = None
    """预估难度（可选）"""
    
    user_notes: Optional[str] = None
    """用户备注"""
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "route_id": "route_wutaishan_001",
                "kml_source": "http://walkbg:8080/static/kml/wutaishan.kml",
                "kml_content": None,
                "enable_content_generation": True,
                "enable_poi_query": True,
                "poi_search_radius": 500,
                "region_name": "五台山",
                "estimated_difficulty": 3,
                "user_notes": "顺时针大朝台路线"
            }
        }
    }
