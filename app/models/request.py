"""
KML Agent Service - 请求数据模型
定义API请求的Pydantic模型
"""

from pydantic import BaseModel
from typing import Optional, List
from enum import IntEnum


class RouteDifficulty(IntEnum):
    """路线难度等级"""
    EASY = 1
    MEDIUM = 2
    HARD = 3
    VERY_HARD = 4
    EXTREME = 5


class PoiFilterItem(BaseModel):
    """待筛选的 POI 条目"""

    name: str
    """POI 名称"""

    category: str = ""
    """当前类别（可选，如 water/camp/supply/photo/pass 等）"""

    latitude: float
    longitude: float
    elevation: Optional[float] = None
    """海拔（米，可选）"""

    description: Optional[str] = None
    """描述（可选，截断后送入 LLM）"""


class PoiFilterRequest(BaseModel):
    """POI LLM 筛选请求"""

    route_id: Optional[str] = None
    """关联路线 ID（可选，仅用于日志）"""

    pois: List[PoiFilterItem]
    """待筛选的 POI 列表"""

    model_config = {
        "json_schema_extra": {
            "example": {
                "route_id": "route_1788106618095_DNukEfsb",
                "pois": [
                    {
                        "name": "东台望海峰",
                        "category": "pass",
                        "latitude": 39.0817,
                        "longitude": 113.6534,
                        "elevation": 2795.0,
                        "description": None
                    }
                ]
            }
        }
    }


class PoiResolvePoi(BaseModel):
    """待判定的路线 POI"""

    name: str
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    exclude_id: Optional[str] = None
    """召回候选时排除的库条目 id（批内互判场景下用于排除自身）"""


class PoiResolveLibraryItem(BaseModel):
    """库内候选条目"""

    id: str
    name: str
    latitude: float
    longitude: float
    category: Optional[str] = None
    elevation: Optional[float] = None


class PoiResolveRequest(BaseModel):
    """POI 位置合并 AI 判定请求"""

    route_id: Optional[str] = None
    """关联路线 ID（可选，仅用于日志）"""

    pois: List[PoiResolvePoi]
    """待判定的新 POI 列表"""

    library: List[PoiResolveLibraryItem]
    """同地区的库内条目（由调用方过滤，agent 内部再做距离召回）"""


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
