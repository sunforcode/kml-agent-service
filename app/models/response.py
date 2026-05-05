"""
KML Agent Service - 请求数据模型
定义API请求的Pydantic模型
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import IntEnum
from datetime import datetime


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


# ========================================
# 响应模型
# ========================================

class TaskSubmitResponse(BaseModel):
    """任务提交响应"""
    
    task_id: str
    """任务ID"""
    
    status: str
    """任务状态：pending"""
    
    message: str
    """消息"""
    
    estimated_seconds: int
    """预估完成时间（秒）"""


class TaskStatusResponse(BaseModel):
    """任务状态查询响应"""
    
    task_id: str
    """任务ID"""
    
    status: str
    """任务状态：pending/processing/completed/failed"""
    
    progress: int
    """进度百分比（0-100）"""
    
    current_step: Optional[str] = None
    """当前执行步骤"""
    
    message: str
    """状态消息"""
    
    result: Optional[Dict[str, Any]] = None
    """分析结果（仅当status=completed时返回）"""
    
    error: Optional[str] = None
    """错误信息（仅当status=failed时返回）"""


class HealthResponse(BaseModel):
    """健康检查响应"""
    
    status: str
    """服务状态：healthy/unhealthy"""
    
    version: str
    """服务版本"""
    
    timestamp: datetime
    """当前时间"""
    
    checks: Optional[Dict[str, str]] = None
    """各组件检查状态"""


# ========================================
# 增强路线输出模型（最终结果）
# ========================================

class TrackPointOutput(BaseModel):
    """轨迹点输出（用于路段的起点/终点）
    
    与 walkbg SegmentDto.start_point / walkfg TrackPointVO 对齐
    """
    
    latitude: float
    """纬度"""
    
    longitude: float
    """经度"""
    
    elevation: Optional[float] = None
    """海拔（米）"""
    
    timestamp: Optional[datetime] = None
    """时间戳（可选）"""


class SegmentOutput(BaseModel):
    """路段输出
    
    与以下模型对齐：
    - walkbg: SegmentDto / SegmentCreateRequest
    - walkfg: SegmentModel
    
    字段命名约定：
    - 使用 snake_case（与 Python 一致）
    - walkbg/walkfg 使用 camelCase，需要通过 @JsonProperty/@JsonKey 映射
    """
    
    id: str
    """唯一ID"""
    
    name: str
    """路段名称"""
    
    sequence_number: int
    """序号（用于排序，从1开始）"""
    
    color: str
    """显示颜色（如 #FF5722）"""
    
    description: Optional[str] = None
    """路段描述"""
    
    # ========================================
    # 数值字段
    # ========================================
    
    distance: float
    """距离（公里）"""
    
    elevation_gain: float
    """爬升（米）"""
    
    elevation_loss: float
    """下降（米）"""
    
    estimated_time: int
    """预计时间（分钟）"""
    
    difficulty: int
    """难度等级（1-5）"""
    
    # ========================================
    # 轨迹点索引范围
    # ========================================
    
    track_start_index: Optional[int] = None
    """轨迹点起始索引（对应完整轨迹的点索引）"""
    
    track_end_index: Optional[int] = None
    """轨迹点结束索引"""
    
    # ========================================
    # 起止坐标（对象结构）
    # ========================================
    
    start_point: Optional[TrackPointOutput] = None
    """起点（包含 lat/lon/elevation）"""
    
    end_point: Optional[TrackPointOutput] = None
    """终点（包含 lat/lon/elevation）"""
    
    # ========================================
    # 路段类型
    # ========================================
    
    segment_type: Optional[str] = None
    """路段类型（坡度方向）：climb/descent/flat/mixed"""
    
    slope_direction: Optional[str] = None
    """坡度方向：climb/descent/flat/mixed（与 segment_type 同义）"""
    
    # ========================================
    # 额外分析信息
    # ========================================
    
    avg_slope_degrees: Optional[float] = None
    """平均坡度（度）"""
    
    max_slope_degrees: Optional[float] = None
    """最大坡度（度）"""
    
    confidence: Optional[float] = None
    """分段置信度（0.0-1.0）"""
    
    # ========================================
    # 可选字段
    # ========================================
    
    notes: Optional[str] = None
    """备注"""


class POIOutput(BaseModel):
    """POI输出基类"""
    
    name: Optional[str] = None
    """名称"""
    
    latitude: float
    """纬度"""
    
    longitude: float
    """经度"""
    
    description: Optional[str] = None
    """描述"""
    
    notes: Optional[str] = None
    """备注"""


class WaterSourceOutput(POIOutput):
    """水源输出"""
    
    source_type: str = "unknown"
    """水源类型：spring/tap/river/unknown"""
    
    reliability: float = 0.5
    """可靠性（0.0-1.0）"""


class CampsiteOutput(POIOutput):
    """营地输出"""
    
    capacity: Optional[int] = None
    """容量"""
    
    has_water: Optional[bool] = None
    """是否有水源"""
    
    has_facilities: Optional[bool] = None
    """是否有设施"""


class SupplyOutput(POIOutput):
    """补给点输出"""
    
    supply_type: str = "unknown"
    """补给类型：shop/restaurant/vending_machine/unknown"""


class MarkerPointOutput(POIOutput):
    """标记点输出"""
    
    elevation: Optional[float] = None
    """海拔（米）"""
    
    type: str = "viewpoint"
    """类型：viewpoint/danger/photo/note"""
    
    image_url: Optional[str] = None
    """图片链接"""
    
    icon_url: Optional[str] = None
    """图标链接"""


class WarningOutput(BaseModel):
    """警告输出"""
    
    level: str
    """级别：info/warning/error"""
    
    message: str
    """消息"""
    
    location: Optional[Dict[str, float]] = None
    """位置：{"lat": ..., "lon": ...}"""
    
    detail: Optional[str] = None
    """详细信息"""


class EnhancedRouteOutput(BaseModel):
    """
    增强路线输出（最终分析结果）
    
    此模型定义了返回给walkbg的完整数据结构
    """
    
    # ========================================
    # 元数据
    # ========================================
    source_kml_url: str
    """来源KML URL"""
    
    analysis_timestamp: datetime
    """分析时间戳"""
    
    quality_score: float
    """质量评分（0-100）"""
    
    # ========================================
    # 路线级别统计
    # ========================================
    total_distance_km: float
    """总距离（公里）"""
    
    total_elevation_gain_m: float
    """总爬升（米）"""
    
    total_elevation_loss_m: float
    """总下降（米）"""
    
    max_elevation: float
    """最高海拔（米）"""
    
    min_elevation: float
    """最低海拔（米）"""
    
    is_loop: bool
    """是否为环线"""
    
    estimated_difficulty: int
    """预估难度等级（1-5）"""
    
    # ========================================
    # 路段数据
    # ========================================
    segments: List[SegmentOutput]
    """路段列表"""
    
    # ========================================
    # POI数据
    # ========================================
    water_sources: List[WaterSourceOutput]
    """水源列表"""
    
    campsites: List[CampsiteOutput]
    """营地列表"""
    
    supplies: List[SupplyOutput]
    """补给点列表"""
    
    marker_points: List[MarkerPointOutput]
    """标记点列表"""
    
    # ========================================
    # 生成内容（可选）
    # ========================================
    generated_description: Optional[str] = None
    """生成的路线描述"""
    
    generated_highlights: List[str] = []
    """生成的路线亮点"""
    
    generated_difficulties: List[str] = []
    """生成的难点提示"""
    
    generated_safety_notes: List[str] = []
    """生成的安全提示"""
    
    equipment_recommendations: List[str] = []
    """装备推荐列表"""
    
    # ========================================
    # 质量信息
    # ========================================
    warnings: List[WarningOutput] = []
    """警告列表"""
    
    # ========================================
    # 原始数据（可选，用于调试）
    # ========================================
    raw_analysis_data: Optional[Dict[str, Any]] = None
    """原始分析数据（可选）"""
