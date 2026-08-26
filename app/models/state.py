"""
KML Agent Service - 状态模型
定义LangGraph工作流使用的状态结构
"""

import operator
from typing import Annotated, TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """
    LangGraph工作流状态
    
    此状态在整个Agent工作流中传递，包含：
    - 输入数据
    - 中间处理结果
    - 执行进度
    - 错误和警告
    
    注意: 使用 total=False 表示所有字段都是可选的
    """
    
    # ========================================
    # 输入数据
    # ========================================
    request: Dict[str, Any]
    """原始分析请求（KmlAnalysisRequest的字典形式）"""
    
    # ========================================
    # 轨迹分析结果（轨迹分析Agent产出）
    # ========================================
    track_points: List[Dict[str, Any]]
    """轨迹点列表；distance_from_start 在 AgentState 中统一使用米。"""
    
    kml_markers: List[Dict[str, Any]]
    """KML标记点列表"""
    
    basic_stats: Optional[Dict[str, Any]]
    """基础统计数据（距离、爬升、下降等）"""
    
    # ========================================
    # 路径分段结果（路径分段Agent产出）
    # ========================================
    segment_schemes: List[Dict[str, Any]]
    """
    分段方案列表
    每条路线可有多套方案，结构为:
    [
        {
            "scheme_type": "slope",
            "label": "按坡度",
            "is_default": True,
            "segments": [...]
        },
        {
            "scheme_type": "day",
            "label": "按天",
            "is_default": False,
            "segments": [...]
        }
    ]
    """

    # ========================================
    # POI识别结果（POI识别Agent产出）
    # ========================================
    poi_points: List[Dict[str, Any]]
    """
    统一附属信息点列表
    结构为:
    [
        {
            "name": str,
            "latitude": float,
            "longitude": float,
            "elevation": float | None,
            "category": str,          # water|camp|supply|photo|pass|valley|weather|danger|start|end
            "sub_category": str | None,
            "source": str,            # kml_marker|algorithm|osm|weather_api|experience
            "description": str | None,
            "confidence": float | None,
            "card_data": dict | None
        }
    ]
    """
    
    # ========================================
    # 内容生成结果（内容生成Agent产出）
    # ========================================
    generated_content: Optional[Dict[str, Any]]
    """生成的内容（描述、亮点、装备推荐等）"""
    
    # ========================================
    # 质量评估结果（质量评估Agent产出）
    # ========================================
    quality_assessment: Optional[Dict[str, Any]]
    """质量评估结果"""
    
    # ========================================
    # 最终结果
    # ========================================
    final_result: Optional[Dict[str, Any]]
    """最终分析结果（EnhancedRouteOutput的字典形式）"""
    
    # ========================================
    # 执行进度
    # ========================================
    current_step: str
    """当前执行步骤"""
    
    overall_progress: int
    """总体进度（0-100）"""
    
    # ========================================
    # 错误处理
    # ========================================
    errors: Annotated[List[str], operator.add]
    """错误列表；各节点仅返回新增项，由 LangGraph reducer 累加。"""
    
    warnings: Annotated[List[Dict[str, Any]], operator.add]
    """警告列表；各节点仅返回新增项，由 LangGraph reducer 累加。"""

    execution_events: Annotated[List[Dict[str, Any]], operator.add]
    """结构化执行事件；各节点仅返回新增项，由 LangGraph reducer 累加。"""

    degraded: Annotated[bool, operator.or_]
    """任一 Agent 进入 fallback 时为 True。"""

    generation_modes: Annotated[Dict[str, str], operator.or_]
    """使用 LLM 的 Agent 对应 generation mode。"""
