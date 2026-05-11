"""
POI识别Agent (POI Agent)

职责:
1. 解析KML标记点并分类为统一 poi_points 格式
2. （Phase 2 TODO）对关键节点进行逆地理编码（Nominatim）
3. （Phase 3 TODO）查询OpenStreetMap的POI数据（Overpass API）
4. （未来）合并去重POI，计算置信度

Phase 1 实现范围：
- 读取 state["kml_markers"]
- 基于名称关键词推断 category
- source 固定为 kml_marker，confidence 固定为 1.0
- 无 kml_markers 时返回空列表（不再生成 mock 数据）

输入:
- kml_markers: KML标记点列表

输出:
- poi_points: 统一附属信息点列表
"""

import logging
from typing import Dict, Any, List, Optional
from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


# KML 标记点名称关键词 → category 映射表
# 按优先级从高到低排列，先匹配到先命中
_CATEGORY_KEYWORDS = [
    ("start",   ["起点", "出发", "trailhead", "入口", "登山口"]),
    ("end",     ["终点", "终止", "到达", "结束"]),
    ("pass",    ["垭口", "哑口", "丫口", "山口", "pass", "col"]),
    ("water",   ["水源", "水", "泉", "溪", "河", "spring", "water", "creek"]),
    ("camp",    ["营地", "扎营", "露营", "camp", "campsite", "帐篷"]),
    ("supply",  ["补给", "村", "镇", "寺", "庙", "temple", "village", "town", "shop"]),
    ("danger",  ["危险", "陡坡", "滑落", "danger", "hazard", "cliff"]),
    ("photo",   ["拍照", "打卡", "观景", "景点", "峰顶", "顶", "peak", "viewpoint", "summit"]),
]

# category 对应的默认 sub_category 推断关键词
_SUBCATEGORY_KEYWORDS = {
    "water": [
        ("spring",  ["泉", "spring"]),
        ("river",   ["河", "溪", "creek", "river"]),
        ("tap",     ["自来水", "tap"]),
    ],
    "supply": [
        ("village", ["村", "镇", "town", "village"]),
        ("temple",  ["寺", "庙", "temple", "shrine"]),
        ("shop",    ["商店", "超市", "shop", "store"]),
        ("restaurant", ["餐", "食堂", "restaurant"]),
    ],
    "photo": [
        ("peak",    ["顶", "峰", "peak", "summit", "顶点"]),
        ("viewpoint", ["观景", "viewpoint", "观"]),
    ],
}


def _infer_category(name: str) -> str:
    """从名称关键词推断 POI category"""
    name_lower = name.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(kw in name_lower for kw in keywords):
            return category
    # 默认归为 photo（观景/打卡点）
    return "photo"


def _infer_sub_category(name: str, category: str) -> Optional[str]:
    """从名称关键词推断 POI sub_category"""
    if category not in _SUBCATEGORY_KEYWORDS:
        return None
    name_lower = name.lower()
    for sub_cat, keywords in _SUBCATEGORY_KEYWORDS[category]:
        if any(kw in name_lower for kw in keywords):
            return sub_cat
    return None


class POIAgent(BaseAgent):
    """
    POI识别Agent（Phase 1：真实实现，基于 KML 标记点）

    Phase 1 工作流:
    1. 读取 state["kml_markers"]
    2. 遍历每个 marker，推断 category 和 sub_category
    3. 组装为统一 poi_points 格式（source=kml_marker, confidence=1.0）
    4. 无 kml_markers 时返回空列表

    后续 Phase:
    - Phase 2: 基于轨迹起终点 + Nominatim 逆地理编码补充 pass/valley 类型 POI
    - Phase 3: OSM Overpass API 查询沿途水源/营地/补给点
    - Phase 4: Open-Meteo 气象点集成
    """
    
    def __init__(self):
        super().__init__(
            name="poi_agent",
            description="识别轨迹沿途的兴趣点（POI）"
        )
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行POI识别（Phase 1：真实实现）
        
        Args:
            state: 当前工作流状态
            
        Returns:
            包含 poi_points 的状态更新
        """
        self.log_start()
        
        kml_markers = state.get("kml_markers", [])
        
        if not kml_markers:
            logger.info("POIAgent: 无 KML 标记点，返回空 poi_points")
            self.log_complete(poi_count=0)
            return {
                "poi_points": [],
                "current_step": "poi_recognition",
                "overall_progress": 45
            }
        
        poi_points = self._kml_markers_to_poi_points(kml_markers)
        
        self.log_complete(poi_count=len(poi_points))
        logger.info(f"POIAgent: 从 {len(kml_markers)} 个 KML 标记点生成 {len(poi_points)} 个 poi_points")
        
        return {
            "poi_points": poi_points,
            "current_step": "poi_recognition",
            "overall_progress": 45
        }
    
    def _kml_markers_to_poi_points(self, kml_markers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将 KML 标记点列表转换为统一 poi_points 格式

        Args:
            kml_markers: KML 标记点列表

        Returns:
            poi_points 列表
        """
        poi_points = []
        for marker in kml_markers:
            poi = self._kml_marker_to_poi_point(marker)
            if poi:
                poi_points.append(poi)
        return poi_points
    
    def _kml_marker_to_poi_point(self, marker: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将单个 KML 标记点转换为 poi_point 格式

        category 基于名称关键词映射，source 固定为 kml_marker，confidence 固定为 1.0。

        Args:
            marker: KML 标记点字典

        Returns:
            poi_point 字典，若坐标无效则返回 None
        """
        lat = marker.get("latitude")
        lon = marker.get("longitude")
        
        if lat is None or lon is None:
            logger.warning(f"POIAgent: 跳过无效坐标的 KML 标记点: {marker.get('name', '?')}")
            return None
        
        name = marker.get("name") or "未命名标记点"
        category = _infer_category(name)
        sub_category = _infer_sub_category(name, category)
        
        # 构建 card_data（各 category 的扩展属性）
        card_data = self._build_card_data(marker, category)
        
        return {
            "name": name,
            "latitude": lat,
            "longitude": lon,
            "elevation": marker.get("elevation"),
            "category": category,
            "sub_category": sub_category,
            "source": "kml_marker",
            "description": marker.get("description"),
            "confidence": 1.0,
            "card_data": card_data if card_data else None
        }
    
    def _build_card_data(self, marker: Dict[str, Any], category: str) -> Optional[Dict[str, Any]]:
        """
        根据 category 构建 card_data 扩展属性

        Args:
            marker: KML 标记点字典
            category: 已推断的 category

        Returns:
            card_data 字典，无扩展数据则返回 None
        """
        card: Dict[str, Any] = {}

        # 所有 category 通用字段
        if marker.get("icon_url"):
            card["icon_url"] = marker["icon_url"]
        if marker.get("image_url"):
            card["image_url"] = marker["image_url"]
        if marker.get("style_url"):
            card["style_url"] = marker["style_url"]

        # 仅当有额外字段时才包含 card_data
        return card if card else None
