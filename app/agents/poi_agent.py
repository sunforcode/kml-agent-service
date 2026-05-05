"""
POI识别Agent (POI Agent)

职责:
1. 解析KML标记点并分类
2. 对关键节点进行逆地理编码（Nominatim）
3. 查询OpenStreetMap的POI数据（Overpass API）
4. 合并去重POI
5. 计算POI置信度

输入:
- kml_markers: KML标记点列表
- track_points: 轨迹点列表（关键节点）
- segments: 路段列表

输出:
- pois: POI列表
"""

import logging
from typing import Dict, Any, List
from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class POIAgent(BaseAgent):
    """
    POI识别Agent
    
    负责识别轨迹沿途的兴趣点。
    
    工作流程:
    1. 处理KML标记点，推断类型和置信度
    2. 选择关键节点（起点、终点、路段端点、海拔极值）
    3. 对关键节点进行逆地理编码（Nominatim）
    4. 查询轨迹附近的OSM POI（Overpass API）
    5. 合并KML标记点和OSM查询结果
    6. 按距离去重
    7. 计算置信度评分
    """
    
    def __init__(self):
        super().__init__(
            name="poi_agent",
            description="识别轨迹沿途的兴趣点（POI）"
        )
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行POI识别
        
        【当前实现】返回假数据，等待后续完善
        
        Args:
            state: 当前工作流状态
            
        Returns:
            包含POI结果的状态更新
        """
        self.log_start()
        
        # TODO: 实现真实的POI识别逻辑
        # 待实现功能:
        # 1. 处理KML标记点（state["kml_markers"]）
        # 2. 关键词匹配推断类型
        # 3. 选择关键节点（起点、终点、路段端点、海拔极值）
        # 4. 调用Nominatim逆地理编码
        # 5. 调用Overpass API查询OSM POI
        #    - 水源: amenity=drinking_water, natural=spring
        #    - 营地: tourism=camp_site, amenity=shelter
        #    - 补给点: shop=convenience, amenity=restaurant
        #    - 观景点: tourism=viewpoint, natural=peak
        # 6. 合并所有POI来源
        # 7. 按距离去重（50米内认为是同一POI）
        # 8. 计算置信度评分
        
        # ========== 临时实现：返回假数据 ==========
        fake_pois = self._create_fake_pois()
        
        result = {
            "pois": fake_pois,
            "current_step": "poi_recognition",
            "overall_progress": 45  # POI识别完成，进度45%
        }
        
        self.log_complete(pois_count=len(fake_pois))
        return result
    
    def _create_fake_pois(self) -> List[Dict[str, Any]]:
        """创建假POI数据"""
        return [
            # 来自KML标记点的POI
            {
                "source": "kml_marker",
                "name": "鸿门岩",
                "latitude": 39.0556,
                "longitude": 113.6565,
                "elevation": 2526.5,
                "category": "start",
                "sub_category": "trailhead",
                "confidence": 1.0,
                "description": "五台山朝台起点",
                "distance_to_track_meters": 0
            },
            {
                "source": "kml_marker",
                "name": "东台顶",
                "latitude": 39.0428,
                "longitude": 113.6604,
                "elevation": 2772.9,
                "category": "viewpoint",
                "sub_category": "peak",
                "confidence": 1.0,
                "description": "东台望海峰，海拔2796米",
                "distance_to_track_meters": 0
            },
            
            # 来自OSM查询的POI
            {
                "source": "osm_query",
                "name": "护银沟村",
                "latitude": 38.9642,
                "longitude": 113.6060,
                "elevation": 1830.0,
                "category": "supply",
                "sub_category": "village",
                "confidence": 0.8,
                "description": "可提供补给和住宿",
                "distance_to_track_meters": 150,
                "osm_type": "node",
                "osm_id": 123456
            },
            {
                "source": "osm_query",
                "name": "龙泉寺",
                "latitude": 38.98,
                "longitude": 113.62,
                "elevation": 1700.0,
                "category": "viewpoint",
                "sub_category": "temple",
                "confidence": 0.75,
                "description": "五台山著名寺庙",
                "distance_to_track_meters": 300,
                "osm_type": "node",
                "osm_id": 789012
            },
            
            # 来自逆地理编码的POI
            {
                "source": "nominatim",
                "name": "北台顶",
                "latitude": 39.0667,
                "longitude": 113.6933,
                "elevation": 3061.0,
                "category": "viewpoint",
                "sub_category": "peak",
                "confidence": 0.9,
                "description": "五台山最高峰，海拔3061米",
                "distance_to_track_meters": 50
            }
        ]
