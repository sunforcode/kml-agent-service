"""
内容生成Agent (Content Agent)

职责:
1. 基于轨迹数据生成路线描述
2. 提取路线亮点和难点
3. 生成安全提示
4. 推荐适合的装备
5. 规划多日行程

输入:
- basic_stats: 基础统计数据
- segments: 路段列表
- pois: POI列表
- kml_markers: KML标记点
- request.enable_content_generation: 是否启用

输出:
- generated_content: 生成的内容
"""

import logging
from typing import Dict, Any, List, Optional
from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class ContentAgent(BaseAgent):
    """
    内容生成Agent
    
    负责使用LLM生成自然语言内容。
    
    工作流程:
    1. 检查是否启用内容生成（request.enable_content_generation）
    2. 构建输入上下文（路段信息、POI分布、关键节点）
    3. 调用LLM生成各部分内容:
       - 路线描述（100-200字）
       - 路线亮点（3-5条）
       - 难点提示（2-3条）
       - 安全提示
       - 装备推荐
       - 多日行程规划（如适用）
    4. 进行内容质量自检
    5. 如LLM调用失败，使用模板降级内容
    """
    
    def __init__(self):
        super().__init__(
            name="content_agent",
            description="使用LLM生成自然语言内容"
        )
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行内容生成
        
        【当前实现】返回假数据，等待后续完善
        
        Args:
            state: 当前工作流状态
            
        Returns:
            包含生成内容的状态更新
        """
        self.log_start()
        
        # 检查是否启用内容生成
        request = state.get("request", {})
        enable_content = request.get("enable_content_generation", True)
        
        if not enable_content:
            logger.info("内容生成功能被禁用，跳过")
            return {
                "generated_content": None,
                "current_step": "content_generation",
                "overall_progress": 65
            }
        
        # TODO: 实现真实的内容生成逻辑
        # 待实现功能:
        # 1. 检查request["enable_content_generation"]
        # 2. 构建Prompt上下文:
        #    - 路段信息汇总
        #    - POI分布统计
        #    - 关键节点提取
        #    - 用户备注整合
        # 3. 调用OpenAI API:
        #    - 路线描述生成
        #    - 亮点难点提取
        #    - 装备推荐
        #    - 多日行程规划
        # 4. 解析LLM响应（JSON格式）
        # 5. 内容质量自检:
        #    - 描述长度检查
        #    - 亮点数量检查
        #    - 装备推荐数量检查
        # 6. 降级策略:
        #    - LLM调用失败时使用模板内容
        #    - 部分失败时跳过该部分
        
        # ========== 临时实现：返回假数据 ==========
        fake_content = self._create_fake_content()
        
        result = {
            "generated_content": fake_content,
            "current_step": "content_generation",
            "overall_progress": 65  # 内容生成完成，进度65%
        }
        
        self.log_complete()
        return result
    
    def _create_fake_content(self) -> Dict[str, Any]:
        """创建假内容数据"""
        return {
            "description": "五台山顺时针大朝台，全程约52公里，累计爬升约2800米。这是一条经典的徒步朝圣路线，途经东台、北台、中台、西台、南台五座山峰，最高海拔3061米。路线整体难度中等偏高，适合有一定徒步经验的爱好者。",
            
            "highlights": [
                "东台顶观日出，视野开阔，云海壮丽",
                "穿越五台核心景区，体验深厚的佛教文化",
                "护银沟段风景秀丽，溪水潺潺，夏季清凉",
                "北台顶为五台山最高峰，海拔3061米",
                "沿途有多座古寺，历史文化底蕴深厚"
            ],
            
            "difficulties": [
                "西台-中台段风大，注意保暖和防风",
                "部分路段无手机信号，需提前下载离线地图",
                "多日行程需合理分配体力，避免第一天过度疲劳"
            ],
            
            "safety_notes": [
                "山区天气多变，注意防雨保暖",
                "多日行程需携带足够水和食物补给",
                "高海拔地区注意防晒和高反预防",
                "建议结伴而行，避免单独行动"
            ],
            
            "equipment_recommendations": [
                "登山杖（必备，节省体力，保护膝盖）",
                "防风外套（高海拔风大）",
                "头灯（可能走夜路或早出发看日出）",
                "20-30L背包",
                "速干衣裤",
                "中帮徒步鞋（防水更佳）",
                "防晒霜、帽子、墨镜",
                "保温杯（喝热水更舒适）",
                "简单急救包"
            ],
            
            "essential_gear": [
                "登山杖",
                "防风外套",
                "徒步鞋",
                "头灯"
            ],
            
            "optional_gear": [
                "护膝",
                "雪套（冬季）",
                "登山表",
                "运动相机"
            ],
            
            "daily_plans": [
                {
                    "day_number": 1,
                    "title": "第一天：鸿门岩-东台-护银沟",
                    "description": "从鸿门岩出发，先登东台看日出，然后经北台下撤到护银沟住宿。",
                    "distance_km": 15.2,
                    "elevation_gain_m": 800,
                    "elevation_loss_m": 1000,
                    "estimated_hours": 6.5,
                    "suggested_start_time": "05:00",
                    "key_points": ["鸿门岩", "东台顶", "护银沟"],
                    "notes": "护银沟有民宿和补给，建议在此住宿",
                    "campsite_suggestion": "护银沟村民宿"
                },
                {
                    "day_number": 2,
                    "title": "第二天：护银沟-狮子窝-金阁寺",
                    "description": "从护银沟出发，经狮子窝到金阁寺，路程较长但相对平缓。",
                    "distance_km": 20.0,
                    "elevation_gain_m": 1000,
                    "elevation_loss_m": 800,
                    "estimated_hours": 8.0,
                    "suggested_start_time": "07:00",
                    "key_points": ["狮子窝", "金阁寺"],
                    "notes": "金阁寺附近有住宿和补给",
                    "campsite_suggestion": "金阁寺附近"
                },
                {
                    "day_number": 3,
                    "title": "第三天：金阁寺-南台-佛母洞",
                    "description": "最后一天，登南台后下撤到佛母洞，完成朝台。",
                    "distance_km": 17.1,
                    "elevation_gain_m": 1000,
                    "elevation_loss_m": 700,
                    "estimated_hours": 7.0,
                    "suggested_start_time": "07:00",
                    "key_points": ["南台顶", "佛母洞"],
                    "notes": "佛母洞有班车回台怀镇",
                    "campsite_suggestion": "无（当日完成）"
                }
            ],
            
            "generation_metadata": {
                "model_used": "gpt-4o-mini",
                "prompt_tokens": 1500,
                "completion_tokens": 800,
                "total_tokens": 2300,
                "generation_time_ms": 4500,
                "temperature_used": 0.7,
                "quality_score": 85.0,
                "needs_human_review": False
            }
        }
