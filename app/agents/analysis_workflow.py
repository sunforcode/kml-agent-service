"""
KML 分析工作流 (Analysis Workflow)

职责:
1. 定义和执行LangGraph工作流
2. 管理状态流转
3. 协调各Agent执行顺序
4. 处理错误和重试
5. 汇总最终结果

工作流节点:
- init: 初始化状态
- analyze_trajectory: 轨迹分析
- segment_route: 路径分段
- merge_segments: AI 分段合并评估（去除噪点段）
- recognize_poi: POI识别
- generate_content: 内容生成（可选）
- assess_quality: 质量评估
- aggregate_result: 汇总结果

工作流边:
- init -> analyze_trajectory
- analyze_trajectory -> segment_route（成功）或 end（失败）
- segment_route -> merge_segments
- merge_segments -> recognize_poi
- recognize_poi -> 条件判断（是否生成内容）
- generate_content -> assess_quality
- assess_quality -> aggregate_result
- aggregate_result -> end
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END

from app.models.state import AgentState
from app.agents.base import BaseAgent, create_initial_state
from app.agents.trajectory_agent import TrajectoryAgent
from app.agents.segmentation_agent import SegmentationAgent
from app.agents.poi_agent import POIAgent
from app.agents.content_agent import ContentAgent
from app.agents.quality_agent import QualityAgent
from app.agents.segment_merge_agent import SegmentMergeAgent

logger = logging.getLogger(__name__)


class AnalysisWorkflow(BaseAgent):
    """
    KML 分析工作流

    负责定义和执行整个分析工作流。

    工作流图:
    init
      -> analyze_trajectory
        -> (成功) segment_route
          -> merge_segments
            -> recognize_poi
              -> (启用内容生成) generate_content -> assess_quality
              -> (跳过内容生成) assess_quality
            -> aggregate_result
              -> END
        -> (失败) END
    """
    
    def __init__(self):
        super().__init__(
            name="analysis_workflow",
            description="协调各Agent执行KML分析工作流"
        )
        
        # 初始化各Agent
        self.trajectory_agent = TrajectoryAgent()
        self.segmentation_agent = SegmentationAgent()
        self.segment_merge_agent = SegmentMergeAgent()
        self.poi_agent = POIAgent()
        self.content_agent = ContentAgent()
        self.quality_agent = QualityAgent()
        
        # 构建工作流
        self._graph = self._build_graph()
        self._compiled = self._graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """
        构建LangGraph状态图

        Returns:
            StateGraph: 工作流图
        """
        graph = StateGraph(AgentState)
        
        # 添加节点
        graph.add_node("init", self._init_node)
        graph.add_node("analyze_trajectory", self._analyze_trajectory_node)
        graph.add_node("segment_route", self._segment_route_node)
        graph.add_node("merge_segments", self._merge_segments_node)
        graph.add_node("recognize_poi", self._recognize_poi_node)
        graph.add_node("generate_content", self._generate_content_node)
        graph.add_node("assess_quality", self._assess_quality_node)
        graph.add_node("aggregate_result", self._aggregate_result_node)
        
        # 设置入口点
        graph.set_entry_point("init")
        
        # 添加边
        graph.add_edge("init", "analyze_trajectory")
        
        # 条件边：轨迹分析失败则结束
        graph.add_conditional_edges(
            "analyze_trajectory",
            self._route_after_trajectory,
            {
                "continue": "segment_route",
                "fail": END
            }
        )
        
        graph.add_edge("segment_route", "merge_segments")
        graph.add_edge("merge_segments", "recognize_poi")
        
        # 条件边：判断是否需要内容生成
        graph.add_conditional_edges(
            "recognize_poi",
            self._should_generate_content,
            {
                "generate": "generate_content",
                "skip": "assess_quality"
            }
        )
        
        graph.add_edge("generate_content", "assess_quality")
        graph.add_edge("assess_quality", "aggregate_result")
        graph.add_edge("aggregate_result", END)
        
        return graph
    
    # ========================================
    # 工作流节点实现
    # ========================================
    
    async def _init_node(self, state: AgentState) -> AgentState:
        """初始化节点：初始化关键字段"""
        self.log_start()
        logger.info("初始化工作流状态")
        
        return {
            "track_points": state.get("track_points", []),
            "kml_markers": state.get("kml_markers", []),
            "segment_schemes": state.get("segment_schemes", []),
            "poi_points": state.get("poi_points", []),
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
            "current_step": "init",
            "overall_progress": 5
        }
    
    async def _analyze_trajectory_node(self, state: AgentState) -> AgentState:
        """轨迹分析节点：调用 TrajectoryAgent 解析KML并计算统计数据"""
        logger.info("执行轨迹分析节点")
        
        try:
            return await self.trajectory_agent.execute(state)
        except Exception as e:
            logger.error(f"轨迹分析失败: {str(e)}")
            return {
                "errors": [f"TRAJ_ANALYSIS_ERROR: {str(e)}"],
                "current_step": "trajectory_analysis_failed",
                "overall_progress": state.get("overall_progress", 0)
            }
    
    async def _segment_route_node(self, state: AgentState) -> AgentState:
        """路径分段节点：调用 SegmentationAgent 进行智能分段（基于坡度算法）"""
        logger.info("执行路径分段节点")
        
        try:
            return await self.segmentation_agent.execute(state)
        except Exception as e:
            logger.error(f"路径分段失败: {str(e)}")
            empty_schemes = [
                {"scheme_type": "slope", "label": "按坡度", "is_default": True, "segments": []},
                {"scheme_type": "day", "label": "按天", "is_default": False, "segments": []},
                {"scheme_type": "terrain", "label": "按地形", "is_default": False, "segments": []},
                {"scheme_type": "road_type", "label": "按路况", "is_default": False, "segments": []},
            ]
            return {
                "segment_schemes": empty_schemes,
                "warnings": [
                    {
                        "level": "warning",
                        "message": f"路径分段失败: {str(e)}",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                ],
                "current_step": "segmentation_failed",
                "overall_progress": state.get("overall_progress", 0)
            }
    
    async def _merge_segments_node(self, state: AgentState) -> AgentState:
        """分段合并节点：调用 SegmentMergeAgent 去除噪点段，保留有地形意义的分段"""
        logger.info("执行分段合并节点")

        try:
            return await self.segment_merge_agent.execute(state)
        except Exception as e:
            logger.error(f"分段合并失败: {str(e)}")
            return {
                "warnings": [
                    {
                        "level": "warning",
                        "message": f"分段合并失败，保持原始分段: {str(e)}",
                    }
                ],
                "current_step": "merge_segments_failed",
                "overall_progress": state.get("overall_progress", 0),
            }

    async def _recognize_poi_node(self, state: AgentState) -> AgentState:
        """POI识别节点：调用 POIAgent 识别兴趣点"""
        logger.info("执行POI识别节点")
        
        try:
            return await self.poi_agent.execute(state)
        except Exception as e:
            logger.error(f"POI识别失败: {str(e)}")
            return {
                "poi_points": [],
                "warnings": [
                    {
                        "level": "warning",
                        "message": f"POI识别失败: {str(e)}",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                ],
                "current_step": "poi_recognition_failed",
                "overall_progress": state.get("overall_progress", 0)
            }
    
    async def _generate_content_node(self, state: AgentState) -> AgentState:
        """内容生成节点：调用 ContentAgent 生成自然语言内容"""
        logger.info("执行内容生成节点")
        
        try:
            return await self.content_agent.execute(state)
        except Exception as e:
            logger.error(f"内容生成失败: {str(e)}")
            return {
                "generated_content": None,
                "warnings": [
                    {
                        "level": "warning",
                        "message": f"内容生成失败: {str(e)}",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                ],
                "current_step": "content_generation_failed",
                "overall_progress": 60
            }
    
    async def _assess_quality_node(self, state: AgentState) -> AgentState:
        """质量评估节点：调用 QualityAgent 评估分析结果质量"""
        logger.info("执行质量评估节点")
        
        try:
            return await self.quality_agent.execute(state)
        except Exception as e:
            logger.error(f"质量评估失败: {str(e)}")
            return {
                "quality_assessment": {
                    "overall_score": 70.0,
                    "warnings": [{"level": "warning", "message": f"质量评估失败: {str(e)}"}]
                },
                "current_step": "quality_assessment_failed",
                "overall_progress": state.get("overall_progress", 0)
            }
    
    async def _aggregate_result_node(self, state: AgentState) -> AgentState:
        """结果汇总节点：汇总所有Agent输出，生成最终结果"""
        logger.info("执行结果汇总节点")
        
        final_result = self._build_final_result(state)
        self.log_complete()
        return {
            "final_result": final_result,
            "current_step": "aggregate_result",
            "overall_progress": 100
        }
    
    # ========================================
    # 条件路由函数
    # ========================================
    
    def _route_after_trajectory(self, state: AgentState) -> str:
        """轨迹分析后的路由判断：返回 "continue" 或 "fail" """
        errors = state.get("errors", [])
        fatal_errors = [
            e for e in errors
            if "KML_PARSE_ERROR" in str(e) or
               "TRAJ_ANALYSIS_ERROR" in str(e) or
               "FATAL" in str(e).upper()
        ]
        
        track_points = state.get("track_points", [])
        if not track_points and not fatal_errors:
            logger.warning("轨迹分析完成但无轨迹点")
        
        if fatal_errors:
            logger.error(f"轨迹分析失败，错误: {fatal_errors}")
            return "fail"
        
        return "continue"
    
    def _should_generate_content(self, state: AgentState) -> str:
        """判断是否需要内容生成：返回 "generate" 或 "skip" """
        request = state.get("request", {})
        enable_content = request.get("enable_content_generation", True)
        
        if enable_content:
            logger.info("启用内容生成")
            return "generate"
        else:
            logger.info("跳过内容生成（用户禁用）")
            return "skip"
    
    # ========================================
    # 结果构建
    # ========================================
    
    def _build_final_result(self, state: AgentState) -> Dict[str, Any]:
        """
        构建最终结果

        将状态中各部分数据汇总为 callback payload 格式：
        - segment_schemes: 多方案分段
        - poi_points: 统一附属信息点
        """
        basic_stats = state.get("basic_stats", {})
        segment_schemes = state.get("segment_schemes", [])
        poi_points = state.get("poi_points", [])
        generated_content = state.get("generated_content") or {}
        quality_assessment = state.get("quality_assessment") or {}
        warnings = state.get("warnings", [])
        errors = state.get("errors", [])
        
        return {
            # 元数据
            "source_kml_url": state.get("request", {}).get("kml_source", ""),
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "quality_score": quality_assessment.get("overall_score", 70.0),
            
            # 路线统计
            "total_distance_km": basic_stats.get("total_distance_km", 0),
            "total_elevation_gain_m": basic_stats.get("total_gain_m", 0),
            "total_elevation_loss_m": basic_stats.get("total_loss_m", 0),
            "max_elevation": basic_stats.get("max_elevation", 0),
            "min_elevation": basic_stats.get("min_elevation", 0),
            "is_loop": basic_stats.get("is_loop", False),
            "estimated_difficulty": basic_stats.get("data_quality_score", 70) // 20,

            # 多方案分段 + 统一 POI
            "segment_schemes": segment_schemes,
            "poi_points": poi_points,

            # 生成内容
            "generated_description": generated_content.get("description"),
            "generated_highlights": generated_content.get("highlights", []),
            "generated_difficulties": generated_content.get("difficulties", []),
            "generated_safety_notes": generated_content.get("safety_notes", []),
            "equipment_recommendations": generated_content.get("equipment_recommendations", []),
            
            # 警告
            "warnings": [
                {
                    "level": w.get("level", "info"),
                    "message": w.get("message", ""),
                    "location": w.get("location"),
                    "detail": w.get("detail")
                }
                for w in warnings
            ] + [
                {
                    "level": "error",
                    "message": e,
                    "location": None,
                    "detail": None
                }
                for e in errors
            ],
            
            # 原始数据
            "raw_analysis_data": {
                "track_points_count": len(state.get("track_points", [])),
                "segment_schemes_count": len(segment_schemes),
                "poi_points_count": len(poi_points),
                "steps_completed": state.get("current_step", "unknown")
            }
        }
    
    # ========================================
    # 执行入口
    # ========================================
    
    async def execute_workflow(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行完整工作流

        Args:
            request_dict: 分析请求的字典形式

        Returns:
            最终分析结果
        """
        self.log_start()
        
        initial_state = create_initial_state(request_dict)
        
        logger.info("开始执行工作流")
        final_state = await self._compiled.ainvoke(initial_state)
        
        result = final_state.get("final_result")
        
        self.log_complete()
        return result or {}
