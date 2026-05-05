"""
编排Agent (Orchestrator Agent)

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
- recognize_poi: POI识别
- generate_content: 内容生成（可选）
- assess_quality: 质量评估
- aggregate_result: 汇总结果

工作流边:
- init -> analyze_trajectory
- analyze_trajectory -> segment_route（成功）或 end（失败）
- segment_route -> recognize_poi
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

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    编排Agent
    
    负责定义和执行整个分析工作流。
    
    工作流图:
    init 
      -> analyze_trajectory 
        -> (成功) segment_route 
          -> recognize_poi 
            -> (启用内容生成) generate_content -> assess_quality
            -> (跳过内容生成) assess_quality
          -> aggregate_result
            -> END
        -> (失败) END
    """
    
    def __init__(self):
        super().__init__(
            name="orchestrator_agent",
            description="协调各Agent执行工作流"
        )
        
        # 初始化各Agent
        self.trajectory_agent = TrajectoryAgent()
        self.segmentation_agent = SegmentationAgent()
        self.poi_agent = POIAgent()
        self.content_agent = ContentAgent()
        self.quality_agent = QualityAgent()
        
        # 构建工作流
        self.workflow = self._build_workflow()
        self.compiled_workflow = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """
        构建LangGraph工作流
        
        Returns:
            StateGraph: 工作流图
        """
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("init", self._init_node)
        workflow.add_node("analyze_trajectory", self._analyze_trajectory_node)
        workflow.add_node("segment_route", self._segment_route_node)
        workflow.add_node("recognize_poi", self._recognize_poi_node)
        workflow.add_node("generate_content", self._generate_content_node)
        workflow.add_node("assess_quality", self._assess_quality_node)
        workflow.add_node("aggregate_result", self._aggregate_result_node)
        
        # 设置入口点
        workflow.set_entry_point("init")
        
        # 添加边
        workflow.add_edge("init", "analyze_trajectory")
        
        # 条件边：轨迹分析失败则结束
        workflow.add_conditional_edges(
            "analyze_trajectory",
            self._route_after_trajectory,
            {
                "continue": "segment_route",
                "fail": END
            }
        )
        
        workflow.add_edge("segment_route", "recognize_poi")
        
        # 条件边：判断是否需要内容生成
        workflow.add_conditional_edges(
            "recognize_poi",
            self._should_generate_content,
            {
                "generate": "generate_content",
                "skip": "assess_quality"
            }
        )
        
        workflow.add_edge("generate_content", "assess_quality")
        workflow.add_edge("assess_quality", "aggregate_result")
        workflow.add_edge("aggregate_result", END)
        
        return workflow
    
    # ========================================
    # 工作流节点实现
    # ========================================
    
    async def _init_node(self, state: AgentState) -> AgentState:
        """
        初始化节点
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        self.log_start()
        logger.info("初始化工作流状态")
        
        # 初始化关键字段（如果不存在）
        result = {
            "track_points": state.get("track_points", []),
            "kml_markers": state.get("kml_markers", []),
            "segments": state.get("segments", []),
            "pois": state.get("pois", []),
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
            "current_step": "init",
            "overall_progress": 5  # 初始化完成，进度5%
        }
        
        return result
    
    async def _analyze_trajectory_node(self, state: AgentState) -> AgentState:
        """
        轨迹分析节点
        
        调用轨迹分析Agent解析KML并计算统计数据。
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        logger.info("执行轨迹分析节点")
        
        try:
            result = await self.trajectory_agent.execute(state)
            return result
        except Exception as e:
            logger.error(f"轨迹分析失败: {str(e)}")
            return {
                "errors": [f"TRAJ_ANALYSIS_ERROR: {str(e)}"],
                "current_step": "trajectory_analysis_failed",
                "overall_progress": state.get("overall_progress", 0)
            }
    
    async def _segment_route_node(self, state: AgentState) -> AgentState:
        """
        路径分段节点
        
        调用路径分段Agent进行智能分段。
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        logger.info("执行路径分段节点")
        
        try:
            result = await self.segmentation_agent.execute(state)
            return result
        except Exception as e:
            logger.error(f"路径分段失败: {str(e)}")
            # 分段失败不阻断流程，记录警告并使用空分段
            return {
                "segments": [],
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
    
    async def _recognize_poi_node(self, state: AgentState) -> AgentState:
        """
        POI识别节点
        
        调用POI识别Agent识别兴趣点。
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        logger.info("执行POI识别节点")
        
        try:
            result = await self.poi_agent.execute(state)
            return result
        except Exception as e:
            logger.error(f"POI识别失败: {str(e)}")
            # POI识别失败不阻断流程
            return {
                "pois": [],
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
        """
        内容生成节点
        
        调用内容生成Agent生成自然语言内容。
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        logger.info("执行内容生成节点")
        
        try:
            result = await self.content_agent.execute(state)
            return result
        except Exception as e:
            logger.error(f"内容生成失败: {str(e)}")
            # 内容生成失败不阻断流程
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
                "overall_progress": 60  # 即使失败也推进进度
            }
    
    async def _assess_quality_node(self, state: AgentState) -> AgentState:
        """
        质量评估节点
        
        调用质量评估Agent评估分析结果质量。
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        logger.info("执行质量评估节点")
        
        try:
            result = await self.quality_agent.execute(state)
            return result
        except Exception as e:
            logger.error(f"质量评估失败: {str(e)}")
            # 质量评估失败不阻断流程，使用默认评分
            return {
                "quality_assessment": {
                    "overall_score": 70.0,
                    "warnings": [{"level": "warning", "message": f"质量评估失败: {str(e)}"}]
                },
                "current_step": "quality_assessment_failed",
                "overall_progress": state.get("overall_progress", 0)
            }
    
    async def _aggregate_result_node(self, state: AgentState) -> AgentState:
        """
        结果汇总节点
        
        汇总所有Agent的输出，生成最终结果。
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        logger.info("执行结果汇总节点")
        
        # 构建最终结果
        final_result = self._build_final_result(state)
        
        result = {
            "final_result": final_result,
            "current_step": "aggregate_result",
            "overall_progress": 100  # 完成
        }
        
        self.log_complete()
        return result
    
    # ========================================
    # 条件路由函数
    # ========================================
    
    def _route_after_trajectory(self, state: AgentState) -> str:
        """
        轨迹分析后的路由判断
        
        Args:
            state: 当前状态
            
        Returns:
            "continue" 或 "fail"
        """
        errors = state.get("errors", [])
        
        # 检查是否有致命错误
        fatal_errors = [
            e for e in errors
            if "KML_PARSE_ERROR" in str(e) or 
               "TRAJ_ANALYSIS_ERROR" in str(e) or
               "FATAL" in str(e).upper()
        ]
        
        # 检查是否有轨迹点
        track_points = state.get("track_points", [])
        if not track_points and not fatal_errors:
            # 如果没有轨迹点但也没有错误，可能是空轨迹
            logger.warning("轨迹分析完成但无轨迹点")
        
        if fatal_errors:
            logger.error(f"轨迹分析失败，错误: {fatal_errors}")
            return "fail"
        
        return "continue"
    
    def _should_generate_content(self, state: AgentState) -> str:
        """
        判断是否需要内容生成
        
        Args:
            state: 当前状态
            
        Returns:
            "generate" 或 "skip"
        """
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
        
        将状态中各部分数据汇总为EnhancedRouteOutput格式。
        
        Args:
            state: 当前状态
            
        Returns:
            最终结果字典
        """
        from datetime import datetime
        
        basic_stats = state.get("basic_stats", {})
        segments = state.get("segments", [])
        pois = state.get("pois", [])
        generated_content = state.get("generated_content") or {}
        quality_assessment = state.get("quality_assessment") or {}
        warnings = state.get("warnings", [])
        errors = state.get("errors", [])
        
        # 分类POI
        water_sources = [p for p in pois if p.get("category") == "water"]
        campsites = [p for p in pois if p.get("category") == "camp"]
        supplies = [p for p in pois if p.get("category") == "supply"]
        marker_points = [p for p in pois if p.get("category") in ["viewpoint", "start", "end", "danger"]]
        
        # 构建最终结果
        result = {
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
            "estimated_difficulty": basic_stats.get("data_quality_score", 70) // 20,  # 简单映射
            
            # 路段
            "segments": [
                {
                    "name": s.get("suggested_name", f"路段{i+1}"),
                    "description": None,
                    "distance_km": s.get("distance_km", 0),
                    "elevation_gain_m": s.get("elevation_gain_m", 0),
                    "elevation_loss_m": s.get("elevation_loss_m", 0),
                    "estimated_time_minutes": s.get("estimated_time_minutes", 0),
                    "difficulty": s.get("suggested_difficulty", 2),
                    "start_lat": s.get("start_lat", 0),
                    "start_lon": s.get("start_lon", 0),
                    "end_lat": s.get("end_lat", 0),
                    "end_lon": s.get("end_lon", 0),
                    "notes": None
                }
                for i, s in enumerate(segments)
            ],
            
            # POI
            "water_sources": [
                {
                    "name": p.get("name", "未知水源"),
                    "latitude": p.get("latitude", 0),
                    "longitude": p.get("longitude", 0),
                    "description": p.get("description"),
                    "source_type": p.get("sub_category", "unknown"),
                    "reliability": p.get("confidence", 0.5),
                    "notes": None
                }
                for p in water_sources
            ],
            
            "campsites": [
                {
                    "name": p.get("name", "未知营地"),
                    "latitude": p.get("latitude", 0),
                    "longitude": p.get("longitude", 0),
                    "description": p.get("description"),
                    "capacity": None,
                    "has_water": None,
                    "has_facilities": None,
                    "notes": None
                }
                for p in campsites
            ],
            
            "supplies": [
                {
                    "name": p.get("name", "未知补给点"),
                    "latitude": p.get("latitude", 0),
                    "longitude": p.get("longitude", 0),
                    "description": p.get("description"),
                    "supply_type": p.get("sub_category", "unknown"),
                    "notes": None
                }
                for p in supplies
            ],
            
            "marker_points": [
                {
                    "name": p.get("name", "未知点"),
                    "latitude": p.get("latitude", 0),
                    "longitude": p.get("longitude", 0),
                    "elevation": p.get("elevation"),
                    "type": p.get("category", "viewpoint"),
                    "description": p.get("description"),
                    "image_url": p.get("image_url"),
                    "icon_url": None
                }
                for p in marker_points
            ],
            
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
                "segments_count": len(segments),
                "pois_count": len(pois),
                "steps_completed": state.get("current_step", "unknown")
            }
        }
        
        return result
    
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
        
        # 创建初始状态
        initial_state = create_initial_state(request_dict)
        
        # 执行工作流
        logger.info("开始执行工作流")
        final_state = await self.compiled_workflow.ainvoke(initial_state)
        
        # 获取结果
        result = final_state.get("final_result")
        
        self.log_complete()
        return result or {}
