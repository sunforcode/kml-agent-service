"""
KML Agent Service - Agent基类和工具函数

所有Agent的公共基类，提供：
- 状态管理
- 错误处理
- 日志记录
- 进度更新
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Agent基类
    
    所有具体Agent都应继承此类，提供统一的接口和工具方法。
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        初始化Agent
        
        Args:
            name: Agent名称
            description: Agent描述
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"agent.{name}")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行Agent逻辑
        
        子类必须重写此方法实现具体逻辑。
        
        Args:
            state: 当前工作流状态
            
        Returns:
            更新后的状态部分
        """
        raise NotImplementedError("子类必须实现execute方法")
    
    def log_start(self, **kwargs):
        """记录开始日志"""
        self.logger.info(
            f"Agent [{self.name}] 开始执行",
            extra={"event": f"{self.name}_start", **kwargs}
        )
    
    def log_complete(self, **kwargs):
        """记录完成日志"""
        self.logger.info(
            f"Agent [{self.name}] 执行完成",
            extra={"event": f"{self.name}_complete", **kwargs}
        )
    
    def log_error(self, error: Exception, **kwargs):
        """记录错误日志"""
        self.logger.error(
            f"Agent [{self.name}] 执行失败: {str(error)}",
            extra={"event": f"{self.name}_error", "error": str(error), **kwargs},
            exc_info=True
        )
    
    def update_progress(
        self,
        state: Dict[str, Any],
        step_name: str,
        progress: int
    ) -> Dict[str, Any]:
        """
        更新进度
        
        Args:
            state: 当前状态
            step_name: 步骤名称
            progress: 进度百分比（0-100）
            
        Returns:
            状态更新部分
        """
        return {
            "current_step": step_name,
            "overall_progress": min(100, max(0, progress))
        }
    
    def add_warning(
        self,
        message: str,
        level: str = "warning",
        **kwargs
    ) -> Dict[str, Any]:
        """
        添加警告
        
        Args:
            message: 警告消息
            level: 级别（info/warning/error）
            
        Returns:
            状态更新部分（用于LangGraph的累加）
        """
        warning = {
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        return {"warnings": [warning]}
    
    def add_error(self, message: str) -> Dict[str, Any]:
        """
        添加错误
        
        Args:
            message: 错误消息
            
        Returns:
            状态更新部分
        """
        return {"errors": [message]}


# ========================================
# 工具函数
# ========================================

def create_initial_state(request_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建初始状态
    
    Args:
        request_dict: 分析请求的字典形式
        
    Returns:
        初始工作流状态
    """
    return {
        # 输入
        "request": request_dict,
        
        # 轨迹分析结果
        "track_points": [],
        "kml_markers": [],
        "basic_stats": None,
        
        # 分段结果
        "segment_schemes": [],
        
        # POI结果
        "poi_points": [],
        
        # 内容生成结果
        "generated_content": None,
        
        # 质量评估结果
        "quality_assessment": None,
        
        # 最终结果
        "final_result": None,
        
        # 进度
        "current_step": "init",
        "overall_progress": 0,
        
        # 错误、警告与可观测性状态
        "errors": [],
        "warnings": [],
        "execution_events": [],
        "degraded": False,
        "generation_modes": {},
    }


def is_state_successful(state: Dict[str, Any]) -> bool:
    """
    检查状态是否成功
    
    Args:
        state: 工作流状态
        
    Returns:
        是否成功（没有致命错误）
    """
    errors = state.get("errors", [])
    
    # 检查是否有致命错误
    fatal_errors = [
        e for e in errors
        if "KML_PARSE_ERROR" in str(e) or "FATAL" in str(e).upper()
    ]
    
    return len(fatal_errors) == 0


def get_quality_score(state: Dict[str, Any]) -> float:
    """
    从状态中获取质量评分
    
    Args:
        state: 工作流状态
        
    Returns:
        质量评分（0-100），如果没有则返回默认值
    """
    assessment = state.get("quality_assessment")
    if assessment:
        return assessment.get("overall_score", 70.0)
    return 70.0
