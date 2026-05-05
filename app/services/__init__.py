"""
任务管理服务

职责:
1. 生成唯一任务ID
2. 管理任务状态
3. 存储中间结果
4. 提供任务查询接口

当前实现:
- 使用内存存储（MVP阶段）
- 后续可扩展到Redis
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskManager:
    """
    任务管理器
    
    负责管理分析任务的生命周期。
    """
    
    def __init__(self):
        # 内存存储（MVP阶段使用）
        self._tasks: Dict[str, Dict[str, Any]] = {}
    
    def create_task(
        self,
        request: Dict[str, Any],
        estimated_seconds: int = 60
    ) -> str:
        """
        创建新任务
        
        Args:
            request: 分析请求
            estimated_seconds: 预估完成时间（秒）
            
        Returns:
            task_id: 任务ID
        """
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        task_data = {
            "task_id": task_id,
            "request": request,
            "status": TaskStatus.PENDING,
            "progress": 0,
            "current_step": "pending",
            "message": "任务已创建，等待执行",
            "result": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "estimated_seconds": estimated_seconds
        }
        
        self._tasks[task_id] = task_data
        logger.info(f"创建任务: {task_id}")
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务信息，不存在返回None
        """
        return self._tasks.get(task_id)
    
    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: Optional[int] = None,
        current_step: Optional[str] = None,
        message: Optional[str] = None
    ) -> bool:
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            progress: 进度（0-100）
            current_step: 当前步骤
            message: 消息
            
        Returns:
            是否成功更新
        """
        task = self._tasks.get(task_id)
        if not task:
            logger.warning(f"任务不存在: {task_id}")
            return False
        
        task["status"] = status
        
        if progress is not None:
            task["progress"] = max(0, min(100, progress))
        
        if current_step is not None:
            task["current_step"] = current_step
        
        if message is not None:
            task["message"] = message
        
        # 记录时间
        if status == TaskStatus.PROCESSING and not task["started_at"]:
            task["started_at"] = datetime.utcnow().isoformat()
        
        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            task["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(
            f"更新任务状态: {task_id} -> {status}, "
            f"进度: {task.get('progress', 0)}%"
        )
        
        return True
    
    def set_task_result(
        self,
        task_id: str,
        result: Dict[str, Any]
    ) -> bool:
        """
        设置任务结果
        
        Args:
            task_id: 任务ID
            result: 分析结果
            
        Returns:
            是否成功设置
        """
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        task["result"] = result
        task["status"] = TaskStatus.COMPLETED
        task["progress"] = 100
        task["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"任务完成: {task_id}")
        return True
    
    def set_task_error(
        self,
        task_id: str,
        error: str
    ) -> bool:
        """
        设置任务错误
        
        Args:
            task_id: 任务ID
            error: 错误信息
            
        Returns:
            是否成功设置
        """
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        task["error"] = error
        task["status"] = TaskStatus.FAILED
        task["completed_at"] = datetime.utcnow().isoformat()
        
        logger.error(f"任务失败: {task_id}, 错误: {error}")
        return True
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        获取所有任务（用于管理）
        
        Returns:
            任务列表
        """
        return list(self._tasks.values())
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """
        清理过期任务
        
        Args:
            max_age_hours: 最大保留时间（小时）
            
        Returns:
            清理的任务数量
        """
        now = datetime.utcnow()
        tasks_to_remove = []
        
        for task_id, task in self._tasks.items():
            created_at = datetime.fromisoformat(task["created_at"])
            age_hours = (now - created_at).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self._tasks[task_id]
        
        if tasks_to_remove:
            logger.info(f"清理过期任务: {len(tasks_to_remove)}个")
        
        return len(tasks_to_remove)


# 全局任务管理器实例
task_manager = TaskManager()
