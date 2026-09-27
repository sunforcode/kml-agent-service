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

import json
import logging
import math
import uuid
from copy import deepcopy
from datetime import datetime
from threading import Lock
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
        self._submission_lock = Lock()

    def create_task(
        self,
        request: Dict[str, Any],
        estimated_seconds: int = 60
    ) -> str:
        """兼容既有调用方；有预分配身份时复用同请求任务。"""
        task_id, _ = self.get_or_create_task(request, estimated_seconds)
        return task_id

    def get_or_create_task(
        self,
        request: Dict[str, Any],
        estimated_seconds: int = 60
    ) -> tuple[str, bool]:
        """原子登记任务，返回身份及是否需要调度；不是跨进程执行保证。"""
        snapshot = deepcopy(request)
        requested_id = snapshot.get("task_id")
        if requested_id is not None and (
            not isinstance(requested_id, str) or not requested_id.strip() or len(requested_id) > 64
        ):
            raise ValueError("task_id 必须是非空且不超过64字符的字符串")

        with self._submission_lock:
            task_id = requested_id or f"task_{uuid.uuid4().hex[:12]}"
            existing = self._tasks.get(task_id)
            if existing is not None:
                if existing["request"] != snapshot:
                    raise ValueError("task_id 已绑定不同的分析请求")
                return task_id, False

            self._tasks[task_id] = {
                "task_id": task_id,
                "request": snapshot,
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
            logger.info(f"创建任务: {task_id}")
            return task_id, True

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
        
        # 将 numpy 等不可序列化类型转换为 Python 原生类型
        task["result"] = _sanitize_for_json(result)
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

        with self._submission_lock:
            for task_id, task in self._tasks.items():
                # 预分配身份保留到进程退出，清理不能重新开放同ID执行资格。
                if task["request"].get("task_id") is not None:
                    continue
                created_at = datetime.fromisoformat(task["created_at"])
                age_hours = (now - created_at).total_seconds() / 3600

                if age_hours > max_age_hours:
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del self._tasks[task_id]

        if tasks_to_remove:
            logger.info(f"清理过期任务: {len(tasks_to_remove)}个")
        
        return len(tasks_to_remove)


def _sanitize_for_json(obj: Any) -> Any:
    """
    递归地将对象中所有不可 JSON 序列化的类型转换为 Python 原生类型。
    处理 numpy 数值类型（int64、float64、ndarray 等）和 NaN/Inf。
    """
    # numpy 类型处理（不直接 import numpy，避免硬依赖）
    module_name = type(obj).__module__ or ""
    if module_name.startswith("numpy"):
        # numpy 标量 -> Python int / float
        if hasattr(obj, "item"):
            obj = obj.item()  # 转换后继续处理（可能是 nan）
        # numpy ndarray -> list
        elif hasattr(obj, "tolist"):
            return [_sanitize_for_json(v) for v in obj.tolist()]

    # 处理 Python float 中的 nan/inf（JSON 不合法）
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]

    # 兜底：用严格 JSON 检查
    try:
        json.dumps(obj, allow_nan=False)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# 全局任务管理器实例
task_manager = TaskManager()
