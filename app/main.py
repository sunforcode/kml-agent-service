"""
KML Agent Service - FastAPI主入口

这是KML智能分析服务的主入口文件。

服务功能:
1. POST /api/v1/analyze - 提交KML分析任务
2. GET /api/v1/tasks/{task_id} - 查询任务状态
3. GET /health - 健康检查

工作流程:
1. 后端(walkbg)调用POST /api/v1/analyze提交KML URL
2. 服务返回task_id，异步执行分析
3. 后端调用GET /api/v1/tasks/{task_id}查询状态
4. 分析完成后返回EnhancedRouteOutput格式数据

当前实现状态:
- 所有Agent通过注释说明职责
- 返回假数据供迭代开发
- 后续逐步完善各Agent真实逻辑
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.config import settings
from app.models.request import (
    KmlAnalysisRequest,
    PoiFilterRequest,
    PoiFilterItem,
    PoiResolveRequest,
)
from app.models.response import (
    TaskSubmitResponse,
    TaskStatusResponse,
    HealthResponse,
    POIFilterResponse,
    POIFilterResultItem,
    POIResolveResponse,
    POIResolveResultItem,
)
from app.services.task_service import task_manager, TaskStatus
from app.services.callback_service import callback_service
from app.services.poi_filter_service import filter_pois_by_llm
from app.services.poi_match_service import resolve_poi_matches
from app.agents.analysis_workflow import AnalysisWorkflow

# 配置日志
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 全局工作流实例
workflow: Optional[AnalysisWorkflow] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI生命周期管理
    
    启动时初始化编排Agent，关闭时清理资源
    """
    global workflow
    
    logger.info("=" * 60)
    logger.info(f"启动 {settings.service_name} v{settings.service_version}")
    logger.info(f"环境: {settings.environment}")
    logger.info(f"调试模式: {settings.debug}")
    logger.info("=" * 60)
    
    # 初始化分析工作流
    logger.info("初始化分析工作流...")
    workflow = AnalysisWorkflow()
    logger.info("分析工作流初始化完成")
    
    yield
    
    # 清理资源
    logger.info("服务关闭，清理资源...")


# 创建FastAPI应用
# 文档端点是否开启由配置控制：本服务不对外提供公开 API，
# 生产环境暴露 /docs 等于公开内部接口与数据结构。
app = FastAPI(
    title="KML Agent Service",
    description="KML智能分析服务 - 使用AI Agent分析徒步路线数据",
    version=settings.service_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_api_docs else None,
    redoc_url="/redoc" if settings.enable_api_docs else None,
    openapi_url="/openapi.json" if settings.enable_api_docs else None,
)

# 配置CORS
# 本服务的调用方是 walkbg 后端（服务端到服务端，不受 CORS 约束），
# 因此默认不开放任何浏览器跨域来源。仅在确实需要时通过
# CORS_ALLOWED_ORIGINS 显式配置。
_cors_origins = settings.cors_origins_list()
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ========================================
# 健康检查端点
# ========================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查端点
    
    返回服务状态和版本信息
    """
    return HealthResponse(
        status="healthy",
        version=settings.service_version,
        timestamp=datetime.utcnow(),
        checks={
            "workflow": "ready" if workflow else "not_initialized"
        }
    )


# ========================================
# 任务管理端点
# ========================================

@app.post("/api/v1/analyze", response_model=TaskSubmitResponse)
async def submit_analysis(
    request: KmlAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    提交KML分析任务
    
    接收KML文件URL或内容，创建分析任务并异步执行。
    
    Args:
        request: KML分析请求
            - route_id: 关联的walkbg路线ID（可选）
            - kml_source: KML文件URL（必填）
            - kml_content: KML内容字符串（优先级高于URL）
            - enable_content_generation: 是否启用内容生成（LLM调用）
            - enable_poi_query: 是否启用OSM POI查询
            - poi_search_radius: POI搜索半径（米）
            - region_name: 区域名称提示
            - estimated_difficulty: 预估难度
            - user_notes: 用户备注
        
        background_tasks: FastAPI后台任务
    
    Returns:
        TaskSubmitResponse: 任务提交响应
            - task_id: 任务ID
            - status: 任务状态（pending）
            - message: 提示消息
            - estimated_seconds: 预估完成时间（秒）
    
    示例:
        POST /api/v1/analyze
        {
            "kml_source": "http://walkbg:8080/static/kml/wutaishan.kml",
            "enable_content_generation": true,
            "region_name": "五台山"
        }
    """
    logger.info(f"收到分析请求: kml_source={request.kml_source}")
    
    # 检查工作流是否就绪
    if not workflow:
        raise HTTPException(
            status_code=503,
            detail="服务未就绪，请稍后重试"
        )
    
    # 创建任务
    task_id = task_manager.create_task(
        request=request.model_dump(),
        estimated_seconds=60
    )
    
    # 添加后台任务执行分析
    background_tasks.add_task(
        execute_analysis_async,
        task_id,
        request.model_dump()
    )
    
    logger.info(f"任务已创建: {task_id}")
    
    return TaskSubmitResponse(
        task_id=task_id,
        status="pending",
        message="分析任务已提交，正在执行中",
        estimated_seconds=60
    )


# ========================================
# POI 筛选端点
# ========================================

@app.post("/api/v1/pois/filter", response_model=POIFilterResponse)
async def filter_pois(request: PoiFilterRequest):
    """
    用 LLM 对 POI 列表做质量筛选

    逐个判断 POI 是否与徒步相关（保留/剔除）、规范化类别，并给出理由。
    同步返回（内部按 30 个一批调用 LLM），大列表耗时可能较长。

    示例:
        POST /api/v1/pois/filter
        {
            "route_id": "route_xxx",
            "pois": [
                {"name": "东台望海峰", "category": "pass", "latitude": 39.08, "longitude": 113.65, "elevation": 2795}
            ]
        }
    """
    logger.info(f"收到 POI 筛选请求: route_id={request.route_id}, 数量={len(request.pois)}")

    if not request.pois:
        return POIFilterResponse(total=0, keep_count=0, reject_count=0, results=[])

    judgements, degraded = await filter_pois_by_llm(request.pois)

    results: list[POIFilterResultItem] = []
    keep_count = 0
    reject_count = 0
    for poi, judge in zip(request.pois, judgements):
        if judge["action"] == "keep":
            keep_count += 1
        else:
            reject_count += 1
        results.append(
            POIFilterResultItem(
                index=judge["index"],
                name=poi.name,
                latitude=poi.latitude,
                longitude=poi.longitude,
                elevation=poi.elevation,
                action=judge["action"],
                category=judge["category"] or poi.category,
                original_category=poi.category or None,
                reason=judge["reason"],
            )
        )

    logger.info(
        f"POI 筛选完成: route_id={request.route_id}, "
        f"总数={len(results)}, 保留={keep_count}, 剔除={reject_count}, degraded={degraded}"
    )
    return POIFilterResponse(
        total=len(results),
        keep_count=keep_count,
        reject_count=reject_count,
        results=results,
        degraded=degraded,
    )


@app.post("/api/v1/pois/resolve", response_model=POIResolveResponse)
async def resolve_pois(request: PoiResolveRequest):
    """
    用 LLM 判定新 POI 与库内条目是否为同一位置

    代码不写合并策略：只做候选召回（同库内距离 500m 内，最多 5 个），
    是否合并/命中由 LLM 依据坐标、海拔、名称参考综合判定。
    LLM 失败时保守回退为不命中（degraded=True）。
    """
    logger.info(
        f"收到 POI 位置判定请求: route_id={request.route_id}, "
        f"pois={len(request.pois)}, library={len(request.library)}"
    )

    if not request.pois:
        return POIResolveResponse(total=0, matched_count=0, results=[])

    pois = [p.model_dump() for p in request.pois]
    library = [item.model_dump() for item in request.library]

    judgements, degraded = await resolve_poi_matches(pois, library)

    results = [
        POIResolveResultItem(
            index=j["index"], library_id=j.get("library_id"), reason=j.get("reason", "")
        )
        for j in judgements
    ]
    matched_count = sum(1 for r in results if r.library_id)

    logger.info(
        f"POI 位置判定完成: route_id={request.route_id}, "
        f"总数={len(results)}, 命中={matched_count}, degraded={degraded}"
    )
    return POIResolveResponse(
        total=len(results), matched_count=matched_count, results=results, degraded=degraded
    )


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    查询任务状态
    
    根据任务ID查询分析任务的执行状态和结果。
    
    Args:
        task_id: 任务ID（从POST /api/v1/analyze返回）
    
    Returns:
        TaskStatusResponse: 任务状态响应
            - task_id: 任务ID
            - status: 任务状态
                - pending: 等待执行
                - processing: 执行中
                - completed: 已完成
                - failed: 失败
            - progress: 进度百分比（0-100）
            - current_step: 当前执行步骤
            - message: 状态消息
            - result: 分析结果（仅当status=completed时返回）
            - error: 错误信息（仅当status=failed时返回）
    
    示例:
        GET /api/v1/tasks/task_abc123def456
        
        响应:
        {
            "task_id": "task_abc123def456",
            "status": "completed",
            "progress": 100,
            "current_step": "aggregate_result",
            "message": "分析完成",
            "result": {
                "total_distance_km": 52.3,
                "total_elevation_gain_m": 2800,
                "segments": [...],
                "water_sources": [...],
                ...
            }
        }
    """
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"任务不存在: {task_id}"
        )
    
    # 构建响应
    response = TaskStatusResponse(
        task_id=task_id,
        status=task["status"].value if hasattr(task["status"], "value") else task["status"],
        progress=task.get("progress", 0),
        current_step=task.get("current_step"),
        message=task.get("message", "")
    )
    
    # 如果完成，返回结果
    if task["status"] == TaskStatus.COMPLETED and task.get("result"):
        response.result = task["result"]
    
    # 如果失败，返回错误
    if task["status"] == TaskStatus.FAILED and task.get("error"):
        response.error = task["error"]
    
    return response


# ========================================
# 后台任务执行
# ========================================

async def execute_analysis_async(task_id: str, request_dict: Dict[str, Any]):
    """
    异步执行KML分析
    
    在后台线程中执行完整的分析工作流。
    
    Args:
        task_id: 任务ID
        request_dict: 分析请求字典
    """
    logger.info(f"开始执行任务: {task_id}")
    
    route_id = request_dict.get("route_id")
    result = None
    
    try:
        task_manager.update_task_status(
            task_id,
            TaskStatus.PROCESSING,
            progress=5,
            current_step="init",
            message="开始分析"
        )
        
        if workflow:
            async def report_progress(step: str, progress: int) -> None:
                task_manager.update_task_status(
                    task_id,
                    TaskStatus.PROCESSING,
                    progress=progress,
                    current_step=step,
                    message=f"已完成步骤: {step}",
                )

            result = await workflow.execute_workflow(
                request_dict,
                progress_callback=report_progress,
                execution_event_callback=lambda event: callback_service.dispatch_execution_event(
                    task_id, event
                ),
            )

            await callback_service.drain_execution_events(task_id)

            # 回调是分析结果落库的唯一途径，结果必须检查：
            # 回调失败意味着 walkbg 侧永远看不到本次分析结果。
            delivered = await callback_service.send_callback(
                task_id=task_id,
                route_id=route_id,
                result=result,
                status="completed"
            )
            if not delivered:
                error_message = "分析完成但结果回调未送达 WalkBG"
                logger.error(
                    f"任务分析成功但回调未送达: task_id={task_id}, route_id={route_id}"
                )
                task_manager.set_task_error(task_id, error_message)
            else:
                task_manager.set_task_result(task_id, result)
                logger.info(f"任务完成: {task_id}")
        else:
            raise Exception("分析工作流未初始化")
            
    except Exception as e:
        logger.error(f"任务执行失败: {task_id}, 错误: {str(e)}", exc_info=True)
        task_manager.set_task_error(task_id, str(e))
        
        error_result = {
            "error": str(e),
            "warnings": [{"level": "error", "message": str(e)}]
        }
        await callback_service.drain_execution_events(task_id)
        delivered = await callback_service.send_callback(
            task_id=task_id,
            route_id=route_id,
            result=error_result,
            status="failed"
        )
        if not delivered:
            # 失败通知也没送到，walkbg 侧任务会一直停在处理中状态
            logger.error(
                f"任务失败且失败回调未送达，walkbg 侧可能残留处理中状态: "
                f"task_id={task_id}, route_id={route_id}"
            )


# ========================================
# 开发辅助端点（仅调试模式可用）
# ========================================

if settings.debug:
    @app.get("/api/v1/debug/tasks")
    async def get_all_tasks():
        """
        [调试] 获取所有任务列表
        
        仅在调试模式(DEBUG=true)下可用。
        """
        return {
            "total": len(task_manager.get_all_tasks()),
            "tasks": task_manager.get_all_tasks()
        }
    
    @app.post("/api/v1/debug/cleanup")
    async def cleanup_tasks():
        """
        [调试] 清理过期任务
        
        仅在调试模式下可用。
        """
        count = task_manager.cleanup_old_tasks(max_age_hours=1)
        return {"cleaned_tasks": count}


# ========================================
# 启动说明
# ========================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"启动服务: http://{settings.host}:{settings.port}")
    logger.info(f"API文档: http://{settings.host}:{settings.port}/docs")
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
