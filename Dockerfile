# kml-agent-service 镜像
#
# 依赖里包含 numpy / scipy / shapely，这些都有预编译 wheel，
# 正常情况下无需在镜像内编译；使用 slim 而非 alpine 正是因为
# alpine 用 musl libc，多数科学计算包没有对应 wheel，会退化为源码编译。

FROM python:3.11-slim

WORKDIR /app

# 时区与 Python 运行时行为：
# - PYTHONUNBUFFERED 保证日志实时写出，否则容器里看不到即时输出
# - PYTHONDONTWRITEBYTECODE 避免生成 .pyc，容器无需持久化字节码
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

# pip 源可覆盖：国内服务器直连 pypi.org 会超时，
# 默认走腾讯云镜像（与服务器同厂，延迟最低）。
# 在能直连 PyPI 的环境构建时，可用 --build-arg PIP_INDEX_URL=https://pypi.org/simple 覆盖。
ARG PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.cloud.tencent.com

# 先装依赖再拷代码：改代码时这一层可命中缓存，不必重装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --index-url "$PIP_INDEX_URL" \
        --trusted-host "$PIP_TRUSTED_HOST" \
        -r requirements.txt

COPY app ./app

# 非 root 运行
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

# 必须单 worker：任务状态保存在进程内存中的 task_manager 里，
# 多 worker 会让「提交任务」与「查询状态」落到不同进程，
# 表现为任务查不到或状态错乱。
# 若将来要多 worker，前提是把任务状态外置到 Redis 或数据库。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
