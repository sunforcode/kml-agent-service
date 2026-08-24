"""
KML Agent Service - core 包

配置的唯一定义位置是 app.core.config，此处仅做转发导出。
历史上本文件曾整份复制了 config.py 的内容，导致存在两个彼此独立的
Settings 类与 settings 实例：从不同路径导入会拿到不同对象，
配置改动只在其中一份生效，属于难以排查的问题。
"""

from app.core.config import Settings, settings

__all__ = ["Settings", "settings"]
