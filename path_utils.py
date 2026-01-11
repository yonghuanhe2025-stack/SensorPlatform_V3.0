# path_utils.py
from pathlib import Path
import sys

def app_dir() -> Path:
    """
    返回“应用根目录”：
    - 源码运行：返回当前脚本所在目录
    - PyInstaller 打包运行：返回 exe 所在目录（onedir 最合适）
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent   # dist/SensorPlatform/
    return Path(__file__).resolve().parent

def rpath(*parts) -> str:
    """资源相对路径 -> 绝对路径字符串"""
    return str(app_dir().joinpath(*parts))
