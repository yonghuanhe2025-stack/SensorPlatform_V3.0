# sitecustomize.py
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

def _add(p: Path):
    if not p or not p.exists():
        return
    try:
        os.add_dll_directory(str(p))
    except Exception:
        os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")

def _bootstrap():
    # 避免 OpenMP/线程导致的初始化失败（WinError 1114 常见诱因之一）
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    roots = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        roots.append(exe_dir)
        mei = Path(getattr(sys, "_MEIPASS", exe_dir))
        if mei not in roots:
            roots.append(mei)
    else:
        roots.append(Path(__file__).resolve().parent)

    for root in roots:
        base = root / "_internal" if (root / "_internal").exists() else root

        # torch 关键目录（必须）
        _add(base / "torch" / "lib")
        _add(base / "Library" / "bin")

        # Qt / 其他扩展（建议）
        _add(base / "PyQt5" / "Qt5" / "bin")
        _add(base / "av.libs")
        _add(base / "numpy.libs")
        _add(base / "scipy.libs")
        _add(base)

_bootstrap()
