# rthook_dll_dirs.py
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
from datetime import datetime

def _log(msg: str):
    try:
        # frozen 时，尽量写到 exe 同级目录
        if getattr(sys, "frozen", False):
            out = Path(sys.executable).resolve().parent / "dll_hook.log"
        else:
            out = Path(__file__).resolve().parent / "dll_hook.log"
        out.write_text((out.read_text(encoding="utf-8", errors="ignore") if out.exists() else "") +
                       f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n",
                       encoding="utf-8")
    except Exception:
        pass

def _add_dir(p: Path):
    if not p or not p.exists():
        return False
    try:
        os.add_dll_directory(str(p))
        return True
    except Exception:
        # 兜底：拼到 PATH
        os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
        return True

def _bootstrap():
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

    added = []
    for root in roots:
        internal = root / "_internal"
        base = internal if internal.exists() else root

        candidates = [
            base / "torch" / "lib",
            base / "Library" / "bin",
            base / "PyQt5" / "Qt5" / "bin",
            base / "av.libs",
            base / "numpy.libs",
            base / "scipy.libs",
            base / "x64" / "vc17" / "bin",
            base,
            root,
        ]
        for p in candidates:
            if _add_dir(p):
                added.append(str(p))

    _log("frozen=%s exe=%s _MEIPASS=%s" % (
        getattr(sys, "frozen", False),
        getattr(sys, "executable", ""),
        getattr(sys, "_MEIPASS", "")
    ))
    _log("added_dirs=" + " | ".join(added))

_bootstrap()
