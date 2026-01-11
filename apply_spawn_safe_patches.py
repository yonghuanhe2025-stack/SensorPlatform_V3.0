# apply_spawn_safe_patches.py
# -*- coding: utf-8 -*-
"""
一键修复（Windows + PyInstaller + multiprocessing）导致的“打包后多窗口”问题。

作用：
1) 扫描同级目录下的以下文件：
   - c16_qt_dense_viewer*.py
   - imu_unity_ui.py
   - open_cam.py
2) 备份为：<name>.bak_YYYYMMDD_HHMMSS
3) 修复规则（尽量保留原逻辑）：
   A. 如果检测到文件尾部存在“无保护”启动（QApplication/show/exec_ 或 main() 直接调用），
      则将其包裹进：
         if __name__ == "__main__":
             import multiprocessing as mp
             mp.freeze_support()
             try: mp.set_start_method("spawn", True)
             except: pass
             main()
   B. 若已存在 __main__，则确保其中包含 freeze_support() 与 set_start_method(spawn)
注意：
- 该脚本不会改变你的目录结构；
- 会自动做备份；
- 建议在源码目录执行：python apply_spawn_safe_patches.py
"""

import re
import sys
from pathlib import Path
from datetime import datetime

TARGET_PATTERNS = [
    "c16_qt_dense_viewer*.py",
    "imu_unity_ui.py",
    "open_cam.py",
]

# 常见“无保护启动”信号
RE_STARTUP = re.compile(
    r"(QApplication\s*\(|\.exec_\s*\(|\.show\s*\(|\bmain\s*\(\s*\)\s*)"
)

RE_MAIN_GUARD = re.compile(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:', re.I)

INJECT_GUARD = """if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", True)
    except Exception:
        pass

    main()
"""

def backup_file(p: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = p.with_suffix(p.suffix + f".bak_{ts}")
    bak.write_bytes(p.read_bytes())
    return bak

def ensure_guard_block(text: str) -> str:
    # 已有 guard：确保包含 freeze_support 与 set_start_method(spawn)
    if RE_MAIN_GUARD.search(text):
        # 在 __main__ 块里补齐 mp.freeze_support / set_start_method
        lines = text.splitlines(True)

        # 找到 __main__ 行号
        idx = None
        for i, ln in enumerate(lines):
            if RE_MAIN_GUARD.search(ln):
                idx = i
                break
        if idx is None:
            return text

        # 在 guard 后插入缺失内容（尽量不破坏缩进）
        indent = ""
        m = re.match(r"^(\s*)if\s+__name__", lines[idx])
        if m:
            indent = m.group(1)

        block_indent = indent + "    "
        need_freeze = "freeze_support" not in text
        need_spawn = "set_start_method" not in text or "spawn" not in text

        insert_lines = []
        if need_freeze or need_spawn:
            insert_lines.append(block_indent + "import multiprocessing as mp\n")
            if need_freeze:
                insert_lines.append(block_indent + "mp.freeze_support()\n")
            if need_spawn:
                insert_lines.append(block_indent + "try:\n")
                insert_lines.append(block_indent + "    mp.set_start_method(\"spawn\", True)\n")
                insert_lines.append(block_indent + "except Exception:\n")
                insert_lines.append(block_indent + "    pass\n")

            # guard 下一行插入
            lines[idx+1:idx+1] = insert_lines

        return "".join(lines)

    # 没有 guard：尝试识别尾部启动代码并替换为 guard
    # 策略：若文件末尾存在典型启动片段，则切掉并追加 INJECT_GUARD
    # 常见尾部片段匹配：
    tail_patterns = [
        re.compile(r"\n\s*main\s*\(\s*\)\s*\n\s*$", re.I),
        re.compile(r"\n\s*app\s*=\s*QApplication\s*\(.*?\)\s*\n.*?exec_\s*\(\s*\)\s*\n\s*$", re.I | re.S),
        re.compile(r"\n\s*sys\.exit\s*\(\s*app\.exec_\s*\(\s*\)\s*\)\s*\n\s*$", re.I),
    ]

    stripped = text.rstrip() + "\n"
    cut_text = stripped
    cut = False
    for pat in tail_patterns:
        m = pat.search(cut_text)
        if m:
            cut_text = cut_text[:m.start()] + "\n"
            cut = True
            break

    # 如果没有切到，但文件里存在 RE_STARTUP 信号，也仍然追加 guard（不切尾部，避免误伤）
    if cut:
        return cut_text + "\n" + INJECT_GUARD

    # 若没有 main() 函数定义，则不强行追加（避免破坏）。仅提示。
    if "def main" not in text:
        return text

    # 仅追加 guard（不删除任何东西）
    return cut_text + "\n" + INJECT_GUARD

def main():
    root = Path(__file__).resolve().parent
    files = []
    for pat in TARGET_PATTERNS:
        files.extend(sorted(root.glob(pat)))

    if not files:
        print("[WARN] 未找到目标文件。请把该脚本放到源码同级目录运行。")
        return 1

    changed = 0
    for p in files:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[SKIP] 读取失败: {p.name}: {e}")
            continue

        new_txt = ensure_guard_block(txt)

        if new_txt != txt:
            bak = backup_file(p)
            p.write_text(new_txt, encoding="utf-8")
            print(f"[OK] patched: {p.name} (backup: {bak.name})")
            changed += 1
        else:
            print(f"[OK] no change: {p.name}")

    print(f"[DONE] changed files: {changed}/{len(files)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
