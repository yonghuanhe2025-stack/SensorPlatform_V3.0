# -*- coding: utf-8 -*-

import sys
import os
import time
import base64
import re
import subprocess
from pathlib import Path
from datetime import datetime, date

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QTimer, QUrl, QCoreApplication
from PyQt5.QtGui import QImage, QDesktopServices
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton,
    QStackedWidget, QLabel, QSizePolicy,
    QFileDialog, QMessageBox, QFrame,
    QDialog, QTextEdit, QLineEdit
)

# ==========================================================
# ✅ onefile 路径策略：
# - rsrc(): 读取内置资源（_MEIPASS 或源码目录）
# - rwrite(): 写入用户文件（exe 同级目录）
# ==========================================================
def setup_dll_search_path():
    """
    双保险：确保 torch / conda 的 DLL 目录进入搜索路径。
    - onedir: <exe_dir>/_internal/torch/lib 及 <exe_dir>/_internal/Library/bin
    - onefile: <_MEIPASS>/torch/lib 及 <_MEIPASS>/Library/bin
    """
    import os, sys
    from pathlib import Path

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    def add(p: Path):
        if not p.exists():
            return
        try:
            os.add_dll_directory(str(p))
        except Exception:
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")

    roots = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
        if hasattr(sys, "_MEIPASS"):
            roots.append(Path(sys._MEIPASS))
    else:
        roots.append(Path(__file__).resolve().parent)

    for root in roots:
        base = root / "_internal" if (root / "_internal").exists() else root
        for p in [
            base / "torch" / "lib",
            base / "Library" / "bin",
            base / "PyQt5" / "Qt5" / "bin",
            base / "av.libs",
            base / "numpy.libs",
            base / "scipy.libs",
        ]:
            add(p)

def exe_dir() -> Path:
    """exe 所在目录（可写、用户可见）"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def app_root() -> Path:
    """
    内置资源根目录：
    - PyInstaller onefile：sys._MEIPASS（临时解压目录，只读）
    - 源码运行：当前 .py 所在目录
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS).resolve()
    return Path(__file__).resolve().parent


def rsrc(*parts) -> Path:
    """读取内置资源路径（只读）"""
    return app_root().joinpath(*parts)


def rwrite(*parts) -> Path:
    """写入用户文件路径（可写）"""
    return exe_dir().joinpath(*parts)


# ==========================================================
# ⭐关键修复：必须在 QApplication/QGuiApplication 构造前设置
# ==========================================================
QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)

# ==========================================================
# ⭐Logo 读取（源码/打包兼容）
# - 优先：exe 同级 logo（方便交付替换）
# - 兜底：onefile 内置资源（_MEIPASS/logo）
# ==========================================================
def load_app_icon() -> QtGui.QIcon:
    p1 = exe_dir() / "logo" / "清泰logo.ico"
    if p1.is_file():
        return QtGui.QIcon(str(p1))

    p2 = rsrc("logo", "清泰logo.ico")
    if p2.is_file():
        return QtGui.QIcon(str(p2))

    return QtGui.QIcon()


# =========================
# 0) 启动声明
# =========================
def show_copyright_notice(parent: QWidget = None) -> None:
    QMessageBox.information(
        parent,
        "声明",
        "版权所有，盗版必究\n\n请合法授权使用本软件。"
    )


# =========================
# 1) License 校验（Win11 兼容，无 wmic）
# =========================
def _norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "", s)
    return s.upper()


def _run_powershell(cmd: str, timeout: int = 6) -> str:
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return (r.stdout or "").strip()


def get_local_serial() -> str:
    # 1) 系统盘(C:)所在物理盘 SerialNumber
    ps_system = r"""
    try {
      $dn = (Get-Partition -DriveLetter C -ErrorAction Stop).DiskNumber
      $d  = Get-CimInstance Win32_DiskDrive | Where-Object { $_.Index -eq $dn } | Select-Object -First 1
      if ($d -and $d.SerialNumber) { $d.SerialNumber } else { "" }
    } catch { "" }
    """
    s = _norm(_run_powershell(ps_system))
    if s:
        return s

    # 2) 兜底：第一个物理盘 SerialNumber
    ps_first = r"""
    try {
      $d = Get-CimInstance Win32_DiskDrive |
           Where-Object { $_.SerialNumber -and $_.SerialNumber.Trim().Length -ge 6 } |
           Select-Object -First 1
      if ($d) { $d.SerialNumber } else { "" }
    } catch { "" }
    """
    return _norm(_run_powershell(ps_first))


def read_license(license_path: Path) -> tuple[str, bytes]:
    lines = license_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        raise ValueError("license.txt 格式错误：至少需要两行")
    if ">" not in lines[0] or ">" not in lines[1]:
        raise ValueError("license.txt 格式错误：需要 'License Data>' 和 'Signature>'")

    license_data = lines[0].split(">", 1)[1].strip()
    sig_b64 = lines[1].split(">", 1)[1].strip()
    try:
        signature = base64.b64decode(sig_b64)
    except Exception as e:
        raise ValueError(f"Signature Base64 解码失败：{e}")
    return license_data, signature


def parse_license_data(license_data: str) -> tuple[str, str]:
    parts = [p.strip() for p in license_data.split(",")]
    if len(parts) < 2:
        raise ValueError("License Data 格式错误：应包含 'User: ... , Expires: ...'")
    if ":" not in parts[0] or ":" not in parts[1]:
        raise ValueError("License Data 格式错误：字段缺少 ':'")

    user = parts[0].split(":", 1)[1].strip()
    exp = parts[1].split(":", 1)[1].strip()
    datetime.strptime(exp, "%Y-%m-%d")  # 校验日期格式
    return user, exp


def verify_signature(public_key_pem: bytes, signature: bytes, data: bytes) -> None:
    # 只在校验时才 import cryptography，避免一上来就因环境缺包崩溃
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    pub = serialization.load_pem_public_key(public_key_pem)
    pub.verify(signature, data, padding.PKCS1v15(), hashes.SHA256())


def is_not_expired(exp_str: str) -> bool:
    exp_d = datetime.strptime(exp_str, "%Y-%m-%d").date()
    return date.today() <= exp_d


def write_expired_log(exp_str: str, local_serial: str, writable_base: Path) -> Path:
    """
    onefile 下必须写到可写目录（exe 同级）
    输出到：<exe目录>/CustomerLicense/license_expired_YYYYMMDD_HHMMSS.log
    """
    out_dir = writable_base / "CustomerLicense"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"license_expired_{ts}.log"

    content = (
        "=== License Expired ===\n"
        f"Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Expires     : {exp_str}\n"
        f"MachineCode : {local_serial}\n"
        "Note        : 请将 MachineCode 发送给管理员用于注册授权。\n"
    )
    log_path.write_text(content, encoding="utf-8")
    return log_path


class LicenseRepairDialog(QDialog):
    """
    校验失败时：显示本机机器码 + 让用户粘贴 license.txt 内容（两行）
    点击写入：覆盖（或创建）<exe目录>/CustomerLicense/license.txt
    """
    def __init__(self, local_serial: str, license_path: Path, parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("授权校验失败 - 粘贴 license.txt")
        self.resize(760, 520)

        self.local_serial = local_serial or ""
        self.license_path = license_path
        self._saved = False

        # ⭐ 给弹窗也加 icon
        icon = load_app_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        lab1 = QLabel("本机机器码（SerialNumber）：")
        lab1.setStyleSheet("color:#e5e7eb;font-weight:800;")
        root.addWidget(lab1)

        self.edit_serial = QLineEdit(self)
        self.edit_serial.setReadOnly(True)
        self.edit_serial.setText(self.local_serial)
        self.edit_serial.setStyleSheet("""
            QLineEdit{
              background:#0b1221; color:#93c5fd; border:1px solid rgba(148,163,184,0.25);
              border-radius:8px; padding:8px; font-weight:800;
            }
        """)
        root.addWidget(self.edit_serial)

        lab2 = QLabel("请将 license.txt 的两行完整内容粘贴到下方（包含 “License Data>” 和 “Signature>”）：")
        lab2.setWordWrap(True)
        lab2.setStyleSheet("color:#cbd5e1;")
        root.addWidget(lab2)

        self.text = QTextEdit(self)
        self.text.setPlaceholderText(
            "示例：\n"
            "License Data>User: XXXXX, Expires: 2099-12-31\n"
            "Signature>BASE64...\n"
        )
        self.text.setStyleSheet("""
            QTextEdit{
              background:#0b1221; color:#e5e7eb; border:1px solid rgba(148,163,184,0.25);
              border-radius:10px; padding:10px; font-family:Consolas, 'Microsoft YaHei';
            }
        """)
        root.addWidget(self.text, 1)

        # 预加载现有 license.txt（如果存在）
        try:
            if self.license_path.is_file():
                self.text.setPlainText(self.license_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass

        btns = QHBoxLayout()
        btns.addStretch(1)

        self.btn_save = QPushButton("写入并退出", self)
        self.btn_cancel = QPushButton("取消", self)

        for b in (self.btn_save, self.btn_cancel):
            b.setFixedHeight(40)
            b.setCursor(Qt.PointingHandCursor)

        self.btn_save.setStyleSheet("""
          QPushButton{
            background-color:rgba(37,99,235,0.35);
            border:1px solid rgba(96,165,250,0.95);
            color:#e5e7eb; border-radius:10px;
            padding:8px 14px; font-size:13px; font-weight:900;
          }
          QPushButton:hover{ background-color:rgba(37,99,235,0.50); }
        """)
        self.btn_cancel.setStyleSheet("""
          QPushButton{
            background-color:#0b1221;
            border:1px solid rgba(148,163,184,0.25);
            color:#e5e7eb; border-radius:10px;
            padding:8px 14px; font-size:13px; font-weight:800;
          }
          QPushButton:hover{ border:1px solid rgba(96,165,250,0.65); }
        """)

        self.btn_save.clicked.connect(self._on_save)
        self.btn_cancel.clicked.connect(self.reject)

        btns.addWidget(self.btn_save)
        btns.addSpacing(10)
        btns.addWidget(self.btn_cancel)
        root.addLayout(btns)

        self.setStyleSheet("QDialog{ background:#020617; }")

    @property
    def saved(self) -> bool:
        return self._saved

    def _on_save(self):
        txt = (self.text.toPlainText() or "").strip()
        if not txt:
            QMessageBox.warning(self, "提示", "未输入 license 内容。")
            return

        lines = [x.strip() for x in txt.splitlines() if x.strip()]
        if len(lines) < 2 or ("License Data>" not in lines[0]) or ("Signature>" not in lines[1]):
            QMessageBox.warning(self, "格式错误", "内容格式不正确：必须包含两行：\nLicense Data>...\nSignature>...\n")
            return

        try:
            self.license_path.parent.mkdir(parents=True, exist_ok=True)
            self.license_path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "写入失败", f"无法写入：\n{self.license_path}\n\n原因：{e}")
            return

        self._saved = True
        self.accept()


def _license_fail_flow(parent: QWidget, local_serial: str, lic_path: Path):
    dlg = LicenseRepairDialog(local_serial=local_serial, license_path=lic_path, parent=parent)
    dlg.exec_()
    QMessageBox.information(parent, "请注册", "请注册  请联系1556630613@qq.com")
    raise SystemExit(100 if dlg.saved else 101)


def verify_or_exit(parent: QWidget = None) -> None:
    # public_key.pem：允许外置（exe 同级）或内置（_MEIPASS）
    pub_path_external = rwrite("CustomerLicense", "public_key.pem")
    pub_path_embedded = rsrc("CustomerLicense", "public_key.pem")
    pub_path = pub_path_external if pub_path_external.is_file() else pub_path_embedded

    # license.txt：必须外置（exe 同级，可写）
    lic_path = rwrite("CustomerLicense", "license.txt")

    local_serial = get_local_serial()

    if not pub_path.is_file():
        _license_fail_flow(parent, local_serial, lic_path)

    if not lic_path.is_file():
        _license_fail_flow(parent, local_serial, lic_path)

    if not local_serial:
        _license_fail_flow(parent, local_serial, lic_path)

    try:
        public_pem = pub_path.read_bytes()
        license_data, signature = read_license(lic_path)
        user_serial, exp = parse_license_data(license_data)
    except Exception:
        _license_fail_flow(parent, local_serial, lic_path)

    try:
        verify_signature(public_pem, signature, license_data.encode("utf-8"))
    except Exception:
        _license_fail_flow(parent, local_serial, lic_path)

    if _norm(user_serial) != _norm(local_serial):
        _license_fail_flow(parent, local_serial, lic_path)

    if not is_not_expired(exp):
        try:
            log_path = write_expired_log(exp, local_serial, exe_dir())
        except Exception:
            log_path = None

        msg = f"授权已过期：{exp}\n\n"
        if log_path:
            msg += f"已生成日志（含机器码）：\n{log_path}\n"
        else:
            msg += "日志生成失败，但机器码如下：\n" + (local_serial or "（空）")

        QMessageBox.critical(parent, "授权过期", msg)
        raise SystemExit(4)

    return


# =========================
# 2) 主 HUB
# =========================
LEARN_URL = "http://47.102.217.160:9876/home"


class ScreenRecorder:
    def __init__(self, target_widget: QWidget, fps: int = 15):
        self.target = target_widget
        self.fps = int(max(fps, 1))
        self._timer = QTimer()
        self._timer.timeout.connect(self._capture_frame)
        self._writer = None
        self._recording = False
        self._start_ts = None
        self._w0 = None
        self._h0 = None
        try:
            import cv2  # noqa
            self._cv2_ok = True
        except Exception:
            self._cv2_ok = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed_sec(self) -> float:
        if not self._recording or self._start_ts is None:
            return 0.0
        return time.time() - self._start_ts

    def start(self, out_path: str):
        if self._recording:
            return True, "已经在录制中"
        if not self._cv2_ok:
            return False, "未安装 opencv-python，无法录制。请先执行：pip install opencv-python"
        import cv2
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        pix = self.target.grab()
        img = pix.toImage().convertToFormat(QImage.Format_RGBA8888)
        w, h = img.width(), img.height()
        self._w0, self._h0 = w, h

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h))
        if not writer.isOpened():
            return False, "VideoWriter 打开失败：%s" % out_path
        self._writer = writer
        self._recording = True
        self._start_ts = time.time()
        self._timer.start(int(1000 / self.fps))
        return True, "开始录制：%s" % out_path

    def stop(self):
        if not self._recording:
            return True, "当前未在录制"
        self._timer.stop()
        try:
            if self._writer is not None:
                self._writer.release()
        except Exception:
            pass
        self._writer = None
        self._recording = False
        self._start_ts = None
        return True, "录制已保存。"

    def _capture_frame(self):
        if not self._recording or self._writer is None:
            return
        import cv2
        import numpy as np

        pix = self.target.grab()
        qimg = pix.toImage().convertToFormat(QImage.Format_RGBA8888)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        if (w, h) != (self._w0, self._h0):
            bgr = cv2.resize(bgr, (self._w0, self._h0), interpolation=cv2.INTER_AREA)
        self._writer.write(bgr)


def make_placeholder(title: str, msg: str, color: str = "#93c5fd") -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(30, 30, 30, 30)
    lay.setSpacing(12)
    t = QLabel(title)
    t.setAlignment(Qt.AlignCenter)
    t.setStyleSheet("font-size:18px;font-weight:800;color:%s;" % color)
    m = QLabel(msg)
    m.setWordWrap(True)
    m.setAlignment(Qt.AlignCenter)
    m.setStyleSheet("font-size:13px;color:#cbd5e1;line-height:1.6;")
    lay.addStretch(1)
    lay.addWidget(t)
    lay.addWidget(m)
    lay.addStretch(2)
    return w


def embed_into_container(container: QWidget, child: QWidget):
    layout = container.layout()
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()
    if isinstance(child, QMainWindow):
        child.setWindowFlags(Qt.Widget)
    layout.addWidget(child)


class HubWindow(QMainWindow):
    """
    ✅ 在原版基础上新增：
    - “超声波测距”作为独立模块页面（stack idx=4）
    - “学习中心”仍为外链（不占 stack）
    """
    def __init__(self, c16_viewer, imu_ui, cam_ui, radar_ui, ultra_ui):
        super().__init__()

        # ⭐主窗口 icon
        icon = load_app_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)

        self.setWindowTitle("智能传感器装调实验平台")
        self.resize(1650, 920)

        self.c16_viewer = c16_viewer
        self.imu_ui = imu_ui
        self.cam_ui = cam_ui
        self.radar_ui = radar_ui
        self.ultra_ui = ultra_ui

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        nav = QFrame(self)
        nav.setFixedWidth(240)
        nav.setObjectName("Nav")
        nav_lay = QVBoxLayout(nav)
        nav_lay.setContentsMargins(12, 12, 12, 12)
        nav_lay.setSpacing(10)

        # ⭐左侧顶部 Logo：图标 + 标题
        logo = QLabel(nav)
        logo.setAlignment(Qt.AlignCenter)
        logo.setObjectName("Logo")

        if not icon.isNull():
            pm = icon.pixmap(72, 72)
            logo.setPixmap(pm)
            logo.setMinimumHeight(86)
        else:
            logo.setText("SENSOR HUB")
        nav_lay.addWidget(logo)

        logo_text = QLabel("清泰 SENSOR HUB", nav)
        logo_text.setAlignment(Qt.AlignCenter)
        logo_text.setStyleSheet("color:#e5e7eb;font-size:13px;font-weight:900;letter-spacing:1px;")
        nav_lay.addWidget(logo_text)

        # ===== 左侧功能按钮（0~4 进入 stack）=====
        self.btns = []

        def add_btn(text, idx):
            b = QPushButton(text, nav)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(42)
            b.setObjectName("NavBtn")
            b.clicked.connect(lambda: self.switch_page(idx))
            nav_lay.addWidget(b)
            self.btns.append(b)

        add_btn("① 激光雷达", 0)
        add_btn("② 惯导", 1)
        add_btn("③ 智能相机", 2)
        add_btn("④ 毫米波雷达", 3)
        add_btn("⑤ 超声波测距", 4)  # ✅ 新增

        # ✅ 学习中心：外链，不占 stack
        self.btn_learn = QPushButton("⑥ 学习中心", nav)
        self.btn_learn.setCheckable(False)
        self.btn_learn.setCursor(Qt.PointingHandCursor)
        self.btn_learn.setFixedHeight(42)
        self.btn_learn.setObjectName("NavBtn")
        self.btn_learn.clicked.connect(self.open_learn_center)
        nav_lay.addWidget(self.btn_learn)

        nav_lay.addSpacing(10)

        sec = QLabel("录制", nav)
        sec.setObjectName("Section")
        nav_lay.addWidget(sec)

        self.btn_rec_start = QPushButton("● 开始录制", nav)
        self.btn_rec_stop = QPushButton("■ 停止录制", nav)
        self.btn_rec_start.setObjectName("RecBtn")
        self.btn_rec_stop.setObjectName("RecBtn")
        self.btn_rec_stop.setEnabled(False)

        self.lab_rec_state = QLabel("REC: OFF", nav)
        self.lab_rec_state.setObjectName("RecState")

        nav_lay.addWidget(self.btn_rec_start)
        nav_lay.addWidget(self.btn_rec_stop)
        nav_lay.addWidget(self.lab_rec_state)

        nav_lay.addStretch(1)
        root.addWidget(nav)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)

        top = QFrame(self)
        top.setObjectName("TopBar")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(14, 10, 14, 10)
        top_lay.setSpacing(10)

        self.lab_title = QLabel("激光雷达", top)
        self.lab_title.setObjectName("TopTitle")
        top_lay.addWidget(self.lab_title)
        top_lay.addStretch(1)
        self.lab_clock = QLabel("", top)
        self.lab_clock.setObjectName("Clock")
        top_lay.addWidget(self.lab_clock)

        right.addWidget(top)

        self.stack = QStackedWidget(self)
        self.stack.setObjectName("Stack")
        right.addWidget(self.stack, 1)
        root.addLayout(right, 1)

        # ✅ 现在有 5 个页面（0~4）
        self.containers = []
        for _ in range(5):
            c = QWidget(self.stack)
            lay = QVBoxLayout(c)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            lay.addWidget(make_placeholder("模块未加载", "首次进入该模块时才初始化（启动更快）。"))
            self.stack.addWidget(c)
            self.containers.append(c)

        self._loaded = [False] * 5
        self.pages = [None] * 5

        self.recorder = ScreenRecorder(self, fps=15)
        self.btn_rec_start.clicked.connect(self.on_start_record)
        self.btn_rec_stop.clicked.connect(self.on_stop_record)

        self._rec_ui_timer = QTimer(self)
        self._rec_ui_timer.timeout.connect(self._update_rec_ui)
        self._rec_ui_timer.start(200)

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(500)
        self._tick_clock()

        self._apply_theme()
        self.switch_page(0)

    def _apply_theme(self):
        self.setStyleSheet("""
        QMainWindow{ background-color:#020617; }
        #Nav{ background-color:#050b18; border-right:1px solid rgba(148,163,184,0.18); }
        #Logo{ color:#e5e7eb; font-size:16px; font-weight:900; letter-spacing:2px; padding:6px; }
        QPushButton#NavBtn{
          background-color:#0b1221; color:#e5e7eb;
          border:1px solid rgba(148,163,184,0.18); border-radius:8px;
          padding:8px 10px; text-align:left; font-size:13px;
        }
        QPushButton#NavBtn:hover{ border:1px solid rgba(96,165,250,0.65); background-color:#111a2e; }
        QPushButton#NavBtn:checked{ background-color:rgba(37,99,235,0.35); border:1px solid rgba(96,165,250,0.95); }
        #Section{ color:#94a3b8; font-size:12px; font-weight:800; margin-top:6px; }
        QPushButton#RecBtn{
          background-color:#0b1221; color:#e5e7eb;
          border:1px solid rgba(148,163,184,0.18); border-radius:8px;
          padding:8px 10px; text-align:left; font-size:13px;
        }
        QPushButton#RecBtn:disabled{ background-color:#0a1020; color:rgba(148,163,184,0.6); border:1px solid rgba(148,163,184,0.12); }
        #RecState{ color:#94a3b8; font-size:12px; padding-left:2px; }
        #TopBar{ background-color:#030a18; border-bottom:1px solid rgba(148,163,184,0.18); }
        #TopTitle{ color:#e5e7eb; font-size:14px; font-weight:900; }
        #Clock{
          color:#cbd5e1; font-size:12px; padding:4px 10px; border-radius:8px;
          border:1px solid rgba(148,163,184,0.18); background-color:#0b1221;
        }
        #Stack{ background-color:#020617; }
        """)

    def _tick_clock(self):
        self.lab_clock.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    # ✅ 学习中心：直接打开系统浏览器，不切页
    def open_learn_center(self):
        ok = QDesktopServices.openUrl(QUrl(LEARN_URL))
        if not ok:
            try:
                import webbrowser
                webbrowser.open(LEARN_URL, new=2)
            except Exception:
                pass

    def switch_page(self, idx: int):
        # 只允许 0~4
        if idx < 0 or idx > 4:
            return

        titles = ["激光雷达", "IMU / Unity", "相机 YOLO", "毫米波雷达", "超声波测距"]
        self.lab_title.setText(titles[idx])
        self.stack.setCurrentIndex(idx)

        for i, b in enumerate(self.btns):
            b.setChecked(i == idx)

        if not self._loaded[idx]:
            self._loaded[idx] = True
            QTimer.singleShot(10, lambda: self._create_real_page(idx))

    def _create_real_page(self, idx: int):
        try:
            if idx == 0:
                page = self.c16_viewer.MainWindow()
            elif idx == 1:
                page = self.imu_ui.MainWindow()
            elif idx == 2:
                page = self.cam_ui.MainWindow()
            elif idx == 3:
                if self.radar_ui is None:
                    page = make_placeholder("毫米波模块缺失", "未找到 ARS408.py / ARS408.py.py。", color="#fb7185")
                else:
                    try:
                        bus, ch = self.radar_ui.open_canalyst_auto(bitrate=500000)
                        page = self.radar_ui.MainWindow(bus, ch)
                    except Exception as e:
                        page = make_placeholder("毫米波雷达启动失败", str(e), color="#fb7185")
            elif idx == 4:
                # ✅ 超声波测距页面
                if self.ultra_ui is None:
                    page = make_placeholder("超声波模块缺失", "未找到 wl300d_ultra_can_scifi_gui.py。", color="#fb7185")
                else:
                    page = self.ultra_ui.MainWindow()
            else:
                page = make_placeholder("无效页面", f"idx={idx}", color="#fb7185")

            self.pages[idx] = page
            embed_into_container(self.containers[idx], page)
        except Exception as e:
            embed_into_container(self.containers[idx], make_placeholder("模块启动失败", str(e), color="#fb7185"))

    def _update_rec_ui(self):
        if self.recorder.is_recording:
            sec = int(self.recorder.elapsed_sec)
            self.lab_rec_state.setText("REC: ON  ●  %02d:%02d:%02d" % (sec // 3600, (sec % 3600) // 60, sec % 60))
            self.lab_rec_state.setStyleSheet("color:#fb7185;font-size:12px;font-weight:800;")
        else:
            self.lab_rec_state.setText("REC: OFF")
            self.lab_rec_state.setStyleSheet("color:#94a3b8;font-size:12px;font-weight:800;")

    def on_start_record(self):
        out_dir = str(rwrite("recordings"))
        os.makedirs(out_dir, exist_ok=True)

        default_name = datetime.now().strftime("hub_record_%Y%m%d_%H%M%S.mp4")
        default_path = os.path.join(out_dir, default_name)

        out_path, _ = QFileDialog.getSaveFileName(
            self, "选择录制保存位置", default_path,
            "MP4 Video (*.mp4);;All Files (*.*)"
        )
        if not out_path:
            return

        ok, msg = self.recorder.start(out_path)
        if not ok:
            QMessageBox.critical(self, "录制失败", msg)
            return
        self.btn_rec_start.setEnabled(False)
        self.btn_rec_stop.setEnabled(True)

    def on_stop_record(self):
        ok, msg = self.recorder.stop()
        if ok:
            QMessageBox.information(self, "录制完成", msg)
        self.btn_rec_start.setEnabled(True)
        self.btn_rec_stop.setEnabled(False)

    def closeEvent(self, event):
        if self.recorder.is_recording:
            self.recorder.stop()
        for p in self.pages:
            if p is None:
                continue
            try:
                p.close()
            except Exception:
                pass
        super().closeEvent(event)
        event.accept()


def main():
    # ==========================================================
    # ⭐关键修复：双保险，必须在 QApplication() 之前设置
    # ==========================================================

    setup_dll_search_path()

    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setFont(QtGui.QFont("Microsoft YaHei", 9))
    app.setStyle("Fusion")

    # ⭐全局 App 图标（任务栏/标题栏）
    app_icon = load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    # 1) 启动声明
    show_copyright_notice()

    # 2) 先校验
    verify_or_exit()

    # 3) 校验通过后再 import 子模块
    try:
        try:
            import c16_qt_dense_viewer as c16_viewer
        except ImportError:
            import c16_qt_dense_viewer_geo_table as c16_viewer

        import imu_unity_ui as imu_ui
        import open_cam as cam_ui

        radar_ui = None
        try:
            import ARS408 as radar_ui
        except ImportError:
            here = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(here, "ARS408.py.py")
            if os.path.exists(alt_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("radar_ui", alt_path)
                radar_ui = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(radar_ui)
            else:
                radar_ui = None

        ultra_ui = None
        try:
            import wl300d_ultra_can_scifi_gui as ultra_ui
        except Exception:
            ultra_ui = None

    except Exception as e:
        QMessageBox.critical(None, "启动失败", f"模块加载失败：\n{e}")
        raise SystemExit(20)

    win = HubWindow(c16_viewer, imu_ui, cam_ui, radar_ui, ultra_ui)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
