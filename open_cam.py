# -*- coding: utf-8 -*-
"""
open_cam.py（完整版：左侧=上视频下日志 / 右侧=控制面板更宽）

功能：
- RTSP(UDP探测) / 本地视频 / 本机摄像头
- YOLO 运行时可开关（不开也能正常看画面）
- 棋盘标定：手动采集 + 视频自动采集（后台线程，避免卡死）
  * 自动采集增强：模糊过滤 / 棋盘面积过滤 / 覆盖度网格选帧 / 去重
  * 两阶段标定：初标定 -> 逐帧重投影误差 -> 剔除离群 -> 再标定
- 矫正模式：HUD 叠加 RMS / K&dist 简写 / frames_used
- A4 棋盘 PDF：自动铺满A4并回填 square(mm)
- 一键下载开源示例数据集（含 chessboard.avi）：自动填入本地视频框
- ✅ 标定过程可视化弹窗：自动采集实时显示角点/进度/指标
"""

import os
import sys
import time
import threading
import zipfile
import urllib.request
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional

import av
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QMessageBox, QFileDialog,
    QSizePolicy, QComboBox, QCheckBox, QSplitter,
    QDialog, QProgressBar
)

from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

# ---- PyInstaller torch dll path（可保留） ----
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    torch_lib = os.path.join(sys._MEIPASS, "torch", "lib")
    if os.path.isdir(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

# YOLO（ultralytics）可选
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# =========================
# 默认配置
# =========================
DEFAULT_IP = "192.168.1.10"
DEFAULT_USER = "admin"
DEFAULT_PASSWORDS = ["", "admin", "12345", "123456"]
DEFAULT_YOLO_WEIGHTS = "yolov11n.pt"

DEFAULT_INFER_WIDTH = 640
DEFAULT_INFER_EVERY_N_FRAMES = 2

DEFAULT_MAX_UI_FPS = 20
DEFAULT_FRAME_MAX_WIDTH = 1280

PORTS = [554, 8554, 10554]
PATHS = [
    "/stream1",
    "/h264/ch1/main/av_stream",
    "/cam/realmonitor?channel=1&subtype=0",
    "/live/ch00_00",
    "/0",
    "/live",
]


# =====================================================================
# A4 棋盘 PDF：自动铺满可用区域
# =====================================================================

def draw_a4_chessboard_pdf(
    out_pdf: str,
    inner_cols: int,
    inner_rows: int,
    margin_mm: float = 8.0,
    add_labels: bool = True,
    landscape: bool = False,
    round_to_mm: float = 0.1,
) -> float:
    page_w, page_h = A4
    if landscape:
        page_w, page_h = page_h, page_w

    margin = float(margin_mm) * mm
    squares_x = inner_cols + 1
    squares_y = inner_rows + 1

    usable_w_mm = (page_w - 2 * margin) / mm
    usable_h_mm = (page_h - 2 * margin) / mm
    if usable_w_mm <= 0 or usable_h_mm <= 0:
        raise ValueError("边距过大导致可用区域为0，请减小边距。")

    sq = min(usable_w_mm / squares_x, usable_h_mm / squares_y)
    if round_to_mm and round_to_mm > 0:
        sq = (sq // round_to_mm) * round_to_mm
    if sq <= 0:
        raise ValueError("自动适配失败：可用区域过小，请减小边距或角点数量。")

    square_mm = float(sq)
    square = square_mm * mm

    board_w = squares_x * square
    board_h = squares_y * square
    x0 = (page_w - board_w) / 2.0
    y0 = (page_h - board_h) / 2.0

    c = rl_canvas.Canvas(out_pdf, pagesize=(page_w, page_h))
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

    c.setFillColorRGB(0, 0, 0)
    for j in range(squares_y):
        for i in range(squares_x):
            if (i + j) % 2 == 0:
                c.rect(x0 + i * square, y0 + j * square, square, square, fill=1, stroke=0)

    c.setLineWidth(0.5)
    c.setStrokeColorRGB(0.2, 0.2, 0.2)
    c.rect(x0, y0, board_w, board_h, fill=0, stroke=1)

    if add_labels:
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 10)
        orient = "A4 Landscape" if landscape else "A4 Portrait"
        info = (
            f"Chessboard | inner corners: {inner_cols}x{inner_rows} | "
            f"square: {square_mm:.2f} mm | {orient} | margin: {margin_mm:.1f}mm"
        )
        c.drawString(12 * mm, 10 * mm, info)
        c.setFont("Helvetica", 9)
        c.drawString(12 * mm, 6 * mm, "Print at 100% (Actual Size). Do NOT scale to fit page.")

    c.showPage()
    c.save()
    return square_mm


# =====================================================================
# RTSP 枚举工具
# =====================================================================

def xm_paths(user: str, pwd: str) -> List[str]:
    base = f"user={user}&password={pwd}&channel=1&stream=0.sdp"
    return [f"/{base}", f"/{base}?real_stream"]


def build_candidates(ip: str, user: str, pwds: List[str]) -> List[Tuple[str, str]]:
    urls: List[Tuple[str, str]] = []
    for port in PORTS:
        for path in PATHS:
            for pwd in pwds:
                if pwd == "":
                    urls.append((f"{user}/<empty>@{port}{path}", f"rtsp://{user}@{ip}:{port}{path}"))
                    urls.append((f"{user}/<empty-colon>@{port}{path}", f"rtsp://{user}:@{ip}:{port}{path}"))
                else:
                    urls.append((f"{user}/{pwd}@{port}{path}", f"rtsp://{user}:{pwd}@{ip}:{port}{path}"))

        for pwd in pwds:
            for p in xm_paths(user, pwd):
                urls.append((f"XM({user}/{pwd})@{port}{p}", f"rtsp://{ip}:{port}{p}"))
    return urls


def open_av_udp(rtsp_url: str):
    opts = {
        "rtsp_transport": "udp",
        "fflags": "nobuffer",
        "flags": "low_delay",
        "probesize": "32",
        "analyzeduration": "0",
        "max_delay": "0",
    }
    try:
        return av.open(rtsp_url, options=opts, timeout=2.0)
    except Exception:
        return None


def open_av_file(path: str):
    try:
        return av.open(path)
    except Exception:
        return None


def frames_from_pyav(ic: av.container.input.InputContainer):
    v = next((s for s in ic.streams if s.type == "video"), None)
    if v is None:
        yield None
        return
    try:
        v.thread_type = "AUTO"
    except Exception:
        pass

    try:
        for packet in ic.demux(v):
            if packet.dts is None:
                continue
            for frame in packet.decode():
                yield frame.to_ndarray(format="bgr24")
    except Exception:
        yield None


# =====================================================================
# 显示：letterbox（保持比例）
# =====================================================================

def letterbox_to_size(bgr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    if out_w <= 2 or out_h <= 2:
        return bgr
    h, w = bgr.shape[:2]
    if w <= 0 or h <= 0:
        return bgr

    scale = min(out_w / w, out_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    if (nw, nh) != (w, h):
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(bgr, (nw, nh), interpolation=interp)
    else:
        resized = bgr

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def resize_max_width(bgr: np.ndarray, max_w: int) -> np.ndarray:
    if max_w <= 0:
        return bgr
    h, w = bgr.shape[:2]
    if w <= max_w:
        return bgr
    scale = max_w / float(w)
    nh = max(1, int(h * scale))
    return cv2.resize(bgr, (max_w, nh), interpolation=cv2.INTER_AREA)


# =====================================================================
# 棋盘角点检测（稳健版）
# =====================================================================

def parse_pattern(s: str) -> Tuple[int, int]:
    s = s.strip().lower().replace("×", "x")
    if "x" not in s:
        raise ValueError("角点格式应为 9x6（列x行，内角点数量）")
    a, b = s.split("x", 1)
    cols = int(a.strip())
    rows = int(b.strip())
    if cols <= 1 or rows <= 1:
        raise ValueError("内角点数必须 > 1")
    return cols, rows


def detect_chessboard_robust(bgr: np.ndarray, pattern_size: Tuple[int, int], strong: bool):
    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray0)
    gray_blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)

    gray_inv = cv2.bitwise_not(gray_clahe)
    gray_inv_blur = cv2.GaussianBlur(gray_inv, (5, 5), 0)

    candidates = [gray0, gray_clahe, gray_blur, gray_inv, gray_inv_blur]

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY)
        if strong:
            sb_flags |= cv2.CALIB_CB_EXHAUSTIVE
        for g in candidates:
            try:
                ok, corners = cv2.findChessboardCornersSB(g, pattern_size, sb_flags)
                if ok and corners is not None:
                    return True, corners.astype(np.float32)
            except Exception:
                pass

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    for g in candidates:
        ok, corners = cv2.findChessboardCorners(g, pattern_size, flags)
        if ok and corners is not None:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(g, corners, (11, 11), (-1, -1), term)
            return True, corners2

    return False, None


# =====================================================================
# 标定可视化弹窗（实时显示采集效果）
# =====================================================================

class CalibVizDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("标定过程可视化（实时）")
        self.resize(1100, 720)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.lab_img = QLabel("等待帧...", self)
        self.lab_img.setAlignment(Qt.AlignCenter)
        self.lab_img.setStyleSheet("background:#111; color:#eee; border:1px solid #444;")
        self.lab_img.setMinimumHeight(420)
        self.lab_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.lab_img, stretch=1)

        row = QHBoxLayout()
        self.prog = QProgressBar(self)
        self.prog.setRange(0, 100)
        self.prog.setValue(0)
        self.prog.setFormat("进度：%p%")
        row.addWidget(self.prog, stretch=1)

        self.btn_close = QPushButton("关闭弹窗（后台继续）", self)
        self.btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_close)
        root.addLayout(row)

        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setStyleSheet(
            "background:#000; color:#00FF66; border:1px solid #444; font-family:Consolas,monospace;"
        )
        self.text.setMinimumHeight(180)
        root.addWidget(self.text)

    def append(self, s: str):
        self.text.append(s)
        self.text.moveCursor(self.text.textCursor().End)

    def set_progress(self, got: int, total: int):
        total = max(1, int(total))
        got = max(0, int(got))
        v = int(min(100, max(0, got * 100 / total)))
        self.prog.setValue(v)

    def show_bgr(self, bgr: np.ndarray):
        if bgr is None or bgr.size == 0:
            return
        w = max(2, self.lab_img.width())
        h = max(2, self.lab_img.height())
        canvas = letterbox_to_size(bgr, w, h)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        hh, ww, ch = rgb.shape
        qimg = QImage(rgb.data, ww, hh, ch * ww, QImage.Format_RGB888).copy()
        self.lab_img.setPixmap(QPixmap.fromImage(qimg))


# =====================================================================
# 自动采集（从视频文件后台抽帧找角点）— 增强版 + 实时预览信号
# =====================================================================

class AutoCollectWorker(QThread):
    log = pyqtSignal(str)
    preview = pyqtSignal(object, str, int, int)  # bgr_vis, info, got, max_frames
    done = pyqtSignal(object, object, object)    # objpoints, imgpoints, image_size(w,h)
    fail = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        pattern: Tuple[int, int],
        square_mm: float,
        every_n: int,
        max_frames: int,
        strong: bool,
        frame_max_width: int,
        dedup_px: float = 15.0,
        min_blur_var: float = 60.0,
        min_board_area_ratio: float = 0.02,
        cover_grid: Tuple[int, int] = (6, 4)
    ):
        super().__init__()
        self.video_path = video_path
        self.pattern = pattern
        self.square_mm = float(square_mm)
        self.every_n = max(1, int(every_n))
        self.max_frames = max(1, int(max_frames))
        self.strong = bool(strong)
        self.frame_max_width = int(frame_max_width)
        self.dedup_px = float(dedup_px)

        self.min_blur_var = float(min_blur_var)
        self.min_board_area_ratio = float(min_board_area_ratio)
        self.cover_grid = (int(cover_grid[0]), int(cover_grid[1]))

    def run(self):
        try:
            if not os.path.isfile(self.video_path):
                self.fail.emit("视频文件不存在。")
                return
            if self.square_mm <= 0:
                self.fail.emit("square(mm) 必须 > 0")
                return

            cols, rows = self.pattern
            objp_template = np.zeros((rows * cols, 3), np.float32)
            objp_template[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.square_mm

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.fail.emit("无法打开视频文件。")
                return

            objpoints, imgpoints = [], []
            image_size = None

            idx = 0
            got = 0
            last_pos = None

            gx, gy = self.cover_grid
            cover = np.zeros((gy, gx), dtype=np.uint8)

            self.log.emit(
                f"自动采集（增强版）：every={self.every_n} max_frames={self.max_frames} "
                f"blur_var>={self.min_blur_var} area_ratio>={self.min_board_area_ratio} grid={gx}x{gy}"
            )

            while got < self.max_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                idx += 1
                if idx % self.every_n != 0:
                    continue

                frame = resize_max_width(frame, self.frame_max_width)
                h, w = frame.shape[:2]
                if image_size is None:
                    image_size = (w, h)

                # 1) 模糊过滤
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur_var < self.min_blur_var:
                    continue

                # 2) 角点检测
                found, corners = detect_chessboard_robust(frame, (cols, rows), strong=self.strong)
                if not found or corners is None:
                    continue

                # 3) 去重
                mean_xy = corners.reshape(-1, 2).mean(axis=0)
                if last_pos is not None:
                    if float(np.linalg.norm(mean_xy - last_pos)) < self.dedup_px:
                        continue

                # 4) 面积过滤
                xys = corners.reshape(-1, 2)
                x1, y1 = xys.min(axis=0)
                x2, y2 = xys.max(axis=0)
                board_area = float(max(1.0, (x2 - x1) * (y2 - y1)))
                area_ratio = board_area / float(w * h)
                if area_ratio < self.min_board_area_ratio:
                    continue

                # 5) 覆盖度
                cx = int(np.clip(mean_xy[0] / max(1, w) * gx, 0, gx - 1))
                cy = int(np.clip(mean_xy[1] / max(1, h) * gy, 0, gy - 1))
                if cover[cy, cx] == 1 and got >= max(6, self.max_frames // 2):
                    continue

                cover[cy, cx] = 1
                last_pos = mean_xy

                objpoints.append(objp_template.copy())
                imgpoints.append(corners)
                got += 1

                msg = f"[OK] {got}/{self.max_frames} frame#{idx} blur={blur_var:.1f} area={area_ratio:.3f} grid={cx},{cy}"
                self.log.emit(msg)

                # ✅ 预览：画角点 + 叠字
                vis = frame.copy()
                cv2.drawChessboardCorners(vis, (cols, rows), corners, True)
                cv2.putText(vis, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                self.preview.emit(vis, msg, got, self.max_frames)

                if got >= 12 and cover.sum() >= int(gx * gy * 0.55):
                    self.log.emit("覆盖度已足够，提前结束采集。")
                    break

            cap.release()

            if len(imgpoints) < 8:
                self.fail.emit(f"自动采集到的有效帧太少：{len(imgpoints)}（建议 >= 8，最好 12~20）")
                return
            if image_size is None:
                self.fail.emit("未能读取到有效视频帧。")
                return

            self.done.emit(objpoints, imgpoints, image_size)
        except Exception as e:
            self.fail.emit(f"自动采集失败：{e}")


# =====================================================================
# 示例数据集下载线程（含 chessboard.avi）
# =====================================================================

class DatasetDownloadWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(str)   # 返回 chessboard.avi 的路径
    fail = pyqtSignal(str)

    def __init__(self, out_dir: str):
        super().__init__()
        self.out_dir = out_dir

    def run(self):
        try:
            zip_url = "https://github.com/smidm/video2calibration/archive/refs/heads/master.zip"
            os.makedirs(self.out_dir, exist_ok=True)
            zip_path = os.path.join(self.out_dir, "video2calibration-master.zip")
            extract_dir = os.path.join(self.out_dir, "video2calibration-master")

            self.log.emit(f"开始下载示例数据集：{zip_url}")
            self._download(zip_url, zip_path)

            if os.path.isdir(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)

            self.log.emit("解压中...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.out_dir)

            candidate = os.path.join(extract_dir, "example_input", "chessboard.avi")
            if not os.path.isfile(candidate):
                found = None
                for root, _dirs, files in os.walk(self.out_dir):
                    for fn in files:
                        if fn.lower() == "chessboard.avi":
                            found = os.path.join(root, fn)
                            break
                    if found:
                        break
                candidate = found

            if not candidate or not os.path.isfile(candidate):
                self.fail.emit("下载/解压完成，但未找到 example_input/chessboard.avi")
                return

            self.log.emit(f"✅ 示例视频已就绪：{candidate}")
            self.done.emit(candidate)
        except Exception as e:
            self.fail.emit(f"下载数据集失败：{e}")

    def _download(self, url: str, out_path: str):
        def _report(blocknum, blocksize, totalsize):
            if totalsize <= 0:
                return
            done = blocknum * blocksize
            pct = min(100.0, done * 100.0 / totalsize)
            if blocknum % 25 == 0:
                self.log.emit(f"下载进度：{pct:.1f}%")

        urllib.request.urlretrieve(url, out_path, reporthook=_report)
        self.log.emit(f"已下载到：{out_path}")


# =====================================================================
# Worker（视频解码 + YOLO可选 + 限帧）
# =====================================================================

@dataclass
class RunConfig:
    mode: str  # "rtsp" / "file" / "camera"
    ip: str = ""
    user: str = ""
    pwds: Optional[List[str]] = None

    file_path: str = ""
    loop_file: bool = False
    cam_index: int = 0

    max_ui_fps: int = DEFAULT_MAX_UI_FPS
    frame_max_width: int = DEFAULT_FRAME_MAX_WIDTH

    enable_yolo: bool = True
    model_path: Optional[str] = None
    infer_width: int = DEFAULT_INFER_WIDTH
    infer_every_n: int = DEFAULT_INFER_EVERY_N_FRAMES


class CameraWorker(QThread):
    frame_ready = pyqtSignal(object)  # (raw_bgr, draw_bgr)
    status_msg = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, cfg: RunConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.cfg = cfg
        self._running = True

        self._lock = threading.Lock()
        self._yolo_enabled = bool(cfg.enable_yolo)

        self._model = None
        self._model_failed = False
        self._last_boxes = None
        self._frame_count = 0

        self._min_emit_interval = 1.0 / max(1, int(cfg.max_ui_fps))
        self._last_emit_t = 0.0

    def stop(self):
        self._running = False

    def set_yolo_enabled(self, enabled: bool):
        with self._lock:
            prev = self._yolo_enabled
            self._yolo_enabled = bool(enabled)
            if not self._yolo_enabled:
                self._last_boxes = None
        if prev != bool(enabled):
            self.status_msg.emit(f"YOLO 推理已{'开启' if enabled else '关闭'}（运行时切换）")

    def _is_yolo_enabled(self) -> bool:
        with self._lock:
            return self._yolo_enabled

    def _try_lazy_load_model(self):
        if not self._is_yolo_enabled():
            return None
        if self._model_failed:
            return None
        if self._model is not None:
            return self._model

        if YOLO is None:
            self.status_msg.emit("⚠️ 未安装 ultralytics，已自动关闭 YOLO（视频仍正常）。")
            self._model_failed = True
            self.set_yolo_enabled(False)
            return None

        try:
            if self.cfg.model_path:
                self.status_msg.emit(f"正在加载本地 YOLO 权重：{self.cfg.model_path}")
                self._model = YOLO(self.cfg.model_path)
            else:
                self.status_msg.emit(f"正在加载默认 YOLO 权重：{DEFAULT_YOLO_WEIGHTS}（无则会下载）")
                self._model = YOLO(DEFAULT_YOLO_WEIGHTS)
            self.status_msg.emit("YOLO 模型加载完成。")
            return self._model
        except Exception as e:
            self.status_msg.emit(f"⚠️ YOLO 加载失败，已自动关闭推理（视频仍正常）。原因：{e}")
            self._model_failed = True
            self.set_yolo_enabled(False)
            return None

    def _infer_and_draw(self, bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        if not self._is_yolo_enabled():
            return bgr
        model = self._try_lazy_load_model()
        if model is None:
            return bgr

        h0, w0 = bgr.shape[:2]
        infer_every = max(int(self.cfg.infer_every_n), 1)
        infer_width = max(int(self.cfg.infer_width), 64)
        need_infer = (frame_idx % infer_every == 0)

        if need_infer:
            try:
                if w0 > infer_width:
                    scale = infer_width / float(w0)
                    infer_w = infer_width
                    infer_h = int(h0 * scale)
                    small = cv2.resize(bgr, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                else:
                    scale = 1.0
                    small = bgr

                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                results = model(rgb_small, verbose=False)[0]

                if results.boxes is not None:
                    boxes = results.boxes
                    xyxy_small = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clses = boxes.cls.cpu().numpy().astype(int)

                    xyxy = xyxy_small.copy()
                    if scale != 1.0:
                        xyxy[:, [0, 2]] /= scale
                        xyxy[:, [1, 3]] /= scale
                    self._last_boxes = (xyxy, confs, clses)
                else:
                    self._last_boxes = None
            except Exception as e:
                self.status_msg.emit(f"⚠️ YOLO 推理出错，已自动关闭推理（视频仍正常）。原因：{e}")
                self._model_failed = True
                self._model = None
                self._last_boxes = None
                self.set_yolo_enabled(False)
                return bgr

        img_draw = bgr.copy()
        if self._last_boxes is not None and self._is_yolo_enabled():
            xyxy, confs, clses = self._last_boxes
            names = model.names if hasattr(model, "names") else {}
            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clses):
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                cls_name = names.get(int(cls_id), str(int(cls_id)))
                label = f"{cls_name} {conf:.2f}"

                cv2.rectangle(img_draw, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_draw, (x1i, y1i - th - baseline), (x1i + tw, y1i), (0, 255, 0), -1)
                cv2.putText(img_draw, label, (x1i, y1i - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img_draw

    def _emit_limited(self, raw_bgr: np.ndarray, draw_bgr: np.ndarray):
        now = time.monotonic()
        if now - self._last_emit_t < self._min_emit_interval:
            return
        self._last_emit_t = now
        self.frame_ready.emit((raw_bgr, draw_bgr))

    def _process_frame(self, bgr: np.ndarray):
        raw = resize_max_width(bgr, int(self.cfg.frame_max_width))
        self._frame_count += 1
        draw = self._infer_and_draw(raw, self._frame_count)
        self._emit_limited(raw, draw)

    def _run_camera(self):
        cam_id = int(self.cfg.cam_index)
        self.status_msg.emit(f"=== 本机摄像头：ID={cam_id} | max_w={self.cfg.frame_max_width} | ui_fps={self.cfg.max_ui_fps} ===")

        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cam_id)

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not cap.isOpened():
            self.error.emit(f"打开摄像头失败：ID={cam_id}（可试 0/1/2...）")
            return

        try:
            while self._running:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    self.status_msg.emit("摄像头读帧失败，停止。")
                    break
                self._process_frame(bgr)
        finally:
            cap.release()

    def _run_file(self):
        path = self.cfg.file_path.strip()
        if not path:
            self.error.emit("未选择本地视频文件。")
            return

        self.status_msg.emit(f"=== 本地视频：{path} | max_w={self.cfg.frame_max_width} | ui_fps={self.cfg.max_ui_fps} ===")

        while self._running:
            ic = open_av_file(path)
            if not ic:
                self.error.emit(f"打开本地视频失败：{path}")
                return

            try:
                for bgr in frames_from_pyav(ic):
                    if not self._running:
                        break
                    if bgr is None:
                        break
                    self._process_frame(bgr)
            finally:
                try:
                    ic.close()
                except Exception:
                    pass

            if not self._running:
                break
            if not self.cfg.loop_file:
                self.status_msg.emit("本地视频播放结束。")
                break
            self.status_msg.emit("本地视频末尾，循环重播...")

    def _run_rtsp(self):
        ip = self.cfg.ip.strip()
        user = self.cfg.user.strip()
        pwds = self.cfg.pwds or [""]

        candidates = build_candidates(ip, user, pwds)
        total = len(candidates)
        self.status_msg.emit(f"=== RTSP 探测：UDP低延迟（{total} 条候选）| max_w={self.cfg.frame_max_width} | ui_fps={self.cfg.max_ui_fps} ===")

        for idx, (label, url) in enumerate(candidates, start=1):
            if not self._running:
                break

            self.status_msg.emit(f"[TRY {idx}/{total}] {label}\n  {url}")
            ic = open_av_udp(url)
            if not ic:
                self.status_msg.emit("  [AV] 打开失败")
                continue

            self.status_msg.emit("[OK] 连接成功，开始解码。")
            try:
                for bgr in frames_from_pyav(ic):
                    if not self._running:
                        break
                    if bgr is None:
                        self.status_msg.emit("解码结束或出错，退出当前流。")
                        break
                    self._process_frame(bgr)
            finally:
                try:
                    ic.close()
                except Exception:
                    pass
            return

        if self._running:
            self.error.emit("× 未能连上任何 RTSP 流，请检查 IP/用户/密码 或 RTSP 设置。")

    def run(self):
        try:
            if self.cfg.mode == "file":
                self._run_file()
            elif self.cfg.mode == "camera":
                self._run_camera()
            else:
                self._run_rtsp()
        except Exception as e:
            self.error.emit(str(e))


# =====================================================================
# UI
# =====================================================================

class CameraPage(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.worker: Optional[CameraWorker] = None
        self.model_path: Optional[str] = None

        self.last_raw_bgr: Optional[np.ndarray] = None
        self.last_draw_bgr: Optional[np.ndarray] = None

        # 标定数据
        self.calib_objpoints: List[np.ndarray] = []
        self.calib_imgpoints: List[np.ndarray] = []
        self.calib_K: Optional[np.ndarray] = None
        self.calib_dist: Optional[np.ndarray] = None
        self.calib_rms: Optional[float] = None
        self.calib_frames_used: int = 0

        # 去畸变映射缓存
        self._map1 = None
        self._map2 = None
        self._map_size = None
        self._map_fill = None
        self._roi = None

        # 线程
        self.auto_collect_thread: Optional[AutoCollectWorker] = None
        self.ds_thread: Optional[DatasetDownloadWorker] = None

        # ✅ 标定可视化弹窗
        self.viz: Optional[CalibVizDialog] = None

        # ========= 主布局：左（视频+日志）/ 右（控件更宽）=========
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet("""
        QSplitter::handle { background: #2a2a2a; }
        QSplitter::handle:hover { background: #3a3a3a; }
        """)

        # ============== 左侧：上视频 + 下日志 ==============
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(6)

        self.lab_title = QLabel("原始（含YOLO框）", self)
        self.lab_title.setStyleSheet("color:#EAEAEA; font-size:12pt; padding:4px;")
        left_layout.addWidget(self.lab_title)

        self.lab_video = QLabel(self)
        self._init_video_label(self.lab_video, "无视频")
        left_layout.addWidget(self.lab_video, stretch=1)

        lab_log = QLabel("日志", self)
        lab_log.setStyleSheet("color:#EAEAEA; font-size:12pt; padding:4px;")
        left_layout.addWidget(lab_log)

        self.text_log = QTextEdit(self)
        self.text_log.setReadOnly(True)
        self.text_log.setStyleSheet(
            "background-color:#000000; color:#00FF00; font-family:Consolas,monospace; font-size:11pt; "
            "border:1px solid #555555;"
        )
        self.text_log.setMinimumHeight(180)
        self.text_log.setMaximumHeight(360)
        left_layout.addWidget(self.text_log)

        # ============== 右：控制区（更宽） ==============
        mid_widget = QWidget(self)
        mid_widget.setStyleSheet("background-color:#000000;")
        mid_layout = QVBoxLayout(mid_widget)
        mid_layout.setContentsMargins(6, 6, 6, 6)
        mid_layout.setSpacing(8)

        # 视频源
        row_mode = QHBoxLayout()
        row_mode.addWidget(self._lab("视频源："))
        self.combo_mode = QComboBox(self)
        self.combo_mode.addItems(["RTSP 探测", "本地视频", "本机摄像头"])
        self.combo_mode.setFixedWidth(140)
        self._style_combo(self.combo_mode)
        row_mode.addWidget(self.combo_mode)

        self.chk_loop = QCheckBox("循环播放", self)
        self.chk_loop.setStyleSheet("color:#FFFFFF;")
        self.chk_loop.setChecked(False)
        row_mode.addWidget(self.chk_loop)
        row_mode.addStretch(1)
        mid_layout.addLayout(row_mode)

        # 显示模式
        row_view = QHBoxLayout()
        row_view.addWidget(self._lab("显示："))
        self.combo_view = QComboBox(self)
        self.combo_view.addItems(["原始（含YOLO框）", "矫正（去畸变）"])
        self.combo_view.setFixedWidth(180)
        self._style_combo(self.combo_view)
        row_view.addWidget(self.combo_view)

        self.chk_undist_fill = QCheckBox("矫正填满(裁切黑边)", self)
        self.chk_undist_fill.setStyleSheet("color:#FFFFFF;")
        self.chk_undist_fill.setChecked(True)
        row_view.addWidget(self.chk_undist_fill)
        row_view.addStretch(1)
        mid_layout.addLayout(row_view)

        # 摄像头ID
        row_cam = QHBoxLayout()
        row_cam.addWidget(self._lab("摄像头ID："))
        self.line_cam = QLineEdit(self); self.line_cam.setText("0")
        self.line_cam.setFixedWidth(80)
        self._style_lineedit(self.line_cam)
        row_cam.addWidget(self.line_cam)
        row_cam.addStretch(1)
        mid_layout.addLayout(row_cam)

        # 本地视频路径
        row_file = QHBoxLayout()
        row_file.addWidget(self._lab("本地视频："))
        self.line_file = QLineEdit(self)
        self._style_lineedit(self.line_file)
        row_file.addWidget(self.line_file, stretch=1)
        self.btn_browse = QPushButton("浏览", self); self._style_button(self.btn_browse)
        row_file.addWidget(self.btn_browse)
        mid_layout.addLayout(row_file)

        # RTSP
        row_rtsp1 = QHBoxLayout()
        row_rtsp1.addWidget(self._lab("IP："))
        self.line_ip = QLineEdit(self); self.line_ip.setText(DEFAULT_IP); self.line_ip.setFixedWidth(160)
        self._style_lineedit(self.line_ip); row_rtsp1.addWidget(self.line_ip)

        row_rtsp1.addWidget(self._lab("用户："))
        self.line_user = QLineEdit(self); self.line_user.setText(DEFAULT_USER); self.line_user.setFixedWidth(120)
        self._style_lineedit(self.line_user); row_rtsp1.addWidget(self.line_user)
        row_rtsp1.addStretch(1)
        mid_layout.addLayout(row_rtsp1)

        row_pw = QHBoxLayout()
        row_pw.addWidget(self._lab("密码："))
        self.line_pwds = QLineEdit(self); self.line_pwds.setText(",".join(DEFAULT_PASSWORDS))
        self._style_lineedit(self.line_pwds); row_pw.addWidget(self.line_pwds)
        mid_layout.addLayout(row_pw)

        # 性能参数
        row_perf = QHBoxLayout()
        row_perf.addWidget(self._lab("UI FPS："))
        self.line_ui_fps = QLineEdit(self); self.line_ui_fps.setText(str(DEFAULT_MAX_UI_FPS)); self.line_ui_fps.setFixedWidth(80)
        self._style_lineedit(self.line_ui_fps); row_perf.addWidget(self.line_ui_fps)

        row_perf.addWidget(self._lab("帧最大宽："))
        self.line_frame_w = QLineEdit(self); self.line_frame_w.setText(str(DEFAULT_FRAME_MAX_WIDTH)); self.line_frame_w.setFixedWidth(100)
        self._style_lineedit(self.line_frame_w); row_perf.addWidget(self.line_frame_w)
        row_perf.addStretch(1)
        mid_layout.addLayout(row_perf)

        # YOLO参数
        row_infer = QHBoxLayout()
        row_infer.addWidget(self._lab("推理宽："))
        self.line_infer_w = QLineEdit(self); self.line_infer_w.setText(str(DEFAULT_INFER_WIDTH)); self.line_infer_w.setFixedWidth(90)
        self._style_lineedit(self.line_infer_w); row_infer.addWidget(self.line_infer_w)

        row_infer.addWidget(self._lab("每N帧："))
        self.line_infer_n = QLineEdit(self); self.line_infer_n.setText(str(DEFAULT_INFER_EVERY_N_FRAMES)); self.line_infer_n.setFixedWidth(90)
        self._style_lineedit(self.line_infer_n); row_infer.addWidget(self.line_infer_n)
        row_infer.addStretch(1)
        mid_layout.addLayout(row_infer)

        self.chk_enable_yolo = QCheckBox("启用YOLO（运行时可切换）", self)
        self.chk_enable_yolo.setStyleSheet("color:#FFFFFF;")
        self.chk_enable_yolo.setChecked(True)
        mid_layout.addWidget(self.chk_enable_yolo)

        # 启停
        row_btn = QHBoxLayout()
        self.btn_start = QPushButton("开始", self); self._style_button(self.btn_start)
        self.btn_stop = QPushButton("停止", self); self._style_button(self.btn_stop); self.btn_stop.setEnabled(False)
        row_btn.addWidget(self.btn_start); row_btn.addWidget(self.btn_stop)
        mid_layout.addLayout(row_btn)

        # 权重
        row_wt = QHBoxLayout()
        self.lab_model = self._lab(f"权重：默认（{DEFAULT_YOLO_WEIGHTS}）")
        row_wt.addWidget(self.lab_model)
        self.btn_sel_wt = QPushButton("选本地", self); self._style_button(self.btn_sel_wt)
        self.btn_rst_wt = QPushButton("默认", self); self._style_button(self.btn_rst_wt)
        row_wt.addWidget(self.btn_sel_wt); row_wt.addWidget(self.btn_rst_wt)
        mid_layout.addLayout(row_wt)

        # 标定区
        mid_layout.addWidget(self._sep("—— 棋盘标定（视频优先，增强+两阶段）——"))

        row_cal0 = QHBoxLayout()
        row_cal0.addWidget(self._lab("角点："))
        self.line_board = QLineEdit(self); self.line_board.setText("9x6"); self.line_board.setFixedWidth(90)
        self._style_lineedit(self.line_board); row_cal0.addWidget(self.line_board)

        row_cal0.addWidget(self._lab("格子mm："))
        self.line_square = QLineEdit(self); self.line_square.setText("25.0"); self.line_square.setFixedWidth(90)
        self._style_lineedit(self.line_square); row_cal0.addWidget(self.line_square)
        row_cal0.addStretch(1)
        mid_layout.addLayout(row_cal0)

        row_cal1 = QHBoxLayout()
        self.chk_strong = QCheckBox("强力检测", self)
        self.chk_strong.setStyleSheet("color:#FFFFFF;"); self.chk_strong.setChecked(False)
        row_cal1.addWidget(self.chk_strong)

        self.chk_show_corners = QCheckBox("采集时画角点", self)
        self.chk_show_corners.setStyleSheet("color:#FFFFFF;"); self.chk_show_corners.setChecked(True)
        row_cal1.addWidget(self.chk_show_corners)
        row_cal1.addStretch(1)
        mid_layout.addLayout(row_cal1)

        row_cal2 = QHBoxLayout()
        self.btn_cap = QPushButton("采集帧(当前画面)", self); self._style_button(self.btn_cap)
        self.btn_clear = QPushButton("清空", self); self._style_button(self.btn_clear)
        self.btn_calib = QPushButton("计算内参(两阶段)", self); self._style_button(self.btn_calib)
        row_cal2.addWidget(self.btn_cap); row_cal2.addWidget(self.btn_clear); row_cal2.addWidget(self.btn_calib)
        mid_layout.addLayout(row_cal2)

        # 自动采集（视频）
        row_auto = QHBoxLayout()
        self.btn_auto = QPushButton("自动采集(视频)+可视化", self); self._style_button(self.btn_auto)
        row_auto.addWidget(self.btn_auto)
        row_auto.addWidget(self._lab("每N帧："))
        self.line_auto_every = QLineEdit(self); self.line_auto_every.setText("10"); self.line_auto_every.setFixedWidth(80)
        self._style_lineedit(self.line_auto_every); row_auto.addWidget(self.line_auto_every)
        row_auto.addWidget(self._lab("帧数："))
        self.line_auto_max = QLineEdit(self); self.line_auto_max.setText("15"); self.line_auto_max.setFixedWidth(80)
        self._style_lineedit(self.line_auto_max); row_auto.addWidget(self.line_auto_max)
        row_auto.addStretch(1)
        mid_layout.addLayout(row_auto)

        # 数据集下载
        row_ds = QHBoxLayout()
        self.btn_ds = QPushButton("下载示例数据集(chessboard.avi)", self); self._style_button(self.btn_ds)
        row_ds.addWidget(self.btn_ds)
        row_ds.addStretch(1)
        mid_layout.addLayout(row_ds)

        # 保存/加载
        row_io = QHBoxLayout()
        self.btn_save = QPushButton("保存YAML", self); self._style_button(self.btn_save)
        self.btn_load = QPushButton("加载YAML", self); self._style_button(self.btn_load)
        self.lab_state = self._lab("未标定")
        row_io.addWidget(self.btn_save); row_io.addWidget(self.btn_load); row_io.addWidget(self.lab_state)
        row_io.addStretch(1)
        mid_layout.addLayout(row_io)

        # A4 棋盘
        mid_layout.addWidget(self._sep("—— A4棋盘PDF ——"))
        row_pdf = QHBoxLayout()
        self.btn_a4 = QPushButton("生成A4 PDF", self); self._style_button(self.btn_a4)
        row_pdf.addWidget(self.btn_a4)

        row_pdf.addWidget(self._lab("边距mm："))
        self.line_margin = QLineEdit(self); self.line_margin.setText("8.0"); self.line_margin.setFixedWidth(90)
        self._style_lineedit(self.line_margin); row_pdf.addWidget(self.line_margin)

        self.chk_land = QCheckBox("横版", self); self.chk_land.setStyleSheet("color:#FFFFFF;"); self.chk_land.setChecked(False)
        row_pdf.addWidget(self.chk_land)
        self.chk_pdf_labels = QCheckBox("标注", self); self.chk_pdf_labels.setStyleSheet("color:#FFFFFF;"); self.chk_pdf_labels.setChecked(True)
        row_pdf.addWidget(self.chk_pdf_labels)

        row_pdf.addStretch(1)
        mid_layout.addLayout(row_pdf)

        mid_layout.addStretch(1)

        # ===== splitter 组装 =====
        splitter.addWidget(left_widget)
        splitter.addWidget(mid_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        # 控件面板更宽
        mid_widget.setMinimumWidth(520)
        mid_widget.setMaximumWidth(900)
        splitter.setSizes([1250, 650])

        left_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mid_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        root.addWidget(splitter)

        # 绑定信号
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
        self.btn_browse.clicked.connect(self.on_browse_video)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)

        self.btn_sel_wt.clicked.connect(self.on_select_weight)
        self.btn_rst_wt.clicked.connect(self.on_reset_weight)
        self.chk_enable_yolo.stateChanged.connect(self.on_toggle_yolo)

        self.btn_cap.clicked.connect(self.on_capture_frame)
        self.btn_clear.clicked.connect(self.on_clear_calib)
        self.btn_calib.clicked.connect(self.on_compute_calib_two_stage)
        self.btn_auto.clicked.connect(self.on_auto_collect)

        self.btn_ds.clicked.connect(self.on_download_dataset)

        self.btn_save.clicked.connect(self.on_save_calib)
        self.btn_load.clicked.connect(self.on_load_calib)

        self.btn_a4.clicked.connect(self.on_make_a4)

        self.combo_view.currentIndexChanged.connect(self.refresh_display)
        self.chk_undist_fill.stateChanged.connect(self._reset_undist_maps)

        self.on_mode_changed()
        self.refresh_display()

    # ---------- 基础样式 ----------
    def _lab(self, s: str) -> QLabel:
        x = QLabel(s)
        x.setStyleSheet("color:#FFFFFF; font-size:11pt;")
        return x

    def _sep(self, s: str) -> QLabel:
        x = QLabel(s)
        x.setStyleSheet("color:#7dd3fc; font-size:11pt; padding-top:6px; padding-bottom:2px;")
        return x

    def _style_lineedit(self, le: QLineEdit):
        le.setStyleSheet(
            "QLineEdit { background:#101010; color:#FFFFFF; border:1px solid #666666; padding:3px; }"
            "QLineEdit:focus { border:1px solid #00CCFF; }"
        )

    def _style_button(self, btn: QPushButton):
        btn.setStyleSheet(
            "QPushButton { background:#DDDDDD; color:#000000; border-radius:4px; border:1px solid #777777; padding:4px 10px; font-weight:bold; }"
            "QPushButton:hover { background:#FFFFFF; }"
            "QPushButton:pressed { background:#BBBBBB; }"
            "QPushButton:disabled { background:#555555; color:#AAAAAA; }"
        )

    def _style_combo(self, cb: Optional[QComboBox] = None):
        if cb is None:
            return
        cb.setStyleSheet(
            "QComboBox { background:#101010; color:#FFFFFF; border:1px solid #666666; padding:3px; }"
            "QComboBox QAbstractItemView { background:#101010; color:#FFFFFF; selection-background-color:#3399FF; }"
        )

    def _init_video_label(self, lab: QLabel, text: str):
        lab.setAlignment(Qt.AlignCenter)
        lab.setText(text)
        lab.setStyleSheet("background:#202020; color:#F0F0F0; border:1px solid #555555; font-size:14pt;")
        lab.setScaledContents(False)
        lab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lab.setMinimumSize(640, 420)

    # ---------- 日志 ----------
    def log(self, s: str):
        self.text_log.append(s)
        self.text_log.moveCursor(self.text_log.textCursor().End)

    # ---------- 模式联动 ----------
    def on_mode_changed(self):
        m = self.combo_mode.currentText()
        is_file = (m == "本地视频")
        is_cam = (m == "本机摄像头")
        is_rtsp = (m == "RTSP 探测")

        self.line_file.setEnabled(is_file)
        self.btn_browse.setEnabled(is_file)
        self.chk_loop.setEnabled(is_file)

        self.line_cam.setEnabled(is_cam)

        self.line_ip.setEnabled(is_rtsp)
        self.line_user.setEnabled(is_rtsp)
        self.line_pwds.setEnabled(is_rtsp)

    # ---------- YOLO 开关 ----------
    def on_toggle_yolo(self):
        if self.worker:
            self.worker.set_yolo_enabled(self.chk_enable_yolo.isChecked())

    # ---------- 权重 ----------
    def on_select_weight(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO 权重文件", "", "PyTorch 权重 (*.pt *.pth);;所有文件 (*.*)")
        if path:
            self.model_path = path
            self.lab_model.setText("权重：本地")
            self.log(f"已选择本地权重：{path}")

    def on_reset_weight(self):
        self.model_path = None
        self.lab_model.setText(f"权重：默认（{DEFAULT_YOLO_WEIGHTS}）")
        self.log("已恢复默认官方权重")

    # ---------- 选择本地视频 ----------
    def on_browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择本地视频文件", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.flv *.wmv);;All Files (*.*)")
        if path:
            self.line_file.setText(path)
            self.log(f"选择本地视频：{path}")

    # ---------- 启动/停止 ----------
    def on_start(self):
        if self.worker is not None:
            QMessageBox.information(self, "提示", "已经在运行中")
            return

        try:
            ui_fps = int(self.line_ui_fps.text().strip())
            frame_max_w = int(self.line_frame_w.text().strip())
            infer_w = int(self.line_infer_w.text().strip())
            infer_n = int(self.line_infer_n.text().strip())
        except Exception:
            QMessageBox.warning(self, "提示", "UI FPS/帧宽/推理参数需为整数")
            return

        enable_yolo = self.chk_enable_yolo.isChecked()

        m = self.combo_mode.currentText()
        if m == "本地视频":
            file_path = self.line_file.text().strip()
            if not file_path:
                QMessageBox.warning(self, "提示", "请先选择本地视频文件")
                return
            cfg = RunConfig(
                mode="file",
                file_path=file_path,
                loop_file=self.chk_loop.isChecked(),
                cam_index=0,
                ip="", user="", pwds=None,
                max_ui_fps=ui_fps,
                frame_max_width=frame_max_w,
                enable_yolo=enable_yolo,
                model_path=self.model_path,
                infer_width=infer_w,
                infer_every_n=infer_n
            )
        elif m == "本机摄像头":
            try:
                cam_id = int(self.line_cam.text().strip())
            except Exception:
                QMessageBox.warning(self, "提示", "摄像头ID需为整数")
                return
            cfg = RunConfig(
                mode="camera",
                cam_index=cam_id,
                max_ui_fps=ui_fps,
                frame_max_width=frame_max_w,
                enable_yolo=enable_yolo,
                model_path=self.model_path,
                infer_width=infer_w,
                infer_every_n=infer_n
            )
        else:
            ip = self.line_ip.text().strip()
            user = self.line_user.text().strip()
            pwds_str = self.line_pwds.text().strip()
            if not ip or not user:
                QMessageBox.warning(self, "提示", "请先输入 IP 和 用户名")
                return
            if pwds_str == "":
                pwds = [""]
            else:
                pwds = [p.strip() for p in pwds_str.split(",")]
                if "" not in pwds:
                    pwds.insert(0, "")
            cfg = RunConfig(
                mode="rtsp",
                ip=ip, user=user, pwds=pwds,
                max_ui_fps=ui_fps,
                frame_max_width=frame_max_w,
                enable_yolo=enable_yolo,
                model_path=self.model_path,
                infer_width=infer_w,
                infer_every_n=infer_n
            )

        self.log("=== 启动线程 ===")
        self.worker = CameraWorker(cfg)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.status_msg.connect(self.log)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def on_stop(self):
        if self.worker:
            self.log("请求停止线程...")
            self.worker.stop()
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)

    def on_worker_error(self, msg: str):
        self.log(msg)
        QMessageBox.critical(self, "错误", msg)

    def on_worker_finished(self):
        self.log("线程结束")
        self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ---------- 画面接收 ----------
    def on_frame_ready(self, pack):
        raw_bgr, draw_bgr = pack
        self.last_raw_bgr = raw_bgr
        self.last_draw_bgr = draw_bgr
        self.refresh_display()

    # ---------- 单画面显示（原始/矫正切换） ----------
    def refresh_display(self):
        mode = self.combo_view.currentText()
        self.lab_title.setText(mode)

        if self.last_raw_bgr is None:
            return

        if mode.startswith("原始"):
            img = self.last_draw_bgr if self.last_draw_bgr is not None else self.last_raw_bgr
            self._show_on_label(img, self.lab_video)
            return

        ud = self._undistort(self.last_raw_bgr)
        if ud is None:
            self.lab_video.setText("矫正失败（未加载参数或参数无效）")
            return
        ud = self._overlay_calib_hud(ud)
        self._show_on_label(ud, self.lab_video)

    def _show_on_label(self, bgr: np.ndarray, lab: QLabel):
        if bgr is None or bgr.size == 0:
            return
        w = max(2, lab.width())
        h = max(2, lab.height())
        canvas = letterbox_to_size(bgr, w, h)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        hh, ww, ch = rgb.shape
        qimg = QImage(rgb.data, ww, hh, ch * ww, QImage.Format_RGB888).copy()
        lab.setPixmap(QPixmap.fromImage(qimg))

    # =================================================================
    # 去畸变
    # =================================================================
    def _reset_undist_maps(self):
        self._map1 = None
        self._map2 = None
        self._map_size = None
        self._roi = None
        self._map_fill = None
        self.refresh_display()

    def _ensure_undist_maps(self, w: int, h: int):
        if self.calib_K is None or self.calib_dist is None:
            return False
        fill = bool(self.chk_undist_fill.isChecked())
        if self._map1 is not None and self._map2 is not None and self._map_size == (w, h) and self._map_fill == fill:
            return True

        K = self.calib_K
        dist = self.calib_dist
        alpha = 0.0 if fill else 1.0
        newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)

        self._map1, self._map2 = map1, map2
        self._map_size = (w, h)
        self._roi = roi
        self._map_fill = fill
        return True

    def _undistort(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.calib_K is None or self.calib_dist is None:
            return None
        h, w = bgr.shape[:2]
        if not self._ensure_undist_maps(w, h):
            return None
        out = cv2.remap(bgr, self._map1, self._map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if self.chk_undist_fill.isChecked() and self._roi is not None:
            x, y, ww, hh = self._roi
            if ww > 10 and hh > 10:
                out = out[y:y + hh, x:x + ww]
        return out

    # =================================================================
    # HUD（矫正模式）
    # =================================================================
    def _fmt_kdist_short(self) -> Tuple[str, str]:
        if self.calib_K is None or self.calib_dist is None:
            return "K: N/A", "dist: N/A"
        K = self.calib_K
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        k_line = f"K fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"

        d = self.calib_dist.reshape(-1).astype(float)

        def get(i):
            return float(d[i]) if i < len(d) else 0.0

        dist_line = f"dist k1={get(0):.4f} k2={get(1):.4f} p1={get(2):.4f} p2={get(3):.4f} k3={get(4):.4f}"
        return k_line, dist_line

    def _overlay_calib_hud(self, bgr: np.ndarray) -> np.ndarray:
        if bgr is None or bgr.size == 0:
            return bgr
        if self.calib_K is None or self.calib_dist is None:
            return bgr

        rms = self.calib_rms
        rms_str = f"{rms:.6f}" if isinstance(rms, (float, int)) else "N/A"
        frames = int(self.calib_frames_used) if self.calib_frames_used > 0 else len(self.calib_imgpoints)

        k_line, dist_line = self._fmt_kdist_short()
        lines = [f"CALIB  RMS={rms_str}  frames_used={frames}", k_line, dist_line]

        h, w = bgr.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55 if min(w, h) >= 900 else 0.50
        thick = 1

        pad = 8
        line_h = int(22 * scale + 10)
        max_tw = 0
        for s in lines:
            (tw, _th), _ = cv2.getTextSize(s, font, scale, thick)
            max_tw = max(max_tw, tw)
        box_w = max_tw + pad * 2
        box_h = len(lines) * line_h + pad * 2

        x0, y0 = 12, 12
        x1, y1 = min(w - 1, x0 + box_w), min(h - 1, y0 + box_h)

        overlay = bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        bgr = cv2.addWeighted(overlay, 0.45, bgr, 0.55, 0)

        y = y0 + pad + int(line_h * 0.8)
        for i, s in enumerate(lines):
            color = (200, 255, 200) if i == 0 else (220, 220, 220)
            cv2.putText(bgr, s, (x0 + pad, y), font, scale, color, thick, cv2.LINE_AA)
            y += line_h
        return bgr

    # =================================================================
    # 标定：采集 / 清空 / 计算（两阶段） / 自动采集（带弹窗）
    # =================================================================
    def _get_pattern_and_square(self) -> Tuple[Tuple[int, int], float]:
        pattern = parse_pattern(self.line_board.text())
        square_mm = float(self.line_square.text().strip())
        if square_mm <= 0:
            raise ValueError("格子边长(mm) 必须 > 0")
        return pattern, square_mm

    def on_capture_frame(self):
        if self.last_raw_bgr is None:
            QMessageBox.warning(self, "提示", "当前没有画面")
            return

        try:
            (cols, rows), square_mm = self._get_pattern_and_square()
        except Exception as e:
            QMessageBox.warning(self, "提示", f"参数错误：{e}")
            return

        frame = self.last_raw_bgr.copy()
        strong = self.chk_strong.isChecked()
        found, corners = detect_chessboard_robust(frame, (cols, rows), strong=strong)
        if not found or corners is None:
            self.log("采集失败：未检测到棋盘角点（请确保棋盘完整可见）。")
            return

        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

        self.calib_objpoints.append(objp)
        self.calib_imgpoints.append(corners)
        self.calib_frames_used = len(self.calib_imgpoints)

        self.log(f"✅ 采集成功：frames_used={self.calib_frames_used}")

        if self.chk_show_corners.isChecked():
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, (cols, rows), corners, True)
            cv2.putText(vis, f"Manual capture OK | frames_used={self.calib_frames_used}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            self.last_draw_bgr = vis
            self.refresh_display()

    def on_clear_calib(self):
        self.calib_objpoints.clear()
        self.calib_imgpoints.clear()
        self.calib_K = None
        self.calib_dist = None
        self.calib_rms = None
        self.calib_frames_used = 0
        self._reset_undist_maps()
        self.lab_state.setText("未标定")
        self.log("已清空标定数据。")

    def on_compute_calib_two_stage(self):
        if self.last_raw_bgr is None:
            QMessageBox.warning(self, "提示", "没有画面（无法获取图像尺寸）")
            return
        if len(self.calib_imgpoints) < 8:
            QMessageBox.warning(self, "提示", f"有效采集帧太少：{len(self.calib_imgpoints)}（建议 >= 8，最好 12~20）")
            return

        try:
            h, w = self.last_raw_bgr.shape[:2]
            image_size = (w, h)
            self.log("开始两阶段标定：Stage-1...")
            rms1, K1, dist1 = self._calibrate_once(self.calib_objpoints, self.calib_imgpoints, image_size)
            self.log(f"Stage-1 RMS={rms1:.6f}")

            errs = self._per_view_reproj_errors(self.calib_objpoints, self.calib_imgpoints, K1, dist1)
            n = len(errs)
            order = np.argsort(errs)

            drop = max(1, int(round(n * 0.15)))
            keep = n - drop
            if keep < 8:
                drop = max(0, n - 8)
                keep = n - drop

            keep_idx = order[:keep].tolist()

            if drop > 0:
                self.log(f"Stage-1 误差统计：min={float(np.min(errs)):.3f}px  mean={float(np.mean(errs)):.3f}px  max={float(np.max(errs)):.3f}px")
                self.log(f"剔除离群帧：{drop} / {n}（按重投影误差 top {int(drop/n*100)}%）")
                self.log("开始 Stage-2（剔除离群后再标定）...")

                obj2 = [self.calib_objpoints[i] for i in keep_idx]
                img2 = [self.calib_imgpoints[i] for i in keep_idx]
                rms2, K2, dist2 = self._calibrate_once(obj2, img2, image_size)
                self.log(f"Stage-2 RMS={rms2:.6f}（对比 Stage-1: {rms1:.6f}）")

                self.calib_rms = float(rms2)
                self.calib_K = K2
                self.calib_dist = dist2
                self.calib_frames_used = len(img2)
                self.calib_objpoints = obj2
                self.calib_imgpoints = img2
            else:
                self.log("样本较少，跳过离群剔除，直接采用 Stage-1 结果。")
                self.calib_rms = float(rms1)
                self.calib_K = K1
                self.calib_dist = dist1
                self.calib_frames_used = len(self.calib_imgpoints)

            self._reset_undist_maps()
            self.lab_state.setText(f"已标定 RMS={self.calib_rms:.6f} frames={self.calib_frames_used}")
            self.log(f"✅ 标定完成：RMS={self.calib_rms:.6f} | frames_used={self.calib_frames_used}")
            self.refresh_display()

        except Exception as e:
            self.log(f"标定失败：{e}")
            QMessageBox.critical(self, "标定失败", str(e))

    def _calibrate_once(self, objpoints, imgpoints, image_size):
        rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
        return float(rms), K, dist

    def _per_view_reproj_errors(self, objpoints, imgpoints, K, dist) -> np.ndarray:
        errs = []
        for objp, imgp in zip(objpoints, imgpoints):
            ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                errs.append(1e9)
                continue
            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
            proj = proj.reshape(-1, 2)
            obs = imgp.reshape(-1, 2)
            e = np.linalg.norm(proj - obs, axis=1).mean()
            errs.append(float(e))
        return np.array(errs, dtype=np.float32)

    # ✅ 自动采集（带标定过程可视化弹窗）
    def on_auto_collect(self):
        video_path = self.line_file.text().strip()
        if not video_path:
            QMessageBox.warning(self, "提示", "自动采集需要先选择“本地视频文件”")
            return
        if not os.path.isfile(video_path):
            QMessageBox.warning(self, "提示", "视频文件不存在")
            return

        try:
            (cols, rows), square_mm = self._get_pattern_and_square()
            every_n = int(self.line_auto_every.text().strip())
            max_frames = int(self.line_auto_max.text().strip())
            frame_max_w = int(self.line_frame_w.text().strip())
        except Exception as e:
            QMessageBox.warning(self, "提示", f"参数错误：{e}")
            return

        if self.auto_collect_thread is not None and self.auto_collect_thread.isRunning():
            QMessageBox.information(self, "提示", "自动采集正在进行中")
            return

        # 打开可视化弹窗
        if self.viz is None:
            self.viz = CalibVizDialog(self)
        self.viz.show()
        self.viz.raise_()
        self.viz.activateWindow()
        self.viz.append("开始自动采集（实时可视化）...")

        self.log("开始自动采集（后台线程，增强版+可视化）...")
        self.auto_collect_thread = AutoCollectWorker(
            video_path=video_path,
            pattern=(cols, rows),
            square_mm=square_mm,
            every_n=every_n,
            max_frames=max_frames,
            strong=self.chk_strong.isChecked(),
            frame_max_width=frame_max_w,
            dedup_px=15.0,
            min_blur_var=60.0,
            min_board_area_ratio=0.02,
            cover_grid=(6, 4)
        )
        self.auto_collect_thread.log.connect(self.log)
        self.auto_collect_thread.preview.connect(self._on_auto_preview)
        self.auto_collect_thread.done.connect(self._on_auto_collect_done)
        self.auto_collect_thread.fail.connect(self._on_auto_collect_fail)
        self.auto_collect_thread.start()

    def _on_auto_preview(self, bgr_vis, info: str, got: int, total: int):
        if self.viz is not None and self.viz.isVisible():
            self.viz.show_bgr(bgr_vis)
            self.viz.append(info)
            self.viz.set_progress(got, total)

    def _on_auto_collect_done(self, objpoints, imgpoints, image_size):
        self.calib_objpoints = list(objpoints)
        self.calib_imgpoints = list(imgpoints)
        self.calib_frames_used = len(self.calib_imgpoints)
        self.log(f"自动采集完成：frames_used={self.calib_frames_used}，准备两阶段标定...")

        if self.viz is not None and self.viz.isVisible():
            self.viz.append("采集完成，开始两阶段标定...")
            self.viz.set_progress(self.calib_frames_used, self.calib_frames_used)

        if self.last_raw_bgr is None:
            try:
                self.last_raw_bgr = np.zeros((int(image_size[1]), int(image_size[0]), 3), np.uint8)
            except Exception:
                pass

        self.on_compute_calib_two_stage()

        if self.viz is not None and self.viz.isVisible():
            self.viz.append("两阶段标定结束（你可以切到“矫正模式”查看效果）。")

    def _on_auto_collect_fail(self, msg: str):
        self.log(msg)
        if self.viz is not None and self.viz.isVisible():
            self.viz.append("❌ 自动采集失败：")
            self.viz.append(msg)
        QMessageBox.critical(self, "自动采集失败", msg)

    # =================================================================
    # 下载示例数据集
    # =================================================================
    def on_download_dataset(self):
        if self.ds_thread is not None and self.ds_thread.isRunning():
            QMessageBox.information(self, "提示", "正在下载中...")
            return
        out_dir = os.path.join(os.getcwd(), "datasets")
        self.log(f"准备下载到：{out_dir}")

        self.ds_thread = DatasetDownloadWorker(out_dir)
        self.ds_thread.log.connect(self.log)
        self.ds_thread.done.connect(self._on_dataset_ready)
        self.ds_thread.fail.connect(self._on_dataset_fail)
        self.ds_thread.start()

    def _on_dataset_ready(self, avi_path: str):
        self.log("示例数据集下载完成，已自动填入本地视频。")
        self.combo_mode.setCurrentText("本地视频")
        self.line_file.setText(avi_path)
        QMessageBox.information(self, "完成", f"示例视频已就绪：\n{avi_path}\n现在可直接点“自动采集(视频)+可视化”。")

    def _on_dataset_fail(self, msg: str):
        self.log(msg)
        QMessageBox.critical(self, "下载失败", msg)

    # =================================================================
    # 保存/加载参数（✅ 修复 FileStorage Emitter 报错）
    # =================================================================
    def on_save_calib(self):
        if self.calib_K is None or self.calib_dist is None:
            QMessageBox.warning(self, "提示", "当前没有标定参数可保存")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "保存相机参数", "camera_calib.yaml",
            "YAML Files (*.yaml *.yml)"
        )
        if not path:
            return

        try:
            if not (path.lower().endswith(".yaml") or path.lower().endswith(".yml")):
                path += ".yaml"

            folder = os.path.dirname(os.path.abspath(path))
            os.makedirs(folder, exist_ok=True)

            fs = cv2.FileStorage()
            ok = fs.open(path, cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_FORMAT_YAML)
            if not ok or not fs.isOpened():
                fs.release()
                raise RuntimeError(
                    "cv2.FileStorage 打开失败（Emitter不可用）。\n"
                    "请尝试：\n"
                    "1) 换到纯英文路径（如 D:\\calib\\camera_calib.yaml）\n"
                    "2) 确保目录存在且有写权限\n"
                    "3) 不要保存在 OneDrive/受保护目录"
                )

            fs.write("rms", float(self.calib_rms) if self.calib_rms is not None else -1.0)
            fs.write("frames_used", int(self.calib_frames_used))
            fs.write("camera_matrix", self.calib_K)
            fs.write("dist_coeffs", self.calib_dist)

            try:
                cols, rows = parse_pattern(self.line_board.text())
                fs.write("pattern_cols", int(cols))
                fs.write("pattern_rows", int(rows))
            except Exception:
                pass
            try:
                fs.write("square_mm", float(self.line_square.text().strip()))
            except Exception:
                pass

            fs.release()
            self.log(f"✅ 已保存参数：{path}")

        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def on_load_calib(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载相机参数", "", "YAML Files (*.yaml *.yml)")
        if not path:
            return
        try:
            fs = cv2.FileStorage()
            ok = fs.open(path, cv2.FILE_STORAGE_READ)
            if not ok or not fs.isOpened():
                fs.release()
                raise RuntimeError("cv2.FileStorage 打开失败：请确认文件存在且路径可读（尽量用纯英文路径）")

            K = fs.getNode("camera_matrix").mat()
            dist = fs.getNode("dist_coeffs").mat()

            rms_node = fs.getNode("rms")
            rms = float(rms_node.real()) if not rms_node.empty() else None

            fu_node = fs.getNode("frames_used")
            frames_used = int(fu_node.real()) if not fu_node.empty() else 0

            pc = fs.getNode("pattern_cols")
            pr = fs.getNode("pattern_rows")
            sm = fs.getNode("square_mm")

            fs.release()

            if K is None or dist is None or K.size == 0 or dist.size == 0:
                raise ValueError("YAML 中缺少 camera_matrix 或 dist_coeffs")

            if (not pc.empty()) and (not pr.empty()):
                self.line_board.setText(f"{int(pc.real())}x{int(pr.real())}")
            if not sm.empty():
                self.line_square.setText(f"{float(sm.real()):.2f}")

            self.calib_K = K
            self.calib_dist = dist
            self.calib_rms = rms
            self.calib_frames_used = frames_used

            self._reset_undist_maps()
            rtxt = f"{rms:.6f}" if isinstance(rms, (float, int)) else "N/A"
            self.lab_state.setText(f"已加载 RMS={rtxt} frames={frames_used if frames_used>0 else 'N/A'}")
            self.log(f"✅ 已加载参数：{path}")
            self.refresh_display()

        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    # =================================================================
    # A4棋盘PDF生成（铺满A4并回填square）
    # =================================================================
    def on_make_a4(self):
        try:
            cols, rows = parse_pattern(self.line_board.text())
            margin_mm = float(self.line_margin.text().strip())
            landscape = self.chk_land.isChecked()
            labels = self.chk_pdf_labels.isChecked()
        except Exception as e:
            QMessageBox.warning(self, "提示", f"参数错误：{e}")
            return

        path, _ = QFileDialog.getSaveFileName(self, "保存A4棋盘PDF", "chessboard_A4.pdf", "PDF Files (*.pdf)")
        if not path:
            return

        try:
            square_mm = draw_a4_chessboard_pdf(
                out_pdf=path,
                inner_cols=cols,
                inner_rows=rows,
                margin_mm=margin_mm,
                add_labels=labels,
                landscape=landscape,
                round_to_mm=0.1
            )
            self.line_square.setText(f"{square_mm:.2f}")
            self.log(f"已生成A4棋盘：{path}")
            self.log(f"自动适配得到 square(mm) = {square_mm:.2f}（已回填）")
            QMessageBox.information(self, "完成", f"已生成：{path}\nsquare(mm) = {square_mm:.2f}\n请用“实际大小/100%”打印，禁用缩放。")
        except Exception as e:
            QMessageBox.critical(self, "生成失败", str(e))

    # =================================================================
    # 关闭事件
    # =================================================================
    def closeEvent(self, event):
        try:
            if self.worker:
                self.worker.stop()
                self.worker.wait(1000)
        except Exception:
            pass
        try:
            if self.auto_collect_thread and self.auto_collect_thread.isRunning():
                self.auto_collect_thread.terminate()
        except Exception:
            pass
        super().closeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("相机上位机（左上视频/左下日志/右控件更宽，标定增强+两阶段+可视化）")
        self.resize(1700, 900)
        self.setCentralWidget(CameraPage(self))


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", True)
    except Exception:
        pass
    main()
