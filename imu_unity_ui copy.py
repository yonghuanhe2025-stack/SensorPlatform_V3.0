#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组合导航融合数据上位机（PyQt5 版本 / 不使用 PySide6 / 已去掉视频回放）

本版改动（按你截图问题）：
1) 右侧“仪表面板”改为更窄：使用 QSplitter 水平分割，右侧默认宽度更小、可拖拽调节
2) 右侧仪表控件最小尺寸下调，避免“面板太宽导致其它区域显示不全”
3) 下拉框/弹出列表文字看不清：补齐 QComboBox 的 view 与 QMenu 的样式（高对比度）

依赖（建议）：
  pip install PyQt5 PyQtWebEngine pyqtgraph numpy pyserial
"""

import os
import sys
import time
import math
import struct
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from collections import deque

import numpy as np

# ----------------------------
# 串口（可选）
# ----------------------------
try:
    import serial
    from serial.tools import list_ports
    HAS_SERIAL = True
except Exception:
    serial = None
    list_ports = None
    HAS_SERIAL = False

# ----------------------------
# PyQt5 UI
# ----------------------------
from PyQt5.QtCore import (
    Qt, QObject, QThread, QTimer, QPointF, QRectF, QCoreApplication, pyqtSignal
)
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QFont, QPolygonF, QPainterPath, QGuiApplication, QFontMetricsF
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QGroupBox, QFormLayout,
    QCheckBox, QDoubleSpinBox, QMessageBox, QSizePolicy, QTabWidget,
    QFileDialog, QSplitter
)

Signal = pyqtSignal

import pyqtgraph as pg

# ----------------------------
# QtWebEngine（可选）
# ----------------------------
HAS_WEBENGINE = False
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    try:
        from PyQt5.QtWebEngineWidgets import QWebEngineProfile, QWebEngineSettings
    except Exception:
        from PyQt5.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
    HAS_WEBENGINE = True
except Exception:
    QWebEngineView = None
    QWebEngineProfile = None
    QWebEngineSettings = None
    HAS_WEBENGINE = False


# ============================================================
# 1) 协议（融合输出帧）— 65 bytes: HEADER(4) + payload + CRC8(1)
# ============================================================

HEADER = b"\xAA\x44\xAA\x45"
FRAME_LEN = 65
CRC_OFFSET = 64
CRC_COVER_LEN = 64
CRC8_POLY = 0x07


def crc8_poly07(data: bytes, init: int = 0x00, xorout: int = 0x00) -> int:
    crc = init & 0xFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) & 0xFF) ^ CRC8_POLY
            else:
                crc = (crc << 1) & 0xFF
    return (crc ^ xorout) & 0xFF


def _s16_le(b: bytes, off: int) -> int:
    return struct.unpack_from("<h", b, off)[0]


def _u16_le(b: bytes, off: int) -> int:
    return struct.unpack_from("<H", b, off)[0]


def _s32_le(b: bytes, off: int) -> int:
    return struct.unpack_from("<i", b, off)[0]


def _u32_le(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]


@dataclass
class FusionFrame:
    lat_deg: float
    lon_deg: float
    h_m: float

    ve_mps: float
    vn_mps: float
    vu_mps: float

    pitch_deg: float
    roll_deg: float
    heading_deg: float

    ins_status: int
    gnss_week: int
    gnss_tow_s: float
    pos_type: int
    sat_main: int
    sat_sub: int

    ax_g: float
    ay_g: float
    az_g: float

    gx_dps: float
    gy_dps: float
    gz_dps: float

    temp_c: float

    hpos_m: float
    hgt_m: float
    hvel_mps: float
    vvel_mps: float
    att_deg: float
    hdg_deg: float

    t_perf: float


def parse_fusion_frame(frame: bytes) -> FusionFrame:
    """
    注意：字段布局/比例请与你的“组合导航串口融合输出协议”一致。
    若你的协议偏移不同，只需要修改本函数的 offset 和 scale。
    """
    if len(frame) != FRAME_LEN:
        raise ValueError(f"bad frame length={len(frame)}")
    if frame[:4] != HEADER:
        raise ValueError("bad header")

    c_calc = crc8_poly07(frame[:CRC_COVER_LEN])
    c_recv = frame[CRC_OFFSET]
    if c_calc != c_recv:
        raise ValueError(f"crc mismatch calc=0x{c_calc:02X} recv=0x{c_recv:02X}")

    lat = _s32_le(frame, 4) * 1e-7
    lon = _s32_le(frame, 8) * 1e-7
    h = _s32_le(frame, 12) * 1e-3

    vel_scale = 1e2 / (2**15)
    ve = _s16_le(frame, 16) * vel_scale
    vn = _s16_le(frame, 18) * vel_scale
    vu = _s16_le(frame, 20) * vel_scale

    ang_scale = 180.0 / (2**15)
    pitch = _s16_le(frame, 22) * ang_scale
    roll = _s16_le(frame, 24) * ang_scale
    heading = _s16_le(frame, 26) * ang_scale

    ins_status = frame[28]
    gnss_week = _u16_le(frame, 29)
    gnss_tow_s = _u32_le(frame, 31) * 1e-3
    pos_type = frame[35]
    sat_main = frame[36]
    sat_sub = frame[37]

    acc_scale = 8.0 / (2**15)
    ax = _s16_le(frame, 38) * acc_scale
    ay = _s16_le(frame, 40) * acc_scale
    az = _s16_le(frame, 42) * acc_scale

    gyr_scale = 250.0 / (2**15)
    gx = _s16_le(frame, 44) * gyr_scale
    gy = _s16_le(frame, 46) * gyr_scale
    gz = _s16_le(frame, 48) * gyr_scale

    tmp_scale = 150.0 / (2**15)
    temp_c = _s16_le(frame, 50) * tmp_scale

    sig_scale = 100.0 / (2**16)
    hpos = _u16_le(frame, 52) * sig_scale
    hgt = _u16_le(frame, 54) * sig_scale
    hvel = _u16_le(frame, 56) * sig_scale
    vvel = _u16_le(frame, 58) * sig_scale
    att = _u16_le(frame, 60) * sig_scale
    hdg = _u16_le(frame, 62) * sig_scale

    return FusionFrame(
        lat_deg=lat, lon_deg=lon, h_m=h,
        ve_mps=ve, vn_mps=vn, vu_mps=vu,
        pitch_deg=pitch, roll_deg=roll, heading_deg=heading,
        ins_status=ins_status, gnss_week=gnss_week, gnss_tow_s=gnss_tow_s,
        pos_type=pos_type, sat_main=sat_main, sat_sub=sat_sub,
        ax_g=ax, ay_g=ay, az_g=az,
        gx_dps=gx, gy_dps=gy, gz_dps=gz,
        temp_c=temp_c,
        hpos_m=hpos, hgt_m=hgt, hvel_mps=hvel, vvel_mps=vvel,
        att_deg=att, hdg_deg=hdg,
        t_perf=time.perf_counter()
    )


# ============================================================
# 2) 流提取：头+CRC 重同步
# ============================================================

class StreamFrameExtractor:
    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes):
        if data:
            self.buf.extend(data)

    def pop_frames(self, max_frames: int = 500) -> List[bytes]:
        out = []
        for _ in range(max_frames):
            idx = self.buf.find(HEADER)
            if idx < 0:
                if len(self.buf) > 3:
                    self.buf = self.buf[-3:]
                break
            if idx > 0:
                del self.buf[:idx]
            if len(self.buf) < FRAME_LEN:
                break

            cand = bytes(self.buf[:FRAME_LEN])
            if crc8_poly07(cand[:CRC_COVER_LEN]) == cand[CRC_OFFSET]:
                out.append(cand)
                del self.buf[:FRAME_LEN]
            else:
                del self.buf[:1]
        return out


# ============================================================
# 3) One Euro Filter
# ============================================================

def _alpha(dt: float, cutoff: float) -> float:
    if cutoff <= 0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x: float, t: float) -> float:
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = float(x)
            self.dx_prev = 0.0
            return float(x)

        dt = max(t - self.t_prev, 1e-6)
        self.t_prev = t

        x = float(x)
        dx = (x - self.x_prev) / dt
        a_d = _alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(dt, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat


# ============================================================
# 4) 原始帧录制（.fusr）与回放
# ============================================================

FUSR_MAGIC = b"FUSR"
FUSR_VER = 1
FUSR_HDR_FMT = "<4sHHd"
FUSR_HDR_SIZE = struct.calcsize(FUSR_HDR_FMT)
FUSR_REC_FMT = "<d65s"
FUSR_REC_SIZE = struct.calcsize(FUSR_REC_FMT)


class RawDataRecorder:
    def __init__(self):
        self.fp = None
        self.path = ""
        self.t0_perf: Optional[float] = None
        self.epoch = 0.0
        self.n = 0

    def is_open(self) -> bool:
        return self.fp is not None

    def start(self, path: str):
        self.stop()
        self.path = path
        self.fp = open(path, "wb")
        self.t0_perf = None
        self.epoch = time.time()
        self.n = 0
        hdr = struct.pack(FUSR_HDR_FMT, FUSR_MAGIC, FUSR_VER, 0, self.epoch)
        self.fp.write(hdr)
        self.fp.flush()

    def write_frame(self, raw65: bytes, t_perf: float):
        if not self.fp:
            return
        if len(raw65) != FRAME_LEN:
            return
        if self.t0_perf is None:
            self.t0_perf = t_perf
        t_rel = float(t_perf - self.t0_perf)
        rec = struct.pack(FUSR_REC_FMT, t_rel, raw65)
        self.fp.write(rec)
        self.n += 1
        if (self.n % 200) == 0:
            self.fp.flush()

    def stop(self):
        try:
            if self.fp:
                self.fp.flush()
                self.fp.close()
        except Exception:
            pass
        self.fp = None
        self.path = ""
        self.t0_perf = None
        self.epoch = 0.0
        self.n = 0

    @staticmethod
    def read_header(path: str) -> Tuple[int, float]:
        with open(path, "rb") as f:
            b = f.read(FUSR_HDR_SIZE)
            if len(b) != FUSR_HDR_SIZE:
                raise RuntimeError("文件过短")
            magic, ver, _, epoch = struct.unpack(FUSR_HDR_FMT, b)
            if magic != FUSR_MAGIC:
                raise RuntimeError("不是有效的 .fusr 文件")
            return int(ver), float(epoch)


# ============================================================
# 5) 串口线程（批量发帧，降低 Qt 信号频率）
# ============================================================

class SerialWorker(QObject):
    frames = Signal(object)         # List[FusionFrame]
    status = Signal(str)
    stats = Signal(int, int)
    raw_count = Signal(int)

    def __init__(self):
        super().__init__()
        self.port = ""
        self.baud = 460800
        self.timeout = 0.2
        self._stop = False
        self._ser = None
        self._ext = StreamFrameExtractor()
        self.ok = 0
        self.bad = 0
        self.recorder: Optional[RawDataRecorder] = None
        self.batch_max = 30

    def open(self) -> bool:
        if not HAS_SERIAL:
            self.status.emit("未安装 pyserial：pip install pyserial")
            return False
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            self.status.emit(f"串口已打开：{self.port} @ {self.baud}")
            return True
        except Exception as e:
            self._ser = None
            self.status.emit(f"打开串口失败：{e}")
            return False

    def request_stop(self):
        self._stop = True

    def close(self):
        try:
            if self._ser and getattr(self._ser, "is_open", False):
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    def run(self):
        self._stop = False
        if not self._ser or not getattr(self._ser, "is_open", False):
            self.status.emit("串口未打开，线程退出")
            return

        self.status.emit("开始接收融合数据…")
        last_stat = time.perf_counter()

        while not self._stop:
            try:
                data = self._ser.read(8192)
                if data:
                    self._ext.feed(data)
                    raws = self._ext.pop_frames(max_frames=800)
                    if not raws:
                        continue

                    batch: List[FusionFrame] = []
                    tnow = time.perf_counter()

                    for raw in raws:
                        if self.recorder and self.recorder.is_open():
                            self.recorder.write_frame(raw, tnow)
                            if (self.recorder.n % 50) == 0:
                                self.raw_count.emit(self.recorder.n)

                        try:
                            fr = parse_fusion_frame(raw)
                            self.ok += 1
                            batch.append(fr)
                            if len(batch) >= self.batch_max:
                                self.frames.emit(batch)
                                batch = []
                        except Exception:
                            self.bad += 1

                    if batch:
                        self.frames.emit(batch)

                now = time.perf_counter()
                if now - last_stat > 1.0:
                    self.stats.emit(self.ok, self.bad)
                    last_stat = now

            except Exception as e:
                self.status.emit(f"串口读取异常：{e}")
                time.sleep(0.05)

        self.close()
        self.status.emit("接收线程已停止")


# ============================================================
# 6) 数据回放线程（.fusr 按时间戳回放）
# ============================================================

class DataReplayWorker(QObject):
    frames = Signal(object)  # List[FusionFrame]
    status = Signal(str)
    progress = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.path = ""
        self.speed = 1.0
        self._stop = False
        self._pause = False
        self.batch_max = 30

    def request_stop(self):
        self._stop = True

    def set_pause(self, pause: bool):
        self._pause = bool(pause)

    def run(self):
        self._stop = False
        self._pause = False

        if not self.path:
            self.status.emit("回放失败：未选择数据文件")
            return

        try:
            ver, _ = RawDataRecorder.read_header(self.path)
            if ver != FUSR_VER:
                self.status.emit(f"提示：文件版本={ver}，当前支持={FUSR_VER}（将尝试继续）")
        except Exception as e:
            self.status.emit(f"回放失败：{e}")
            return

        try:
            with open(self.path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size < FUSR_HDR_SIZE:
                    raise RuntimeError("文件过短")
                total = (size - FUSR_HDR_SIZE) // FUSR_REC_SIZE
                f.seek(FUSR_HDR_SIZE, os.SEEK_SET)

                self.status.emit(f"开始回放数据：{os.path.basename(self.path)}（{total} 帧）x{self.speed:g}")
                last_t_rel = None

                batch: List[FusionFrame] = []
                for i in range(int(total)):
                    if self._stop:
                        break
                    while self._pause and (not self._stop):
                        time.sleep(0.02)

                    b = f.read(FUSR_REC_SIZE)
                    if len(b) != FUSR_REC_SIZE:
                        break
                    t_rel, raw = struct.unpack(FUSR_REC_FMT, b)

                    if last_t_rel is None:
                        last_t_rel = float(t_rel)
                    else:
                        dt = float(t_rel) - float(last_t_rel)
                        last_t_rel = float(t_rel)
                        dt_play = dt / max(float(self.speed), 1e-6)
                        if dt_play > 0:
                            remain = min(dt_play, 0.8)
                            while remain > 0 and (not self._stop):
                                while self._pause and (not self._stop):
                                    time.sleep(0.02)
                                s = min(0.05, remain)
                                time.sleep(s)
                                remain -= s

                    try:
                        fr = parse_fusion_frame(raw)
                        fr.t_perf = time.perf_counter()
                        batch.append(fr)
                        if len(batch) >= self.batch_max:
                            self.frames.emit(batch)
                            batch = []
                    except Exception:
                        pass

                    if (i % 80) == 0:
                        self.progress.emit(i + 1, int(total))

                if batch:
                    self.frames.emit(batch)

        except Exception as e:
            self.status.emit(f"回放异常：{e}")
            return

        if self._stop:
            self.status.emit("数据回放已停止")
        else:
            self.status.emit("数据回放完成")


# ============================================================
# 7) 轨迹记录器（点记录/导出）
# ============================================================

class TrackRecorder:
    def __init__(self):
        self.points: List[Dict] = []

    def clear(self):
        self.points.clear()

    def append(self, rec: Dict):
        self.points.append(rec)

    def __len__(self):
        return len(self.points)

    def export_csv(self, path: str):
        import csv
        if not self.points:
            raise RuntimeError("轨迹为空，无法导出")
        keys = list(self.points[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.points:
                w.writerow(r)

    def export_gpx(self, path: str):
        if not self.points:
            raise RuntimeError("轨迹为空，无法导出")

        def esc(s: str) -> str:
            return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    .replace('"', "&quot;").replace("'", "&apos;"))

        with open(path, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<gpx version="1.1" creator="FusionUpper" xmlns="http://www.topografix.com/GPX/1/1">\n')
            f.write("  <trk><name>Fusion Track</name><trkseg>\n")
            for r in self.points:
                lat = r.get("lat_deg")
                lon = r.get("lon_deg")
                if lat is None or lon is None:
                    continue
                f.write(f'    <trkpt lat="{float(lat):.8f}" lon="{float(lon):.8f}">')
                spd = r.get("spd_kmh", 0.0)
                hdg = r.get("heading_deg", 0.0)
                f.write("<extensions>")
                f.write(f"<spd_kmh>{esc(str(spd))}</spd_kmh>")
                f.write(f"<heading_deg>{esc(str(hdg))}</heading_deg>")
                f.write("</extensions>")
                f.write("</trkpt>\n")
            f.write("  </trkseg></trk></gpx>\n")

    def export_kml(self, path: str):
        if not self.points:
            raise RuntimeError("轨迹为空，无法导出")
        coords = []
        for r in self.points:
            lat = r.get("lat_deg")
            lon = r.get("lon_deg")
            if lat is None or lon is None:
                continue
            coords.append(f"{float(lon):.8f},{float(lat):.8f},0")
        with open(path, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
            f.write("  <Document>\n")
            f.write("    <Placemark>\n")
            f.write("      <name>Fusion Track</name>\n")
            f.write("      <Style><LineStyle><color>ff00f0ff</color><width>4</width></LineStyle></Style>\n")
            f.write("      <LineString><tessellate>1</tessellate><coordinates>\n")
            f.write("\n".join(coords))
            f.write("\n      </coordinates></LineString>\n")
            f.write("    </Placemark>\n")
            f.write("  </Document>\n")
            f.write("</kml>\n")

    @staticmethod
    def load_csv(path: str) -> List[Dict]:
        import csv
        out = []
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                def to_float(x):
                    try:
                        return float(x)
                    except Exception:
                        return x
                out.append({k: to_float(v) for k, v in row.items()})
        return out


# ============================================================
# 8) 仪表控件（姿态球/航向/速度表）
# ============================================================

def _clamp(x, a, b):
    return a if x < a else b if x > b else x


class AttitudeBall(QWidget):
    def __init__(self):
        super().__init__()
        self.pitch = 0.0
        self.roll = 0.0
        # ↓ 调小最小尺寸，右侧面板可更窄
        self.setMinimumSize(190, 190)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_att(self, pitch_deg: float, roll_deg: float):
        self.pitch = float(pitch_deg)
        self.roll = float(roll_deg)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            w, h = self.width(), self.height()
            s = min(w, h)
            cx, cy = w / 2, h / 2
            r = s * 0.43

            p.setPen(QPen(QColor(0, 240, 255, 70), 10))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QRectF(cx - r - 8, cy - r - 8, 2 * (r + 8), 2 * (r + 8)))

            p.setPen(QPen(QColor(0, 240, 255, 190), 3))
            p.drawEllipse(QRectF(cx - r - 4, cy - r - 4, 2 * (r + 4), 2 * (r + 4)))

            clip = QPainterPath()
            clip.addEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
            p.save()
            p.setClipPath(clip)

            for k in range(26):
                t = k / 25.0
                rr = r * (1.0 - 0.02 * t)
                glow = int(70 * (1.0 - t))
                col = QColor(6 + glow, 10 + glow, 18 + glow)
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                p.drawEllipse(QRectF(cx - rr - 0.65 * t * r, cy - rr - 0.65 * t * r, 2 * rr, 2 * rr))

            p.translate(cx, cy)
            p.rotate(-self.roll)

            pitch = _clamp(self.pitch, -45.0, 45.0)
            px_per_deg = r / 40.0
            p.translate(0, pitch * px_per_deg)

            sky = QColor(40, 160, 255, 175)
            ground = QColor(240, 140, 60, 170)
            p.setPen(Qt.NoPen)
            p.setBrush(sky)
            p.drawRect(QRectF(-2 * r, -2 * r, 4 * r, 2 * r))
            p.setBrush(ground)
            p.drawRect(QRectF(-2 * r, 0, 4 * r, 2 * r))

            p.setPen(QPen(QColor(0, 240, 255, 70), 9))
            p.drawLine(QPointF(-1.35 * r, 0), QPointF(1.35 * r, 0))
            p.setPen(QPen(QColor(0, 240, 255, 210), 3))
            p.drawLine(QPointF(-1.35 * r, 0), QPointF(1.35 * r, 0))

            p.setPen(QPen(QColor(200, 240, 255, 150), 2))
            for deg in [-30, -20, -10, 10, 20, 30]:
                yy = -deg * px_per_deg
                L = 0.55 * r if abs(deg) == 10 else 0.78 * r
                p.drawLine(QPointF(-L, yy), QPointF(L, yy))

            p.restore()

            p.setPen(QPen(QColor(0, 240, 255, 80), 10))
            p.drawLine(QPointF(cx - 0.40 * r, cy), QPointF(cx + 0.40 * r, cy))
            p.setPen(QPen(QColor(0, 240, 255, 230), 4))
            p.drawLine(QPointF(cx - 0.38 * r, cy), QPointF(cx - 0.08 * r, cy))
            p.drawLine(QPointF(cx + 0.08 * r, cy), QPointF(cx + 0.38 * r, cy))
            p.drawLine(QPointF(cx, cy - 0.05 * r), QPointF(cx, cy + 0.12 * r))
        finally:
            p.end()


class HeadingDial(QWidget):
    def __init__(self):
        super().__init__()
        self.heading = 0.0
        # ↓ 调小最小尺寸，右侧面板可更窄
        self.setMinimumSize(190, 190)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_heading(self, heading_deg: float):
        h = float(heading_deg) % 360.0
        if h < 0:
            h += 360.0
        self.heading = h
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            w, h = self.width(), self.height()
            s = min(w, h)
            cx, cy = w / 2, h / 2
            r = s * 0.43

            p.setPen(QPen(QColor(0, 240, 255, 70), 10))
            p.setBrush(QColor(8, 12, 18))
            p.drawEllipse(QRectF(cx - r - 8, cy - r - 8, 2 * (r + 8), 2 * (r + 8)))

            p.setPen(QPen(QColor(0, 240, 255, 190), 3))
            p.setBrush(QColor(10, 14, 22))
            p.drawEllipse(QRectF(cx - r - 4, cy - r - 4, 2 * (r + 4), 2 * (r + 4)))

            p.save()
            p.translate(cx, cy)
            p.rotate(self.heading)

            for deg in range(0, 360, 5):
                major = (deg % 30 == 0)
                L = 18 if major else 10
                col = QColor(0, 240, 255, 180) if major else QColor(180, 220, 255, 110)
                p.setPen(QPen(col, 2 if major else 1))
                a = math.radians(deg)
                x1 = (r - 6) * math.sin(a)
                y1 = -(r - 6) * math.cos(a)
                x2 = (r - 6 - L) * math.sin(a)
                y2 = -(r - 6 - L) * math.cos(a)
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            p.setPen(QColor(220, 245, 255, 200))
            p.setFont(QFont("Microsoft YaHei", max(10, int(r * 0.14)), QFont.Bold))
            for txt, a0 in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
                a = math.radians(a0)
                tx = (r * 0.70) * math.sin(a)
                ty = -(r * 0.70) * math.cos(a)
                p.drawText(QRectF(tx - 20, ty - 16, 40, 32), Qt.AlignCenter, txt)

            p.restore()

            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 240, 255, 80))
            tri2 = QPolygonF([
                QPointF(cx, cy - r + 8),
                QPointF(cx - 13, cy - r + 34),
                QPointF(cx + 13, cy - r + 34),
            ])
            p.drawPolygon(tri2)

            p.setBrush(QColor(0, 240, 255, 235))
            tri = QPolygonF([
                QPointF(cx, cy - r + 12),
                QPointF(cx - 9, cy - r + 30),
                QPointF(cx + 9, cy - r + 30),
            ])
            p.drawPolygon(tri)

            p.setPen(QColor(0, 240, 255, 220))
            p.setFont(QFont("Consolas", max(10, int(r * 0.16)), QFont.Bold))
            p.drawText(QRectF(cx - 70, cy + 0.56 * r, 140, 30), Qt.AlignCenter, f"{self.heading:06.2f}")
        finally:
            p.end()


class SpeedGauge(QWidget):
    def __init__(self):
        super().__init__()
        self.kmh = 0.0
        self.vmax = 140.0
        self.redline_from = 110.0
        # ↓ 调小最小尺寸，右侧面板可更窄
        self.setMinimumSize(260, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_speed(self, kmh: float):
        self.kmh = max(0.0, float(kmh))
        self.update()

    @staticmethod
    def _polar(cx: float, cy: float, R: float, rad: float) -> QPointF:
        return QPointF(cx + R * math.cos(rad), cy - R * math.sin(rad))

    def paintEvent(self, _):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            w, h = self.width(), self.height()
            m_left, m_right, m_top, m_bot = 22, 22, 22, 22
            cx = w * 0.52
            cy = h - m_bot - 8
            r_lr = min(cx - m_left, (w - m_right) - cx)
            r_tb = (cy - m_top)
            r = max(10.0, min(r_lr, r_tb) - 8.0)

            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0))
            p.drawRect(QRectF(0, 0, w, h))

            for k in range(20):
                t = k / 19.0
                rr = r * (1.0 - 0.012 * t)
                col = QColor(6 + int(18 * (1 - t)), 6 + int(18 * (1 - t)), 8 + int(22 * (1 - t)))
                p.setBrush(col)
                p.drawEllipse(QRectF(cx - rr, cy - rr, 2 * rr, 2 * rr))

            start_deg = 180.0
            span_deg = -180.0
            vmax = max(self.vmax, 1e-6)
            v = min(self.kmh, vmax)

            p.setPen(QPen(QColor(255, 255, 255, 40), 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QRectF(cx - r - 3, cy - r - 3, 2 * (r + 3), 2 * (r + 3)))

            minor_step, mid_step, major_step = 2, 10, 20
            for vv in range(0, int(vmax) + 1, minor_step):
                f = vv / vmax
                ang = start_deg + span_deg * f
                rad = math.radians(ang)
                is_major = (vv % major_step == 0)
                is_mid = (vv % mid_step == 0)

                if is_major:
                    L = r * 0.13; thick = 3; alpha = 235
                elif is_mid:
                    L = r * 0.09; thick = 2; alpha = 210
                else:
                    L = r * 0.06; thick = 2; alpha = 170

                col = QColor(255, 145, 0, 240) if vv >= self.redline_from else QColor(255, 255, 255, alpha)
                p.setPen(QPen(col, thick, Qt.SolidLine, Qt.RoundCap))
                pt1 = self._polar(cx, cy, r - 8, rad)
                pt2 = self._polar(cx, cy, r - 8 - L, rad)
                p.drawLine(pt1, pt2)

            p.setPen(QColor(255, 255, 255, 235))
            p.setFont(QFont("Consolas", max(12, int(r * 0.12)), QFont.Bold))
            for vv in range(0, int(vmax) + 1, major_step):
                f = vv / vmax
                ang = start_deg + span_deg * f
                rad = math.radians(ang)
                pt = self._polar(cx, cy, r * 0.68, rad)
                p.drawText(QRectF(pt.x() - 28, pt.y() - 18, 56, 36), Qt.AlignCenter, str(vv))

            unit_font = QFont("Microsoft YaHei", max(9, int(r * 0.10)), QFont.DemiBold)
            p.setFont(unit_font)
            p.setPen(QColor(255, 255, 255, 220))
            fm = QFontMetricsF(unit_font)
            tw = fm.horizontalAdvance("km/h")
            th = fm.height()
            x = (cx - r) + 10
            y = h - th - 10
            x = max(8, min(x, w - tw - 8))
            y = max(8, min(y, h - th - 8))
            p.drawText(QRectF(x, y, tw + 6, th + 2), Qt.AlignLeft | Qt.AlignVCenter, "km/h")

            frac = v / vmax
            ang = start_deg + span_deg * frac
            rad = math.radians(ang)
            tip = self._polar(cx, cy, r * 0.80, rad)

            p.setPen(QPen(QColor(0, 0, 0, 150), 10, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(cx + 2, cy + 2), QPointF(tip.x() + 2, tip.y() + 2))
            p.setPen(QPen(QColor(255, 120, 0, 245), 8, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(cx, cy), tip)

            p.setPen(Qt.NoPen)
            p.setBrush(QColor(10, 10, 10, 255))
            p.drawEllipse(QRectF(cx - 18, cy - 18, 36, 36))
            p.setBrush(QColor(255, 140, 0, 220))
            p.drawEllipse(QRectF(cx - 6, cy - 6, 12, 12))

            p.setPen(QColor(255, 255, 255, 235))
            p.setFont(QFont("Consolas", max(14, int(r * 0.14)), QFont.Bold))
            p.drawText(QRectF(cx - 140, cy - r * 0.36, 280, 40), Qt.AlignCenter, f"{v:05.1f}")
        finally:
            p.end()


# ============================================================
# 9) 地图 HTML（定位点始终居中：follow=true）
# ============================================================

LEAFLET_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
html, body { height:100%; margin:0; background:#050812; overflow:hidden; }
#mapWrap { position:relative; height:100%; width:100%; }
#map { position:absolute; inset:0; height:100%; width:100%; background:#050812;
  filter: contrast(1.18) saturate(1.18) brightness(1.03);
  transform: translateZ(0);
}
.hud-frame{ position:absolute; inset:10px; border-radius:14px;
  border:1px solid rgba(0,240,255,0.18);
  box-shadow: 0 0 0 1px rgba(0,240,255,0.10) inset, 0 0 24px rgba(0,240,255,0.10);
  pointer-events:none;
}
.hud-grid{ position:absolute; inset:10px; border-radius:14px;
  background:
    linear-gradient(rgba(0,240,255,0.06) 1px, transparent 1px) 0 0/36px 36px,
    linear-gradient(90deg, rgba(0,240,255,0.06) 1px, transparent 1px) 0 0/36px 36px;
  mask-image: radial-gradient(circle at 50% 50%, rgba(0,0,0,0.95), rgba(0,0,0,0.15));
  pointer-events:none;
}
.crosshair{ position:absolute; left:50%; top:50%; width:26px; height:26px;
  transform: translate(-50%,-50%); pointer-events:none;
}
.crosshair:before,.crosshair:after{
  content:""; position:absolute; left:50%; top:50%;
  background: rgba(0,240,255,0.75);
  box-shadow: 0 0 10px rgba(0,240,255,0.25);
}
.crosshair:before{ width:26px; height:2px; transform: translate(-50%,-50%); }
.crosshair:after{ width:2px; height:26px; transform: translate(-50%,-50%); }
.hud-panel{
  position:absolute; left:18px; top:16px;
  padding:10px 12px;
  border-radius:12px;
  background: rgba(6,10,18,0.62);
  border: 1px solid rgba(0,240,255,0.22);
  box-shadow: 0 0 18px rgba(0,240,255,0.10);
  color: rgba(210,250,255,0.90);
  font-family: Consolas, "Microsoft YaHei", monospace;
  font-size: 12.5px;
  letter-spacing: 0.3px;
  pointer-events:none;
}
.hud-title{ font-weight: 800; color: rgba(0,240,255,0.95); margin-bottom:6px; }
.hud-row{ display:flex; gap:10px; }
.hud-k{ color: rgba(170,240,255,0.85); min-width: 64px; }
.hud-v{ color: rgba(235,255,255,0.92); }
.leaflet-control-zoom a { background:#0b1220; color:#8fe9ff; border:1px solid rgba(0,240,255,0.35); }
.leaflet-control-attribution{ background:rgba(8,12,18,0.55); color:#9bdfff; border:1px solid rgba(0,240,255,0.2); }
</style>
</head>
<body>
<div id="mapWrap">
  <div id="map"></div>
  <div class="hud-frame"></div>
  <div class="hud-grid"></div>
  <div class="crosshair"></div>

  <div class="hud-panel" id="hud">
    <div class="hud-title">SAT NAV • FUSION HUD</div>
    <div class="hud-row"><div class="hud-k">LAT</div><div class="hud-v" id="h_lat">--</div></div>
    <div class="hud-row"><div class="hud-k">LON</div><div class="hud-v" id="h_lon">--</div></div>
    <div class="hud-row"><div class="hud-k">SPD</div><div class="hud-v" id="h_spd">--</div></div>
    <div class="hud-row"><div class="hud-k">SATS</div><div class="hud-v" id="h_sat">--</div></div>
  </div>
</div>

<script>
const map = L.map('map', {
  zoomControl:true,
  preferCanvas:true,
  zoomSnap: 0.25,
  zoomDelta: 0.25,
  updateWhenIdle: true,
  updateWhenZooming: false,
  scrollWheelZoom: true
}).setView([30.67, 104.06], 12);

function tileOpt(maxZoom, maxNativeZoom, attribution){
  return {
    maxZoom: maxZoom, maxNativeZoom: maxNativeZoom,
    keepBuffer: 4,
    updateWhenIdle: true,
    updateWhenZooming: false,
    reuseTiles: true,
    crossOrigin: true,
    attribution: attribution || ''
  };
}

const baseLayers = {
  "esri": L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    tileOpt(20, 19, 'Tiles © Esri')
  ),
  "eox": L.tileLayer(
    'https://{s}.tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2019_3857/default/g/{z}/{y}/{x}.jpg',
    Object.assign(tileOpt(20, 19, 'EOX S2 Cloudless'), { subdomains: ['a','b','c','d'] })
  ),
  "gibs": L.tileLayer(
    'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg',
    tileOpt(9, 9, 'NASA GIBS')
  )
};

const labels = L.tileLayer(
  'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
  Object.assign(tileOpt(20, 19, ''), { opacity: 0.82 })
);

let currentBase = null;
function setBase(name){
  try{
    if(currentBase) map.removeLayer(currentBase);
    currentBase = baseLayers[name] || baseLayers["esri"];
    currentBase.addTo(map);

    if(name === "gibs"){
      if(map.hasLayer(labels)) map.removeLayer(labels);
    }else{
      if(!map.hasLayer(labels)) labels.addTo(map);
    }
  }catch(e){}
}
setBase("esri");

let trackLatLngs = [];

let polyGlow = L.polyline(trackLatLngs, {
  color:'#00f0ff', weight:8, opacity:0.18, lineCap:'round', lineJoin:'round'
}).addTo(map);

let poly = L.polyline(trackLatLngs, {
  color:'#00f0ff', weight:3, opacity:0.92, lineCap:'round', lineJoin:'round'
}).addTo(map);

let dotGlow = L.circleMarker([30.67,104.06], {
  radius:12, color:'#00f0ff', weight:1, opacity:0.10, fillColor:'#00f0ff', fillOpacity:0.12
}).addTo(map);

let dot = L.circleMarker([30.67,104.06], {
  radius:5.5, color:'#00f0ff', weight:2, opacity:0.85, fillColor:'#00f0ff', fillOpacity:0.85
}).addTo(map);

let follow = true;
function setFollow(v){ follow = !!v; }

function addPointsBulk(arr){
  if(!arr || arr.length===0) return;
  for(let i=0;i<arr.length;i++){
    const a = arr[i];
    if(a && a.length>=2){
      const lat = a[0], lon = a[1];
      if(isFinite(lat) && isFinite(lon)){
        trackLatLngs.push([lat,lon]);
      }
    }
  }
  poly.setLatLngs(trackLatLngs);
  polyGlow.setLatLngs(trackLatLngs);

  const last = trackLatLngs[trackLatLngs.length-1];
  dot.setLatLng(last); dotGlow.setLatLng(last);

  if(follow){
    map.panTo(last, {animate:false});
  }
}

function clearTrack(){
  trackLatLngs = [];
  poly.setLatLngs(trackLatLngs);
  polyGlow.setLatLngs(trackLatLngs);
}

function setHud(lat, lon, spd, satm, sats){
  const f = (x, n)=> (isFinite(x) ? x.toFixed(n) : "--");
  document.getElementById("h_lat").innerText = f(lat, 7);
  document.getElementById("h_lon").innerText = f(lon, 7);
  document.getElementById("h_spd").innerText = (isFinite(spd) ? (spd.toFixed(1)+" km/h") : "--");
  document.getElementById("h_sat").innerText = `${satm||0} / ${sats||0}`;
}
</script>
</body>
</html>
"""


# ============================================================
# 10) 科技风 QSS（补齐下拉/菜单文字）
# ============================================================

SCIFI_QSS = """
QMainWindow { background: #050812; }

QGroupBox {
  color: rgba(200, 245, 255, 235);
  border: 1px solid rgba(0, 240, 255, 85);
  border-radius: 12px;
  margin-top: 12px;
  background: rgba(8, 14, 24, 200);
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 12px;
  padding: 0 8px;
  color: rgba(180, 245, 255, 235);
}

QLabel { color: rgba(220, 250, 255, 235); }

QPushButton {
  color: rgba(230, 255, 255, 240);
  background: rgba(8, 16, 28, 240);
  border: 1px solid rgba(0, 240, 255, 120);
  border-radius: 10px;
  padding: 7px 12px;
}
QPushButton:hover {
  border: 1px solid rgba(0, 240, 255, 210);
  background: rgba(10, 22, 40, 255);
}
QPushButton:pressed { background: rgba(6, 12, 20, 255); }

QComboBox, QDoubleSpinBox {
  color: rgba(235, 255, 255, 240);
  background: rgba(8, 16, 28, 240);
  border: 1px solid rgba(0, 240, 255, 120);
  border-radius: 10px;
  padding: 5px 10px;
}
QComboBox::drop-down {
  subcontrol-origin: padding;
  subcontrol-position: top right;
  width: 26px;
  border-left: 1px solid rgba(0,240,255,120);
}
QComboBox::down-arrow {
  width: 10px; height: 10px;
}

/* 关键：下拉列表（popup）文字可见 */
QComboBox QAbstractItemView {
  background: rgba(8, 12, 18, 245);
  color: rgba(235, 255, 255, 240);
  selection-background-color: rgba(0, 240, 255, 70);
  selection-color: rgba(255, 255, 255, 255);
  outline: 0;
  border: 1px solid rgba(0, 240, 255, 140);
}
QComboBox QAbstractItemView::item {
  padding: 6px 10px;
}

/* 关键：菜单文字可见（如果你后面加了 QMenu / 右键菜单） */
QMenu {
  background: rgba(8, 12, 18, 245);
  color: rgba(235,255,255,240);
  border: 1px solid rgba(0,240,255,140);
}
QMenu::item {
  padding: 8px 14px;
}
QMenu::item:selected {
  background: rgba(0,240,255,70);
  color: rgba(255,255,255,255);
}

QCheckBox { color: rgba(215, 250, 255, 235); }

QTabWidget::pane {
  border: 1px solid rgba(0, 240, 255, 80);
  border-radius: 12px;
  background: rgba(8, 14, 24, 180);
}
QTabBar::tab {
  background: rgba(8, 12, 18, 220);
  color: rgba(220, 250, 255, 235);
  border: 1px solid rgba(0,240,255,80);
  padding: 7px 14px;
  border-top-left-radius: 10px;
  border-top-right-radius: 10px;
}
QTabBar::tab:selected {
  background: rgba(10, 22, 40, 245);
  border: 1px solid rgba(0,240,255,170);
}
"""


# ============================================================
# 11) 主窗口
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("组合导航融合数据上位机（PyQt5/去视频/优化版）")
        self.resize(1700, 980)
        pg.setConfigOptions(antialias=True)

        self.track = TrackRecorder()
        self.raw_rec = RawDataRecorder()

        # 串口线程
        self.worker = SerialWorker()
        self.worker.recorder = self.raw_rec
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.frames.connect(self.on_frames_batch)
        self.worker.status.connect(self.on_status)
        self.worker.stats.connect(self.on_stats)
        self.worker.raw_count.connect(self.on_raw_count)

        # 数据回放线程
        self.replay_worker = DataReplayWorker()
        self.replay_thread = QThread(self)
        self.replay_worker.moveToThread(self.replay_thread)
        self.replay_thread.started.connect(self.replay_worker.run)
        self.replay_worker.frames.connect(self.on_frames_batch)
        self.replay_worker.status.connect(self.on_status)
        self.replay_worker.progress.connect(self.on_replay_progress)

        # 轨迹点CSV回放（地图端）
        self.track_replay_timer = QTimer(self)
        self.track_replay_timer.timeout.connect(self._track_replay_tick)
        self.track_replay_data: Optional[List[Dict]] = None
        self.track_replay_idx = 0
        self.track_replay_speed = 1.0

        # 缓存上限
        self.max_points = 3000

        self.tbuf = deque(maxlen=self.max_points)
        self.gx = deque(maxlen=self.max_points)
        self.gy = deque(maxlen=self.max_points)
        self.gz = deque(maxlen=self.max_points)
        self.pitch_buf = deque(maxlen=self.max_points)
        self.roll_buf = deque(maxlen=self.max_points)
        self.head_unwrap_buf = deque(maxlen=self.max_points)
        self.satm = deque(maxlen=self.max_points)
        self.sats = deque(maxlen=self.max_points)

        self._t0_perf: Optional[float] = None
        self._last_t: Optional[float] = None
        self.latest: Optional[FusionFrame] = None

        self._head_unwrap_prev: Optional[float] = None
        self._head_unwrap_acc: float = 0.0

        # 滤波
        self.f_pitch = OneEuroFilter(min_cutoff=0.8, beta=0.02, d_cutoff=1.0)
        self.f_roll  = OneEuroFilter(min_cutoff=0.8, beta=0.02, d_cutoff=1.0)
        self.f_head  = OneEuroFilter(min_cutoff=0.8, beta=0.02, d_cutoff=1.0)
        self.f_gx = OneEuroFilter(min_cutoff=1.5, beta=0.02, d_cutoff=1.0)
        self.f_gy = OneEuroFilter(min_cutoff=1.5, beta=0.02, d_cutoff=1.0)
        self.f_gz = OneEuroFilter(min_cutoff=1.5, beta=0.02, d_cutoff=1.0)
        self.f_lat = OneEuroFilter(min_cutoff=0.4, beta=0.01, d_cutoff=1.0)
        self.f_lon = OneEuroFilter(min_cutoff=0.4, beta=0.01, d_cutoff=1.0)
        self.f_spd = OneEuroFilter(min_cutoff=0.8, beta=0.03, d_cutoff=1.0)

        # 地图批量队列（10Hz 刷新）
        self._map_queue: List[List[float]] = []
        self._map_last_hud = None
        self._map_flush_timer = QTimer(self)
        self._map_flush_timer.timeout.connect(self.flush_map_queue)
        self._map_flush_timer.start(100)

        self._build_ui()
        self.setStyleSheet(SCIFI_QSS)

        # 曲线刷新（10Hz）
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.refresh_plots)
        self.plot_timer.start(100)

        self.scan_ports()
        QTimer.singleShot(900, lambda: self.set_map_source("esri"))

    # ---------------- UI ----------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # 顶部栏
        top = QHBoxLayout()
        top.setSpacing(8)
        root.addLayout(top)

        self.cb_port = QComboBox()
        self.cb_baud = QComboBox()
        for b in [460800, 115200, 921600]:
            self.cb_baud.addItem(str(b), b)
        self.cb_baud.setCurrentText("460800")

        self.btn_scan = QPushButton("检测串口")
        self.btn_open = QPushButton("打开")
        self.btn_close = QPushButton("关闭")
        self.btn_close.setEnabled(False)

        self.cb_deadband = QCheckBox("角速度死区")
        self.cb_deadband.setChecked(True)
        self.sp_deadband = QDoubleSpinBox()
        self.sp_deadband.setDecimals(3)
        self.sp_deadband.setRange(0.0, 2.0)
        self.sp_deadband.setValue(0.03)
        self.sp_deadband.setSuffix(" deg/s")

        self.cb_map = QComboBox()
        self.cb_map.addItem("Esri World Imagery（高清）", "esri")
        self.cb_map.addItem("EOX Sentinel-2 Cloudless（S2）", "eox")
        self.cb_map.addItem("NASA GIBS（MODIS）", "gibs")
        self.cb_map.currentIndexChanged.connect(self.on_map_changed)

        self.btn_clear_map = QPushButton("清空地图轨迹")
        self.btn_clear_all = QPushButton("清空全部记录")

        top.addWidget(QLabel("COM"))
        top.addWidget(self.cb_port, 2)
        top.addWidget(QLabel("BAUD"))
        top.addWidget(self.cb_baud, 1)
        top.addWidget(self.btn_scan)
        top.addWidget(self.btn_open)
        top.addWidget(self.btn_close)
        top.addSpacing(10)
        top.addWidget(self.cb_deadband)
        top.addWidget(self.sp_deadband)
        top.addSpacing(10)
        top.addWidget(QLabel("卫星源"))
        top.addWidget(self.cb_map, 2)
        top.addWidget(self.btn_clear_map)
        top.addWidget(self.btn_clear_all)
        top.addStretch(1)

        self.btn_scan.clicked.connect(self.scan_ports)
        self.btn_open.clicked.connect(self.open_serial)
        self.btn_close.clicked.connect(self.close_serial)
        self.btn_clear_map.clicked.connect(self.clear_map_track)
        self.btn_clear_all.clicked.connect(self.clear_all)

        # 第二行：轨迹导出/导入回放 + 数据录制/回放
        bar2 = QHBoxLayout()
        bar2.setSpacing(8)
        root.addLayout(bar2)

        bar2.addWidget(QLabel("轨迹点数:"))
        self.lb_track_n = QLabel("0")
        bar2.addWidget(self.lb_track_n)
        bar2.addSpacing(10)

        self.btn_export_csv = QPushButton("导出CSV")
        self.btn_export_gpx = QPushButton("导出GPX")
        self.btn_export_kml = QPushButton("导出KML")
        self.btn_import_csv = QPushButton("导入CSV轨迹")
        self.btn_track_replay = QPushButton("回放轨迹点")
        self.btn_track_stop = QPushButton("停止轨迹回放")
        self.btn_track_stop.setEnabled(False)

        self.sp_track_replay = QDoubleSpinBox()
        self.sp_track_replay.setDecimals(1)
        self.sp_track_replay.setRange(0.1, 50.0)
        self.sp_track_replay.setValue(5.0)
        self.sp_track_replay.setSuffix(" x")

        bar2.addWidget(self.btn_export_csv)
        bar2.addWidget(self.btn_export_gpx)
        bar2.addWidget(self.btn_export_kml)
        bar2.addWidget(self.btn_import_csv)
        bar2.addWidget(self.btn_track_replay)
        bar2.addWidget(self.btn_track_stop)
        bar2.addWidget(QLabel("倍率"))
        bar2.addWidget(self.sp_track_replay)
        bar2.addSpacing(18)

        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_gpx.clicked.connect(self.export_gpx)
        self.btn_export_kml.clicked.connect(self.export_kml)
        self.btn_import_csv.clicked.connect(self.import_csv)
        self.btn_track_replay.clicked.connect(self.start_track_replay)
        self.btn_track_stop.clicked.connect(self.stop_track_replay)

        # 数据录制/回放（原始帧）
        self.lb_raw_n = QLabel("raw=0")
        self.btn_data_rec_start = QPushButton("开始数据录制(.fusr)")
        self.btn_data_rec_stop = QPushButton("停止数据录制")
        self.btn_data_rec_stop.setEnabled(False)

        self.btn_data_open = QPushButton("打开数据文件")
        self.btn_data_play = QPushButton("回放数据")
        self.btn_data_pause = QPushButton("暂停")
        self.btn_data_stop = QPushButton("停止回放")
        self.btn_data_pause.setEnabled(False)
        self.btn_data_stop.setEnabled(False)

        self.sp_data_speed = QDoubleSpinBox()
        self.sp_data_speed.setDecimals(1)
        self.sp_data_speed.setRange(0.2, 20.0)
        self.sp_data_speed.setValue(1.0)
        self.sp_data_speed.setSuffix(" x")

        self.lb_data_prog = QLabel("0/0")

        bar2.addWidget(self.lb_raw_n)
        bar2.addWidget(self.btn_data_rec_start)
        bar2.addWidget(self.btn_data_rec_stop)
        bar2.addWidget(self.btn_data_open)
        bar2.addWidget(self.btn_data_play)
        bar2.addWidget(self.btn_data_pause)
        bar2.addWidget(self.btn_data_stop)
        bar2.addWidget(QLabel("回放倍率"))
        bar2.addWidget(self.sp_data_speed)
        bar2.addWidget(self.lb_data_prog)
        bar2.addStretch(1)

        self.btn_data_rec_start.clicked.connect(self.start_data_recording)
        self.btn_data_rec_stop.clicked.connect(self.stop_data_recording)
        self.btn_data_open.clicked.connect(self.pick_data_file)
        self.btn_data_play.clicked.connect(self.start_data_replay)
        self.btn_data_pause.clicked.connect(self.toggle_data_pause)
        self.btn_data_stop.clicked.connect(self.stop_data_replay)

        # ===========================
        # 中间：改为 QSplitter（右侧默认更窄、可拖拽）
        # ===========================
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)
        root.addWidget(splitter, 1)

        map_box = QGroupBox("高清卫星地图轨迹（定位点始终居中，可缩放查看）")
        map_lay = QVBoxLayout(map_box)
        map_lay.setContentsMargins(6, 6, 6, 6)

        if HAS_WEBENGINE:
            cache_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "webcache")
            os.makedirs(cache_dir, exist_ok=True)
            profile = QWebEngineProfile.defaultProfile()
            profile.setCachePath(cache_dir)
            profile.setPersistentStoragePath(cache_dir)
            try:
                profile.setHttpCacheType(QWebEngineProfile.DiskHttpCache)
            except Exception:
                profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.DiskHttpCache)
            profile.setHttpCacheMaximumSize(256 * 1024 * 1024)

            self.web = QWebEngineView()
            s = self.web.settings()
            try:
                s.setAttribute(QWebEngineSettings.Accelerated2dCanvasEnabled, True)
                s.setAttribute(QWebEngineSettings.WebGLEnabled, True)
                s.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
            except Exception:
                s.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
                s.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
                s.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)

            self.web.setHtml(LEAFLET_HTML)
            map_lay.addWidget(self.web, 1)
        else:
            self.web = None
            map_lay.addWidget(QLabel("缺少 PyQtWebEngine：pip install PyQtWebEngine"), 1)

        # 右侧仪表面板容器（可控最大宽度，默认更窄）
        right_panel = QWidget()
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)

        # 关键：限制右侧最大宽度，避免“面板显示不全/挤压其它区域”
        right_panel.setMaximumWidth(460)  # 你要更窄就改成 420 / 380

        box_g = QGroupBox("姿态 / 航向")
        lay_g = QHBoxLayout(box_g)
        lay_g.setContentsMargins(6, 6, 6, 6)
        self.w_att = AttitudeBall()
        self.w_head = HeadingDial()
        lay_g.addWidget(self.w_att, 1)
        lay_g.addWidget(self.w_head, 1)
        right.addWidget(box_g, 2)

        box_s = QGroupBox("速度（0~140 黑白橙）")
        lay_s = QVBoxLayout(box_s)
        lay_s.setContentsMargins(6, 6, 6, 6)
        self.w_spd = SpeedGauge()
        lay_s.addWidget(self.w_spd, 1)
        right.addWidget(box_s, 2)

        box_v = QGroupBox("融合数值")
        form = QFormLayout(box_v)
        form.setContentsMargins(10, 6, 10, 6)
        self.lb_pitch = QLabel("-")
        self.lb_roll = QLabel("-")
        self.lb_head = QLabel("-")
        self.lb_lat = QLabel("-")
        self.lb_lon = QLabel("-")
        self.lb_spd = QLabel("-")
        self.lb_sats = QLabel("-")
        self.lb_stat = QLabel("-")
        form.addRow("Pitch (deg)", self.lb_pitch)
        form.addRow("Roll  (deg)", self.lb_roll)
        form.addRow("Heading (deg)", self.lb_head)
        form.addRow("Lat", self.lb_lat)
        form.addRow("Lon", self.lb_lon)
        form.addRow("Speed (km/h)", self.lb_spd)
        form.addRow("Sats (M/S)", self.lb_sats)
        form.addRow("Frames ok/bad", self.lb_stat)
        right.addWidget(box_v, 1)

        splitter.addWidget(map_box)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 2)

        # 默认尺寸：右侧更窄（你截图的问题主要靠这个解决）
        splitter.setSizes([1320, 380])

        # 下方：曲线
        tabs = QTabWidget()
        tabs.setMinimumHeight(320)
        root.addWidget(tabs, 0)

        self.p_gyr = self._mk_plot("角速度（滤波后）", "deg/s")
        self.c_gx = self.p_gyr.plot([], [], pen=pg.mkPen((0, 240, 255), width=2), name="w_x")
        self.c_gy = self.p_gyr.plot([], [], pen=pg.mkPen((255, 120, 255), width=2), name="w_y")
        self.c_gz = self.p_gyr.plot([], [], pen=pg.mkPen((120, 255, 160), width=2), name="w_z")
        tabs.addTab(self._wrap_widget(self.p_gyr), "角速度")

        self.p_ang = self._mk_plot("角度（滤波后）", "deg")
        self.c_pitch = self.p_ang.plot([], [], pen=pg.mkPen((0, 240, 255), width=2), name="Pitch")
        self.c_roll  = self.p_ang.plot([], [], pen=pg.mkPen((255, 220, 100), width=2), name="Roll")
        self.c_head  = self.p_ang.plot([], [], pen=pg.mkPen((120, 255, 160), width=2), name="Heading(unwrap)")
        tabs.addTab(self._wrap_widget(self.p_ang), "角度")

        self.p_sat = self._mk_plot("卫星数", "count")
        self.c_sm = self.p_sat.plot([], [], pen=pg.mkPen((0, 240, 255), width=2), name="主")
        self.c_ss = self.p_sat.plot([], [], pen=pg.mkPen((255, 220, 100), width=2), name="副")
        tabs.addTab(self._wrap_widget(self.p_sat), "卫星")

        self.status_label = QLabel("就绪")
        root.addWidget(self.status_label)

    def _wrap_widget(self, w: QWidget) -> QWidget:
        c = QWidget()
        l = QVBoxLayout(c)
        l.setContentsMargins(6, 6, 6, 6)
        l.addWidget(w)
        return c

    def _mk_plot(self, title: str, ylab: str) -> pg.PlotWidget:
        pw = pg.PlotWidget()
        pw.setBackground((7, 10, 16))
        pw.showGrid(x=True, y=True, alpha=0.25)
        pw.setTitle(title, color="#9befff")
        pw.setLabel("bottom", "t", units="s", **{"color": "#8fe9ff"})
        pw.setLabel("left", ylab, **{"color": "#8fe9ff"})
        pw.getPlotItem().setDownsampling(mode="peak")
        pw.getPlotItem().setClipToView(True)
        pw.addLegend(offset=(10, 10),
                    brush=pg.mkBrush(8, 12, 18, 220),
                    pen=pg.mkPen(0, 240, 255, 120))
        return pw

    def _js(self, code: str):
        if not HAS_WEBENGINE or self.web is None:
            return
        self.web.page().runJavaScript(code)

    # ---------------- 地图批量刷新 ----------------

    def flush_map_queue(self):
        if not HAS_WEBENGINE or self.web is None:
            return

        if self._map_queue:
            pts = self._map_queue
            self._map_queue = []
            self._js(f"addPointsBulk({pts});")

        if self._map_last_hud is not None:
            lat, lon, spd, satm, sats = self._map_last_hud
            self._map_last_hud = None
            self._js(f"setHud({lat}, {lon}, {spd}, {satm}, {sats});")

    # ---------------- 地图控制 ----------------

    def set_map_source(self, src_key: str):
        self._js(f"setBase('{src_key}');")

    def on_map_changed(self):
        key = self.cb_map.currentData()
        if key:
            self.set_map_source(str(key))

    def clear_map_track(self):
        self._map_queue = []
        self._map_last_hud = None
        self._js("clearTrack();")

    # ---------------- 串口 ----------------

    def scan_ports(self):
        self.cb_port.clear()
        if not HAS_SERIAL:
            self.cb_port.addItem("未安装 pyserial", "")
            return
        ports = list(list_ports.comports())
        for p in ports:
            self.cb_port.addItem(f"{p.device} ({p.description})", p.device)
        if not ports:
            self.cb_port.addItem("未发现串口", "")

    def open_serial(self):
        port = self.cb_port.currentData()
        if not port:
            QMessageBox.warning(self, "提示", "未选择有效串口")
            return
        if self.thread.isRunning():
            QMessageBox.information(self, "提示", "串口线程已在运行")
            return

        self.stop_data_replay()
        self._reset_runtime(clear_track=True)

        self.worker.port = port
        self.worker.baud = int(self.cb_baud.currentData())
        ok = self.worker.open()
        if not ok:
            return

        self.thread.start()
        self.btn_open.setEnabled(False)
        self.btn_close.setEnabled(True)

    def close_serial(self):
        try:
            self.worker.request_stop()
        except Exception:
            pass
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(2000)
        self.worker.close()
        self.btn_open.setEnabled(True)
        self.btn_close.setEnabled(False)

    # ---------------- 轨迹导出/导入/回放 ----------------

    def export_csv(self):
        if len(self.track) == 0:
            QMessageBox.information(self, "提示", "轨迹为空")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", "track.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            self.track.export_csv(path)
            self.on_status(f"已导出 CSV：{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def export_gpx(self):
        if len(self.track) == 0:
            QMessageBox.information(self, "提示", "轨迹为空")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 GPX", "track.gpx", "GPX (*.gpx)")
        if not path:
            return
        try:
            self.track.export_gpx(path)
            self.on_status(f"已导出 GPX：{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def export_kml(self):
        if len(self.track) == 0:
            QMessageBox.information(self, "提示", "轨迹为空")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 KML", "track.kml", "KML (*.kml)")
        if not path:
            return
        try:
            self.track.export_kml(path)
            self.on_status(f"已导出 KML：{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def import_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入 CSV 轨迹", "", "CSV (*.csv)")
        if not path:
            return
        try:
            data = TrackRecorder.load_csv(path)
            if not data:
                QMessageBox.information(self, "提示", "CSV 无有效数据")
                return
            self.track_replay_data = data
            self.track_replay_idx = 0
            self.on_status(f"已导入轨迹：{path}（{len(data)} 点）")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

    def start_track_replay(self):
        if not self.track_replay_data:
            QMessageBox.information(self, "提示", "请先导入 CSV 轨迹")
            return
        self.stop_track_replay()
        self.clear_map_track()

        self.track_replay_speed = float(self.sp_track_replay.value())
        self.track_replay_idx = 0
        self.btn_track_stop.setEnabled(True)

        bulk = []
        n0 = min(1500, len(self.track_replay_data))
        for i in range(n0):
            lat = self._get_float(self.track_replay_data[i].get("lat_deg"))
            lon = self._get_float(self.track_replay_data[i].get("lon_deg"))
            if self._valid_latlon(lat, lon):
                bulk.append([lat, lon])
        if bulk:
            self._js(f"addPointsBulk({bulk});")
        self.track_replay_idx = n0

        self.track_replay_timer.start(15)

    def stop_track_replay(self):
        if self.track_replay_timer.isActive():
            self.track_replay_timer.stop()
        self.btn_track_stop.setEnabled(False)

    def _track_replay_tick(self):
        if not self.track_replay_data:
            self.stop_track_replay()
            return
        if self.track_replay_idx >= len(self.track_replay_data):
            self.on_status("轨迹点回放完成")
            self.stop_track_replay()
            return

        step = max(1, int(self.track_replay_speed))
        bulk = []
        end = min(self.track_replay_idx + step, len(self.track_replay_data))
        for i in range(self.track_replay_idx, end):
            lat = self._get_float(self.track_replay_data[i].get("lat_deg"))
            lon = self._get_float(self.track_replay_data[i].get("lon_deg"))
            if self._valid_latlon(lat, lon):
                bulk.append([lat, lon])
        if bulk:
            self._js(f"addPointsBulk({bulk});")
        self.track_replay_idx = end

    # ---------------- 数据录制 / 回放 ----------------

    def start_data_recording(self):
        if self.raw_rec.is_open():
            QMessageBox.information(self, "提示", "数据录制已在进行")
            return
        path, _ = QFileDialog.getSaveFileName(self, "选择数据录制文件", "fusion_data.fusr", "Fusion Raw (*.fusr)")
        if not path:
            return
        try:
            self.raw_rec.start(path)
            self.btn_data_rec_start.setEnabled(False)
            self.btn_data_rec_stop.setEnabled(True)
            self.on_status(f"开始数据录制：{path}")
        except Exception as e:
            QMessageBox.critical(self, "失败", str(e))

    def stop_data_recording(self):
        if not self.raw_rec.is_open():
            return
        n = self.raw_rec.n
        path = self.raw_rec.path
        self.raw_rec.stop()
        self.btn_data_rec_start.setEnabled(True)
        self.btn_data_rec_stop.setEnabled(False)
        self.on_status(f"数据录制已停止：{path}（{n} 帧）")

    def pick_data_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "Fusion Raw (*.fusr)")
        if not path:
            return
        self.replay_worker.path = path
        self.on_status(f"已选择数据文件：{path}")

    def start_data_replay(self):
        if self.replay_thread.isRunning():
            QMessageBox.information(self, "提示", "数据回放已在进行")
            return
        if not self.replay_worker.path:
            QMessageBox.information(self, "提示", "请先打开数据文件")
            return

        self.close_serial()
        self.stop_track_replay()

        self._reset_runtime(clear_track=True)
        self.clear_map_track()

        self.replay_worker.speed = float(self.sp_data_speed.value())
        self._data_paused = False
        self.btn_data_pause.setText("暂停")
        self.btn_data_pause.setEnabled(True)
        self.btn_data_stop.setEnabled(True)

        self.replay_thread.start()

    def toggle_data_pause(self):
        if not self.replay_thread.isRunning():
            return
        self._data_paused = not getattr(self, "_data_paused", False)
        self.replay_worker.set_pause(self._data_paused)
        self.btn_data_pause.setText("继续" if self._data_paused else "暂停")

    def stop_data_replay(self):
        try:
            self.replay_worker.request_stop()
        except Exception:
            pass
        if self.replay_thread.isRunning():
            self.replay_thread.quit()
            self.replay_thread.wait(2500)
        self.btn_data_pause.setEnabled(False)
        self.btn_data_stop.setEnabled(False)

    def on_replay_progress(self, cur: int, total: int):
        self.lb_data_prog.setText(f"{cur}/{total}")

    def on_raw_count(self, n: int):
        self.lb_raw_n.setText(f"raw={n}")

    # ---------------- 状态 ----------------

    def on_status(self, s: str):
        self.status_label.setText(s)

    def on_stats(self, okc: int, badc: int):
        self.lb_stat.setText(f"{okc} / {badc}")

    # ---------------- 批量帧入口 ----------------

    def on_frames_batch(self, batch: List[FusionFrame]):
        for fr in batch:
            self.on_frame(fr)

    # ---------------- 数据处理核心 ----------------

    def clear_all(self):
        self.stop_track_replay()
        self.stop_data_replay()
        self.stop_data_recording()
        self._reset_runtime(clear_track=True)
        self.on_status("已清空全部记录（轨迹+曲线缓存）")

    def _reset_runtime(self, clear_track: bool):
        self.latest = None
        self._t0_perf = None
        self._last_t = None
        self._head_unwrap_prev = None
        self._head_unwrap_acc = 0.0

        self.tbuf.clear()
        self.gx.clear(); self.gy.clear(); self.gz.clear()
        self.pitch_buf.clear(); self.roll_buf.clear(); self.head_unwrap_buf.clear()
        self.satm.clear(); self.sats.clear()

        for f in [self.f_pitch, self.f_roll, self.f_head,
                  self.f_gx, self.f_gy, self.f_gz,
                  self.f_lat, self.f_lon, self.f_spd]:
            f.reset()

        self._map_queue = []
        self._map_last_hud = None

        if clear_track:
            self.track.clear()
            self.lb_track_n.setText("0")
            self.clear_map_track()

    def _next_t(self, t_perf: float) -> float:
        if self._t0_perf is None:
            self._t0_perf = t_perf
            self._last_t = 0.0
            return 0.0
        t = t_perf - self._t0_perf
        if self._last_t is not None and t <= self._last_t:
            t = self._last_t + 0.001
        self._last_t = t
        return t

    def _deadband(self, v: float) -> float:
        if not self.cb_deadband.isChecked():
            return v
        thr = float(self.sp_deadband.value())
        return 0.0 if abs(v) < thr else v

    def _unwrap_heading(self, head_deg_0_360: float) -> float:
        h = float(head_deg_0_360) % 360.0
        if self._head_unwrap_prev is None:
            self._head_unwrap_prev = h
            self._head_unwrap_acc = 0.0
            return h
        prev = self._head_unwrap_prev
        delta = h - prev
        if delta > 180.0:
            self._head_unwrap_acc -= 360.0
        elif delta < -180.0:
            self._head_unwrap_acc += 360.0
        self._head_unwrap_prev = h
        return h + self._head_unwrap_acc

    @staticmethod
    def _get_float(x) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    @staticmethod
    def _valid_latlon(lat: float, lon: float) -> bool:
        return (isinstance(lat, float) and isinstance(lon, float)
                and math.isfinite(lat) and math.isfinite(lon)
                and -90 <= lat <= 90 and -180 <= lon <= 180
                and (abs(lat) > 1e-9 or abs(lon) > 1e-9))

    def on_frame(self, fr: FusionFrame):
        t = self._next_t(fr.t_perf)
        self.latest = fr

        pitch = self.f_pitch.filter(fr.pitch_deg, t)
        roll  = self.f_roll.filter(fr.roll_deg, t)
        head  = self.f_head.filter(fr.heading_deg, t)
        head_unwrap = self._unwrap_heading(head)

        gx = self._deadband(self.f_gx.filter(fr.gx_dps, t))
        gy = self._deadband(self.f_gy.filter(fr.gy_dps, t))
        gz = self._deadband(self.f_gz.filter(fr.gz_dps, t))

        spd_kmh = math.sqrt(fr.ve_mps**2 + fr.vn_mps**2) * 3.6
        spd_kmh = self.f_spd.filter(spd_kmh, t)

        lat_ok = self._valid_latlon(fr.lat_deg, fr.lon_deg)
        if lat_ok:
            lat_f = self.f_lat.filter(fr.lat_deg, t)
            lon_f = self.f_lon.filter(fr.lon_deg, t)

            self._map_queue.append([float(lat_f), float(lon_f)])
            self._map_last_hud = (float(lat_f), float(lon_f), float(spd_kmh), int(fr.sat_main), int(fr.sat_sub))

            self.track.append({
                "t_s": float(t),
                "lat_deg": float(lat_f),
                "lon_deg": float(lon_f),
                "h_m": float(fr.h_m),
                "spd_kmh": float(spd_kmh),
                "ve_mps": float(fr.ve_mps),
                "vn_mps": float(fr.vn_mps),
                "vu_mps": float(fr.vu_mps),
                "pitch_deg": float(pitch),
                "roll_deg": float(roll),
                "heading_deg": float(head),
                "ins_status": int(fr.ins_status),
                "pos_type": int(fr.pos_type),
                "sat_main": int(fr.sat_main),
                "sat_sub": int(fr.sat_sub),
            })
            self.lb_track_n.setText(str(len(self.track)))
        else:
            self._map_last_hud = (float("nan"), float("nan"), float(spd_kmh), int(fr.sat_main), int(fr.sat_sub))

        self._push_plot(t, gx, gy, gz, pitch, roll, head_unwrap, fr.sat_main, fr.sat_sub)

        self.w_att.set_att(pitch, roll)
        self.w_head.set_heading(head)
        self.w_spd.set_speed(spd_kmh)

        self.lb_pitch.setText(f"{pitch:.6f}")
        self.lb_roll.setText(f"{roll:.6f}")
        self.lb_head.setText(f"{head:.6f}")
        self.lb_lat.setText(f"{fr.lat_deg:.7f}")
        self.lb_lon.setText(f"{fr.lon_deg:.7f}")
        self.lb_spd.setText(f"{spd_kmh:.2f}")
        self.lb_sats.setText(f"{fr.sat_main} / {fr.sat_sub}")

    def _push_plot(self, t, gx, gy, gz, pitch, roll, head_unwrap, satm, sats):
        self.tbuf.append(t)
        self.gx.append(gx); self.gy.append(gy); self.gz.append(gz)
        self.pitch_buf.append(pitch)
        self.roll_buf.append(roll)
        self.head_unwrap_buf.append(head_unwrap)
        self.satm.append(satm); self.sats.append(sats)

    def refresh_plots(self):
        if len(self.tbuf) < 10:
            return

        t = np.asarray(self.tbuf, dtype=float)
        tmax = float(t[-1])
        tmin = tmax - 30.0
        i0 = int(np.searchsorted(t, tmin, side="left"))

        tt = t[i0:]

        def sl(dq):
            a = np.asarray(dq, dtype=float)
            return a[i0:]

        self.c_gx.setData(tt, sl(self.gx))
        self.c_gy.setData(tt, sl(self.gy))
        self.c_gz.setData(tt, sl(self.gz))

        self.c_pitch.setData(tt, sl(self.pitch_buf))
        self.c_roll.setData(tt, sl(self.roll_buf))
        self.c_head.setData(tt, sl(self.head_unwrap_buf))

        self.c_sm.setData(tt, sl(self.satm))
        self.c_ss.setData(tt, sl(self.sats))

    # ---------------- 关闭 ----------------

    def closeEvent(self, event):
        self.stop_track_replay()
        self.stop_data_replay()
        self.stop_data_recording()
        self.close_serial()
        super().closeEvent(event)


# ============================================================
# 12) main（注意：Qt 属性必须在 QApplication 之前）
# ============================================================

def _pre_qt_init():
    os.environ.setdefault(
        "QTWEBENGINE_CHROMIUM_FLAGS",
        "--enable-gpu --enable-zero-copy --ignore-gpu-blocklist --disable-features=RendererCodeIntegrity"
    )

    try:
        QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    except Exception:
        pass
    try:
        QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass
    try:
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    except Exception:
        pass

    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass


def main():
    _pre_qt_init()
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
