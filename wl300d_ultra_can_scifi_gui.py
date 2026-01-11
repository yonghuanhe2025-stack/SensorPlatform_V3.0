#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WL300D-4 超声波 CAN 上位机（PyQt5 / Sci-Fi HUD）

严格按协议（PDF：CAN总线）
- J1939 扩展帧：Priority=0x18, PGN=0xEC00(60416), SA=设备ID
  => arbitration_id = 0x18EC00<ID> (29-bit, is_extended_id=True)
- 数据区 DLC=8：
  Byte0..3 = CH1..CH4 距离（cm）
  Byte4..7 = 保留位，通常 FF FF FF FF

兼容模式（为适配你当前抓到的 0x639 ext=False）：
- 标准帧：ID=0x600+设备ID（常见做法，非PDF的J1939写法）
- 数据区同样按 Byte0..3 = CH1..CH4（cm），Byte4..7 保留

功能：
- 4路实时显示（卡片 + 环形仪表）
- 实时曲线（pyqtgraph）
- 帧HUD信息（ID/EXT/PGN/设备ID/帧率/延迟）
- CSV 记录（含 raw 数据）
"""

import sys
import time
import csv
from dataclasses import dataclass
from collections import deque
from typing import Optional, List, Tuple

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF
from PyQt5.QtGui import QColor, QPainter, QPen, QFont, QLinearGradient, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QComboBox,
    QSpinBox, QCheckBox, QFileDialog, QMessageBox, QFrame, QLineEdit
)

import pyqtgraph as pg

try:
    import can  # python-can
except Exception:
    can = None


# =========================
# 协议常量（严格按PDF）
# =========================
J1939_PRIORITY = 0x18
J1939_PGN_EC00 = 0xEC00  # 60416


# =========================
# 数据结构
# =========================
@dataclass
class UltraFrame:
    ts: float
    arb_id: int
    is_ext: bool
    mode: str               # "J1939" or "STD600"
    dev_id: Optional[int]   # 0~255
    pgn: Optional[int]      # 0xEC00 for J1939
    d_cm: List[Optional[int]]  # 4 channels
    raw: bytes


# =========================
# 协议解析（严格实现 + 兼容）
# =========================
def parse_j1939_ec00(arb_id: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    解析 29-bit J1939 arbitration_id：
    典型字节序展示为：18 EC 00 SA
    - priority/reserved => 0x18（我们只做匹配，不强制等于0x18，避免某些设备改优先级）
    - PF = 0xEC, PS = 0x00 => PGN = 0xEC00
    - SA = device_id

    返回 (priority_byte, pgn, sa)
    """
    # 取低 4 字节（常见显示 18 EC 00 SA）
    b0 = (arb_id >> 24) & 0xFF
    b1 = (arb_id >> 16) & 0xFF
    b2 = (arb_id >> 8) & 0xFF
    b3 = arb_id & 0xFF

    pgn = (b1 << 8) | b2  # EC00
    sa = b3
    pri = b0
    return pri, pgn, sa


def decode_payload_4ch_cm(data: bytes, invalid_ff: bool, invalid_00: bool) -> List[Optional[int]]:
    """
    严格按PDF：data[0..3]为四路距离（cm），data[4..7]为保留位
    无效规则：PDF提到0xFF无效；你现场也见到0x00，因此提供开关
    """
    if len(data) < 8:
        return [None, None, None, None]

    out: List[Optional[int]] = []
    for i in range(4):
        v = int(data[i])
        if invalid_ff and v == 0xFF:
            out.append(None)
        elif invalid_00 and v == 0x00:
            out.append(None)
        else:
            out.append(v)
    return out


def decode_wl300d_can(msg, mode: str, filter_dev_id: Optional[int],
                     invalid_ff: bool, invalid_00: bool) -> Optional[UltraFrame]:
    """
    mode:
      - "strict_j1939": 只接收 is_extended_id=True 且 PGN=0xEC00 的帧
      - "compat_std600": 只接收 标准帧 ID=0x600+dev_id
      - "auto": 两者都接收，优先 J1939

    filter_dev_id:
      - None 表示不过滤
      - 0~255 只接收特定设备ID
    """
    if msg is None:
        return None
    data = bytes(msg.data) if msg.data is not None else b""
    if len(data) < 8:
        return None

    arb_id = int(msg.arbitration_id)
    is_ext = bool(getattr(msg, "is_extended_id", False))

    # ---------- J1939 ----------
    if is_ext:
        pri, pgn, sa = parse_j1939_ec00(arb_id)
        if pgn == J1939_PGN_EC00:
            if mode in ("strict_j1939", "auto"):
                if filter_dev_id is not None and sa != filter_dev_id:
                    return None
                d_cm = decode_payload_4ch_cm(data, invalid_ff=invalid_ff, invalid_00=invalid_00)
                return UltraFrame(
                    ts=time.time(),
                    arb_id=arb_id,
                    is_ext=True,
                    mode="J1939",
                    dev_id=sa,
                    pgn=pgn,
                    d_cm=d_cm,
                    raw=data
                )
        # 不是EC00，忽略
        return None

    # ---------- 标准帧兼容（0x600 + ID） ----------
    # 你抓到的是 0x639，符合 0x600 + 0x39
    if mode in ("compat_std600", "auto"):
        if (arb_id & 0x700) == 0x600:
            dev = arb_id & 0xFF
            if filter_dev_id is not None and dev != filter_dev_id:
                return None
            d_cm = decode_payload_4ch_cm(data, invalid_ff=invalid_ff, invalid_00=invalid_00)
            return UltraFrame(
                ts=time.time(),
                arb_id=arb_id,
                is_ext=False,
                mode="STD600",
                dev_id=dev,
                pgn=None,
                d_cm=d_cm,
                raw=data
            )

    return None


# =========================
# Sci-Fi 环形仪表控件
# =========================
class NeonGauge(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._value: Optional[int] = None
        self._unit = "cm"
        self._max = 250  # cm，用于仪表占比显示，可在UI里改
        self._pulse = 0.0
        self.setMinimumSize(160, 160)
        self.setStyleSheet("background: transparent;")

        self._title_font = QFont("Segoe UI", 10)
        self._value_font = QFont("Segoe UI", 20, QFont.Bold)
        self._small_font = QFont("Segoe UI", 9)

        self._anim = QTimer(self)
        self._anim.setInterval(40)
        self._anim.timeout.connect(self._tick)
        self._anim.start()

    def set_max(self, m: int):
        self._max = max(1, int(m))
        self.update()

    def set_unit(self, u: str):
        self._unit = u
        self.update()

    def set_value(self, v: Optional[int]):
        self._value = v
        self.update()

    def _tick(self):
        self._pulse += 0.06
        if self._pulse > 6.28:
            self._pulse = 0.0
        self.update()

    def paintEvent(self, e):
        w = self.width()
        h = self.height()
        side = min(w, h)

        cx = w / 2.0
        cy = h / 2.0
        r_outer = side * 0.42
        r_inner = side * 0.32

        # 归一化
        if self._value is None:
            frac = 0.0
        else:
            frac = max(0.0, min(1.0, float(self._value) / float(self._max)))

        start_deg = 210
        span_deg = 300  # 仪表弧长
        sweep = span_deg * frac

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # 背景环
        rect = QRectF(cx - r_outer, cy - r_outer, 2*r_outer, 2*r_outer)

        base_pen = QPen(QColor(80, 90, 120, 120), 10)
        base_pen.setCapStyle(Qt.RoundCap)
        p.setPen(base_pen)
        p.drawArc(rect, int((90 - start_deg - span_deg) * 16), int(span_deg * 16))

        # 霓虹弧（带脉冲）
        glow = int(120 + 80 * (0.5 + 0.5 * __import__("math").sin(self._pulse)))
        neon = QColor(80, 180, 255, glow)  # 青蓝霓虹
        pen = QPen(neon, 10)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, int((90 - start_deg - sweep) * 16), int(sweep * 16))

        # 内圈微光
        rect2 = QRectF(cx - r_inner, cy - r_inner, 2*r_inner, 2*r_inner)
        inner_pen = QPen(QColor(120, 220, 255, 60), 2)
        p.setPen(inner_pen)
        p.drawEllipse(rect2)

        # 扫描线（科幻感）
        scan_pen = QPen(QColor(120, 220, 255, 80), 1)
        p.setPen(scan_pen)
        p.drawLine(int(cx - r_outer), int(cy), int(cx + r_outer), int(cy))

        # 文本
        p.setPen(QColor(230, 240, 255, 220))
        p.setFont(self._title_font)
        p.drawText(QRectF(0, 10, w, 20), Qt.AlignCenter, self._title)

        p.setFont(self._value_font)
        if self._value is None:
            val_text = "--"
        else:
            val_text = f"{self._value:d}"
        p.drawText(QRectF(0, cy - 18, w, 40), Qt.AlignCenter, val_text)

        p.setFont(self._small_font)
        p.setPen(QColor(230, 240, 255, 160))
        p.drawText(QRectF(0, cy + 18, w, 20), Qt.AlignCenter, self._unit)

        # 状态点
        dot_r = 5
        dot_color = QColor(0, 255, 160, 200) if self._value is not None else QColor(255, 90, 90, 180)
        p.setBrush(QBrush(dot_color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(int(cx + r_outer*0.75), int(cy - r_outer*0.75), dot_r*2, dot_r*2)

        p.end()


# =========================
# CAN 接收线程
# =========================
class CanRxThread(QThread):
    sig_frame = pyqtSignal(object)   # UltraFrame
    sig_status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bus = None
        self._running = False

        # config
        self.interface = "canalystii"
        self.device = 0
        self.channel = 1
        self.bitrate = 500000
        self.timeout = 0.2

        self.mode = "auto"  # strict_j1939 / compat_std600 / auto
        self.filter_dev_id: Optional[int] = None

        self.invalid_ff = True
        self.invalid_00 = False

    def stop(self):
        self._running = False

    def _open_bus(self):
        if can is None:
            raise RuntimeError("python-can 未安装：pip install python-can")
        # python-can 不同版本参数差异，做双写法兼容
        try:
            return can.Bus(interface=self.interface, device=self.device, channel=self.channel, bitrate=self.bitrate)
        except TypeError:
            return can.Bus(bustype=self.interface, device=self.device, channel=self.channel, bitrate=self.bitrate)

    def run(self):
        try:
            self.sig_status.emit(
                f"Opening CAN: interface={self.interface}, device={self.device}, channel={self.channel}, bitrate={self.bitrate} ..."
            )
            try:
                self._bus = self._open_bus()
            except Exception as e:
                self.sig_status.emit(
                    f"[ERROR] CAN 打开失败：{repr(e)}\n"
                    f"若提示 ControlCAN.dll：请将厂商驱动目录中的 ControlCAN.dll 复制到本脚本同级目录，或把其目录加入系统 PATH。"
                )
                return

            self.sig_status.emit("CAN opened OK. Receiving ...")
            self._running = True

            while self._running:
                msg = self._bus.recv(timeout=self.timeout)
                if not self._running:
                    break
                if msg is None:
                    continue

                uf = decode_wl300d_can(
                    msg,
                    mode=self.mode,
                    filter_dev_id=self.filter_dev_id,
                    invalid_ff=self.invalid_ff,
                    invalid_00=self.invalid_00
                )
                if uf is None:
                    continue

                self.sig_frame.emit(uf)

        except Exception as e:
            self.sig_status.emit(f"[FATAL] RX thread error: {repr(e)}")
        finally:
            try:
                if self._bus is not None:
                    self._bus.shutdown()
            except Exception:
                pass
            self._bus = None
            self.sig_status.emit("CAN closed.")


# =========================
# 主窗口（Sci-Fi HUD）
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WL300D-4 超声波 CAN 上位机（Sci-Fi HUD / 严格协议）")
        self.resize(1280, 760)

        # --- state ---
        self.latest: List[Optional[int]] = [None, None, None, None]
        self.latest_ts = 0.0
        self.latest_frame: Optional[UltraFrame] = None

        # fps
        self._fps_cnt = 0
        self._fps = 0.0
        self._fps_t0 = time.time()

        # csv
        self._csv_fp = None
        self._csv_wr = None

        # plot buffers
        self.t0 = time.time()
        self.max_points = 900  # ~90s@10Hz
        self.buf_t = deque(maxlen=self.max_points)
        self.buf = [deque(maxlen=self.max_points) for _ in range(4)]

        # rx thread
        self.rx = CanRxThread()
        self.rx.sig_frame.connect(self.on_frame)
        self.rx.sig_status.connect(self.append_log)

        # --- build UI ---
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(14, 14, 14, 14)
        main.setSpacing(14)

        # ========== 左侧控制面板 ==========
        left = QVBoxLayout()
        left.setSpacing(12)
        main.addLayout(left, 0)

        gb_conn = QGroupBox("LINK / BUS")
        left.addWidget(gb_conn)
        gl = QGridLayout(gb_conn)
        gl.setContentsMargins(12, 18, 12, 12)
        gl.setHorizontalSpacing(10)
        gl.setVerticalSpacing(10)

        self.sp_device = QSpinBox()
        self.sp_device.setRange(0, 15)
        self.sp_device.setValue(0)

        self.sp_channel = QSpinBox()
        self.sp_channel.setRange(0, 1)
        self.sp_channel.setValue(1)

        self.cb_bitrate = QComboBox()
        self.cb_bitrate.addItems(["500000", "250000"])
        self.cb_bitrate.setCurrentText("500000")

        self.cb_mode = QComboBox()
        self.cb_mode.addItems([
            "auto（推荐：J1939优先+兼容0x600+ID）",
            "strict_j1939（严格：仅扩展帧PGN=0xEC00）",
            "compat_std600（兼容：仅标准帧0x600+ID）",
        ])

        self.ed_filter_id = QLineEdit("")
        self.ed_filter_id.setPlaceholderText("设备ID过滤(十进制/0x..)，留空=不过滤")

        self.ck_inv_ff = QCheckBox("0xFF 视为无效（协议推荐）")
        self.ck_inv_ff.setChecked(True)
        self.ck_inv_00 = QCheckBox("0x00 视为无效（现场可选）")
        self.ck_inv_00.setChecked(False)

        self.sp_gauge_max = QSpinBox()
        self.sp_gauge_max.setRange(10, 2000)
        self.sp_gauge_max.setValue(250)

        self.btn_start = QPushButton("CONNECT")
        self.btn_stop = QPushButton("DISCONNECT")
        self.btn_stop.setEnabled(False)

        self.btn_csv = QPushButton("REC CSV")
        self.btn_csv_stop = QPushButton("STOP REC")
        self.btn_csv_stop.setEnabled(False)

        r = 0
        gl.addWidget(QLabel("Device"), r, 0); gl.addWidget(self.sp_device, r, 1)
        gl.addWidget(QLabel("Channel"), r, 2); gl.addWidget(self.sp_channel, r, 3)
        r += 1
        gl.addWidget(QLabel("Bitrate"), r, 0); gl.addWidget(self.cb_bitrate, r, 1)
        gl.addWidget(QLabel("Mode"), r, 2); gl.addWidget(self.cb_mode, r, 3)
        r += 1
        gl.addWidget(QLabel("DevID Filter"), r, 0); gl.addWidget(self.ed_filter_id, r, 1, 1, 3)
        r += 1
        gl.addWidget(self.ck_inv_ff, r, 0, 1, 2); gl.addWidget(self.ck_inv_00, r, 2, 1, 2)
        r += 1
        gl.addWidget(QLabel("Gauge Max(cm)"), r, 0); gl.addWidget(self.sp_gauge_max, r, 1)
        gl.addWidget(self.btn_start, r, 2); gl.addWidget(self.btn_stop, r, 3)
        r += 1
        gl.addWidget(self.btn_csv, r, 2); gl.addWidget(self.btn_csv_stop, r, 3)

        # HUD
        gb_hud = QGroupBox("HUD")
        left.addWidget(gb_hud)
        v = QVBoxLayout(gb_hud)
        v.setContentsMargins(12, 18, 12, 12)
        self.lb_hud = QLabel("NO SIGNAL")
        self.lb_hud.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lb_hud.setMinimumHeight(110)
        v.addWidget(self.lb_hud)

        # Log
        gb_log = QGroupBox("LOG")
        left.addWidget(gb_log, 1)
        vv = QVBoxLayout(gb_log)
        vv.setContentsMargins(12, 18, 12, 12)
        self.lb_log = QLabel("")
        self.lb_log.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lb_log.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lb_log.setWordWrap(True)
        vv.addWidget(self.lb_log)

        # ========== 右侧显示区 ==========
        right = QVBoxLayout()
        right.setSpacing(12)
        main.addLayout(right, 1)

        # 上：四路仪表
        top = QHBoxLayout()
        top.setSpacing(12)
        right.addLayout(top, 0)

        self.g1 = NeonGauge("CH1")
        self.g2 = NeonGauge("CH2")
        self.g3 = NeonGauge("CH3")
        self.g4 = NeonGauge("CH4")
        top.addWidget(self.g1)
        top.addWidget(self.g2)
        top.addWidget(self.g3)
        top.addWidget(self.g4)

        # 中：实时曲线
        gb_plot = QGroupBox("WAVEFORM")
        right.addWidget(gb_plot, 1)
        pv = QVBoxLayout(gb_plot)
        pv.setContentsMargins(12, 18, 12, 12)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "t", units="s")
        self.plot.setLabel("left", "distance", units="cm")

        # sci-fi plot style
        self.plot.setBackground((12, 16, 28))
        ax = self.plot.getAxis('bottom'); ax.setPen(pg.mkPen((140, 220, 255, 160)))
        ax = self.plot.getAxis('left');   ax.setPen(pg.mkPen((140, 220, 255, 160)))
        self.plot.getViewBox().setBorder(pg.mkPen((80, 180, 255, 80), width=1))

        # curves (颜色用霓虹风)
        pens = [
            pg.mkPen((80, 200, 255, 220), width=2),
            pg.mkPen((0, 255, 170, 200), width=2),
            pg.mkPen((255, 210, 90, 200), width=2),
            pg.mkPen((255, 90, 160, 200), width=2),
        ]
        self.curves = [self.plot.plot([], [], pen=pens[i]) for i in range(4)]
        pv.addWidget(self.plot)

        # 连接信号
        self.btn_start.clicked.connect(self.on_connect)
        self.btn_stop.clicked.connect(self.on_disconnect)
        self.btn_csv.clicked.connect(self.start_csv)
        self.btn_csv_stop.clicked.connect(self.stop_csv)
        self.sp_gauge_max.valueChanged.connect(self._apply_gauge_max)

        # 定时刷新 HUD/超时置灰
        self.timer = QTimer(self)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start()

        self.plot_timer = QTimer(self)
        self.plot_timer.setInterval(200)
        self.plot_timer.timeout.connect(self.refresh_plot)
        self.plot_timer.start()

        self._apply_styles()
        self._apply_gauge_max(self.sp_gauge_max.value())

    # ---------- Sci-Fi 样式 ----------
    def _apply_styles(self):
        # 整体霓虹暗色风格
        self.setStyleSheet("""
        QMainWindow { background: #070b16; }
        QLabel { color: rgba(230,240,255,210); font-family: "Segoe UI"; }
        QGroupBox {
            border: 1px solid rgba(90, 190, 255, 80);
            border-radius: 14px;
            margin-top: 10px;
            padding: 10px;
            background: rgba(10, 14, 28, 180);
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 14px;
            padding: 0 8px 0 8px;
            color: rgba(150, 230, 255, 220);
            font-weight: 600;
            letter-spacing: 1px;
        }
        QLineEdit, QComboBox, QSpinBox {
            background: rgba(255,255,255,10);
            border: 1px solid rgba(120,220,255,60);
            border-radius: 10px;
            padding: 6px 8px;
            color: rgba(230,240,255,220);
        }
        QPushButton {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 rgba(0,170,255,40), stop:1 rgba(0,255,190,30));
            border: 1px solid rgba(120,220,255,120);
            border-radius: 10px;
            padding: 8px 10px;
            color: rgba(230,240,255,230);
            font-weight: 700;
            letter-spacing: 1px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 rgba(0,170,255,70), stop:1 rgba(0,255,190,50));
        }
        QPushButton:disabled {
            color: rgba(230,240,255,90);
            border: 1px solid rgba(120,220,255,40);
            background: rgba(255,255,255,8);
        }
        QCheckBox { color: rgba(230,240,255,190); }
        """)

    def _apply_gauge_max(self, v):
        m = int(v)
        for g in (self.g1, self.g2, self.g3, self.g4):
            g.set_max(m)

    # ---------- CSV ----------
    def start_csv(self):
        if self._csv_fp is not None:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(self, "保存 CSV", f"wl300d_can_{ts}.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            fp = open(path, "w", newline="", encoding="utf-8-sig")
            wr = csv.writer(fp)
            wr.writerow([
                "unix_ts", "t_rel_s", "mode", "is_ext", "arb_id_hex", "dev_id",
                "pgn_hex", "ch1_cm", "ch2_cm", "ch3_cm", "ch4_cm", "raw_hex"
            ])
            self._csv_fp = fp
            self._csv_wr = wr
            self.btn_csv.setEnabled(False)
            self.btn_csv_stop.setEnabled(True)
            self.append_log(f"CSV recording -> {path}")
        except Exception as e:
            QMessageBox.critical(self, "CSV 打开失败", repr(e))

    def stop_csv(self):
        try:
            if self._csv_fp:
                self._csv_fp.close()
        except Exception:
            pass
        self._csv_fp = None
        self._csv_wr = None
        self.btn_csv.setEnabled(True)
        self.btn_csv_stop.setEnabled(False)
        self.append_log("CSV recording stopped")

    # ---------- 连接/断开 ----------
    def on_connect(self):
        if self.rx.isRunning():
            return

        # mode
        idx = self.cb_mode.currentIndex()
        if idx == 0:
            self.rx.mode = "auto"
        elif idx == 1:
            self.rx.mode = "strict_j1939"
        else:
            self.rx.mode = "compat_std600"

        # bus params
        self.rx.device = int(self.sp_device.value())
        self.rx.channel = int(self.sp_channel.value())
        self.rx.bitrate = int(self.cb_bitrate.currentText())

        # invalid rules
        self.rx.invalid_ff = bool(self.ck_inv_ff.isChecked())
        self.rx.invalid_00 = bool(self.ck_inv_00.isChecked())

        # filter device id
        s = self.ed_filter_id.text().strip()
        if not s:
            self.rx.filter_dev_id = None
        else:
            try:
                self.rx.filter_dev_id = int(s, 0)
                if not (0 <= self.rx.filter_dev_id <= 255):
                    raise ValueError()
            except Exception:
                QMessageBox.warning(self, "参数错误", "设备ID过滤应为 0~255（支持 0x..）或留空")
                return

        # reset state
        self.latest = [None, None, None, None]
        self.latest_frame = None
        self.latest_ts = 0.0
        self._fps_cnt = 0
        self._fps = 0.0
        self._fps_t0 = time.time()
        self.buf_t.clear()
        for b in self.buf:
            b.clear()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.append_log(f"CONNECT: mode={self.rx.mode}, filter_dev_id={self.rx.filter_dev_id}")
        self.rx.start()

    def on_disconnect(self):
        if self.rx.isRunning():
            self.append_log("DISCONNECT requested ...")
            self.rx.stop()
            self.rx.wait(1500)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lb_hud.setText("NO SIGNAL")

    # ---------- 数据到达 ----------
    def on_frame(self, uf: UltraFrame):
        self.latest_frame = uf
        self.latest_ts = uf.ts
        self.latest = uf.d_cm[:]

        # gauge
        self.g1.set_value(self.latest[0])
        self.g2.set_value(self.latest[1])
        self.g3.set_value(self.latest[2])
        self.g4.set_value(self.latest[3])

        # fps
        self._fps_cnt += 1
        now = time.time()
        dt = now - self._fps_t0
        if dt >= 1.0:
            self._fps = self._fps_cnt / dt
            self._fps_cnt = 0
            self._fps_t0 = now

        # plot buffer
        t_rel = uf.ts - self.t0
        self.buf_t.append(t_rel)
        for i in range(4):
            v = uf.d_cm[i]
            self.buf[i].append(float("nan") if v is None else float(v))

        # csv
        if self._csv_wr is not None and self._csv_fp is not None:
            try:
                raw_hex = uf.raw.hex(" ").upper()
                self._csv_wr.writerow([
                    f"{uf.ts:.6f}",
                    f"{t_rel:.3f}",
                    uf.mode,
                    "1" if uf.is_ext else "0",
                    f"0x{uf.arb_id:08X}" if uf.is_ext else f"0x{uf.arb_id:03X}",
                    "" if uf.dev_id is None else str(int(uf.dev_id)),
                    "" if uf.pgn is None else f"0x{uf.pgn:04X}",
                    "" if uf.d_cm[0] is None else str(int(uf.d_cm[0])),
                    "" if uf.d_cm[1] is None else str(int(uf.d_cm[1])),
                    "" if uf.d_cm[2] is None else str(int(uf.d_cm[2])),
                    "" if uf.d_cm[3] is None else str(int(uf.d_cm[3])),
                    raw_hex
                ])
                self._csv_fp.flush()
            except Exception:
                pass

    # ---------- 定时刷新 HUD / 超时 ----------
    def on_tick(self):
        uf = self.latest_frame
        if uf is None:
            return

        age_ms = (time.time() - self.latest_ts) * 1000.0
        rid = f"0x{uf.arb_id:08X}" if uf.is_ext else f"0x{uf.arb_id:03X}"
        dev = "--" if uf.dev_id is None else f"{uf.dev_id} (0x{uf.dev_id:02X})"
        pgn = "--" if uf.pgn is None else f"0x{uf.pgn:04X}"
        raw = uf.raw.hex(" ").upper()

        self.lb_hud.setText(
            f"MODE: {uf.mode}    FPS: {self._fps:.1f}    AGE: {age_ms:.0f} ms\n"
            f"ID: {rid}    EXT: {uf.is_ext}\n"
            f"PGN: {pgn}    DEV_ID: {dev}\n"
            f"DATA: {raw}\n"
            f"CH: {uf.d_cm}"
        )

        # 超时置灰
        if age_ms > 800:
            self.g1.set_value(None)
            self.g2.set_value(None)
            self.g3.set_value(None)
            self.g4.set_value(None)

    def refresh_plot(self):
        if len(self.buf_t) < 2:
            return
        x = list(self.buf_t)
        for i in range(4):
            self.curves[i].setData(x, list(self.buf[i]))

    # ---------- 日志 ----------
    def append_log(self, s: str):
        now = time.strftime("%H:%M:%S")
        old = self.lb_log.text().splitlines() if self.lb_log.text() else []
        old.append(f"{now}  {s}")
        if len(old) > 10:
            old = old[-10:]
        self.lb_log.setText("\n".join(old))

    def closeEvent(self, e):
        try:
            self.stop_csv()
        except Exception:
            pass
        try:
            self.on_disconnect()
        except Exception:
            pass
        e.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
