# -*- coding: utf-8 -*-
"""
ARS408-21 毫米波雷达 上位机（Cluster 模式）- HURYS 风格增强版
- 顶部菜单/工具栏（操作/雷达配置/过滤器/检测区/标定/雷达数据/自定义数据帧/多边形/关于）
- 左侧：雷达网格散点（滚轮缩放、右键重置、Shift+左键画多边形、双击结束）
- 右侧：Tab（Cluster 列表 / 原始数据流输入）
- 右侧 Dock：快速调参（实时生效）
- 抗噪：TTL + hits + EMA + 跳变剔除 + RCS/速度过滤 + 距离窗口
- 原始流：增量追加 + 最大行数限制，不会卡顿
- 标定：yaw(偏航) + 平移偏置（x/y），支持保存/加载；支持“选中目标设为原点”
"""

import sys
import os
import json
import threading
import math
import time
from dataclasses import dataclass, field, asdict
from collections import deque
from typing import List, Tuple, Optional, Set, Dict

import numpy as np
import can

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.patches import Rectangle, Polygon

# -------------------------
# Matplotlib 字体
# -------------------------
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
rcParams["axes.unicode_minus"] = False


# =========================
# 配置结构
# =========================
@dataclass
class ViewConfig:
    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = 0.0
    y_max: float = 100.0
    show_id_label: bool = True
    size_by_rcs: bool = False
    show_grid: bool = True
    major_step: float = 10.0
    minor_step: float = 2.0


@dataclass
class FilterConfig:
    rcs_min: float = -10.0
    speed_max: float = 60.0
    min_hits: int = 2
    ttl_s: float = 0.8
    ema_alpha: float = 0.35
    max_jump_m: float = 18.0

    dist_long_min: float = 0.0
    dist_long_max: float = 200.0
    dist_lat_min: float = -100.0
    dist_lat_max: float = 100.0


@dataclass
class ZoneConfig:
    enabled: bool = False
    mode: str = "rect"   # rect | poly
    rect: Tuple[float, float, float, float] = (-20.0, 5.0, 40.0, 80.0)  # x,y,w,h
    poly: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class RawViewConfig:
    enable_id_filter: bool = False
    id_filter_set: Set[int] = field(default_factory=set)  # e.g. {0x600, 0x701}


@dataclass
class CalibConfig:
    """雷达安装位姿标定（对显示/区域/过滤统一生效）"""
    enabled: bool = True
    yaw_deg: float = 0.0     # 逆时针为正：对 (x,y) 旋转
    x_offset: float = 0.0    # 平移：x' = x_rot + x_offset
    y_offset: float = 0.0    # 平移：y' = y_rot + y_offset

    def apply_xy(self, x: float, y: float) -> Tuple[float, float]:
        if not self.enabled:
            return float(x), float(y)
        yaw = math.radians(float(self.yaw_deg))
        cy, sy = math.cos(yaw), math.sin(yaw)
        xr = cy * x - sy * y
        yr = sy * x + cy * y
        return float(xr + self.x_offset), float(yr + self.y_offset)


@dataclass
class AppConfig:
    view: ViewConfig = field(default_factory=ViewConfig)
    filt: FilterConfig = field(default_factory=FilterConfig)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    rawview: RawViewConfig = field(default_factory=RawViewConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
    simulate: bool = False

    def to_json_dict(self) -> dict:
        d = asdict(self)
        # set 序列化
        d["rawview"]["id_filter_set"] = sorted(list(self.rawview.id_filter_set))
        return d

    @staticmethod
    def from_json_dict(d: dict) -> "AppConfig":
        cfg = AppConfig()
        try:
            if "view" in d:
                for k, v in d["view"].items():
                    setattr(cfg.view, k, v)
            if "filt" in d:
                for k, v in d["filt"].items():
                    setattr(cfg.filt, k, v)
            if "zone" in d:
                for k, v in d["zone"].items():
                    setattr(cfg.zone, k, v)
            if "rawview" in d:
                rv = d["rawview"]
                cfg.rawview.enable_id_filter = bool(rv.get("enable_id_filter", False))
                cfg.rawview.id_filter_set = set(int(x) for x in rv.get("id_filter_set", []) if x is not None)
            if "calib" in d:
                for k, v in d["calib"].items():
                    setattr(cfg.calib, k, v)
            cfg.simulate = bool(d.get("simulate", False))
        except Exception:
            pass
        return cfg


# =========================
# CAN 打开
# =========================
def open_canalyst_auto(bitrate=500000):
    """自动扫描 CANalyst-II 通道 0~3，返回 (bus, channel)。"""
    last_err = None
    for ch in range(4):
        try:
            print(f"尝试打开：CANalyst-II 通道 {ch} ...")
            bus = can.Bus(interface="canalystii", channel=ch, bitrate=bitrate)
            print(f"✅ 成功：CANalyst-II 通道 {ch}")
            return bus, ch
        except Exception as e:
            last_err = e
            print(f"❌ 失败：CANalyst-II 通道 {ch} -> {e}")
    raise RuntimeError(f"打开 CAN 失败：未找到可用的 CANalyst-II 通道，最后一次错误：{last_err}")


# =========================
# ARS408 Cluster 解码
# =========================
def decode_ars408_cluster_status(data_bytes):
    """解析 Cluster_0_Status (0x600)"""
    if len(data_bytes) < 4:
        return None
    b0, b1, b2, b3 = data_bytes[:4]
    near = b0
    far = b1
    meas_counter = b2 | (b3 << 8)
    return {"near": near, "far": far, "meas_counter": meas_counter}


def decode_ars408_cluster_general(msg: can.Message):
    """解析 Cluster_1_General (0x701)"""
    data = msg.data
    if len(data) < 8:
        return None

    b0, b1, b2, b3, b4, b5, b6, b7 = data
    cluster_id = b0

    raw_dist_long = (b1 << 5) + (b2 >> 3)
    dist_long = raw_dist_long * 0.2 - 500.0

    raw_dist_lat = ((b2 & 0x07) << 8) + b3
    dist_lat = raw_dist_lat * 0.2 - 102.3

    raw_vrel_long = (b4 << 2) + (b5 >> 6)
    vrel_long = raw_vrel_long * 0.25 - 128.0

    raw_vrel_lat = ((b5 & 0x3F) << 3) + (b6 >> 5)
    vrel_lat = raw_vrel_lat * 0.25 - 64.0

    rcs = b7 * 0.5 - 64.0

    # 雷达坐标：x=DistLat，y=DistLong（与你原逻辑一致）
    x = float(dist_lat)
    y = float(dist_long)

    return {
        "id": int(cluster_id),
        "dist_long": float(dist_long),
        "dist_lat": float(dist_lat),
        "vrel_long": float(vrel_long),
        "vrel_lat": float(vrel_lat),
        "rcs": float(rcs),
        "x": x,
        "y": y,
    }


# =========================
# Track 抗噪
# =========================
@dataclass
class Track:
    tid: int
    last_seen_ts: float = 0.0
    hits: int = 0
    ema_x: float = 0.0
    ema_y: float = 0.0
    last_x: float = 0.0
    last_y: float = 0.0
    last_obj: dict = field(default_factory=dict)

    def update(self, obj: dict, ts: float, alpha: float, max_jump: float) -> bool:
        x, y = float(obj["x"]), float(obj["y"])

        if self.hits == 0:
            self.ema_x, self.ema_y = x, y
            self.last_x, self.last_y = x, y
            self.hits = 1
            self.last_seen_ts = ts
            self.last_obj = obj
            return True

        jump = math.hypot(x - self.last_x, y - self.last_y)
        if max_jump > 0 and jump > max_jump:
            return False

        a = float(np.clip(alpha, 0.01, 0.99))
        self.ema_x = a * x + (1 - a) * self.ema_x
        self.ema_y = a * y + (1 - a) * self.ema_y

        self.last_x, self.last_y = x, y
        self.last_seen_ts = ts
        self.hits += 1
        self.last_obj = obj
        return True

    def to_display_obj(self) -> dict:
        o = dict(self.last_obj)
        o["x"] = float(self.ema_x)
        o["y"] = float(self.ema_y)
        o["hits"] = int(self.hits)
        return o


# =========================
# 雷达画布（仿 HURYS）
# =========================
class RadarCanvas(FigureCanvas):
    sig_polygon_finished = QtCore.pyqtSignal(list)  # [(x,y), ...]
    sig_double_clicked = QtCore.pyqtSignal(float, float)

    def __init__(self, cfg: AppConfig, parent=None):
        self.cfg = cfg
        self.fig = Figure(figsize=(6, 6), facecolor="#0b1220")
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.scatter = None
        self._objs = []
        self._id_texts = []
        self._hover_annot = None

        self._zone_artist = None
        self._poly_temp_artist = None

        self._poly_drawing = False
        self._poly_points: List[Tuple[float, float]] = []

        self._init_axes()

        self.mpl_connect("scroll_event", self.on_scroll)
        self.mpl_connect("motion_notify_event", self.on_hover)
        self.mpl_connect("button_press_event", self.on_mouse_press)

    def _init_axes(self):
        self.ax.cla()
        self.ax.set_facecolor("#0b1220")

        self.ax.set_xlim(self.cfg.view.x_min, self.cfg.view.x_max)
        self.ax.set_ylim(self.cfg.view.y_min, self.cfg.view.y_max)
        self.ax.set_aspect("auto")

        self.ax.xaxis.set_major_locator(MultipleLocator(self.cfg.view.major_step))
        self.ax.yaxis.set_major_locator(MultipleLocator(self.cfg.view.major_step))
        self.ax.xaxis.set_minor_locator(MultipleLocator(self.cfg.view.minor_step))
        self.ax.yaxis.set_minor_locator(MultipleLocator(self.cfg.view.minor_step))
        self.ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        self.ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

        if self.cfg.view.show_grid:
            self.ax.grid(which="major", linestyle="-", linewidth=0.9, color="#38bdf8", alpha=0.22)
            self.ax.grid(which="minor", linestyle=":", linewidth=0.7, color="#38bdf8", alpha=0.10)
        else:
            self.ax.grid(False)

        for sp in self.ax.spines.values():
            sp.set_color("#334155")
            sp.set_linewidth(1.0)

        self.ax.tick_params(axis="x", colors="#a5f3fc", labelsize=8, pad=2)
        self.ax.tick_params(axis="y", colors="#e5e7eb", labelsize=8, pad=2)

        # 雷达中心线
        self.ax.axvline(0, color="#b8ff5a", linewidth=1.1, alpha=0.85)

        self._draw_left_big_numbers()

        self.scatter = self.ax.scatter([], [], s=22, c="#22d3ee", alpha=0.95)

        self._hover_annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.35", fc="#020617", ec="#22d3ee", alpha=0.95),
            color="#e5e7eb",
            fontsize=9,
        )
        self._hover_annot.set_visible(False)

        self.fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.12)
        self._update_zone_artist()
        self.draw_idle()

    def _draw_left_big_numbers(self):
        y0, y1 = self.ax.get_ylim()
        step = max(int(self.cfg.view.major_step), 2)
        ys = list(range(int(math.ceil(y0 / step) * step), int(y1) + 1, step))
        x_left = self.cfg.view.x_min - (self.cfg.view.x_max - self.cfg.view.x_min) * 0.06
        for y in ys:
            self.ax.text(
                x_left, y, str(y),
                va="center", ha="right",
                fontsize=10, color="#e5e7eb", fontweight="bold", alpha=0.85
            )

    def apply_view(self):
        self._init_axes()

    def reset_view(self):
        self.cfg.view.x_min, self.cfg.view.x_max = -50, 50
        self.cfg.view.y_min, self.cfg.view.y_max = 0, 100
        self.apply_view()

    def _clear_id_texts(self):
        for t in self._id_texts:
            try:
                t.remove()
            except Exception:
                pass
        self._id_texts = []

    def update_objects(self, objs: List[dict]):
        self._objs = list(objs) if objs else []
        if self._objs:
            offsets = np.array([(float(o["x"]), float(o["y"])) for o in self._objs], dtype=float)
        else:
            offsets = np.empty((0, 2), dtype=float)
        self.scatter.set_offsets(offsets)

        if self.cfg.view.size_by_rcs and self._objs:
            rcs = np.array([float(o.get("rcs", 0.0)) for o in self._objs], dtype=float)
            sizes = 10.0 + np.clip((rcs + 64.0) / 128.0, 0.0, 1.0) * 70.0
            self.scatter.set_sizes(sizes)
        else:
            self.scatter.set_sizes(np.full((len(self._objs),), 22.0, dtype=float))

        self._clear_id_texts()
        if self.cfg.view.show_id_label:
            for o in self._objs:
                self._id_texts.append(
                    self.ax.text(
                        float(o["x"]) + 0.6, float(o["y"]) + 0.6,
                        str(o.get("id", "")),
                        fontsize=9, color="#22d3ee", alpha=0.95
                    )
                )

        self._update_zone_artist()
        if self._hover_annot:
            self._hover_annot.set_visible(False)
        self.draw_idle()

    def _update_zone_artist(self):
        for art in [self._zone_artist, self._poly_temp_artist]:
            if art is not None:
                try:
                    art.remove()
                except Exception:
                    pass
        self._zone_artist = None
        self._poly_temp_artist = None

        if self.cfg.zone.enabled:
            if self.cfg.zone.mode == "rect":
                x, y, w, h = self.cfg.zone.rect
                self._zone_artist = Rectangle(
                    (x, y), w, h, fill=False,
                    edgecolor="#22d3ee", linewidth=1.3, linestyle="-", alpha=0.9
                )
                self.ax.add_patch(self._zone_artist)
            else:
                pts = self.cfg.zone.poly
                if len(pts) >= 3:
                    self._zone_artist = Polygon(
                        pts, closed=True, fill=False,
                        edgecolor="#22d3ee", linewidth=1.3, linestyle="-", alpha=0.9
                    )
                    self.ax.add_patch(self._zone_artist)

        if self._poly_drawing and len(self._poly_points) >= 1:
            self._poly_temp_artist = Polygon(
                self._poly_points, closed=False, fill=False,
                edgecolor="#b8ff5a", linewidth=1.0, linestyle="--", alpha=0.9
            )
            self.ax.add_patch(self._poly_temp_artist)

    def on_hover(self, event):
        if not self._objs or event.inaxes != self.ax:
            if self._hover_annot and self._hover_annot.get_visible():
                self._hover_annot.set_visible(False)
                self.draw_idle()
            return

        cont, ind = self.scatter.contains(event)
        if not cont:
            if self._hover_annot and self._hover_annot.get_visible():
                self._hover_annot.set_visible(False)
                self.draw_idle()
            return

        i = ind["ind"][0]
        if i < 0 or i >= len(self._objs):
            return
        o = self._objs[i]

        self._hover_annot.xy = (float(o["x"]), float(o["y"]))
        self._hover_annot.set_text(
            f"ID: {o.get('id','-')}\n"
            f"Hits: {o.get('hits',0)}\n"
            f"Y(DistLong): {float(o.get('dist_long',0.0)):.2f} m\n"
            f"X(DistLat):  {float(o.get('dist_lat',0.0)):.2f} m\n"
            f"Vlong: {float(o.get('vrel_long',0.0)):.2f} m/s\n"
            f"Vlat:  {float(o.get('vrel_lat',0.0)):.2f} m/s\n"
            f"RCS: {float(o.get('rcs',0.0)):.1f} dB\n"
            f"XY(calib): ({float(o.get('x',0.0)):.2f}, {float(o.get('y',0.0)):.2f})"
        )
        self._hover_annot.set_visible(True)
        self.draw_idle()

    def on_scroll(self, event):
        if event.xdata is None or event.ydata is None:
            return
        base_scale = 1.2
        if event.button == "up":
            scale_factor = 1 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        cur_width = cur_xlim[1] - cur_xlim[0]
        cur_height = cur_ylim[1] - cur_ylim[0]

        new_width = cur_width * scale_factor
        new_height = cur_height * scale_factor

        relx = (cur_xlim[1] - xdata) / cur_width
        rely = (cur_ylim[1] - ydata) / cur_height

        new_xmin = xdata - new_width * (1 - relx)
        new_xmax = xdata + new_width * relx
        new_ymin = ydata - new_height * (1 - rely)
        new_ymax = ydata + new_height * rely

        self.cfg.view.x_min, self.cfg.view.x_max = new_xmin, new_xmax
        self.cfg.view.y_min, self.cfg.view.y_max = new_ymin, new_ymax
        self.apply_view()

    def on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return

        # 右键：重置
        if event.button == 3:
            self.reset_view()
            return

        # Shift+左键：画多边形点
        mods = QtWidgets.QApplication.keyboardModifiers()
        if (mods & Qt.ShiftModifier) and event.button == 1:
            if event.xdata is None or event.ydata is None:
                return
            if not self._poly_drawing:
                self._poly_drawing = True
                self._poly_points = []
            self._poly_points.append((float(event.xdata), float(event.ydata)))
            self._update_zone_artist()
            self.draw_idle()
            return

        # 双击：结束多边形 或 发射双击坐标（给标定用）
        if event.dblclick:
            if self._poly_drawing:
                if len(self._poly_points) >= 3:
                    pts = list(self._poly_points)
                    self._poly_drawing = False
                    self._poly_points = []
                    self.sig_polygon_finished.emit(pts)
                else:
                    self._poly_drawing = False
                    self._poly_points = []
                self._update_zone_artist()
                self.draw_idle()
            else:
                if event.xdata is not None and event.ydata is not None:
                    self.sig_double_clicked.emit(float(event.xdata), float(event.ydata))


# =========================
# 右侧面板：Cluster 表格 + 原始数据流
# =========================
class RightPanel(QtWidgets.QWidget):
    sig_selected_obj = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet("""
            QWidget{background:#0b1220;}
            QGroupBox{color:#e5e7eb;border:1px solid #334155;border-radius:10px;margin-top:8px;}
            QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 10px;}
            QTabWidget::pane{border:1px solid #334155;border-radius:10px;}
            QTabBar::tab{background:#111827;color:#e5e7eb;padding:8px 12px;border:1px solid #334155;border-bottom:none;border-top-left-radius:8px;border-top-right-radius:8px;}
            QTabBar::tab:selected{background:#020617;}
            QLabel{color:#e5e7eb;}
        """)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        self.lab_status = QtWidgets.QLabel("未连接")
        self.lab_status.setStyleSheet("color:#e5e7eb;font-size:12px;")
        gb_status = QtWidgets.QGroupBox("状态")
        v = QtWidgets.QVBoxLayout(gb_status)
        v.addWidget(self.lab_status)
        lay.addWidget(gb_status)

        gb_tabs = QtWidgets.QGroupBox("数据")
        tabs_lay = QtWidgets.QVBoxLayout(gb_tabs)

        self.tabs = QtWidgets.QTabWidget()

        # Cluster 表格
        tab_table = QtWidgets.QWidget()
        tlay = QtWidgets.QVBoxLayout(tab_table)
        tlay.setContentsMargins(0, 0, 0, 0)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Hits", "Y(DistLong)", "X(DistLat)", "Vlong", "Vlat", "RCS"]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setStyleSheet(
            "QTableWidget{background-color:#020617;color:#e5e7eb;gridline-color:#334155;border-radius:8px;}"
            "QHeaderView::section{background-color:#0b1220;color:#e5e7eb;border:none;padding:6px;}"
            "QTableWidget::item:selected{background-color:#1d4ed8;}"
        )
        self.table.setFont(QtGui.QFont("Microsoft YaHei", 9))
        self.table.itemSelectionChanged.connect(self._emit_selected)
        tlay.addWidget(self.table)

        # 原始流
        tab_raw = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(tab_raw)
        rlay.setContentsMargins(0, 0, 0, 0)

        self.raw_edit = QtWidgets.QPlainTextEdit()
        self.raw_edit.setReadOnly(True)
        self.raw_edit.document().setMaximumBlockCount(2500)  # 防卡
        self.raw_edit.setStyleSheet(
            "QPlainTextEdit{background:#020617;color:#b7f7ff;border:1px solid #334155;border-radius:8px;}"
        )
        self.raw_edit.setFont(QtGui.QFont("Consolas", 9))
        rlay.addWidget(self.raw_edit)

        self.tabs.addTab(tab_table, "Cluster 列表")
        self.tabs.addTab(tab_raw, "原始数据流输入")
        tabs_lay.addWidget(self.tabs)

        lay.addWidget(gb_tabs, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_clear_raw = QtWidgets.QPushButton("清空原始流")
        self.btn_clear_raw.setCursor(Qt.PointingHandCursor)
        self.btn_clear_raw.setStyleSheet(
            "QPushButton{background:#111827;color:#e5e7eb;border:1px solid #22d3ee;border-radius:10px;padding:8px 12px;}"
            "QPushButton:hover{background:#020617;}"
        )
        btn_row.addWidget(self.btn_clear_raw)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        self._last_display_cache: List[dict] = []

    def _emit_selected(self):
        r = self.table.currentRow()
        if r < 0 or r >= len(self._last_display_cache):
            return
        self.sig_selected_obj.emit(dict(self._last_display_cache[r]))

    def set_status_text(self, text: str):
        self.lab_status.setText(text)

    def update_table(self, display_objs: List[dict]):
        self._last_display_cache = list(display_objs) if display_objs else []
        self.table.setRowCount(len(self._last_display_cache))

        def set_item(row, col, txt):
            it = QtWidgets.QTableWidgetItem(txt)
            it.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, col, it)

        for r, o in enumerate(self._last_display_cache):
            set_item(r, 0, str(o.get("id", "")))
            set_item(r, 1, str(o.get("hits", 0)))
            set_item(r, 2, f"{float(o.get('dist_long', 0.0)):.2f}")
            set_item(r, 3, f"{float(o.get('dist_lat', 0.0)):.2f}")
            set_item(r, 4, f"{float(o.get('vrel_long', 0.0)):.2f}")
            set_item(r, 5, f"{float(o.get('vrel_lat', 0.0)):.2f}")
            set_item(r, 6, f"{float(o.get('rcs', 0.0)):.1f}")

    def append_raw_lines(self, lines: List[str]):
        if not lines:
            return
        self.raw_edit.appendPlainText("\n".join(lines))

    def clear_raw(self):
        self.raw_edit.setPlainText("")


# =========================
# 对话框：雷达配置 / 过滤器 / 检测区 / 自定义数据帧 / 雷达数据 / 标定 / 关于
# =========================
class RadarConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent, cfg: AppConfig, connect_cb, disconnect_cb):
        super().__init__(parent)
        self.cfg = cfg
        self.connect_cb = connect_cb
        self.disconnect_cb = disconnect_cb

        self.setWindowTitle("雷达配置")
        self.setModal(True)
        self.resize(520, 260)

        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.cmb_bitrate = QtWidgets.QComboBox()
        self.cmb_bitrate.addItems(["500000", "250000", "125000"])
        self.cmb_bitrate.setCurrentText("500000")

        self.cb_auto = QtWidgets.QCheckBox("自动扫描通道 (0~3)")
        self.cb_auto.setChecked(True)

        self.sp_channel = QtWidgets.QSpinBox()
        self.sp_channel.setRange(0, 3)
        self.sp_channel.setValue(0)
        self.sp_channel.setEnabled(False)

        self.cb_sim = QtWidgets.QCheckBox("仿真模式（无雷达也可运行）")
        self.cb_sim.setChecked(self.cfg.simulate)

        self.cb_auto.toggled.connect(lambda v: self.sp_channel.setEnabled(not v))

        form.addRow("波特率", self.cmb_bitrate)
        form.addRow("", self.cb_auto)
        form.addRow("通道（手动）", self.sp_channel)
        form.addRow("", self.cb_sim)

        lay.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        btn_connect = QtWidgets.QPushButton("连接")
        btn_disconnect = QtWidgets.QPushButton("断开")
        btn_close = QtWidgets.QPushButton("关闭")
        btn_connect.clicked.connect(self.on_connect)
        btn_disconnect.clicked.connect(self.on_disconnect)
        btn_close.clicked.connect(self.accept)
        btns.addWidget(btn_connect)
        btns.addWidget(btn_disconnect)
        btns.addStretch(1)
        btns.addWidget(btn_close)
        lay.addLayout(btns)

    def on_connect(self):
        bitrate = int(self.cmb_bitrate.currentText())
        self.cfg.simulate = self.cb_sim.isChecked()

        if self.cfg.simulate:
            self.connect_cb(None, None, bitrate, auto=True)
            QtWidgets.QMessageBox.information(self, "提示", "已进入仿真模式。")
            return

        if self.cb_auto.isChecked():
            self.connect_cb(None, None, bitrate, auto=True)
        else:
            self.connect_cb(int(self.sp_channel.value()), None, bitrate, auto=False)

    def on_disconnect(self):
        self.disconnect_cb()
        QtWidgets.QMessageBox.information(self, "提示", "已断开。")


class FilterDialog(QtWidgets.QDialog):
    def __init__(self, parent, cfg: AppConfig):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("过滤器设置")
        self.setModal(True)
        self.resize(560, 380)

        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        def dspin(mn, mx, val, step, dec=2):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(mn, mx)
            sp.setValue(val)
            sp.setSingleStep(step)
            sp.setDecimals(dec)
            return sp

        def ispin(mn, mx, val):
            sp = QtWidgets.QSpinBox()
            sp.setRange(mn, mx)
            sp.setValue(val)
            return sp

        f = self.cfg.filt
        self.sp_rcs = dspin(-64, 64, f.rcs_min, 0.5, 1)
        self.sp_speed = dspin(0, 200, f.speed_max, 1.0, 1)
        self.sp_hits = ispin(1, 10, f.min_hits)
        self.sp_ttl = dspin(0.1, 10.0, f.ttl_s, 0.1, 2)
        self.sp_alpha = dspin(0.05, 0.95, f.ema_alpha, 0.05, 2)
        self.sp_jump = dspin(0.0, 200.0, f.max_jump_m, 1.0, 1)

        self.sp_dy0 = dspin(-500, 500, f.dist_long_min, 1.0, 1)
        self.sp_dy1 = dspin(-500, 500, f.dist_long_max, 1.0, 1)
        self.sp_dx0 = dspin(-500, 500, f.dist_lat_min, 1.0, 1)
        self.sp_dx1 = dspin(-500, 500, f.dist_lat_max, 1.0, 1)

        form.addRow("RCS 最小 (dB)", self.sp_rcs)
        form.addRow("速度上限 |V| (m/s)", self.sp_speed)
        form.addRow("最小连续命中 Hits", self.sp_hits)
        form.addRow("保留时间 TTL (s)", self.sp_ttl)
        form.addRow("EMA 平滑 alpha", self.sp_alpha)
        form.addRow("最大跳变剔除 (m)", self.sp_jump)
        form.addRow("DistLong 范围 min", self.sp_dy0)
        form.addRow("DistLong 范围 max", self.sp_dy1)
        form.addRow("DistLat  范围 min", self.sp_dx0)
        form.addRow("DistLat  范围 max", self.sp_dx1)

        lay.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton("应用")
        btn_close = QtWidgets.QPushButton("关闭")
        btn_ok.clicked.connect(self.on_apply)
        btn_close.clicked.connect(self.accept)
        btns.addWidget(btn_ok)
        btns.addStretch(1)
        btns.addWidget(btn_close)
        lay.addLayout(btns)

    def on_apply(self):
        f = self.cfg.filt
        f.rcs_min = float(self.sp_rcs.value())
        f.speed_max = float(self.sp_speed.value())
        f.min_hits = int(self.sp_hits.value())
        f.ttl_s = float(self.sp_ttl.value())
        f.ema_alpha = float(self.sp_alpha.value())
        f.max_jump_m = float(self.sp_jump.value())
        f.dist_long_min = float(self.sp_dy0.value())
        f.dist_long_max = float(self.sp_dy1.value())
        f.dist_lat_min = float(self.sp_dx0.value())
        f.dist_lat_max = float(self.sp_dx1.value())
        QtWidgets.QMessageBox.information(self, "提示", "过滤器参数已应用。")


class ZoneDialog(QtWidgets.QDialog):
    def __init__(self, parent, cfg: AppConfig):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("检测区设置")
        self.setModal(True)
        self.resize(560, 320)

        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.cb_enable = QtWidgets.QCheckBox("启用检测区")
        self.cb_enable.setChecked(cfg.zone.enabled)

        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["rect", "poly"])
        self.cmb_mode.setCurrentText(cfg.zone.mode)

        x, y, w, h = cfg.zone.rect
        self.sp_rx = QtWidgets.QDoubleSpinBox(); self.sp_rx.setRange(-1000, 1000); self.sp_rx.setValue(x); self.sp_rx.setDecimals(2)
        self.sp_ry = QtWidgets.QDoubleSpinBox(); self.sp_ry.setRange(-1000, 1000); self.sp_ry.setValue(y); self.sp_ry.setDecimals(2)
        self.sp_rw = QtWidgets.QDoubleSpinBox(); self.sp_rw.setRange(0, 2000); self.sp_rw.setValue(w); self.sp_rw.setDecimals(2)
        self.sp_rh = QtWidgets.QDoubleSpinBox(); self.sp_rh.setRange(0, 2000); self.sp_rh.setValue(h); self.sp_rh.setDecimals(2)

        rect_row = QtWidgets.QWidget()
        rlay = QtWidgets.QHBoxLayout(rect_row); rlay.setContentsMargins(0,0,0,0)
        for lab, sp in [("x", self.sp_rx), ("y", self.sp_ry), ("w", self.sp_rw), ("h", self.sp_rh)]:
            rlay.addWidget(QtWidgets.QLabel(lab))
            rlay.addWidget(sp)
        rlay.addStretch(1)

        self.txt_poly = QtWidgets.QPlainTextEdit()
        self.txt_poly.setFixedHeight(120)
        self.txt_poly.setPlaceholderText("poly 模式：每行一个点 x,y，例如：10,20")
        if cfg.zone.poly:
            self.txt_poly.setPlainText("\n".join([f"{p[0]},{p[1]}" for p in cfg.zone.poly]))

        tips = QtWidgets.QLabel("提示：也可在左侧画布 Shift+左键点选，双击结束，会自动写入多边形。")
        tips.setStyleSheet("color:#94a3b8;")

        form.addRow("", self.cb_enable)
        form.addRow("检测区模式", self.cmb_mode)
        form.addRow("矩形 (x,y,w,h)", rect_row)
        form.addRow("多边形点列表", self.txt_poly)

        lay.addLayout(form)
        lay.addWidget(tips)

        btns = QtWidgets.QHBoxLayout()
        btn_apply = QtWidgets.QPushButton("应用")
        btn_close = QtWidgets.QPushButton("关闭")
        btn_apply.clicked.connect(self.on_apply)
        btn_close.clicked.connect(self.accept)
        btns.addWidget(btn_apply)
        btns.addStretch(1)
        btns.addWidget(btn_close)
        lay.addLayout(btns)

        self.cmb_mode.currentTextChanged.connect(self._refresh_enable_state)
        self._refresh_enable_state(self.cmb_mode.currentText())

    def _refresh_enable_state(self, mode):
        is_poly = (mode == "poly")
        self.txt_poly.setEnabled(is_poly)
        for w in [self.sp_rx, self.sp_ry, self.sp_rw, self.sp_rh]:
            w.setEnabled(not is_poly)

    def on_apply(self):
        z = self.cfg.zone
        z.enabled = self.cb_enable.isChecked()
        z.mode = self.cmb_mode.currentText()

        if z.mode == "rect":
            z.rect = (float(self.sp_rx.value()), float(self.sp_ry.value()),
                      float(self.sp_rw.value()), float(self.sp_rh.value()))
        else:
            pts = []
            for line in self.txt_poly.toPlainText().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    a, b = line.split(",")
                    pts.append((float(a), float(b)))
                except Exception:
                    pass
            z.poly = pts

        QtWidgets.QMessageBox.information(self, "提示", "检测区设置已应用。")


class CustomFrameDialog(QtWidgets.QDialog):
    """自定义数据帧：设置右侧“原始数据流输入”的显示过滤 ID"""
    def __init__(self, parent, cfg: AppConfig):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("自定义数据帧")
        self.setModal(True)
        self.resize(560, 260)

        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.cb_enable = QtWidgets.QCheckBox("启用原始流 ID 过滤（只影响右侧显示）")
        self.cb_enable.setChecked(cfg.rawview.enable_id_filter)

        self.edit_ids = QtWidgets.QLineEdit()
        self.edit_ids.setPlaceholderText("例如：0x600,0x701 或 1536,1793（十六进制/十进制都支持）")
        if cfg.rawview.id_filter_set:
            self.edit_ids.setText(",".join([f"0x{v:X}" for v in sorted(cfg.rawview.id_filter_set)]))

        form.addRow("", self.cb_enable)
        form.addRow("显示 ID 列表", self.edit_ids)

        lay.addLayout(form)

        tips = QtWidgets.QLabel("说明：过滤只影响“原始数据流输入”Tab 的显示，缓存仍保留全部帧。")
        tips.setStyleSheet("color:#94a3b8;")
        lay.addWidget(tips)

        btns = QtWidgets.QHBoxLayout()
        btn_apply = QtWidgets.QPushButton("应用")
        btn_close = QtWidgets.QPushButton("关闭")
        btn_apply.clicked.connect(self.on_apply)
        btn_close.clicked.connect(self.accept)
        btns.addWidget(btn_apply)
        btns.addStretch(1)
        btns.addWidget(btn_close)
        lay.addLayout(btns)

    def on_apply(self):
        self.cfg.rawview.enable_id_filter = self.cb_enable.isChecked()
        s = self.edit_ids.text().strip()
        ids = set()
        if s:
            for token in s.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    if token.lower().startswith("0x"):
                        ids.add(int(token, 16))
                    else:
                        ids.add(int(token, 10))
                except Exception:
                    pass
        self.cfg.rawview.id_filter_set = ids
        QtWidgets.QMessageBox.information(self, "提示", "自定义数据帧已应用。")


class DataDialog(QtWidgets.QDialog):
    def __init__(self, parent, raw_buffer: deque):
        super().__init__(parent)
        self.raw_buffer = raw_buffer

        self.setWindowTitle("雷达数据（原始帧缓存）")
        self.setModal(False)
        self.resize(820, 460)

        lay = QtWidgets.QVBoxLayout(self)
        self.txt = QtWidgets.QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setFont(QtGui.QFont("Consolas", 9))
        lay.addWidget(self.txt)

        btns = QtWidgets.QHBoxLayout()
        btn_refresh = QtWidgets.QPushButton("刷新")
        btn_export = QtWidgets.QPushButton("导出 TXT")
        btn_close = QtWidgets.QPushButton("关闭")
        btn_refresh.clicked.connect(self.refresh)
        btn_export.clicked.connect(self.export_txt)
        btn_close.clicked.connect(self.close)
        btns.addWidget(btn_refresh)
        btns.addWidget(btn_export)
        btns.addStretch(1)
        btns.addWidget(btn_close)
        lay.addLayout(btns)

        self.refresh()

    def refresh(self):
        lines = [line for _, line in list(self.raw_buffer)]
        self.txt.setPlainText("\n".join(lines[-800:]))

    def export_txt(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出为 TXT", "radar_raw.txt", "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            for _, line in self.raw_buffer:
                f.write(line + "\n")
        QtWidgets.QMessageBox.information(self, "提示", "导出成功。")


class CalibDialog(QtWidgets.QDialog):
    """标定：yaw + x/y offset；支持保存/加载；支持选中目标设原点"""
    def __init__(self, parent, cfg: AppConfig, get_selected_cb):
        super().__init__(parent)
        self.cfg = cfg
        self.get_selected_cb = get_selected_cb

        self.setWindowTitle("标定（安装位姿）")
        self.setModal(True)
        self.resize(560, 280)

        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.cb_enable = QtWidgets.QCheckBox("启用标定（对显示/过滤/检测区统一生效）")
        self.cb_enable.setChecked(cfg.calib.enabled)

        self.sp_yaw = QtWidgets.QDoubleSpinBox()
        self.sp_yaw.setRange(-180.0, 180.0)
        self.sp_yaw.setDecimals(2)
        self.sp_yaw.setSingleStep(0.5)
        self.sp_yaw.setValue(cfg.calib.yaw_deg)

        self.sp_xoff = QtWidgets.QDoubleSpinBox()
        self.sp_xoff.setRange(-1000.0, 1000.0)
        self.sp_xoff.setDecimals(3)
        self.sp_xoff.setSingleStep(0.05)
        self.sp_xoff.setValue(cfg.calib.x_offset)

        self.sp_yoff = QtWidgets.QDoubleSpinBox()
        self.sp_yoff.setRange(-1000.0, 1000.0)
        self.sp_yoff.setDecimals(3)
        self.sp_yoff.setSingleStep(0.05)
        self.sp_yoff.setValue(cfg.calib.y_offset)

        form.addRow("", self.cb_enable)
        form.addRow("Yaw 偏航角 (deg)", self.sp_yaw)
        form.addRow("X 平移偏置 (m)", self.sp_xoff)
        form.addRow("Y 平移偏置 (m)", self.sp_yoff)

        lay.addLayout(form)

        tips = QtWidgets.QLabel(
            "说明：\n"
            "1) yaw 对 (x,y) 旋转，逆时针为正；2) 再加 x/y 偏置；\n"
            "3) 表格选中一个目标后，可“一键设为原点”（把该目标平移到 0,0）。"
        )
        tips.setStyleSheet("color:#94a3b8;")
        lay.addWidget(tips)

        btns = QtWidgets.QHBoxLayout()
        btn_apply = QtWidgets.QPushButton("应用")
        btn_zero = QtWidgets.QPushButton("以选中目标设为原点")
        btn_save = QtWidgets.QPushButton("保存标定...")
        btn_load = QtWidgets.QPushButton("加载标定...")
        btn_close = QtWidgets.QPushButton("关闭")

        btn_apply.clicked.connect(self.apply)
        btn_zero.clicked.connect(self.zero_by_selected)
        btn_save.clicked.connect(self.save_calib)
        btn_load.clicked.connect(self.load_calib)
        btn_close.clicked.connect(self.accept)

        btns.addWidget(btn_apply)
        btns.addWidget(btn_zero)
        btns.addStretch(1)
        btns.addWidget(btn_save)
        btns.addWidget(btn_load)
        btns.addWidget(btn_close)
        lay.addLayout(btns)

    def apply(self):
        c = self.cfg.calib
        c.enabled = self.cb_enable.isChecked()
        c.yaw_deg = float(self.sp_yaw.value())
        c.x_offset = float(self.sp_xoff.value())
        c.y_offset = float(self.sp_yoff.value())
        QtWidgets.QMessageBox.information(self, "提示", "标定参数已应用。")

    def zero_by_selected(self):
        sel = self.get_selected_cb()
        if not sel:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在右侧表格选中一个目标。")
            return
        # 让选中目标的“当前标定后坐标”落在 (0,0)：offset += (-x,-y)
        x = float(sel.get("x", 0.0))
        y = float(sel.get("y", 0.0))
        self.sp_xoff.setValue(float(self.sp_xoff.value()) - x)
        self.sp_yoff.setValue(float(self.sp_yoff.value()) - y)
        self.apply()

    def save_calib(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存标定参数", "calib.json", "JSON Files (*.json)")
        if not path:
            return
        self.apply()
        d = asdict(self.cfg.calib)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        QtWidgets.QMessageBox.information(self, "提示", "保存成功。")

    def load_calib(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "加载标定参数", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            d = json.loads(open(path, "r", encoding="utf-8").read())
            for k, v in d.items():
                if hasattr(self.cfg.calib, k):
                    setattr(self.cfg.calib, k, v)
            self.cb_enable.setChecked(bool(self.cfg.calib.enabled))
            self.sp_yaw.setValue(float(self.cfg.calib.yaw_deg))
            self.sp_xoff.setValue(float(self.cfg.calib.x_offset))
            self.sp_yoff.setValue(float(self.cfg.calib.y_offset))
            QtWidgets.QMessageBox.information(self, "提示", "加载成功。")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载失败：{e}")


class AboutDialog(QtWidgets.QMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("关于")
        self.setIcon(QtWidgets.QMessageBox.Information)
        self.setText(
            "ARS408-21 上位机（Cluster 模式）- HURYS 风格增强版\n\n"
            "交互：\n"
            "- 滚轮：缩放\n"
            "- 右键：重置视野\n"
            "- Shift+左键：画多边形点\n"
            "- 双击：结束多边形\n\n"
            "增强：\n"
            "- 右侧 Dock 快速调参（实时）\n"
            "- 标定：Yaw + X/Y Offset（保存/加载、选中目标设原点）\n"
        )


# =========================
# 右侧 Dock：快速调参面板（实时）
# =========================
class TuningDock(QtWidgets.QDockWidget):
    def __init__(self, parent, cfg: AppConfig, on_changed_cb):
        super().__init__("快速调参", parent)
        self.cfg = cfg
        self.on_changed_cb = on_changed_cb

        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)

        w = QtWidgets.QWidget()
        self.setWidget(w)

        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        def box(title):
            gb = QtWidgets.QGroupBox(title)
            gb.setStyleSheet(
                "QGroupBox{color:#e5e7eb;border:1px solid #334155;border-radius:10px;margin-top:8px;}"
                "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 10px;}"
            )
            v = QtWidgets.QVBoxLayout(gb)
            v.setContentsMargins(10, 10, 10, 10)
            v.setSpacing(8)
            lay.addWidget(gb)
            return v

        # 视图
        v1 = box("视图")
        self.cb_id = QtWidgets.QCheckBox("显示 ID 标签")
        self.cb_id.setChecked(cfg.view.show_id_label)
        self.cb_rcs_size = QtWidgets.QCheckBox("点大小按 RCS")
        self.cb_rcs_size.setChecked(cfg.view.size_by_rcs)
        self.cb_grid = QtWidgets.QCheckBox("显示网格")
        self.cb_grid.setChecked(cfg.view.show_grid)
        v1.addWidget(self.cb_id)
        v1.addWidget(self.cb_rcs_size)
        v1.addWidget(self.cb_grid)

        # 滤波
        v2 = box("抗噪滤波（实时）")

        def row_spin(label, widget):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(label))
            row.addStretch(1)
            row.addWidget(widget)
            v2.addLayout(row)

        f = cfg.filt
        self.sp_rcs = QtWidgets.QDoubleSpinBox(); self.sp_rcs.setRange(-64, 64); self.sp_rcs.setDecimals(1); self.sp_rcs.setSingleStep(0.5); self.sp_rcs.setValue(f.rcs_min)
        self.sp_speed = QtWidgets.QDoubleSpinBox(); self.sp_speed.setRange(0, 200); self.sp_speed.setDecimals(1); self.sp_speed.setSingleStep(1.0); self.sp_speed.setValue(f.speed_max)
        self.sp_ttl = QtWidgets.QDoubleSpinBox(); self.sp_ttl.setRange(0.1, 10.0); self.sp_ttl.setDecimals(2); self.sp_ttl.setSingleStep(0.1); self.sp_ttl.setValue(f.ttl_s)
        self.sp_hits = QtWidgets.QSpinBox(); self.sp_hits.setRange(1, 10); self.sp_hits.setValue(f.min_hits)
        self.sp_alpha = QtWidgets.QDoubleSpinBox(); self.sp_alpha.setRange(0.05, 0.95); self.sp_alpha.setDecimals(2); self.sp_alpha.setSingleStep(0.05); self.sp_alpha.setValue(f.ema_alpha)
        self.sp_jump = QtWidgets.QDoubleSpinBox(); self.sp_jump.setRange(0.0, 200.0); self.sp_jump.setDecimals(1); self.sp_jump.setSingleStep(1.0); self.sp_jump.setValue(f.max_jump_m)

        row_spin("RCS 最小(dB)", self.sp_rcs)
        row_spin("速度上限(m/s)", self.sp_speed)
        row_spin("TTL(s)", self.sp_ttl)
        row_spin("最小 Hits", self.sp_hits)
        row_spin("EMA alpha", self.sp_alpha)
        row_spin("最大跳变(m)", self.sp_jump)

        # 距离窗口
        v3 = box("距离窗口（Dist）")
        self.sp_dy0 = QtWidgets.QDoubleSpinBox(); self.sp_dy0.setRange(-500, 500); self.sp_dy0.setDecimals(1); self.sp_dy0.setSingleStep(1.0); self.sp_dy0.setValue(f.dist_long_min)
        self.sp_dy1 = QtWidgets.QDoubleSpinBox(); self.sp_dy1.setRange(-500, 500); self.sp_dy1.setDecimals(1); self.sp_dy1.setSingleStep(1.0); self.sp_dy1.setValue(f.dist_long_max)
        self.sp_dx0 = QtWidgets.QDoubleSpinBox(); self.sp_dx0.setRange(-500, 500); self.sp_dx0.setDecimals(1); self.sp_dx0.setSingleStep(1.0); self.sp_dx0.setValue(f.dist_lat_min)
        self.sp_dx1 = QtWidgets.QDoubleSpinBox(); self.sp_dx1.setRange(-500, 500); self.sp_dx1.setDecimals(1); self.sp_dx1.setSingleStep(1.0); self.sp_dx1.setValue(f.dist_lat_max)

        def row4(a, wa, b, wb):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(a)); row.addWidget(wa)
            row.addSpacing(8)
            row.addWidget(QtWidgets.QLabel(b)); row.addWidget(wb)
            v3.addLayout(row)

        row4("DistLong min", self.sp_dy0, "DistLong max", self.sp_dy1)
        row4("DistLat  min", self.sp_dx0, "DistLat  max", self.sp_dx1)

        lay.addStretch(1)

        # style
        w.setStyleSheet("""
            QWidget{background:#0b1220;color:#e5e7eb;}
            QLabel{color:#cbd5e1;}
            QSpinBox,QDoubleSpinBox{background:#020617;border:1px solid #334155;border-radius:8px;padding:4px;color:#e5e7eb;}
            QCheckBox{color:#e5e7eb;}
        """)

        # signals -> live apply
        for sigw in [
            self.cb_id, self.cb_rcs_size, self.cb_grid,
            self.sp_rcs, self.sp_speed, self.sp_ttl, self.sp_hits, self.sp_alpha, self.sp_jump,
            self.sp_dy0, self.sp_dy1, self.sp_dx0, self.sp_dx1
        ]:
            if isinstance(sigw, QtWidgets.QCheckBox):
                sigw.toggled.connect(self.apply_live)
            else:
                sigw.valueChanged.connect(self.apply_live)

    def apply_live(self):
        v = self.cfg.view
        v.show_id_label = self.cb_id.isChecked()
        v.size_by_rcs = self.cb_rcs_size.isChecked()
        v.show_grid = self.cb_grid.isChecked()

        f = self.cfg.filt
        f.rcs_min = float(self.sp_rcs.value())
        f.speed_max = float(self.sp_speed.value())
        f.ttl_s = float(self.sp_ttl.value())
        f.min_hits = int(self.sp_hits.value())
        f.ema_alpha = float(self.sp_alpha.value())
        f.max_jump_m = float(self.sp_jump.value())

        f.dist_long_min = float(self.sp_dy0.value())
        f.dist_long_max = float(self.sp_dy1.value())
        f.dist_lat_min = float(self.sp_dx0.value())
        f.dist_lat_max = float(self.sp_dx1.value())

        self.on_changed_cb()


# =========================
# 主窗口
# =========================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, bus=None, channel=None, bitrate=500000):
        super().__init__()

        self.cfg = AppConfig()

        self.bus: Optional[can.Bus] = bus
        self.channel = channel
        self.bitrate = bitrate

        self.setWindowTitle("HURYS雷达目标显示与标定软件V1.0（增强仿）")
        self.resize(1560, 940)

        self._running = True
        self._paused = False
        self._lock = threading.Lock()

        self.frame_count = 0
        self.cluster_near = 0
        self.cluster_far = 0
        self.meas_counter = None

        self.tracks: Dict[int, Track] = {}

        # 原始数据缓存：保存全部帧（右侧显示可过滤）
        self.raw_buffer = deque(maxlen=8000)
        self._raw_last_len = 0

        self._selected_obj: Optional[dict] = None
        self._latest_display: List[dict] = []

        # 左侧画布 + 右侧面板
        self.canvas = RadarCanvas(self.cfg, self)
        self.canvas.sig_polygon_finished.connect(self.on_polygon_finished)

        self.right = RightPanel(self)
        self.right.btn_clear_raw.clicked.connect(self.on_clear_raw)
        self.right.sig_selected_obj.connect(self._on_selected_obj)

        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1120, 420])
        self.setCentralWidget(splitter)

        # 顶部菜单/工具栏
        self._build_top_controls()

        # 右侧 Dock：快速调参
        self.tuning_dock = TuningDock(self, self.cfg, self._on_tuning_changed)
        self.addDockWidget(Qt.RightDockWidgetArea, self.tuning_dock)

        # 底部状态栏
        self.status_label = QtWidgets.QLabel("未连接")
        self.statusBar().addPermanentWidget(self.status_label)

        # UI 刷新
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start(80)

        # 仿真
        self._sim_phase = 0.0

        # 接收线程
        self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.recv_thread.start()

        if self.bus is not None and self.channel is not None:
            self.cfg.simulate = False

        # 全局样式
        self._apply_app_style()

    def _apply_app_style(self):
        self.setStyleSheet("""
            QMainWindow{background:#0b1220;}
            QMenuBar{background:#0b1220;color:#e5e7eb;border-bottom:1px solid #334155;}
            QMenuBar::item:selected{background:#111827;}
            QMenu{background:#0b1220;color:#e5e7eb;border:1px solid #334155;}
            QMenu::item:selected{background:#1d4ed8;}
            QToolBar{background:#0b1220;border-bottom:1px solid #334155;}
            QStatusBar{background:#0b1220;color:#e5e7eb;border-top:1px solid #334155;}
        """)

    def _icon(self, sp):
        return self.style().standardIcon(sp)

    def _build_top_controls(self):
        tb = QtWidgets.QToolBar()
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(18, 18))
        self.addToolBar(Qt.TopToolBarArea, tb)

        act_op = QtWidgets.QAction(self._icon(QtWidgets.QStyle.SP_FileDialogListView), "操作", self)
        act_radar = QtWidgets.QAction(self._icon(QtWidgets.QStyle.SP_DriveNetIcon), "雷达配置", self)
        act_filter = QtWidgets.QAction(self._icon(QtWidgets.QStyle.SP_FileDialogDetailedView), "过滤器", self)
        act_zone = QtWidgets.QAction(self._icon(QtWidgets.QStyle.SP_DialogYesButton), "检测区", self)
        act_calib = QtWidgets.QAction(self._icon(QtWidgets.QStyle.SP_BrowserReload), "标定", self)
        act_data = QtWidgets.QAction(self._icon(QtWidgets.QStyle.SP_FileIcon), "雷达数据", self)
        act_custom = QtWidgets.QAction(self._icon(QtWidgets.QStyle.SP_ComputerIcon), "自定义帧", self)
        act_poly = QtWidgets.QAction(self._icon(QtWidgets.QStyle.SP_DirIcon), "多边形", self)
        act_about = QtWidgets.QAction(self._icon(QtWidgets.QStyle.SP_MessageBoxInformation), "关于", self)

        tb.addAction(act_op)
        tb.addSeparator()
        tb.addAction(act_radar)
        tb.addAction(act_filter)
        tb.addAction(act_zone)
        tb.addAction(act_calib)
        tb.addAction(act_data)
        tb.addAction(act_custom)
        tb.addAction(act_poly)
        tb.addSeparator()
        tb.addAction(act_about)

        act_op.triggered.connect(self.menu_operation)
        act_radar.triggered.connect(self.menu_radar_config)
        act_filter.triggered.connect(self.menu_filter)
        act_zone.triggered.connect(self.menu_zone)
        act_calib.triggered.connect(self.menu_calib)
        act_data.triggered.connect(self.menu_data)
        act_custom.triggered.connect(self.menu_custom_frame)
        act_poly.triggered.connect(self.menu_polygon_settings)
        act_about.triggered.connect(self.menu_about)

        mb = self.menuBar()

        m_op = mb.addMenu("操作")
        m_radar = mb.addMenu("雷达配置")
        m_filter = mb.addMenu("过滤器设置")
        m_zone = mb.addMenu("检测区设置")
        m_calib = mb.addMenu("标定")
        m_data = mb.addMenu("雷达数据")
        m_custom = mb.addMenu("自定义数据帧")
        m_poly = mb.addMenu("多边形设置")
        m_about = mb.addMenu("关于")

        m_op.addAction("自动连接", self.auto_connect)
        m_op.addAction("暂停/继续接收", self.toggle_pause)
        m_op.addAction("清空目标/轨迹", self.clear_tracks)
        m_op.addAction("清空原始数据流输入", self.on_clear_raw)
        m_op.addSeparator()
        m_op.addAction("重置视野", self.canvas.reset_view)
        m_op.addSeparator()
        m_op.addAction("保存全部配置...", self.save_config)
        m_op.addAction("加载全部配置...", self.load_config)
        m_op.addSeparator()
        m_op.addAction("退出", self.close)

        m_radar.addAction("雷达配置", self.menu_radar_config)
        m_filter.addAction("过滤器设置", self.menu_filter)
        m_zone.addAction("检测区设置", self.menu_zone)
        m_calib.addAction("标定（Yaw + Offset）", self.menu_calib)
        m_data.addAction("雷达数据（缓存查看/导出）", self.menu_data)
        m_custom.addAction("自定义数据帧（原始流过滤）", self.menu_custom_frame)

        m_poly.addAction("清空多边形", self.clear_polygon)
        m_poly.addAction("导入多边形...", self.import_polygon)
        m_poly.addAction("导出多边形...", self.export_polygon)
        m_poly.addSeparator()
        m_poly.addAction("提示：Shift+左键点选，双击结束", self.menu_polygon_tip)

        m_about.addAction("关于", self.menu_about)

    def _on_tuning_changed(self):
        # 视图参数变化需要重绘坐标轴
        self.canvas.apply_view()

    def _on_selected_obj(self, obj: dict):
        self._selected_obj = obj

    def get_selected_obj(self) -> Optional[dict]:
        return dict(self._selected_obj) if self._selected_obj else None

    # -------- 菜单动作 --------
    def menu_operation(self):
        QtWidgets.QMessageBox.information(
            self, "操作",
            "常用操作：\n"
            "- 操作→自动连接 / 暂停 / 清空 / 保存/加载配置\n"
            "- 右键重置视野，滚轮缩放\n"
            "- Shift+左键绘制多边形，双击结束\n"
            "- 自定义数据帧：可过滤右侧原始流显示 ID\n"
            "- 标定：Yaw + Offset（可选中目标一键设为原点）\n"
        )

    def menu_radar_config(self):
        dlg = RadarConfigDialog(self, self.cfg, self.connect_radar, self.disconnect_radar)
        dlg.exec_()

    def menu_filter(self):
        dlg = FilterDialog(self, self.cfg)
        dlg.exec_()

    def menu_zone(self):
        dlg = ZoneDialog(self, self.cfg)
        dlg.exec_()
        self.canvas.apply_view()

    def menu_calib(self):
        dlg = CalibDialog(self, self.cfg, self.get_selected_obj)
        dlg.exec_()
        self.canvas.apply_view()

    def menu_data(self):
        dlg = DataDialog(self, self.raw_buffer)
        dlg.show()

    def menu_custom_frame(self):
        dlg = CustomFrameDialog(self, self.cfg)
        dlg.exec_()
        self._raw_last_len = 0
        self.right.clear_raw()

    def menu_polygon_settings(self):
        QtWidgets.QMessageBox.information(
            self, "多边形设置",
            "多边形交互：\n"
            "- Shift + 左键：依次点选多边形顶点\n"
            "- 双击：结束并应用\n"
            "- 菜单：可清空/导入/导出\n"
        )

    def menu_polygon_tip(self):
        QtWidgets.QMessageBox.information(self, "提示", "Shift+左键点选多边形顶点，双击结束并应用。")

    def menu_about(self):
        AboutDialog(self).exec_()

    # -------- 配置保存/加载 --------
    def save_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存配置", "ars408_config.json", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.cfg.to_json_dict(), f, ensure_ascii=False, indent=2)
            QtWidgets.QMessageBox.information(self, "提示", "保存成功。")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"保存失败：{e}")

    def load_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "加载配置", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            d = json.loads(open(path, "r", encoding="utf-8").read())
            self.cfg = AppConfig.from_json_dict(d)

            # 让各组件引用最新 cfg
            self.canvas.cfg = self.cfg
            self.tuning_dock.cfg = self.cfg
            self.tuning_dock.apply_live()  # 同步回 UI 并触发重绘

            self._raw_last_len = 0
            self.right.clear_raw()
            self.canvas.apply_view()
            QtWidgets.QMessageBox.information(self, "提示", "加载成功。")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载失败：{e}")

    # -------- 连接/断开 --------
    def connect_radar(self, channel, _unused, bitrate: int, auto: bool = True):
        self.disconnect_radar()
        self.bitrate = bitrate

        if self.cfg.simulate:
            self.bus = None
            self.channel = -1
            self.status_label.setText("仿真模式：已启用")
            return

        if auto:
            bus, ch = open_canalyst_auto(bitrate=bitrate)
            self.bus = bus
            self.channel = ch
        else:
            ch = int(channel)
            bus = can.Bus(interface="canalystii", channel=ch, bitrate=bitrate)
            self.bus = bus
            self.channel = ch

        self.status_label.setText(f"已连接：CANalyst-II ch{self.channel} @ {self.bitrate}")

    def disconnect_radar(self):
        try:
            if self.bus is not None:
                self.bus.shutdown()
        except Exception:
            pass
        self.bus = None
        self.channel = None

    def auto_connect(self):
        self.cfg.simulate = False
        try:
            bus, ch = open_canalyst_auto(bitrate=500000)
            self.disconnect_radar()
            self.bus = bus
            self.channel = ch
            self.bitrate = 500000
            self.status_label.setText(f"已连接：CANalyst-II ch{ch} @ 500000")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "毫米波雷达启动失败", str(e))
            self.status_label.setText("未连接（可切换仿真）")

    # -------- 操作 --------
    def toggle_pause(self):
        self._paused = not self._paused
        self.status_label.setText("已暂停接收" if self._paused else "继续接收")

    def clear_tracks(self):
        with self._lock:
            self.tracks.clear()
        self.status_label.setText("已清空轨迹/目标")

    def on_clear_raw(self):
        with self._lock:
            self.raw_buffer.clear()
        self._raw_last_len = 0
        self.right.clear_raw()
        self.status_label.setText("已清空原始数据流输入")

    # -------- 多边形 --------
    def on_polygon_finished(self, pts):
        self.cfg.zone.enabled = True
        self.cfg.zone.mode = "poly"
        self.cfg.zone.poly = pts
        self.canvas.apply_view()
        self.status_label.setText(f"多边形检测区已更新：{len(pts)} 点")

    def clear_polygon(self):
        self.cfg.zone.enabled = False
        self.cfg.zone.poly = []
        if self.cfg.zone.mode == "poly":
            self.cfg.zone.mode = "rect"
        self.canvas.apply_view()
        self.status_label.setText("已清空多边形检测区")

    def export_polygon(self):
        if not self.cfg.zone.poly or len(self.cfg.zone.poly) < 3:
            QtWidgets.QMessageBox.warning(self, "提示", "当前没有有效多边形可导出。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出多边形", "polygon.txt", "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            for x, y in self.cfg.zone.poly:
                f.write(f"{x},{y}\n")
        QtWidgets.QMessageBox.information(self, "提示", "导出成功。")

    def import_polygon(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "导入多边形", "", "Text Files (*.txt)")
        if not path:
            return
        pts = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    a, b = line.split(",")
                    pts.append((float(a), float(b)))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"导入失败：{e}")
            return

        if len(pts) < 3:
            QtWidgets.QMessageBox.warning(self, "提示", "导入的点数不足 3，无法形成多边形。")
            return

        self.cfg.zone.enabled = True
        self.cfg.zone.mode = "poly"
        self.cfg.zone.poly = pts
        self.canvas.apply_view()
        self.status_label.setText(f"已导入多边形：{len(pts)} 点")

    # -------- 原始数据缓存与显示 --------
    def _push_raw(self, msg: can.Message):
        ts = time.strftime("%H:%M:%S", time.localtime())
        data_str = " ".join([f"{b:02X}" for b in msg.data])
        line = f"{ts}  ID=0x{msg.arbitration_id:03X}  DLC={msg.dlc}  {data_str}"
        self.raw_buffer.append((msg.arbitration_id, line))

    def _raw_lines_for_ui_increment(self):
        cur_len = len(self.raw_buffer)
        if cur_len < self._raw_last_len:
            self._raw_last_len = 0
        new = list(self.raw_buffer)[self._raw_last_len:cur_len]
        self._raw_last_len = cur_len
        if not new:
            return []

        rv = self.cfg.rawview
        if rv.enable_id_filter and rv.id_filter_set:
            lines = [line for mid, line in new if mid in rv.id_filter_set]
        else:
            lines = [line for _, line in new]
        return lines

    # -------- 标定：把原始 obj 转为标定后坐标 --------
    def _apply_calib_to_obj(self, obj: dict) -> dict:
        # 统一在进入 Track 之前做标定：保证 EMA/跳变/区域判断与显示一致
        x0, y0 = float(obj.get("x", 0.0)), float(obj.get("y", 0.0))
        x1, y1 = self.cfg.calib.apply_xy(x0, y0)
        o = dict(obj)
        o["x"] = x1
        o["y"] = y1
        return o

    # -------- 接收线程 --------
    def recv_loop(self):
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            if self.cfg.simulate:
                time.sleep(0.05)
                self._simulate_step()
                continue

            if self.bus is None:
                time.sleep(0.05)
                continue

            try:
                msg = self.bus.recv(0.1)
            except Exception:
                continue
            if msg is None:
                continue

            now_ts = time.time()

            with self._lock:
                self.frame_count += 1
                self._push_raw(msg)

            mid = msg.arbitration_id
            if mid == 0x600:
                st = decode_ars408_cluster_status(bytes(msg.data))
                if st:
                    with self._lock:
                        self.cluster_near = st["near"]
                        self.cluster_far = st["far"]
                        self.meas_counter = st["meas_counter"]

            elif mid == 0x701:
                obj = decode_ars408_cluster_general(msg)
                if not obj:
                    continue

                # 距离窗口过滤（先用原始 dist_* 做）
                f = self.cfg.filt
                if not (f.dist_long_min <= obj["dist_long"] <= f.dist_long_max):
                    continue
                if not (f.dist_lat_min <= obj["dist_lat"] <= f.dist_lat_max):
                    continue

                # 标定后坐标（进入 Track）
                obj = self._apply_calib_to_obj(obj)

                with self._lock:
                    tid = int(obj["id"])
                    tr = self.tracks.get(tid)
                    if tr is None:
                        tr = Track(tid=tid)
                        self.tracks[tid] = tr
                    tr.update(obj, now_ts, alpha=f.ema_alpha, max_jump=f.max_jump_m)

    def _simulate_step(self):
        now_ts = time.time()
        f = self.cfg.filt

        self._sim_phase += 0.06
        base_targets = [
            (10.0 * math.sin(self._sim_phase * 0.8), 25.0 + 2.0 * math.cos(self._sim_phase * 0.6), 8.0),
            (-18.0 + 1.5 * math.cos(self._sim_phase), 60.0 + 2.5 * math.sin(self._sim_phase * 0.7), -2.0),
            (20.0 + 2.0 * math.cos(self._sim_phase * 0.5), 85.0, 12.0),
        ]

        with self._lock:
            fake_line = f"{time.strftime('%H:%M:%S')}  SIM  targets={len(base_targets)}"
            self.raw_buffer.append((0xFFFF, fake_line))

        for k, (x, y, rcs) in enumerate(base_targets, start=1):
            obj = {
                "id": k, "dist_long": y, "dist_lat": x,
                "vrel_long": 0.0, "vrel_lat": 0.0,
                "rcs": rcs, "x": x, "y": y,
            }
            # 距离窗口同样按 dist_* 筛
            if not (f.dist_long_min <= obj["dist_long"] <= f.dist_long_max):
                continue
            if not (f.dist_lat_min <= obj["dist_lat"] <= f.dist_lat_max):
                continue

            obj = self._apply_calib_to_obj(obj)

            with self._lock:
                tr = self.tracks.get(k)
                if tr is None:
                    tr = Track(tid=k)
                    self.tracks[k] = tr
                tr.update(obj, now_ts, alpha=f.ema_alpha, max_jump=f.max_jump_m)

        if np.random.rand() < 0.25:
            nid = int(np.random.randint(30, 60))
            x = float(np.random.uniform(-60, 60))
            y = float(np.random.uniform(-10, 140))
            rcs = float(np.random.uniform(-35, 5))
            obj = {
                "id": nid, "dist_long": y, "dist_lat": x,
                "vrel_long": float(np.random.uniform(-20, 20)),
                "vrel_lat": float(np.random.uniform(-10, 10)),
                "rcs": rcs, "x": x, "y": y,
            }
            if not (f.dist_long_min <= obj["dist_long"] <= f.dist_long_max):
                return
            if not (f.dist_lat_min <= obj["dist_lat"] <= f.dist_lat_max):
                return

            obj = self._apply_calib_to_obj(obj)

            with self._lock:
                tr = self.tracks.get(nid)
                if tr is None:
                    tr = Track(tid=nid)
                    self.tracks[nid] = tr
                tr.update(obj, now_ts, alpha=f.ema_alpha, max_jump=f.max_jump_m)

    # -------- 检测区/过滤 --------
    def _passes_zone(self, x: float, y: float) -> bool:
        z = self.cfg.zone
        if not z.enabled:
            return True
        if z.mode == "rect":
            rx, ry, w, h = z.rect
            return (rx <= x <= rx + w) and (ry <= y <= ry + h)
        pts = z.poly
        if len(pts) < 3:
            return True
        inside = False
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            inter = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            if inter:
                inside = not inside
            j = i
        return inside

    def _passes_filters(self, o: dict) -> bool:
        f = self.cfg.filt
        if float(o.get("rcs", -999)) < f.rcs_min:
            return False
        v = math.hypot(float(o.get("vrel_long", 0.0)), float(o.get("vrel_lat", 0.0)))
        if f.speed_max > 0 and v > f.speed_max:
            return False
        return True

    def _purge_tracks(self, now_ts: float):
        ttl = self.cfg.filt.ttl_s
        dead = []
        for tid, tr in self.tracks.items():
            if (now_ts - tr.last_seen_ts) > ttl:
                dead.append(tid)
        for tid in dead:
            self.tracks.pop(tid, None)

    # -------- UI 刷新 --------
    def refresh_ui(self):
        now_ts = time.time()

        with self._lock:
            self._purge_tracks(now_ts)

            display = []
            for tr in self.tracks.values():
                if tr.hits < self.cfg.filt.min_hits:
                    continue
                o = tr.to_display_obj()

                # 视野裁剪
                if not (self.cfg.view.x_min <= o["x"] <= self.cfg.view.x_max):
                    continue
                if not (self.cfg.view.y_min <= o["y"] <= self.cfg.view.y_max):
                    continue

                if not self._passes_filters(o):
                    continue
                if not self._passes_zone(o["x"], o["y"]):
                    continue

                display.append(o)

            display.sort(key=lambda d: int(d.get("id", 0)))
            self._latest_display = list(display)

            fc = self.frame_count
            near, far = self.cluster_near, self.cluster_far
            mc = self.meas_counter

            new_lines = self._raw_lines_for_ui_increment()

        self.canvas.update_objects(display)
        self.right.update_table(display)

        if new_lines:
            self.right.append_raw_lines(new_lines)

        conn = "仿真" if self.cfg.simulate else ("未连接" if self.bus is None else f"ch{self.channel}@{self.bitrate}")
        status = f"{conn} | 帧数:{fc} | tracks:{len(self.tracks)} | 显示:{len(display)} | 近:{near} 远:{far}"
        if mc is not None:
            status += f" | meas:{mc}"
        if self.cfg.calib.enabled:
            status += f" | calib:yaw={self.cfg.calib.yaw_deg:.1f}° off=({self.cfg.calib.x_offset:.2f},{self.cfg.calib.y_offset:.2f})"

        self.status_label.setText(status)
        self.right.set_status_text(status)

    def closeEvent(self, event):
        self._running = False
        time.sleep(0.1)
        try:
            if self.bus is not None:
                self.bus.shutdown()
        except Exception:
            pass
        super().closeEvent(event)


# =========================
# 入口
# =========================
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Microsoft YaHei", 9))
    app.setStyle("Fusion")

    win = MainWindow()

    # 启动时尝试自动连接（失败也能进界面）
    try:
        bus, ch = open_canalyst_auto(bitrate=500000)
        win.bus = bus
        win.channel = ch
        win.bitrate = 500000
        win.cfg.simulate = False
    except Exception as e:
        QtWidgets.QMessageBox.critical(win, "毫米波雷达启动失败", str(e))
        # 可选：默认进仿真
        # win.cfg.simulate = True

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
