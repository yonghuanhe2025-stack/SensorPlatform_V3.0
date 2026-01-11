# c16_qt_dense_viewer_geo_table.py
# -*- coding: utf-8 -*-

import os
import sys
import socket
import struct
import math
import time
import queue
import multiprocessing as mp
from dataclasses import dataclass
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTableWidget, QTableWidgetItem,
    QHBoxLayout, QSplitter, QToolBar, QAction,
    QLabel, QComboBox, QLineEdit, QCheckBox,
    QDialog, QFormLayout, QPushButton, QMessageBox,
    QListWidget, QListWidgetItem, QFileDialog
)

import pyqtgraph as pg
import pyqtgraph.opengl as gl


# ============================================================
# 1) CPU 亲和性（可选 psutil）
# ============================================================

def set_process_affinity(core_ids):
    core_ids = list(core_ids or [])
    if not core_ids:
        return
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(core_ids))
            return
    except Exception:
        pass
    try:
        import psutil
        psutil.Process().cpu_affinity(core_ids)
    except Exception:
        pass


def normalize_core_list(requested, cpu_count):
    cpu_count = cpu_count or 8
    requested = [c for c in requested if isinstance(c, int) and 0 <= c < cpu_count]
    out, seen = [], set()
    for c in requested:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


CPU_COUNT = os.cpu_count() or 8
WORKER_CORES = normalize_core_list([0, 1, 2, 3], CPU_COUNT)
GUI_CORES = normalize_core_list([4, 5, 6, 7], CPU_COUNT)
if not WORKER_CORES:
    WORKER_CORES = normalize_core_list(list(range(min(4, CPU_COUNT))), CPU_COUNT)
if not GUI_CORES:
    rest = [c for c in range(CPU_COUNT) if c not in set(WORKER_CORES)]
    GUI_CORES = rest if rest else WORKER_CORES[:]


# ============================================================
# 2) C16-500B 参数
# ============================================================

VERT_ANGLES_DEG = {
    0: -16, 1: 0, 2: -14, 3: 2,
    4: -12, 5: 4, 6: -10, 7: 6,
    8: -8, 9: 8, 10: -6, 11: 10,
    12: -4, 13: 12, 14: -2, 15: 14,
}

UDP_IP = "0.0.0.0"
UDP_PORT = 2368


# ============================================================
# 3) 显示 & 实时性（观感 / 丝滑 / 防闪）
# ============================================================

MAX_POINTS = 140000
SCALE = 1.0 / 8.0

POINT_SIZE_WORLD_BASE = 0.020 / 3.0
SCATTER_PX_MODE = False

UPDATE_INTERVAL_SEC = 0.020
POLL_QUEUE_INTERVAL_MS = 5

MAX_GL_POINTS = 420000

TABLE_ROWS = 220
TABLE_UPDATE_INTERVAL = 0.18
TABLE_RESIZE_INTERVAL = 1.0

RAW_QUEUE_MAXSIZE = 384
PARSED_QUEUE_MAXSIZE = 128
MAX_DRAIN_PER_TICK = 18
POLL_APPLY_FRAMES = 10

WINDOW_HYST_SEC = 0.06

PERSIST_SEC_DEFAULT = 2.6
FADE_TAU = 0.62
ALPHA_FLOOR = 0.05
ALPHA_POWER = 0.90

ADAPT_ENABLE_DEFAULT = True
ADAPT_TARGET_FPS = 38.0
ADAPT_MIN_DENSITY = 0.8
ADAPT_MAX_DENSITY = 8.0
ADAPT_DENSITY_STEP = 0.15
ADAPT_PERSIST_MIN = 1.2
ADAPT_PERSIST_MAX = 3.2
ADAPT_STEP_SEC = 0.12

USE_FIXED_COLOR_RANGE_DEFAULT = True
FIXED_R_MIN_DEFAULT = 0.0
FIXED_R_MAX_DEFAULT = 120.0
COLOR_RANGE_EMA_BETA = 0.10

HEAT_BIN_M = 0.25
HEAT_MAX_BINS = 256
HEAT_LOG_COMPRESS = True
HEAT_GAMMA = 0.85
HEAT_MAX_EMA_BETA = 0.15

YELLOW_START = 0.30
YELLOW_END = 0.82

DYNAMIC_POINT_SIZE_DEFAULT = True
POINT_SIZE_NEAR = POINT_SIZE_WORLD_BASE * 1.55
POINT_SIZE_FAR = POINT_SIZE_WORLD_BASE * 0.70
POINT_SIZE_R_MIN = 2.0
POINT_SIZE_R_MAX = 90.0

Z_MIN_DEFAULT = -2.0
Z_MAX_DEFAULT = 4.0


# ============================================================
# 4) 渲染模式：点云 / 立体马赛克(方块)
# ============================================================

MODE_POINT = 0
MODE_CUBE_MOSAIC = 2
DEFAULT_MODE = MODE_POINT

GRID_SIZE = 50.0 * SCALE
GRID_SPACING = 1.0 * SCALE

DEFAULT_VOXEL_M = 0.80
MAX_CUBES_RENDER = 7000

VOXEL_EMA_ENABLE_DEFAULT = True
VOXEL_EMA_BETA = 0.35
VOXEL_CACHE_MAX = 25000


# ============================================================
# 5) 地理坐标
# ============================================================

R_EARTH = 6378137.0


def euler_to_rot(roll, pitch, yaw):
    cr = math.cos(roll); sr = math.sin(roll)
    cp = math.cos(pitch); sp = math.sin(pitch)
    cy = math.cos(yaw); sy = math.sin(yaw)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,    0, 1]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp],
                   [0,  1, 0],
                   [-sp, 0, cp]], dtype=np.float32)
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float32)
    return Rz @ Ry @ Rx


def quat_to_rot(qx, qy, qz, qw):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3, dtype=np.float32)
    qw, qx, qy, qz = q / n
    R = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx ** 2 + qy ** 2)]
    ], dtype=np.float32)
    return R


def enu_scaled_to_lla_sphere(p_enu_scaled, lat0_deg, lon0_deg, h0, scale=SCALE):
    p_enu = p_enu_scaled / scale
    east = p_enu[..., 0]
    north = p_enu[..., 1]
    up = p_enu[..., 2]

    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    lat = lat0 + north / R_EARTH
    lon = lon0 + east / (R_EARTH * math.cos(lat0))
    h = h0 + up

    return math.degrees(lat), math.degrees(lon), h


class GeoConverter:
    def __init__(self):
        self._pyproj_ok = False
        try:
            import pyproj  # noqa
            self._pyproj_ok = True
        except Exception:
            self._pyproj_ok = False

    @property
    def pyproj_ok(self) -> bool:
        return self._pyproj_ok

    def enu_to_lla(self, p_enu_scaled, lat0, lon0, h0, scale=SCALE, high_precision=False):
        if not high_precision or (not self._pyproj_ok):
            return enu_scaled_to_lla_sphere(p_enu_scaled, lat0, lon0, h0, scale=scale)

        import pyproj
        transformer_to_ecef = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
        transformer_to_lla = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
        x0, y0, z0 = transformer_to_ecef.transform(lon0, lat0, h0)

        lat0r = math.radians(lat0)
        lon0r = math.radians(lon0)
        slat, clat = math.sin(lat0r), math.cos(lat0r)
        slon, clon = math.sin(lon0r), math.cos(lon0r)
        R = np.array([[-slon, -slat*clon,  clat*clon],
                      [ clon, -slat*slon,  clat*slon],
                      [ 0.0,       clat,       slat]], dtype=np.float64)

        p = (p_enu_scaled / scale).astype(np.float64)
        if p.ndim == 1:
            p = p[None, :]
        ecef = (p @ R.T) + np.array([x0, y0, z0], dtype=np.float64)[None, :]
        lon, lat, h = transformer_to_lla.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
        return lat, lon, h


GEO = GeoConverter()


# ============================================================
# 6) MSOP 解析
# ============================================================

def extract_msop_payload(udp_data: bytes):
    l = len(udp_data)
    if l == 1206:
        return udp_data
    if l >= 1248:
        return udp_data[42:42 + 1206]
    return None


def parse_msop(payload: bytes):
    if payload is None or len(payload) != 1206:
        return None

    rows = []
    offset = 0

    for _ in range(12):
        block = payload[offset:offset + 100]
        offset += 100

        flag_le, = struct.unpack_from("<H", block, 0)
        if flag_le not in (0xEEFF, 0xFFEE):
            continue

        az_raw, = struct.unpack_from("<H", block, 2)
        az_deg = az_raw / 100.0
        az_rad = math.radians(az_deg)

        for i in range(32):
            ch_offset = 4 + i * 3
            dist_raw, intensity = struct.unpack_from("<HB", block, ch_offset)
            if dist_raw == 0:
                continue

            laser_id = i % 16
            v_deg = VERT_ANGLES_DEG.get(laser_id)
            if v_deg is None:
                continue
            v_rad = math.radians(v_deg)

            dist_m = dist_raw * 0.01
            r = dist_m
            x = r * math.cos(v_rad) * math.cos(az_rad)
            y = r * math.cos(v_rad) * math.sin(az_rad)
            z = r * math.sin(v_rad)

            rows.append((x, y, z, dist_m, az_deg, float(intensity)))

    if not rows:
        return None

    arr = np.array(
        rows,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("distance", "f4"),
            ("azimuth", "f4"),
            ("intensity", "f4"),
        ],
    )
    return arr


# ============================================================
# 7) 颜色映射
# ============================================================

def colormap_distance(norm_r: np.ndarray) -> np.ndarray:
    r = np.clip(norm_r, 0.0, 1.0)
    c = np.zeros((r.shape[0], 4), dtype=np.float32)
    c[:, 3] = 1.0

    m = (r >= 0.0) & (r < 0.25)
    c[m, 2] = 1.0
    c[m, 1] = r[m] / 0.25

    m = (r >= 0.25) & (r < 0.5)
    c[m, 2] = 1.0 - (r[m] - 0.25) / 0.25
    c[m, 1] = 1.0

    m = (r >= 0.5) & (r < 0.75)
    c[m, 0] = (r[m] - 0.5) / 0.25
    c[m, 1] = 1.0

    m = (r >= 0.75)
    c[m, 0] = 1.0
    c[m, 1] = 1.0 - (r[m] - 0.75) / 0.25
    return c


def colormap_blue_yellow_red_wide(norm_v: np.ndarray,
                                 yellow_start: float = YELLOW_START,
                                 yellow_end: float = YELLOW_END) -> np.ndarray:
    v = np.clip(norm_v, 0.0, 1.0).astype(np.float32)

    ys = float(yellow_start)
    ye = float(yellow_end)
    ys = max(1e-6, min(ys, 0.999999))
    ye = max(1e-6, min(ye, 0.999999))
    if ye <= ys + 1e-6:
        ys = 0.5
        ye = 0.5

    c = np.zeros((v.shape[0], 4), dtype=np.float32)
    c[:, 3] = 1.0

    m1 = v <= ys
    if np.any(m1):
        t1 = v[m1] / ys
        c[m1, 0] = t1
        c[m1, 1] = t1
        c[m1, 2] = 1.0 - t1

    m2 = (v > ys) & (v < ye)
    if np.any(m2):
        c[m2, 0] = 1.0
        c[m2, 1] = 1.0
        c[m2, 2] = 0.0

    m3 = v >= ye
    if np.any(m3):
        t3 = (v[m3] - ye) / max(1e-6, (1.0 - ye))
        c[m3, 0] = 1.0
        c[m3, 1] = 1.0 - t3
        c[m3, 2] = 0.0
    return c


def density_heat_norm_xy(xy_scaled: np.ndarray, bin_m=HEAT_BIN_M, max_ema=None):
    if xy_scaled.size == 0:
        return np.zeros((0,), dtype=np.float32), 0.0

    bin_scaled = float(bin_m) * float(SCALE)
    if bin_scaled <= 1e-9:
        bin_scaled = 1e-6

    x = xy_scaled[:, 0].astype(np.float32)
    y = xy_scaled[:, 1].astype(np.float32)

    xmin = float(x.min()); xmax = float(x.max())
    ymin = float(y.min()); ymax = float(y.max())

    span_x = max(1e-6, xmax - xmin)
    span_y = max(1e-6, ymax - ymin)

    nx = int(math.ceil(span_x / bin_scaled))
    ny = int(math.ceil(span_y / bin_scaled))

    if nx > HEAT_MAX_BINS or ny > HEAT_MAX_BINS or nx * ny > (HEAT_MAX_BINS * HEAT_MAX_BINS):
        sx = span_x / float(HEAT_MAX_BINS)
        sy = span_y / float(HEAT_MAX_BINS)
        bin_scaled = max(bin_scaled, sx, sy)
        nx = int(math.ceil(span_x / bin_scaled))
        ny = int(math.ceil(span_y / bin_scaled))
        nx = max(1, min(nx, HEAT_MAX_BINS))
        ny = max(1, min(ny, HEAT_MAX_BINS))

    ix = ((x - xmin) / bin_scaled).astype(np.int32)
    iy = ((y - ymin) / bin_scaled).astype(np.int32)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    lin = ix + iy * nx
    grid_size = nx * ny
    counts = np.bincount(lin, minlength=grid_size).astype(np.float32)
    per_pt = counts[lin]

    if HEAT_LOG_COMPRESS:
        per_pt = np.log1p(per_pt)

    cur_max = float(per_pt.max()) if per_pt.size else 0.0
    use_max = float(max_ema) if (max_ema is not None and max_ema > 1e-6) else cur_max
    if use_max <= 1e-6:
        return np.zeros_like(per_pt, dtype=np.float32), cur_max

    norm = (per_pt / use_max).astype(np.float32)
    norm = np.clip(norm, 0.0, 1.0)

    g = float(HEAT_GAMMA)
    if abs(g - 1.0) > 1e-6:
        norm = np.power(norm, g).astype(np.float32)

    return norm, cur_max


# ============================================================
# 8) 体素化（用于方块）
# ============================================================

def voxelize_points(pts_world: np.ndarray,
                    dists: np.ndarray,
                    intens: np.ndarray,
                    times_sel: np.ndarray,
                    voxel_size_scaled: float):
    if pts_world.size == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.int32),
                np.empty((0, 3), dtype=np.int32))

    vs = float(voxel_size_scaled)
    if vs <= 1e-9:
        vs = 1e-6

    ijk = np.floor(pts_world / vs).astype(np.int32)

    dtype = np.dtype([('i', np.int32), ('j', np.int32), ('k', np.int32)])
    s = ijk.view(dtype).reshape(-1)
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]

    change = np.empty((s_sorted.shape[0],), dtype=bool)
    change[0] = True
    change[1:] = s_sorted[1:] != s_sorted[:-1]
    group_starts = np.nonzero(change)[0]
    ends = np.r_[group_starts[1:], s_sorted.shape[0]]
    counts = (ends - group_starts).astype(np.int32)

    uniq = s_sorted[group_starts]
    v_ijk = np.vstack([uniq['i'], uniq['j'], uniq['k']]).T.astype(np.int32)

    idx = order
    d_sorted = dists[idx].astype(np.float32)
    i_sorted = intens[idx].astype(np.float32)
    t_sorted = times_sel[idx].astype(np.float64)

    sum_d = np.add.reduceat(d_sorted, group_starts)
    sum_i = np.add.reduceat(i_sorted, group_starts)

    max_t = np.empty((group_starts.shape[0],), dtype=np.float64)
    for gi, (a, b) in enumerate(zip(group_starts, ends)):
        max_t[gi] = float(np.max(t_sorted[a:b]))

    v_dist = sum_d / np.maximum(1, counts).astype(np.float32)
    v_inten = sum_i / np.maximum(1, counts).astype(np.float32)
    v_center = (v_ijk.astype(np.float32) + 0.5) * vs

    return v_center.astype(np.float32), v_dist.astype(np.float32), v_inten.astype(np.float32), max_t, counts, v_ijk


# ============================================================
# 9) Mesh：立方体（批量 / 单个）
# ============================================================

def build_cube_mesh_vectorized(centers: np.ndarray, cube_size_scaled: float, cube_colors: np.ndarray):
    m = int(centers.shape[0])
    if m <= 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.int32),
                np.empty((0, 4), dtype=np.float32))

    s = float(cube_size_scaled) * 0.5
    v_off = np.array([
        [-s, -s, -s],
        [ s, -s, -s],
        [ s,  s, -s],
        [-s,  s, -s],
        [-s, -s,  s],
        [ s, -s,  s],
        [ s,  s,  s],
        [-s,  s,  s],
    ], dtype=np.float32)

    f0 = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int32)

    verts = centers[:, None, :] + v_off[None, :, :]
    verts = verts.reshape(m * 8, 3).astype(np.float32)

    base = (np.arange(m, dtype=np.int32) * 8)[:, None, None]
    faces = f0[None, :, :] + base
    faces = faces.reshape(m * 12, 3).astype(np.int32)

    face_cols = np.repeat(cube_colors[:, None, :], 12, axis=1)
    face_cols = face_cols.reshape(m * 12, 4).astype(np.float32)

    return verts, faces, face_cols


def build_single_cube_mesh(center: np.ndarray, cube_size_scaled: float, rgba: np.ndarray):
    c = np.asarray(center, dtype=np.float32).reshape(1, 3)
    col = np.asarray(rgba, dtype=np.float32).reshape(1, 4)
    verts, faces, face_cols = build_cube_mesh_vectorized(c, cube_size_scaled, col)
    return verts, faces, face_cols


# ============================================================
# 10) 多进程：Receiver + Parser
# ============================================================

@dataclass
class SharedStats:
    rx_drop: mp.Value
    parse_drop: mp.Value
    rx_cnt: mp.Value
    parse_cnt: mp.Value


def _inc(v: mp.Value, n=1):
    try:
        with v.get_lock():
            v.value += int(n)
    except Exception:
        pass


class UdpReceiverProc(mp.Process):
    def __init__(self, ip, port, raw_q: mp.Queue, stop_evt: mp.Event, stats: SharedStats, core_id=None):
        super().__init__()
        self.ip = ip
        self.port = int(port)
        self.raw_q = raw_q
        self.stop_evt = stop_evt
        self.stats = stats
        self.core_id = core_id

    def run(self):
        if self.core_id is not None:
            set_process_affinity([self.core_id])

        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.ip, self.port))
            sock.settimeout(0.2)

            while not self.stop_evt.is_set():
                try:
                    data, _ = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break

                payload = extract_msop_payload(data)
                if payload is None:
                    continue

                _inc(self.stats.rx_cnt, 1)
                try:
                    self.raw_q.put_nowait(payload)
                except Exception:
                    _inc(self.stats.rx_drop, 1)
                    try:
                        _ = self.raw_q.get_nowait()
                        self.raw_q.put_nowait(payload)
                    except Exception:
                        pass
        finally:
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass


class ParserProc(mp.Process):
    def __init__(self, raw_q: mp.Queue, out_q: mp.Queue, stop_evt: mp.Event, stats: SharedStats,
                 core_id=None, batch_ms=20):
        super().__init__()
        self.raw_q = raw_q
        self.out_q = out_q
        self.stop_evt = stop_evt
        self.stats = stats
        self.core_id = core_id
        self.batch_ms = max(5, int(batch_ms))

    def run(self):
        if self.core_id is not None:
            set_process_affinity([self.core_id])

        buf = []
        t0 = time.time()

        while not self.stop_evt.is_set():
            try:
                payload = self.raw_q.get(timeout=0.2)
            except Exception:
                payload = None

            now = time.time()

            if payload is not None:
                arr = parse_msop(payload)
                if arr is not None and arr.size:
                    buf.append(arr)
                    _inc(self.stats.parse_cnt, 1)

            if buf and ((now - t0) * 1000.0 >= self.batch_ms):
                try:
                    merged = np.concatenate(buf, axis=0)
                except Exception:
                    merged = buf[-1]
                buf.clear()
                t0 = now
                ts = now
                try:
                    self.out_q.put_nowait((ts, merged))
                except Exception:
                    _inc(self.stats.parse_drop, 1)
                    try:
                        _ = self.out_q.get_nowait()
                        self.out_q.put_nowait((ts, merged))
                    except Exception:
                        pass

            if (payload is None) and (not buf):
                t0 = now


# ============================================================
# 11) 标定对话框（直角 + 点选经纬度）
# ============================================================

class CalibDialog(QDialog):
    def __init__(self, parent: "MainWindow"):
        super().__init__(parent)
        self.parent_win = parent
        self.setWindowTitle("激光雷达标定（直角坐标模式）")
        self.setModal(True)

        layout = QFormLayout(self)

        self.edit_tx = QLineEdit("0.0")
        self.edit_ty = QLineEdit("0.0")
        self.edit_tz = QLineEdit("0.0")

        self.edit_roll = QLineEdit("0.0")
        self.edit_pitch = QLineEdit("0.0")
        self.edit_yaw = QLineEdit("0.0")

        self.edit_qx = QLineEdit("0.0")
        self.edit_qy = QLineEdit("0.0")
        self.edit_qz = QLineEdit("0.0")
        self.edit_qw = QLineEdit("1.0")

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["欧拉角(roll,pitch,yaw)", "四元数(qx,qy,qz,qw)"])

        layout.addRow("姿态模式", self.combo_mode)
        layout.addRow("Tx (米)", self.edit_tx)
        layout.addRow("Ty (米)", self.edit_ty)
        layout.addRow("Tz (米)", self.edit_tz)
        layout.addRow("Roll (度)", self.edit_roll)
        layout.addRow("Pitch (度)", self.edit_pitch)
        layout.addRow("Yaw (度)", self.edit_yaw)
        layout.addRow("qx", self.edit_qx)
        layout.addRow("qy", self.edit_qy)
        layout.addRow("qz", self.edit_qz)
        layout.addRow("qw", self.edit_qw)

        self.list_targets = QListWidget()
        layout.addRow(QLabel("已知障碍物（用于标定验证，直角坐标）："))
        layout.addRow(self.list_targets)

        self.edit_target_name = QLineEdit()
        self.edit_target_x = QLineEdit("0.0")
        self.edit_target_y = QLineEdit("0.0")
        self.edit_target_z = QLineEdit("0.0")
        layout.addRow("障碍物名称", self.edit_target_name)
        layout.addRow("障碍物 X (米)", self.edit_target_x)
        layout.addRow("障碍物 Y (米)", self.edit_target_y)
        layout.addRow("障碍物 Z (米)", self.edit_target_z)

        btn_add_update = QPushButton("添加 / 更新障碍物")
        btn_del = QPushButton("删除选中障碍物")
        btn_apply = QPushButton("应用并关闭")

        btn_add_update.clicked.connect(self.on_add_update_target)
        btn_del.clicked.connect(self.on_del_target)
        btn_apply.clicked.connect(self.on_apply_clicked)

        layout.addRow(btn_add_update)
        layout.addRow(btn_del)
        layout.addRow(btn_apply)

        self._init_from_parent()
        self.list_targets.currentRowChanged.connect(self.on_target_selected)

    def _init_from_parent(self):
        t = self.parent_win.t_extr
        self.edit_tx.setText(f"{t[0]:.4f}")
        self.edit_ty.setText(f"{t[1]:.4f}")
        self.edit_tz.setText(f"{t[2]:.4f}")

        self.list_targets.clear()
        for tgt in self.parent_win.known_targets:
            item = QListWidgetItem(tgt["name"])
            item.setData(Qt.UserRole, {"name": tgt["name"], "pos": tgt["pos"].tolist()})
            self.list_targets.addItem(item)

    def on_target_selected(self, row: int):
        if row < 0:
            return
        item = self.list_targets.item(row)
        data = item.data(Qt.UserRole)
        self.edit_target_name.setText(data["name"])
        x, y, z = data["pos"]
        self.edit_target_x.setText(str(x))
        self.edit_target_y.setText(str(y))
        self.edit_target_z.setText(str(z))

    def on_add_update_target(self):
        try:
            name = self.edit_target_name.text().strip()
            if not name:
                QMessageBox.warning(self, "错误", "障碍物名称不能为空。")
                return
            x = float(self.edit_target_x.text())
            y = float(self.edit_target_y.text())
            z = float(self.edit_target_z.text())
            target = {"name": name, "pos": [x, y, z]}

            row = self.list_targets.currentRow()
            if row >= 0:
                item = self.list_targets.item(row)
                item.setText(name)
                item.setData(Qt.UserRole, target)
            else:
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, target)
                self.list_targets.addItem(item)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"障碍物输入无效: {e}")

    def on_del_target(self):
        row = self.list_targets.currentRow()
        if row >= 0:
            self.list_targets.takeItem(row)

    def on_apply_clicked(self):
        try:
            tx = float(self.edit_tx.text())
            ty = float(self.edit_ty.text())
            tz = float(self.edit_tz.text())
            t = np.array([tx, ty, tz], dtype=np.float32)

            if self.combo_mode.currentIndex() == 0:
                roll = math.radians(float(self.edit_roll.text()))
                pitch = math.radians(float(self.edit_pitch.text()))
                yaw = math.radians(float(self.edit_yaw.text()))
                R = euler_to_rot(roll, pitch, yaw)
            else:
                qx = float(self.edit_qx.text())
                qy = float(self.edit_qy.text())
                qz = float(self.edit_qz.text())
                qw = float(self.edit_qw.text())
                R = quat_to_rot(qx, qy, qz, qw)

            targets = []
            for i in range(self.list_targets.count()):
                item = self.list_targets.item(i)
                data = item.data(Qt.UserRole)
                targets.append({"name": data["name"], "pos": data["pos"]})

            self.parent_win.update_calibration(R, t, targets)
            QMessageBox.information(self, "成功", "标定参数已应用（直角坐标模式）。")
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"标定参数输入无效: {e}")


class CalibFromPointDialog(QDialog):
    def __init__(self, parent: "MainWindow", sensor_point_scaled: np.ndarray):
        super().__init__(parent)
        self.parent_win = parent
        self.sensor_point_scaled = sensor_point_scaled.astype(np.float32)

        self.setWindowTitle("通过经纬度标定（以选中点为原点）")
        self.setModal(True)

        layout = QFormLayout(self)

        self.edit_px = QLineEdit(f"{self.sensor_point_scaled[0]:.4f}")
        self.edit_py = QLineEdit(f"{self.sensor_point_scaled[1]:.4f}")
        self.edit_pz = QLineEdit(f"{self.sensor_point_scaled[2]:.4f}")
        for e in (self.edit_px, self.edit_py, self.edit_pz):
            e.setReadOnly(True)

        layout.addRow(QLabel("雷达坐标系下该点坐标（已缩放，仅用于参考）："))
        layout.addRow("px", self.edit_px)
        layout.addRow("py", self.edit_py)
        layout.addRow("pz", self.edit_pz)

        self.edit_lat = QLineEdit("0.0")
        self.edit_lon = QLineEdit("0.0")
        self.edit_alt = QLineEdit("0.0")

        layout.addRow(QLabel("标定点的地理位置（世界坐标 = WGS84 经纬度）："))
        layout.addRow("纬度 lat(°)", self.edit_lat)
        layout.addRow("经度 lon(°)", self.edit_lon)
        layout.addRow("高程 h(m)", self.edit_alt)

        self.chk_high_prec = QCheckBox("高精度（pyproj 椭球）")
        self.chk_high_prec.setChecked(False)
        self.chk_high_prec.setEnabled(GEO.pyproj_ok)
        if not GEO.pyproj_ok:
            self.chk_high_prec.setToolTip("未检测到 pyproj，保持球体近似。")
        layout.addRow(self.chk_high_prec)

        self.edit_qx = QLineEdit("0.0")
        self.edit_qy = QLineEdit("0.0")
        self.edit_qz = QLineEdit("0.0")
        self.edit_qw = QLineEdit("1.0")

        layout.addRow(QLabel("雷达在该点时刻的姿态四元数（ENU坐标系）："))
        layout.addRow("qx", self.edit_qx)
        layout.addRow("qy", self.edit_qy)
        layout.addRow("qz", self.edit_qz)
        layout.addRow("qw", self.edit_qw)

        btn_apply = QPushButton("根据该点计算并应用标定（经纬度）")
        btn_apply.clicked.connect(self.on_apply_clicked)
        layout.addRow(btn_apply)

    def on_apply_clicked(self):
        try:
            lat0 = float(self.edit_lat.text())
            lon0 = float(self.edit_lon.text())
            h0 = float(self.edit_alt.text())

            qx = float(self.edit_qx.text())
            qy = float(self.edit_qy.text())
            qz = float(self.edit_qz.text())
            qw = float(self.edit_qw.text())
            R = quat_to_rot(qx, qy, qz, qw)

            p_sensor_scaled = self.sensor_point_scaled
            t = - (R @ p_sensor_scaled)

            targets = [{"name": tgt["name"], "pos": tgt["pos"].tolist()} for tgt in self.parent_win.known_targets]
            self.parent_win.update_geo_calibration(
                R, t, lat0, lon0, h0, targets, high_precision=bool(self.chk_high_prec.isChecked())
            )
            self.parent_win.action_calib_from_point.setChecked(False)

            QMessageBox.information(
                self, "成功",
                "已根据选中点的经纬度与四元数更新标定参数（R, t）。\n"
                "后续左侧表格将显示对应点的经纬度。"
            )
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"输入无效: {e}")


# ============================================================
# 12) ROI / 过滤对话框
# ============================================================

@dataclass
class ROIConfig:
    enabled: bool = False
    xmin: float = -9999.0
    xmax: float =  9999.0
    ymin: float = -9999.0
    ymax: float =  9999.0
    zmin: float = -9999.0
    zmax: float =  9999.0

    def apply(self, pts_world: np.ndarray) -> np.ndarray:
        if (not self.enabled) or pts_world.size == 0:
            return np.ones((pts_world.shape[0],), dtype=bool)
        x = pts_world[:, 0]; y = pts_world[:, 1]; z = pts_world[:, 2]
        m = (x >= self.xmin) & (x <= self.xmax) & (y >= self.ymin) & (y <= self.ymax) & (z >= self.zmin) & (z <= self.zmax)
        return m


class ROIDialog(QDialog):
    def __init__(self, parent: "MainWindow"):
        super().__init__(parent)
        self.parent_win = parent
        self.setWindowTitle("ROI / 裁剪过滤（世界坐标，缩放后单位）")
        self.setModal(True)

        self.cfg = parent.roi_cfg

        layout = QFormLayout(self)
        self.chk = QCheckBox("启用 ROI 裁剪")
        self.chk.setChecked(bool(self.cfg.enabled))
        layout.addRow(self.chk)

        self.ed_xmin = QLineEdit(str(self.cfg.xmin))
        self.ed_xmax = QLineEdit(str(self.cfg.xmax))
        self.ed_ymin = QLineEdit(str(self.cfg.ymin))
        self.ed_ymax = QLineEdit(str(self.cfg.ymax))
        self.ed_zmin = QLineEdit(str(self.cfg.zmin))
        self.ed_zmax = QLineEdit(str(self.cfg.zmax))

        layout.addRow("X min", self.ed_xmin)
        layout.addRow("X max", self.ed_xmax)
        layout.addRow("Y min", self.ed_ymin)
        layout.addRow("Y max", self.ed_ymax)
        layout.addRow("Z min", self.ed_zmin)
        layout.addRow("Z max", self.ed_zmax)

        btn_apply = QPushButton("应用")
        btn_cancel = QPushButton("取消")
        btn_apply.clicked.connect(self._apply)
        btn_cancel.clicked.connect(self.reject)

        row = QHBoxLayout()
        row.addWidget(btn_apply)
        row.addWidget(btn_cancel)
        w = QWidget()
        w.setLayout(row)
        layout.addRow(w)

    def _apply(self):
        try:
            self.cfg.enabled = bool(self.chk.isChecked())
            self.cfg.xmin = float(self.ed_xmin.text())
            self.cfg.xmax = float(self.ed_xmax.text())
            self.cfg.ymin = float(self.ed_ymin.text())
            self.cfg.ymax = float(self.ed_ymax.text())
            self.cfg.zmin = float(self.ed_zmin.text())
            self.cfg.zmax = float(self.ed_zmax.text())
            self.parent_win.roi_cfg = self.cfg
            self.parent_win.update_gl_view(force=True)
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"ROI 参数无效：{e}")


# ============================================================
# 13) CAD 级拾取：相机射线 + 候选集索引（2D 屏幕栅格 / 3D 体素哈希）
# ============================================================

PICK_CELL_PX = 24
PICK_RADIUS_PX = 10
PICK_NEIGHBOR_CELLS = 2
PICK_MAX_CANDS = 4096

RAY_MAX_STEPS = 8192
RAY_T_EPS = 1e-4


def _normalize(v):
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v
    return v / n


def mat4_look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    f = _normalize(target - eye)
    r = _normalize(np.cross(f, up))
    u = np.cross(r, f)

    M = np.eye(4, dtype=np.float32)
    M[0, 0:3] = r
    M[1, 0:3] = u
    M[2, 0:3] = -f
    M[0, 3] = -np.dot(r, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = np.dot(f, eye)
    return M


def mat4_perspective(fov_y_deg, aspect, z_near, z_far):
    fov = math.radians(float(fov_y_deg))
    f = 1.0 / math.tan(fov / 2.0)
    a = float(aspect)
    zn = float(z_near)
    zf = float(z_far)
    if zf <= zn + 1e-6:
        zf = zn + 1.0

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / a
    M[1, 1] = f
    M[2, 2] = (zf + zn) / (zn - zf)
    M[2, 3] = (2 * zf * zn) / (zn - zf)
    M[3, 2] = -1.0
    return M


def ray_aabb_intersect(o, d, bmin, bmax):
    o = np.asarray(o, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)
    bmin = np.asarray(bmin, dtype=np.float32)
    bmax = np.asarray(bmax, dtype=np.float32)

    inv = np.where(np.abs(d) < 1e-12, 1e12, 1.0 / d)
    t0 = (bmin - o) * inv
    t1 = (bmax - o) * inv
    tmin = np.maximum(np.minimum(t0, t1), -1e18)
    tmax = np.minimum(np.maximum(t0, t1),  1e18)
    t_enter = float(np.max(tmin))
    t_exit = float(np.min(tmax))
    if t_exit < max(t_enter, 0.0):
        return None
    return t_enter, t_exit


# ============================================================
# 14) 主界面
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("C16 点云（点/马赛克·并行解析·CAD拾取·悬停高亮·连续测距N段·导出测量CSV·标定·手动方块WSADQE）")
        self.resize(1760, 940)

        # ring buffer
        self.xyz_buf = np.zeros((MAX_POINTS, 3), dtype=np.float32)
        self.time_buf = np.zeros((MAX_POINTS,), dtype=np.float64)
        self.dist_buf = np.zeros((MAX_POINTS,), dtype=np.float32)
        self.az_buf = np.zeros((MAX_POINTS,), dtype=np.float32)
        self.inten_buf = np.zeros((MAX_POINTS,), dtype=np.float32)
        self.write_idx = 0

        self.last_gl_update = 0.0
        self.last_table_update = 0.0
        self.last_table_resize = 0.0
        self.need_gl_update = True

        self._last_frame_time = None
        self._fps_ema = None

        self.density_factor = 2.4
        self.mirror_lr = False

        # 0=距离 1=强度 2=密度 3=高度Z
        self.color_mode = 2

        self.render_mode = DEFAULT_MODE
        self.voxel_size_m = float(DEFAULT_VOXEL_M)

        # calibration (sensor->world)
        self.calib_enabled = False
        self.R_extr = np.eye(3, dtype=np.float32)
        self.t_extr = np.zeros(3, dtype=np.float32)

        self.ref_lat = None
        self.ref_lon = None
        self.ref_alt = 0.0
        self.geo_enabled = False
        self.geo_high_precision = False

        self.last_pts_world = None
        self.last_pts_sensor = None

        # ======== 用于拾取（马赛克+手动方块的“统一候选集”）========
        self.last_cubes_world = None     # 合并后的中心(世界)
        self.last_cubes_sensor = None    # 合并后的中心(传感器)
        self.last_cubes_is_manual = None # 合并后的是否手动
        self.last_cubes_manual_index = None  # 合并索引->手动索引(非手动为-1)

        self._cube_voxel_map = {}
        self._cube_aabb_min = None
        self._cube_aabb_max = None
        self._cube_vs = None

        self.known_targets = []
        self.current_target_idx = -1
        self.calib_by_click_mode = False

        self._subsample_phase = 0
        self._rmin_ema = None
        self._rmax_ema = None
        self._heat_max_ema = None

        self.use_fixed_color_range = bool(USE_FIXED_COLOR_RANGE_DEFAULT)
        self.fixed_r_min = float(FIXED_R_MIN_DEFAULT)
        self.fixed_r_max = float(FIXED_R_MAX_DEFAULT)

        self.z_min = float(Z_MIN_DEFAULT)
        self.z_max = float(Z_MAX_DEFAULT)

        self.dynamic_point_size = bool(DYNAMIC_POINT_SIZE_DEFAULT)

        self.roi_cfg = ROIConfig()

        self.adapt_enable = bool(ADAPT_ENABLE_DEFAULT)
        self.persist_sec = float(PERSIST_SEC_DEFAULT)

        self.voxel_ema_enable = bool(VOXEL_EMA_ENABLE_DEFAULT)
        self._voxel_cache = {}

        # 连续测距：N段 polyline
        self.measure_enable = False
        self.measure_points_world = []
        self.measure_points_sensor = []
        self.measure_segments = []
        self.measure_total = 0.0

        # 选中/悬停方块高亮（紫色实体）
        self.selected_cube_valid = False
        self.selected_cube_world = None
        self.selected_cube_sensor = None
        self.selected_is_manual = False
        self.selected_manual_idx = -1

        self.hover_cube_valid = False
        self.hover_cube_idx = None
        self.hover_cube_world = None
        self.hover_is_manual = False

        # 2D 屏幕栅格索引（点云）
        self._vp = np.eye(4, dtype=np.float32)
        self._inv_vp = None
        self._cam_pos = np.zeros(3, dtype=np.float32)
        self._screen_grid = {}
        self._screen_w = 1
        self._screen_h = 1
        self._screen_grid_valid = False
        self._points_proj_xy = None
        self._points_proj_z = None
        self._points_proj_valid = None

        # ======== 手动添加方块（绿色，尺寸随 voxel_size_m 实时变化）========
        self.manual_cubes_world = []  # list[np.ndarray(3,)]
        self.active_manual_idx = -1   # WSADQE 控制的当前手动方块
        self._manual_dirty = True

        # ---------------- Toolbar（两行、固定不折叠） ----------------
        self.toolbar1 = QToolBar("主工具栏-1")
        self.toolbar1.setMovable(False)
        self.toolbar1.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar1)

        self.addToolBarBreak(Qt.TopToolBarArea)
        self.toolbar2 = QToolBar("主工具栏-2")
        self.toolbar2.setMovable(False)
        self.toolbar2.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar2)

        # ========== Row 1 ==========
        self.action_pause = QAction("暂停", self)
        self.action_pause.setCheckable(True)
        self.action_pause.toggled.connect(self.on_pause_toggled)

        action_clear = QAction("清空", self)
        action_clear.triggered.connect(self.on_clear_clicked)

        self.toolbar1.addAction(self.action_pause)
        self.toolbar1.addAction(action_clear)
        self.toolbar1.addSeparator()

        self.toolbar1.addWidget(QLabel("视角"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["自由", "前视", "左视", "右视", "俯视"])
        self.view_combo.currentIndexChanged.connect(self.on_view_changed)
        self.toolbar1.addWidget(self.view_combo)

        self.toolbar1.addSeparator()
        self.toolbar1.addWidget(QLabel("显示模式"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["点云", "马赛克(方块)"])
        self.mode_combo.setCurrentIndex(0 if self.render_mode != MODE_CUBE_MOSAIC else 1)
        self.mode_combo.currentIndexChanged.connect(self.on_render_mode_changed)
        self.toolbar1.addWidget(self.mode_combo)

        self.chk_grid_overlay = QCheckBox("网格")
        self.chk_grid_overlay.setChecked(False)
        self.chk_grid_overlay.toggled.connect(self.on_grid_overlay_toggled)
        self.toolbar1.addWidget(self.chk_grid_overlay)

        self.toolbar1.addSeparator()
        self.action_measure = QAction("连续测距(N段)", self)
        self.action_measure.setCheckable(True)
        self.action_measure.toggled.connect(self.on_measure_toggled)
        self.toolbar1.addAction(self.action_measure)

        act_measure_undo = QAction("测距撤销", self)
        act_measure_undo.triggered.connect(self.on_measure_undo)
        self.toolbar1.addAction(act_measure_undo)

        act_measure_clear = QAction("测距清空", self)
        act_measure_clear.triggered.connect(self.on_measure_clear)
        self.toolbar1.addAction(act_measure_clear)

        act_measure_export = QAction("导出测距CSV", self)
        act_measure_export.triggered.connect(self.on_measure_export_csv)
        self.toolbar1.addAction(act_measure_export)

        self.toolbar1.addSeparator()
        self.toolbar1.addWidget(QLabel("UDP"))
        self.port_edit = QLineEdit(str(UDP_PORT))
        self.port_edit.setFixedWidth(70)
        self.port_edit.setReadOnly(True)
        self.toolbar1.addWidget(self.port_edit)

        # ========== Row 2 ==========
        self.toolbar2.addWidget(QLabel("点密度"))
        self.density_combo = QComboBox()
        self.density_combo.addItems(["低(0.8x)", "中(1.6x)", "高(2.4x)", "超高(5x)", "极高(8x)"])
        self.density_combo.setCurrentIndex(2)
        self.density_combo.currentIndexChanged.connect(self.on_density_changed)
        self.toolbar2.addWidget(self.density_combo)

        self.toolbar2.addSeparator()
        self.toolbar2.addWidget(QLabel("颜色"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["距离(彩虹)", "强度(蓝黄红·宽黄)", "密度(蓝黄红·宽黄)", "高度Z(蓝黄红·宽黄)"])
        self.color_combo.setCurrentIndex(self.color_mode)
        self.color_combo.currentIndexChanged.connect(self.on_color_mode_changed)
        self.toolbar2.addWidget(self.color_combo)

        self.toolbar2.addSeparator()
        self.toolbar2.addWidget(QLabel("方块"))
        self.voxel_combo = QComboBox()
        self.voxel_combo.addItems(["0.30", "0.50", "0.80", "1.00", "1.50", "2.00"])
        self.voxel_combo.setCurrentText(f"{self.voxel_size_m:.2f}")
        self.voxel_combo.currentIndexChanged.connect(self.on_voxel_changed)
        self.toolbar2.addWidget(self.voxel_combo)

        self.toolbar2.addSeparator()
        self.mirror_checkbox = QCheckBox("左右镜像(显示)")
        self.mirror_checkbox.toggled.connect(self.on_mirror_toggled)
        self.toolbar2.addWidget(self.mirror_checkbox)

        self.chk_dynamic_size = QCheckBox("动态点大小")
        self.chk_dynamic_size.setChecked(self.dynamic_point_size)
        self.chk_dynamic_size.toggled.connect(self.on_dynamic_size_toggled)
        self.toolbar2.addWidget(self.chk_dynamic_size)

        self.chk_adapt = QCheckBox("自适应降载")
        self.chk_adapt.setChecked(self.adapt_enable)
        self.chk_adapt.toggled.connect(self.on_adapt_toggled)
        self.toolbar2.addWidget(self.chk_adapt)

        self.toolbar2.addSeparator()
        self.chk_fixed_range = QCheckBox("固定距离范围")
        self.chk_fixed_range.setChecked(self.use_fixed_color_range)
        self.chk_fixed_range.toggled.connect(self.on_fixed_range_toggled)
        self.toolbar2.addWidget(self.chk_fixed_range)

        self.toolbar2.addWidget(QLabel("Rmin"))
        self.ed_rmin = QLineEdit(f"{self.fixed_r_min:.1f}")
        self.ed_rmin.setFixedWidth(60)
        self.ed_rmin.editingFinished.connect(self.on_fixed_range_changed)
        self.toolbar2.addWidget(self.ed_rmin)

        self.toolbar2.addWidget(QLabel("Rmax"))
        self.ed_rmax = QLineEdit(f"{self.fixed_r_max:.1f}")
        self.ed_rmax.setFixedWidth(60)
        self.ed_rmax.editingFinished.connect(self.on_fixed_range_changed)
        self.toolbar2.addWidget(self.ed_rmax)

        self.toolbar2.addSeparator()
        act_roi = QAction("ROI裁剪", self)
        act_roi.triggered.connect(self.on_roi_clicked)
        self.toolbar2.addAction(act_roi)

        self.toolbar2.addSeparator()
        self.action_calib = QAction("标定设置(直角)", self)
        self.action_calib.setCheckable(True)
        self.action_calib.toggled.connect(self.on_calib_toggled)
        self.toolbar2.addAction(self.action_calib)

        self.action_calib_from_point = QAction("点选经纬度标定(吸附中心)", self)
        self.action_calib_from_point.setCheckable(True)
        self.action_calib_from_point.toggled.connect(self.on_calib_from_point_toggled)
        self.toolbar2.addAction(self.action_calib_from_point)

        # ======== 手动方块按钮（绿色，WSADQE移动）========
        self.toolbar2.addSeparator()
        act_add_manual = QAction("添加手动方块(绿)", self)
        act_add_manual.triggered.connect(self.on_add_manual_cube)
        self.toolbar2.addAction(act_add_manual)

        act_del_manual = QAction("删除手动方块", self)
        act_del_manual.triggered.connect(self.on_delete_manual_cube)
        self.toolbar2.addAction(act_del_manual)

        self.toolbar2.addSeparator()
        self.toolbar2.addWidget(QLabel("验证障碍物"))
        self.target_combo = QComboBox()
        self.target_combo.addItem("无")
        self.target_combo.currentIndexChanged.connect(self.on_target_changed)
        self.toolbar2.addWidget(self.target_combo)

        # 状态栏
        self.status = QLabel("状态: -")
        self.statusBar().addPermanentWidget(self.status)

        # ---------------- Layout ----------------
        central = QWidget()
        self.setCentralWidget(central)
        splitter = QSplitter(Qt.Horizontal)
        layout = QHBoxLayout(central)
        layout.addWidget(splitter)

        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels(
            ["点ID", "X(m)", "Y(m)", "Z(m)",
             "纬度(°)", "经度(°)", "高程(m)",
             "水平角(°)", "距离(m)", "强度"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)

        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor("k")
        self.gl_view.opts["distance"] = 30
        self.gl_view.opts["elevation"] = 25
        self.gl_view.opts["azimuth"] = -45
        self.gl_view.opts["fov"] = 60

        self.gl_view.setMouseTracking(True)
        central.setMouseTracking(True)

        # axes
        axis_len = 2.0 * SCALE / 5.0
        self.axis_items = []
        self.axis_visible = True
        self._create_axes(axis_len)

        # grid
        self.grid = gl.GLGridItem()
        self.grid.setSize(GRID_SIZE, GRID_SIZE)
        self.grid.setSpacing(GRID_SPACING, GRID_SPACING)
        self.grid.setVisible(False)
        self.gl_view.addItem(self.grid)

        # scatter
        self.scatter = gl.GLScatterPlotItem()
        self.scatter.setGLOptions("translucent")
        self.gl_view.addItem(self.scatter)

        # mosaic merged mesh
        self.cube_mesh = gl.GLMeshItem(
            vertexes=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int32),
            faceColors=np.empty((0, 4), dtype=np.float32),
            smooth=False,
            drawEdges=False
        )
        self.cube_mesh.setGLOptions("translucent")
        self.cube_mesh.setVisible(False)
        self.gl_view.addItem(self.cube_mesh)

        # manual cubes mesh (green)
        self.manual_mesh = gl.GLMeshItem(
            vertexes=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int32),
            faceColors=np.empty((0, 4), dtype=np.float32),
            smooth=False,
            drawEdges=False
        )
        self.manual_mesh.setGLOptions("translucent")
        self.manual_mesh.setVisible(True)
        self.gl_view.addItem(self.manual_mesh)

        # hover cube highlight mesh (deep purple, opaque)
        self.hover_cube_mesh = gl.GLMeshItem(
            vertexes=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int32),
            faceColors=np.empty((0, 4), dtype=np.float32),
            smooth=False,
            drawEdges=False
        )
        self.hover_cube_mesh.setGLOptions("translucent")
        self.hover_cube_mesh.setVisible(False)
        self.gl_view.addItem(self.hover_cube_mesh)

        # selected cube highlight mesh
        self.sel_cube_mesh = gl.GLMeshItem(
            vertexes=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int32),
            faceColors=np.empty((0, 4), dtype=np.float32),
            smooth=False,
            drawEdges=False
        )
        self.sel_cube_mesh.setGLOptions("translucent")
        self.sel_cube_mesh.setVisible(False)
        self.gl_view.addItem(self.sel_cube_mesh)

        # measurement line
        self.measure_line_item = gl.GLLinePlotItem(pos=np.empty((0, 3), dtype=np.float32),
                                                   color=(0.2, 0.6, 1.0, 1.0),
                                                   width=2,
                                                   antialias=True,
                                                   mode="line_strip")
        self.measure_line_item.setVisible(False)
        self.gl_view.addItem(self.measure_line_item)

        splitter.addWidget(self.table)
        splitter.addWidget(self.gl_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([620, 1140])

        # events override
        self._orig_gl_key_press = self.gl_view.keyPressEvent
        self.gl_view.keyPressEvent = self.gl_key_press_event

        self._orig_gl_mouse_press = self.gl_view.mousePressEvent
        self.gl_view.mousePressEvent = self.gl_mouse_press_event

        self._orig_gl_mouse_move = self.gl_view.mouseMoveEvent
        self.gl_view.mouseMoveEvent = self.gl_mouse_move_event

        self._orig_gl_mouse_double = getattr(self.gl_view, "mouseDoubleClickEvent", None)
        self.gl_view.mouseDoubleClickEvent = self.gl_mouse_double_click_event

        # hover throttling
        self._hover_last_pos = None
        self._hover_timer = QTimer(self)
        self._hover_timer.setInterval(16)  # ~60Hz
        self._hover_timer.timeout.connect(self._hover_tick)
        self._hover_timer.start()

        # timers
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gl_view)
        self.timer.start(int(UPDATE_INTERVAL_SEC * 1000))

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_queue)
        self.poll_timer.start(POLL_QUEUE_INTERVAL_MS)

        # stats
        self.stats = SharedStats(
            rx_drop=mp.Value("i", 0),
            parse_drop=mp.Value("i", 0),
            rx_cnt=mp.Value("i", 0),
            parse_cnt=mp.Value("i", 0),
        )
        self._rx_drop_last = 0
        self._parse_drop_last = 0
        self._rx_cnt_last = 0
        self._parse_cnt_last = 0
        self._last_stat_ts = time.time()

        # multiprocessing
        self._paused = False
        self._stop_evt = mp.Event()
        self._raw_q = mp.Queue(maxsize=RAW_QUEUE_MAXSIZE)
        self._parsed_q = mp.Queue(maxsize=PARSED_QUEUE_MAXSIZE)
        self._receiver = None
        self._parsers = []
        self.start_udp_processes()

        # init visibility
        self._apply_render_visibility()
        self._refresh_manual_mesh(force=True)

    # ---------- 多进程启动/停止 ----------
    def start_udp_processes(self):
        self.stop_udp_processes()

        recv_core = None
        candidates = [c for c in range(CPU_COUNT) if c not in set(WORKER_CORES)]
        recv_core = candidates[0] if candidates else (WORKER_CORES[0] if WORKER_CORES else None)

        self._stop_evt.clear()

        self._receiver = UdpReceiverProc(UDP_IP, UDP_PORT, self._raw_q, self._stop_evt, self.stats, core_id=recv_core)
        self._receiver.daemon = True
        self._receiver.start()

        self._parsers = []
        for c in WORKER_CORES:
            p = ParserProc(self._raw_q, self._parsed_q, self._stop_evt, self.stats, core_id=c, batch_ms=20)
            p.daemon = True
            p.start()
            self._parsers.append(p)

    def stop_udp_processes(self):
        try:
            if hasattr(self, "_stop_evt") and self._stop_evt is not None:
                self._stop_evt.set()
        except Exception:
            pass

        def _safe_join(p, t1=1.2, t2=0.8):
            try:
                if p is None:
                    return
                if p.is_alive():
                    p.join(timeout=t1)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=t2)
            except Exception:
                pass

        try:
            _safe_join(self._receiver, 1.2, 0.8)
        except Exception:
            pass

        try:
            for p in getattr(self, "_parsers", []):
                _safe_join(p, 1.2, 0.8)
        except Exception:
            pass

        self._receiver = None
        self._parsers = []

    # ---------- 坐标轴 ----------
    def _create_axes(self, length):
        x = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [length, 0, 0]]),
                              color=(1, 0, 0, 1), width=2, antialias=True)
        y = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, length, 0]]),
                              color=(0, 1, 0, 1), width=2, antialias=True)
        z = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, length]]),
                              color=(0, 0, 1, 1), width=2, antialias=True)
        self.gl_view.addItem(x)
        self.gl_view.addItem(y)
        self.gl_view.addItem(z)
        self.axis_items = [x, y, z]

    # ============================================================
    # 手动方块：创建/删除/刷新/移动
    # ============================================================

    def _get_voxel_size_scaled(self):
        return float(self.voxel_size_m) * float(SCALE)

    def _refresh_manual_mesh(self, force=False):
        if (not force) and (not self._manual_dirty):
            return
        self._manual_dirty = False

        if not self.manual_cubes_world:
            self.manual_mesh.setMeshData(
                vertexes=np.empty((0, 3), dtype=np.float32),
                faces=np.empty((0, 3), dtype=np.int32),
                faceColors=np.empty((0, 4), dtype=np.float32)
            )
            self.manual_mesh.setVisible(True)
            return

        centers = np.vstack([c.reshape(1, 3) for c in self.manual_cubes_world]).astype(np.float32)
        vs = self._get_voxel_size_scaled()

        # 绿色：可见但不喧宾夺主
        col = np.tile(np.array([[0.05, 0.95, 0.10, 0.55]], dtype=np.float32), (centers.shape[0], 1))
        verts, faces, face_cols = build_cube_mesh_vectorized(centers, vs, col)
        self.manual_mesh.setMeshData(vertexes=verts, faces=faces, faceColors=face_cols)
        self.manual_mesh.setVisible(True)

    def on_add_manual_cube(self):
        # 优先：鼠标悬停方块中心；其次：当前选中方块；否则：视图中心
        p = None
        if self.hover_cube_valid and self.hover_cube_world is not None:
            p = self.hover_cube_world.copy()
        elif self.selected_cube_valid and self.selected_cube_world is not None:
            p = self.selected_cube_world.copy()
        else:
            center = self.gl_view.opts.get("center", pg.Vector(0, 0, 0))
            try:
                p = np.array([center.x(), center.y(), center.z()], dtype=np.float32)
            except AttributeError:
                p = np.array(center, dtype=np.float32)

        self.manual_cubes_world.append(p.astype(np.float32))
        self.active_manual_idx = len(self.manual_cubes_world) - 1
        self._manual_dirty = True

        # 将“选中”切换到该手动方块，便于立即 WSADQE 调整
        self.selected_cube_valid = True
        self.selected_cube_world = p.astype(np.float32)
        self.selected_cube_sensor = self._world_to_sensor_for_pick(self.selected_cube_world)
        self.selected_is_manual = True
        self.selected_manual_idx = self.active_manual_idx
        self._refresh_selected_cube_mesh()

        self.need_gl_update = True
        self.update_gl_view(force=True)

        QMessageBox.information(
            self, "手动方块",
            "已添加绿色手动方块。\n"
            "提示：点击任意方块可切换选中；当选中为手动方块时，可用 WSADQE 移动。\n"
            "Shift=大步，Ctrl=小步。"
        )

    def on_delete_manual_cube(self):
        if self.active_manual_idx < 0 or self.active_manual_idx >= len(self.manual_cubes_world):
            QMessageBox.information(self, "删除手动方块", "当前没有激活的手动方块可删除（先点击/添加一个手动方块）。")
            return
        self.manual_cubes_world.pop(self.active_manual_idx)
        if not self.manual_cubes_world:
            self.active_manual_idx = -1
        else:
            self.active_manual_idx = min(self.active_manual_idx, len(self.manual_cubes_world) - 1)

        # 若当前选中的是手动方块，也同步清理
        if self.selected_is_manual:
            self._clear_selected()

        self._manual_dirty = True
        self.need_gl_update = True
        self.update_gl_view(force=True)

    def _move_active_manual_cube(self, dx, dy, dz, step_scaled):
        if self.active_manual_idx < 0 or self.active_manual_idx >= len(self.manual_cubes_world):
            return
        p = self.manual_cubes_world[self.active_manual_idx].astype(np.float32)
        p = p + np.array([dx, dy, dz], dtype=np.float32) * float(step_scaled)
        self.manual_cubes_world[self.active_manual_idx] = p
        self._manual_dirty = True

        # 如果“选中”就是该手动方块，保持选中中心一致
        if self.selected_is_manual and self.selected_manual_idx == self.active_manual_idx:
            self.selected_cube_world = p.copy()
            self.selected_cube_sensor = self._world_to_sensor_for_pick(self.selected_cube_world)
            self._refresh_selected_cube_mesh()

        self.need_gl_update = True
        self.update_gl_view(force=True)

    # ============================================================
    # 工具栏事件
    # ============================================================

    def on_pause_toggled(self, checked):
        self._paused = bool(checked)
        if self._paused:
            self._drain_queue_all()

    def on_clear_clicked(self):
        self.xyz_buf[:] = 0
        self.time_buf[:] = 0
        self.dist_buf[:] = 0
        self.az_buf[:] = 0
        self.inten_buf[:] = 0
        self.write_idx = 0

        self.scatter.setData(pos=np.empty((0, 3)), size=float(POINT_SIZE_WORLD_BASE), pxMode=SCATTER_PX_MODE)
        self.cube_mesh.setMeshData(
            vertexes=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int32),
            faceColors=np.empty((0, 4), dtype=np.float32)
        )

        self._clear_hover()
        self._clear_selected()

        self.table.setRowCount(0)

        self.need_gl_update = True
        self.last_pts_world = None
        self.last_pts_sensor = None

        self.last_cubes_world = None
        self.last_cubes_sensor = None
        self.last_cubes_is_manual = None
        self.last_cubes_manual_index = None
        self._cube_voxel_map.clear()
        self._cube_aabb_min = None
        self._cube_aabb_max = None
        self._cube_vs = None

        self._subsample_phase = 0
        self._rmin_ema = None
        self._rmax_ema = None
        self._heat_max_ema = None
        self._voxel_cache.clear()
        self._drain_queue_all()

        self.on_measure_clear()

        # 手动方块也清空
        self.manual_cubes_world = []
        self.active_manual_idx = -1
        self._manual_dirty = True
        self._refresh_manual_mesh(force=True)

    def on_view_changed(self, index):
        if index == 1:
            self.gl_view.opts["elevation"] = 0
            self.gl_view.opts["azimuth"] = 0
        elif index == 2:
            self.gl_view.opts["elevation"] = 0
            self.gl_view.opts["azimuth"] = 90
        elif index == 3:
            self.gl_view.opts["elevation"] = 0
            self.gl_view.opts["azimuth"] = -90
        elif index == 4:
            self.gl_view.opts["elevation"] = 90
            self.gl_view.opts["azimuth"] = -90
        self.gl_view.update()
        self.need_gl_update = True

    def on_density_changed(self, index: int):
        factors = [0.8, 1.6, 2.4, 5.0, 8.0]
        index = max(0, min(len(factors) - 1, int(index)))
        self.density_factor = float(factors[index])
        self.update_gl_view(force=True)

    def on_color_mode_changed(self, index: int):
        self.color_mode = int(index)
        self.update_gl_view(force=True)

    def on_render_mode_changed(self, index: int):
        self.render_mode = MODE_POINT if int(index) == 0 else MODE_CUBE_MOSAIC
        self._apply_render_visibility()
        if self.render_mode != MODE_CUBE_MOSAIC:
            self._clear_hover()
        self.update_gl_view(force=True)

    def on_grid_overlay_toggled(self, checked: bool):
        self.grid.setVisible(bool(checked))
        self.gl_view.update()

    def on_voxel_changed(self, _index: int):
        try:
            self.voxel_size_m = float(self.voxel_combo.currentText())
        except Exception:
            self.voxel_size_m = float(DEFAULT_VOXEL_M)

        # 关键：手动方块尺寸与马赛克尺寸保持一致（切换后立即跟随）
        self._manual_dirty = True
        self._refresh_manual_mesh(force=True)

        # 高亮/选中也要跟随新尺寸重建
        self._refresh_selected_cube_mesh()
        if self.hover_cube_valid and self.hover_cube_world is not None:
            self._set_hover_cube(self.hover_cube_idx)

        self.update_gl_view(force=True)

    def on_mirror_toggled(self, checked: bool):
        self.mirror_lr = checked
        self.update_gl_view(force=True)

    def on_dynamic_size_toggled(self, checked: bool):
        self.dynamic_point_size = bool(checked)
        self.update_gl_view(force=True)

    def on_adapt_toggled(self, checked: bool):
        self.adapt_enable = bool(checked)

    def on_fixed_range_toggled(self, checked: bool):
        self.use_fixed_color_range = bool(checked)
        self.update_gl_view(force=True)

    def on_fixed_range_changed(self):
        try:
            self.fixed_r_min = float(self.ed_rmin.text())
            self.fixed_r_max = float(self.ed_rmax.text())
            if self.fixed_r_max <= self.fixed_r_min:
                self.fixed_r_max = self.fixed_r_min + 1.0
            self.ed_rmin.setText(f"{self.fixed_r_min:.1f}")
            self.ed_rmax.setText(f"{self.fixed_r_max:.1f}")
            self.update_gl_view(force=True)
        except Exception:
            pass

    def on_roi_clicked(self):
        dlg = ROIDialog(self)
        dlg.exec_()

    # ---------- 测距 ----------
    def on_measure_toggled(self, checked: bool):
        self.measure_enable = bool(checked)
        if self.measure_enable:
            QMessageBox.information(self, "连续测距(N段)", "已进入测距模式：左键连续点击点/方块中心，形成多段折线。")
        self._refresh_selected_cube_mesh()
        self.update_gl_view(force=True)

    def on_measure_undo(self):
        if not self.measure_points_world:
            return
        self.measure_points_world.pop()
        self.measure_points_sensor.pop()
        self._rebuild_measure_segments()
        self._update_measure_line()
        self._refresh_selected_cube_mesh()
        self.update_gl_view(force=True)

    def on_measure_clear(self):
        self.measure_points_world = []
        self.measure_points_sensor = []
        self.measure_segments = []
        self.measure_total = 0.0
        self._update_measure_line()
        self._refresh_selected_cube_mesh()
        self.update_gl_view(force=True)

    def on_measure_export_csv(self):
        if not self.measure_segments:
            QMessageBox.information(self, "导出测距CSV", "当前没有测距结果可导出。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出测距结果CSV", "", "CSV (*.csv);;All Files (*.*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("seg_id,x1,y1,z1,x2,y2,z2,dist_scaled,dist_m,cum_scaled,cum_m\n")
                for s in self.measure_segments:
                    p1 = s["p1"]
                    p2 = s["p2"]
                    d_sc = float(s["dist_scaled"])
                    d_m = d_sc / SCALE
                    c_sc = float(s["cum_scaled"])
                    c_m = c_sc / SCALE
                    f.write(f"{s['id']},{p1[0]:.6f},{p1[1]:.6f},{p1[2]:.6f},"
                            f"{p2[0]:.6f},{p2[1]:.6f},{p2[2]:.6f},"
                            f"{d_sc:.6f},{d_m:.6f},{c_sc:.6f},{c_m:.6f}\n")
            QMessageBox.information(self, "导出测距CSV", f"已导出：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{e}")

    def _rebuild_measure_segments(self):
        self.measure_segments = []
        self.measure_total = 0.0
        if len(self.measure_points_world) < 2:
            return
        cum = 0.0
        for i in range(1, len(self.measure_points_world)):
            p1 = self.measure_points_world[i - 1]
            p2 = self.measure_points_world[i]
            d = float(np.linalg.norm(p2 - p1))
            cum += d
            self.measure_segments.append({
                "id": i,
                "p1": p1.copy(),
                "p2": p2.copy(),
                "dist_scaled": d,
                "cum_scaled": cum
            })
        self.measure_total = cum

    def _update_measure_line(self):
        if not self.measure_enable or len(self.measure_points_world) < 2:
            self.measure_line_item.setData(pos=np.empty((0, 3), dtype=np.float32))
            self.measure_line_item.setVisible(False)
            return
        pts = np.vstack(self.measure_points_world).astype(np.float32)
        self.measure_line_item.setData(pos=pts)
        self.measure_line_item.setVisible(True)

    # ---------- 标定 ----------
    def update_calibration(self, R, t, targets):
        self.R_extr = np.array(R, dtype=np.float32)
        self.t_extr = np.array(t, dtype=np.float32)
        self.known_targets = [{"name": tgt["name"], "pos": np.array(tgt["pos"], dtype=np.float32)} for tgt in targets]
        self.refresh_target_combo()
        self.calib_enabled = True
        self.update_gl_view(force=True)

    def update_geo_calibration(self, R, t, lat0, lon0, h0, targets, high_precision=False):
        self.update_calibration(R, t, targets)
        self.ref_lat = float(lat0)
        self.ref_lon = float(lon0)
        self.ref_alt = float(h0)
        self.geo_enabled = True
        self.geo_high_precision = bool(high_precision)

    def refresh_target_combo(self):
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItem("无")
        for tgt in self.known_targets:
            self.target_combo.addItem(tgt["name"])
        self.target_combo.blockSignals(False)
        self.current_target_idx = -1

    def on_calib_toggled(self, checked: bool):
        if checked:
            self.calib_enabled = True
            dlg = CalibDialog(self)
            dlg.exec_()
        else:
            self.calib_enabled = False
            self.update_gl_view(force=True)

    def on_calib_from_point_toggled(self, checked: bool):
        self.calib_by_click_mode = bool(checked)
        if checked:
            QMessageBox.information(
                self, "提示",
                "已进入“点选经纬度标定”模式：\n"
                "左键点击一个标定点。\n"
                "若存在方块（马赛克或绿色手动方块），将优先吸附到被射线命中的方块中心。"
            )

    def on_target_changed(self, index: int):
        self.current_target_idx = -1 if index <= 0 else index - 1

    # ---------- 视图可见性 ----------
    def _apply_render_visibility(self):
        if self.render_mode == MODE_CUBE_MOSAIC:
            self.scatter.setVisible(False)
            self.cube_mesh.setVisible(True)
            self.hover_cube_mesh.setVisible(bool(self.hover_cube_valid))
            self.sel_cube_mesh.setVisible(bool(self.selected_cube_valid))
        else:
            self.scatter.setVisible(True)
            self.cube_mesh.setVisible(False)
            self.hover_cube_mesh.setVisible(False)
            self.sel_cube_mesh.setVisible(False)

        # 手动方块：两种模式都显示（用于标定/测距）
        self.manual_mesh.setVisible(True)

        self.grid.setVisible(bool(self.chk_grid_overlay.isChecked()))
        self.measure_line_item.setVisible(bool(self.measure_enable and len(self.measure_points_world) >= 2))

    # ============================================================
    # GL 键盘（含手动方块 WSADQE）
    # ============================================================

    def gl_key_press_event(self, event):
        # 若当前激活的是手动方块，则 WSADQE 直接移动（像 CAD 里移动参考体一样）
        if self.active_manual_idx >= 0 and self.active_manual_idx < len(self.manual_cubes_world):
            k = event.key()
            mods = event.modifiers()

            vs = self._get_voxel_size_scaled()
            step = vs * 0.25
            if mods & Qt.ShiftModifier:
                step *= 4.0
            if mods & Qt.ControlModifier:
                step *= 0.20

            moved = False

            # 约定：X前方、Y左方、Z上方（与常见雷达坐标一致）
            if k == Qt.Key_W:
                self._move_active_manual_cube(+1, 0, 0, step); moved = True
            elif k == Qt.Key_S:
                self._move_active_manual_cube(-1, 0, 0, step); moved = True
            elif k == Qt.Key_A:
                self._move_active_manual_cube(0, +1, 0, step); moved = True
            elif k == Qt.Key_D:
                self._move_active_manual_cube(0, -1, 0, step); moved = True
            elif k == Qt.Key_Q:
                self._move_active_manual_cube(0, 0, -1, step); moved = True
            elif k == Qt.Key_E:
                self._move_active_manual_cube(0, 0, +1, step); moved = True

            if moved:
                return

        if event.key() == Qt.Key_R:
            self.gl_view.opts["elevation"] = 25
            self.gl_view.opts["azimuth"] = -45
            self.gl_view.opts["distance"] = 30
            self.gl_view.update()
            self.need_gl_update = True
        elif event.key() == Qt.Key_A:
            if self.axis_visible:
                for item in self.axis_items:
                    self.gl_view.removeItem(item)
                self.axis_visible = False
            else:
                for item in self.axis_items:
                    self.gl_view.addItem(item)
                self.axis_visible = True
        elif event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            if self.measure_enable:
                self.on_measure_undo()
        else:
            self._orig_gl_key_press(event)

    # ============================================================
    # 屏幕/射线缓存 & 点云 2D 栅格索引（略：保持原逻辑）
    # ============================================================

    def _update_viewproj_cache(self):
        w = max(1, int(self.gl_view.width()))
        h = max(1, int(self.gl_view.height()))
        self._screen_w = w
        self._screen_h = h

        opts = self.gl_view.opts
        az = math.radians(float(opts.get("azimuth", -45.0)))
        el = math.radians(float(opts.get("elevation", 25.0)))
        dist = float(opts.get("distance", 30.0))
        fov = float(opts.get("fov", 60.0))

        center = opts.get("center", pg.Vector(0, 0, 0))
        try:
            c = np.array([center.x(), center.y(), center.z()], dtype=np.float32)
        except AttributeError:
            c = np.array(center, dtype=np.float32)

        ce = math.cos(el)
        se = math.sin(el)
        ca = math.cos(az)
        sa = math.sin(az)
        cam = c + dist * np.array([ce * ca, ce * sa, se], dtype=np.float32)
        self._cam_pos = cam.astype(np.float32)

        view = mat4_look_at(cam, c, up=(0.0, 0.0, 1.0))
        aspect = float(w) / float(max(1, h))
        z_near = max(1e-3, dist * 0.01)
        z_far = max(50.0, dist * 50.0)
        proj = mat4_perspective(fov, aspect, z_near, z_far)

        vp = proj @ view
        self._vp = vp.astype(np.float32)
        try:
            self._inv_vp = np.linalg.inv(self._vp).astype(np.float32)
        except Exception:
            self._inv_vp = None

    def _unproject_ndc(self, x_ndc, y_ndc, z_ndc):
        if self._inv_vp is None:
            return None
        p = np.array([x_ndc, y_ndc, z_ndc, 1.0], dtype=np.float32)
        w = self._inv_vp @ p
        if abs(float(w[3])) < 1e-12:
            return None
        return (w[:3] / w[3]).astype(np.float32)

    def _screen_to_ray(self, x_pix, y_pix):
        self._update_viewproj_cache()
        if self._inv_vp is None:
            return None

        w = float(self._screen_w)
        h = float(self._screen_h)

        x_ndc = (2.0 * (float(x_pix) / w)) - 1.0
        y_ndc = 1.0 - (2.0 * (float(y_pix) / h))

        p_far = self._unproject_ndc(x_ndc, y_ndc, 1.0)
        if p_far is None:
            return None

        o = self._cam_pos.copy()
        d = _normalize(p_far - o)
        return o.astype(np.float32), d.astype(np.float32)

    def _project_points_to_screen(self, pts_world: np.ndarray):
        if pts_world is None or pts_world.size == 0:
            return None, None, None

        self._update_viewproj_cache()
        vp = self._vp

        pts = pts_world.astype(np.float32)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        P = np.hstack([pts, ones])
        clip = (vp @ P.T).T

        w = clip[:, 3]
        valid = np.abs(w) > 1e-9
        ndc = np.zeros((pts.shape[0], 3), dtype=np.float32)
        ndc[valid, 0] = clip[valid, 0] / w[valid]
        ndc[valid, 1] = clip[valid, 1] / w[valid]
        ndc[valid, 2] = clip[valid, 2] / w[valid]

        in_clip = valid & (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0) & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)

        w_px = float(self._screen_w)
        h_px = float(self._screen_h)
        x = (ndc[:, 0] * 0.5 + 0.5) * w_px
        y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h_px

        xy = np.vstack([x, y]).T.astype(np.float32)
        return xy, ndc[:, 2].astype(np.float32), in_clip

    def _rebuild_point_screen_index(self, pts_world: np.ndarray):
        xy, z, valid = self._project_points_to_screen(pts_world)
        self._screen_grid = {}
        self._screen_grid_valid = False
        self._points_proj_xy = None
        self._points_proj_z = None
        self._points_proj_valid = None

        if xy is None:
            return

        self._points_proj_xy = xy
        self._points_proj_z = z
        self._points_proj_valid = valid

        cell = int(PICK_CELL_PX)
        idxs = np.nonzero(valid)[0]
        for i in idxs:
            px = float(xy[i, 0])
            py = float(xy[i, 1])
            gx = int(px // cell)
            gy = int(py // cell)
            key = (gx, gy)
            lst = self._screen_grid.get(key)
            if lst is None:
                self._screen_grid[key] = [int(i)]
            else:
                if len(lst) < PICK_MAX_CANDS:
                    lst.append(int(i))

        self._screen_grid_valid = True

    def pick_point_stable(self, x_pix, y_pix):
        if not self._screen_grid_valid or self._points_proj_xy is None:
            return None

        cell = int(PICK_CELL_PX)
        gx = int(float(x_pix) // cell)
        gy = int(float(y_pix) // cell)

        cand = []
        for dx in range(-PICK_NEIGHBOR_CELLS, PICK_NEIGHBOR_CELLS + 1):
            for dy in range(-PICK_NEIGHBOR_CELLS, PICK_NEIGHBOR_CELLS + 1):
                lst = self._screen_grid.get((gx + dx, gy + dy))
                if lst:
                    cand.extend(lst)
                    if len(cand) >= PICK_MAX_CANDS:
                        break
            if len(cand) >= PICK_MAX_CANDS:
                break

        if not cand:
            return None

        cand = np.array(cand, dtype=np.int32)
        xy = self._points_proj_xy[cand]
        dz = self._points_proj_z[cand]
        dx = xy[:, 0] - float(x_pix)
        dy = xy[:, 1] - float(y_pix)
        dist2 = dx * dx + dy * dy

        r2 = float(PICK_RADIUS_PX * PICK_RADIUS_PX)
        m = dist2 <= r2
        if not np.any(m):
            return None

        sub = cand[m]
        d2 = dist2[m]
        zsub = dz[m]
        order = np.lexsort((zsub, d2))
        return int(sub[order[0]])

    # ============================================================
    # 方块拾取：3D 体素哈希 + 射线 DDA（合并 马赛克 + 手动方块）
    # ============================================================

    def _world_to_sensor_for_pick(self, p_world_displayed: np.ndarray) -> np.ndarray:
        p = np.asarray(p_world_displayed, dtype=np.float32).copy()

        # displayed world -> unmirror
        if self.mirror_lr:
            p[1] *= -1.0

        # world -> sensor inverse: p_sensor = (p_world - t) @ R
        if self.calib_enabled:
            p = (p - self.t_extr) @ self.R_extr
        return p.astype(np.float32)

    def _rebuild_cube_voxel_index_combined(self,
                                          mosaic_centers_world: np.ndarray,
                                          mosaic_ijk: np.ndarray,
                                          voxel_size_scaled: float):
        """
        合并索引用于拾取：
          - 马赛克中心（mosaic_centers_world/mosaic_ijk）
          - 手动方块中心（self.manual_cubes_world）
        """
        vs = float(voxel_size_scaled)
        if vs <= 1e-9:
            self._cube_voxel_map = {}
            self._cube_aabb_min = None
            self._cube_aabb_max = None
            self._cube_vs = None
            self.last_cubes_world = None
            self.last_cubes_sensor = None
            self.last_cubes_is_manual = None
            self.last_cubes_manual_index = None
            return

        # 合并 centers
        m0 = 0
        if mosaic_centers_world is not None and mosaic_centers_world.size:
            m0 = int(mosaic_centers_world.shape[0])
        m1 = len(self.manual_cubes_world)

        if m0 == 0 and m1 == 0:
            self._cube_voxel_map = {}
            self._cube_aabb_min = None
            self._cube_aabb_max = None
            self._cube_vs = None
            self.last_cubes_world = None
            self.last_cubes_sensor = None
            self.last_cubes_is_manual = None
            self.last_cubes_manual_index = None
            return

        parts_world = []
        parts_ijk = []
        is_manual = []
        manual_index = []

        if m0 > 0:
            parts_world.append(mosaic_centers_world.astype(np.float32))
            parts_ijk.append(mosaic_ijk.astype(np.int32) if mosaic_ijk is not None else np.floor(mosaic_centers_world / vs).astype(np.int32))
            is_manual.extend([False] * m0)
            manual_index.extend([-1] * m0)

        if m1 > 0:
            man_world = np.vstack([c.reshape(1, 3) for c in self.manual_cubes_world]).astype(np.float32)
            man_ijk = np.floor(man_world / vs).astype(np.int32)
            parts_world.append(man_world)
            parts_ijk.append(man_ijk)
            is_manual.extend([True] * m1)
            manual_index.extend(list(range(m1)))

        centers_world = np.vstack(parts_world).astype(np.float32)
        ijk = np.vstack(parts_ijk).astype(np.int32)
        is_manual = np.array(is_manual, dtype=bool)
        manual_index = np.array(manual_index, dtype=np.int32)

        # 合并后的 sensor centers
        centers_sensor = np.zeros_like(centers_world, dtype=np.float32)
        for i in range(int(centers_world.shape[0])):
            centers_sensor[i] = self._world_to_sensor_for_pick(centers_world[i])

        # 建立 voxel->index（同体素冲突时：后写覆盖，手动方块在后面会“更容易点到”）
        voxel_map = {}
        for idx in range(int(ijk.shape[0])):
            key = (int(ijk[idx, 0]), int(ijk[idx, 1]), int(ijk[idx, 2]))
            voxel_map[key] = int(idx)

        half = 0.5 * vs
        aabb_min = (centers_world.min(axis=0) - half).astype(np.float32)
        aabb_max = (centers_world.max(axis=0) + half).astype(np.float32)

        self._cube_voxel_map = voxel_map
        self._cube_aabb_min = aabb_min
        self._cube_aabb_max = aabb_max
        self._cube_vs = vs

        self.last_cubes_world = centers_world
        self.last_cubes_sensor = centers_sensor
        self.last_cubes_is_manual = is_manual
        self.last_cubes_manual_index = manual_index

    def pick_cube_stable(self, x_pix, y_pix):
        if (not self._cube_voxel_map) or self.last_cubes_world is None:
            return None

        ray = self._screen_to_ray(x_pix, y_pix)
        if ray is None:
            return None
        o, d = ray

        if self._cube_aabb_min is None or self._cube_aabb_max is None:
            return None

        hit = ray_aabb_intersect(o, d, self._cube_aabb_min, self._cube_aabb_max)
        if hit is None:
            return None
        t_enter, t_exit = hit
        t_enter = max(0.0, t_enter)

        p = o + d * (t_enter + RAY_T_EPS)

        vs = float(self._cube_vs)
        if vs <= 1e-9:
            return None

        pv = p / vs
        dv = d / vs

        i = int(math.floor(float(pv[0])))
        j = int(math.floor(float(pv[1])))
        k = int(math.floor(float(pv[2])))

        step_x = 1 if dv[0] > 0 else (-1 if dv[0] < 0 else 0)
        step_y = 1 if dv[1] > 0 else (-1 if dv[1] < 0 else 0)
        step_z = 1 if dv[2] > 0 else (-1 if dv[2] < 0 else 0)

        def _tmax(pv_comp, dv_comp, i_comp, step):
            if step == 0:
                return 1e18
            next_boundary = (i_comp + 1.0) if step > 0 else float(i_comp)
            return (next_boundary - pv_comp) / dv_comp

        def _tdelta(dv_comp, step):
            if step == 0:
                return 1e18
            return 1.0 / abs(float(dv_comp))

        tMaxX = _tmax(float(pv[0]), float(dv[0]), i, step_x)
        tMaxY = _tmax(float(pv[1]), float(dv[1]), j, step_y)
        tMaxZ = _tmax(float(pv[2]), float(dv[2]), k, step_z)

        tDeltaX = _tdelta(float(dv[0]), step_x)
        tDeltaY = _tdelta(float(dv[1]), step_y)
        tDeltaZ = _tdelta(float(dv[2]), step_z)

        t_limit = (t_exit - t_enter) / vs + 2.0

        steps = 0
        t_cur = 0.0
        while steps < RAY_MAX_STEPS and t_cur <= t_limit:
            key = (i, j, k)
            idx = self._cube_voxel_map.get(key)
            if idx is not None:
                idx = int(idx)
                c = self.last_cubes_world[idx]
                half = 0.5 * vs
                bmin = c - half
                bmax = c + half
                h2 = ray_aabb_intersect(o, d, bmin, bmax)
                if h2 is not None:
                    return idx

            if tMaxX < tMaxY:
                if tMaxX < tMaxZ:
                    i += step_x
                    t_cur = tMaxX
                    tMaxX += tDeltaX
                else:
                    k += step_z
                    t_cur = tMaxZ
                    tMaxZ += tDeltaZ
            else:
                if tMaxY < tMaxZ:
                    j += step_y
                    t_cur = tMaxY
                    tMaxY += tDeltaY
                else:
                    k += step_z
                    t_cur = tMaxZ
                    tMaxZ += tDeltaZ

            steps += 1

        return None

    # ============================================================
    # 悬停高亮（不透明深紫色）
    # ============================================================

    def _clear_hover(self):
        self.hover_cube_valid = False
        self.hover_cube_idx = None
        self.hover_cube_world = None
        self.hover_is_manual = False
        self.hover_cube_mesh.setVisible(False)

    def _set_hover_cube(self, cube_idx: int):
        if self.last_cubes_world is None or cube_idx is None:
            self._clear_hover()
            return

        cube_idx = int(cube_idx)
        if cube_idx < 0 or cube_idx >= int(self.last_cubes_world.shape[0]):
            self._clear_hover()
            return

        if self.hover_cube_valid and self.hover_cube_idx == cube_idx:
            return

        self.hover_cube_idx = cube_idx
        self.hover_cube_world = self.last_cubes_world[cube_idx].astype(np.float32)
        self.hover_is_manual = bool(self.last_cubes_is_manual[cube_idx]) if self.last_cubes_is_manual is not None else False
        self.hover_cube_valid = True

        voxel_size_scaled = self._get_voxel_size_scaled()

        rgba = np.array([0.20, 0.00, 0.35, 1.0], dtype=np.float32)  # 深紫不透明
        verts, faces, face_cols = build_single_cube_mesh(self.hover_cube_world, voxel_size_scaled, rgba)
        self.hover_cube_mesh.setMeshData(vertexes=verts, faces=faces, faceColors=face_cols)
        self.hover_cube_mesh.setVisible(True)

    # ============================================================
    # 选中方块（用于测距/标定点击落点；手动方块可被选中并 WSADQE 移动）
    # ============================================================

    def _clear_selected(self):
        self.selected_cube_valid = False
        self.selected_cube_world = None
        self.selected_cube_sensor = None
        self.selected_is_manual = False
        self.selected_manual_idx = -1
        self.sel_cube_mesh.setVisible(False)

    def _refresh_selected_cube_mesh(self):
        if (not self.selected_cube_valid) or (self.selected_cube_world is None):
            self.sel_cube_mesh.setVisible(False)
            self._apply_render_visibility()
            return

        voxel_size_scaled = self._get_voxel_size_scaled()
        a = 1.0 if self.measure_enable else 0.45
        rgba = np.array([0.75, 0.10, 0.95, a], dtype=np.float32)  # 紫色（测距时不透明）

        verts, faces, face_cols = build_single_cube_mesh(self.selected_cube_world, voxel_size_scaled, rgba)
        self.sel_cube_mesh.setMeshData(vertexes=verts, faces=faces, faceColors=face_cols)
        self.sel_cube_mesh.setVisible(True)
        self._apply_render_visibility()

    def _set_selected_cube(self, cube_idx: int):
        if self.last_cubes_world is None or cube_idx is None:
            self._clear_selected()
            self._apply_render_visibility()
            return

        cube_idx = int(cube_idx)
        if cube_idx < 0 or cube_idx >= int(self.last_cubes_world.shape[0]):
            self._clear_selected()
            self._apply_render_visibility()
            return

        self.selected_cube_world = self.last_cubes_world[cube_idx].astype(np.float32)
        self.selected_cube_sensor = self.last_cubes_sensor[cube_idx].astype(np.float32) if self.last_cubes_sensor is not None else self._world_to_sensor_for_pick(self.selected_cube_world)

        self.selected_is_manual = bool(self.last_cubes_is_manual[cube_idx]) if self.last_cubes_is_manual is not None else False
        self.selected_manual_idx = int(self.last_cubes_manual_index[cube_idx]) if (self.last_cubes_manual_index is not None) else -1
        if self.selected_is_manual and self.selected_manual_idx >= 0:
            self.active_manual_idx = self.selected_manual_idx

        self.selected_cube_valid = True
        self._refresh_selected_cube_mesh()

    # ============================================================
    # Mouse Move / Hover
    # ============================================================

    def gl_mouse_move_event(self, event):
        try:
            self._hover_last_pos = (event.pos().x(), event.pos().y())
        except Exception:
            self._hover_last_pos = None
        self._orig_gl_mouse_move(event)

    def _hover_tick(self):
        # 仅在马赛克模式下启用“悬停高亮”
        if self.render_mode != MODE_CUBE_MOSAIC:
            if self.hover_cube_valid:
                self._clear_hover()
            return

        if self._hover_last_pos is None:
            if self.hover_cube_valid:
                self._clear_hover()
            return

        x, y = self._hover_last_pos

        idx = self.pick_cube_stable(x, y)
        if idx is None:
            if self.hover_cube_valid:
                self._clear_hover()
            return

        self._set_hover_cube(idx)
        self._apply_render_visibility()

    # ============================================================
    # GL 鼠标（单击：测距/标定/验证；方块优先吸附中心）
    # ============================================================

    def gl_mouse_press_event(self, event):
        if event.button() != Qt.LeftButton:
            self._orig_gl_mouse_press(event)
            return

        x = event.pos().x()
        y = event.pos().y()

        cube_hit = self.pick_cube_stable(x, y)
        pt_hit = self.pick_point_stable(x, y)

        picked_world = None
        picked_sensor = None

        if cube_hit is not None and self.last_cubes_world is not None:
            self._set_selected_cube(cube_hit)
            picked_world = self.last_cubes_world[cube_hit].astype(np.float32)
            picked_sensor = self.last_cubes_sensor[cube_hit].astype(np.float32) if self.last_cubes_sensor is not None else self._world_to_sensor_for_pick(picked_world)
        elif pt_hit is not None and self.last_pts_world is not None and self.last_pts_sensor is not None:
            picked_world = self.last_pts_world[pt_hit].astype(np.float32)
            picked_sensor = self.last_pts_sensor[pt_hit].astype(np.float32)

        # 测距：连续多段
        if self.measure_enable:
            if picked_world is None:
                QMessageBox.warning(self, "测距", "未能拾取到目标，请放大/调整视角后重试。")
                self._orig_gl_mouse_press(event)
                return
            self.measure_points_world.append(picked_world.copy())
            self.measure_points_sensor.append(picked_sensor.copy())
            self._rebuild_measure_segments()
            self._update_measure_line()
            self._refresh_selected_cube_mesh()
            self._orig_gl_mouse_press(event)
            return

        # 点选经纬度标定：优先方块中心（包括绿色手动方块）
        if self.calib_by_click_mode:
            if picked_sensor is None:
                QMessageBox.warning(self, "提示", "未能拾取到目标，请放大/调整视角后重试。")
                self._orig_gl_mouse_press(event)
                return
            dlg = CalibFromPointDialog(self, picked_sensor)
            dlg.exec_()
            self._orig_gl_mouse_press(event)
            return

        # 标定验证
        if self.current_target_idx >= 0 and 0 <= self.current_target_idx < len(self.known_targets):
            if picked_world is not None:
                tgt = self.known_targets[self.current_target_idx]
                target_pos = tgt["pos"]
                err_scaled = float(np.linalg.norm(picked_world - target_pos))
                err_m = err_scaled / SCALE
                err_cm = err_m * 100.0
                QMessageBox.information(
                    self, "标定验证结果",
                    f"障碍物: {tgt['name']}\n"
                    f"点击坐标(缩放): [{picked_world[0]:.4f}, {picked_world[1]:.4f}, {picked_world[2]:.4f}]\n"
                    f"理论坐标(缩放): [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]\n"
                    f"误差: {err_m:.4f} m  ({err_cm:.1f} cm)"
                )
            else:
                QMessageBox.warning(self, "提示", "未能拾取到目标，请放大/调整视角后重试。")

        self._orig_gl_mouse_press(event)

    # ---------- GL 鼠标（双击：弹出经纬度） ----------
    def gl_mouse_double_click_event(self, event):
        if event.button() != Qt.LeftButton:
            if self._orig_gl_mouse_double:
                self._orig_gl_mouse_double(event)
            return

        use_geo = (self.geo_enabled and self.calib_enabled and self.ref_lat is not None and self.ref_lon is not None)
        if not use_geo:
            QMessageBox.information(self, "提示", "尚未启用经纬度标定。请先用“点选经纬度标定”。")
            if self._orig_gl_mouse_double:
                self._orig_gl_mouse_double(event)
            return

        x = event.pos().x()
        y = event.pos().y()

        cube_hit = self.pick_cube_stable(x, y)
        if cube_hit is not None and self.last_cubes_world is not None:
            p_world = self.last_cubes_world[cube_hit].astype(np.float32)
            p_sensor = self.last_cubes_sensor[cube_hit].astype(np.float32) if self.last_cubes_sensor is not None else self._world_to_sensor_for_pick(p_world)
        else:
            idx = self.pick_point_stable(x, y)
            if idx is None or self.last_pts_world is None or self.last_pts_sensor is None:
                QMessageBox.warning(self, "提示", "未能拾取到目标，请放大或调整视角后重试。")
                if self._orig_gl_mouse_double:
                    self._orig_gl_mouse_double(event)
                return
            p_world = self.last_pts_world[idx].astype(np.float32)
            p_sensor = self.last_pts_sensor[idx].astype(np.float32)

        p_enu_scaled = (p_sensor @ self.R_extr.T) + self.t_extr
        lat, lon, alt = GEO.enu_to_lla(p_enu_scaled, self.ref_lat, self.ref_lon, self.ref_alt, scale=SCALE,
                                       high_precision=self.geo_high_precision)
        if np.ndim(lat) > 0:
            lat, lon, alt = float(lat[0]), float(lon[0]), float(alt[0])

        QMessageBox.information(
            self, "点信息（经纬度）",
            f"X = {p_world[0]:.4f}\nY = {p_world[1]:.4f}\nZ = {p_world[2]:.4f}\n\n"
            f"lat = {lat:.7f}°\nlon = {lon:.7f}°\nh   = {alt:.3f} m"
        )
        if self._orig_gl_mouse_double:
            self._orig_gl_mouse_double(event)

    # ============================================================
    # 队列轮询/写环形缓冲/表格更新（保持原逻辑）
    # ============================================================

    def poll_queue(self):
        if self._paused:
            return

        items = []
        drained = 0
        while drained < MAX_DRAIN_PER_TICK:
            try:
                items.append(self._parsed_q.get_nowait())
                drained += 1
            except queue.Empty:
                break
            except Exception:
                break

        if not items:
            return

        items = items[-POLL_APPLY_FRAMES:]
        for ts, arr in items:
            self.on_new_packet(arr, ts)

    def _drain_queue_all(self):
        for qx in (self._raw_q, self._parsed_q):
            while True:
                try:
                    _ = qx.get_nowait()
                except Exception:
                    break

    def _append_to_ring_buffer(self, xyz, times, dist, az, inten):
        n = int(xyz.shape[0])
        if n <= 0:
            return

        if n >= MAX_POINTS:
            xyz = xyz[-MAX_POINTS:]
            times = times[-MAX_POINTS:]
            dist = dist[-MAX_POINTS:]
            az = az[-MAX_POINTS:]
            inten = inten[-MAX_POINTS:]
            n = MAX_POINTS

        end = self.write_idx + n
        if end <= MAX_POINTS:
            self.xyz_buf[self.write_idx:end] = xyz
            self.time_buf[self.write_idx:end] = times
            self.dist_buf[self.write_idx:end] = dist
            self.az_buf[self.write_idx:end] = az
            self.inten_buf[self.write_idx:end] = inten
        else:
            first = MAX_POINTS - self.write_idx
            self.xyz_buf[self.write_idx:] = xyz[:first]
            self.time_buf[self.write_idx:] = times[:first]
            self.dist_buf[self.write_idx:] = dist[:first]
            self.az_buf[self.write_idx:] = az[:first]
            self.inten_buf[self.write_idx:] = inten[:first]

            second = end - MAX_POINTS
            self.xyz_buf[:second] = xyz[first:]
            self.time_buf[:second] = times[first:]
            self.dist_buf[:second] = dist[first:]
            self.az_buf[:second] = az[first:]
            self.inten_buf[:second] = inten[first:]

        self.write_idx = (self.write_idx + n) % MAX_POINTS

    def on_new_packet(self, arr, ts):
        xyz = np.vstack([arr["x"], arr["y"], arr["z"]]).T.astype(np.float32)
        xyz *= SCALE
        times = np.full((xyz.shape[0],), float(ts), dtype=np.float64)

        dist = arr["distance"].astype(np.float32)
        az = arr["azimuth"].astype(np.float32)
        inten = arr["intensity"].astype(np.float32)

        self._append_to_ring_buffer(xyz, times, dist, az, inten)
        self.need_gl_update = True

        if float(ts) - self.last_table_update > TABLE_UPDATE_INTERVAL:
            self.update_table(arr, float(ts))
            self.last_table_update = float(ts)

    def update_table(self, arr, now):
        rows = min(int(arr.shape[0]), TABLE_ROWS)
        self.table.setRowCount(rows)
        data = arr[-rows:]

        use_geo = (self.geo_enabled and self.calib_enabled and self.ref_lat is not None and self.ref_lon is not None)

        for i in range(rows):
            x = float(data["x"][i])
            y = float(data["y"][i])
            z = float(data["z"][i])
            az = float(data["azimuth"][i])
            dist = float(data["distance"][i])
            inten = float(data["intensity"][i])

            lat_str = lon_str = h_str = "-"

            if use_geo:
                p_sensor_scaled = np.array([x, y, z], dtype=np.float32) * SCALE
                p_enu_scaled = (p_sensor_scaled @ self.R_extr.T) + self.t_extr
                lat, lon, h = GEO.enu_to_lla(p_enu_scaled, self.ref_lat, self.ref_lon, self.ref_alt,
                                             scale=SCALE, high_precision=self.geo_high_precision)
                if np.ndim(lat) > 0:
                    lat, lon, h = float(lat[0]), float(lon[0]), float(h[0])
                lat_str = f"{lat:.7f}"
                lon_str = f"{lon:.7f}"
                h_str = f"{h:.3f}"

            values = [
                str(i),
                f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
                lat_str, lon_str, h_str,
                f"{az:.2f}", f"{dist:.4f}", f"{inten:.0f}",
            ]

            for col, v in enumerate(values):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, col, item)

        if now - self.last_table_resize > TABLE_RESIZE_INTERVAL:
            self.table.resizeColumnsToContents()
            self.last_table_resize = now

    # ============================================================
    # 颜色计算 / 动态点大小（保持原逻辑）
    # ============================================================

    def _compute_colors_for_points(self, pts_world, dists, intens):
        if pts_world is None or pts_world.size == 0:
            return np.zeros((0, 4), dtype=np.float32)

        if self.color_mode == 0:
            r = dists
            if self.use_fixed_color_range:
                r_min = self.fixed_r_min
                r_max = self.fixed_r_max
            else:
                cur_min = float(r.min())
                cur_max = float(r.max()) if float(r.max()) > cur_min else cur_min + 1e-3
                if self._rmin_ema is None:
                    self._rmin_ema = cur_min
                    self._rmax_ema = cur_max
                else:
                    b = COLOR_RANGE_EMA_BETA
                    self._rmin_ema = (1 - b) * self._rmin_ema + b * cur_min
                    self._rmax_ema = (1 - b) * self._rmax_ema + b * cur_max
                r_min = self._rmin_ema
                r_max = self._rmax_ema

            den = float(r_max - r_min)
            if den < 1e-6:
                den = 1e-6
            norm = (r - r_min) / den
            colors = colormap_distance(norm)

        elif self.color_mode == 1:
            v = intens.astype(np.float32)
            v_min = float(v.min())
            v_max = float(v.max()) if float(v.max()) > v_min else v_min + 1e-3
            norm = (v - v_min) / (v_max - v_min)
            colors = colormap_blue_yellow_red_wide(norm, YELLOW_START, YELLOW_END)

        elif self.color_mode == 2:
            xy = pts_world[:, :2]
            norm, cur_max = density_heat_norm_xy(xy, bin_m=HEAT_BIN_M, max_ema=self._heat_max_ema)
            if cur_max > 1e-6:
                if self._heat_max_ema is None:
                    self._heat_max_ema = cur_max
                else:
                    b = HEAT_MAX_EMA_BETA
                    self._heat_max_ema = (1 - b) * self._heat_max_ema + b * cur_max
            colors = colormap_blue_yellow_red_wide(norm, YELLOW_START, YELLOW_END)

        else:
            z = pts_world[:, 2].astype(np.float32) / SCALE
            zmin = float(self.z_min)
            zmax = float(self.z_max)
            if zmax <= zmin + 1e-6:
                zmax = zmin + 1.0
            norm = (z - zmin) / (zmax - zmin)
            colors = colormap_blue_yellow_red_wide(norm, YELLOW_START, YELLOW_END)

        return colors

    def _compute_point_sizes(self, dists):
        if (not self.dynamic_point_size) or dists is None or dists.size == 0:
            return float(POINT_SIZE_WORLD_BASE)

        r = dists.astype(np.float32)
        rmin = float(POINT_SIZE_R_MIN)
        rmax = float(POINT_SIZE_R_MAX)
        t = (r - rmin) / max(1e-6, (rmax - rmin))
        t = np.clip(t, 0.0, 1.0)
        s = (1.0 - t) ** 0.7
        sizes = POINT_SIZE_FAR + (POINT_SIZE_NEAR - POINT_SIZE_FAR) * s
        return sizes.astype(np.float32)

    # ============================================================
    # 3D 刷新（关键增强：每次刷新后更新“合并方块拾取索引”，手动方块也参与拾取）
    # ============================================================

    def update_gl_view(self, force=False):
        now = time.time()

        if not force:
            if now - self.last_gl_update < UPDATE_INTERVAL_SEC:
                return
            if not self.need_gl_update:
                self._update_status_only()
                return

        self.last_gl_update = now
        self._adaptive_update(now)

        t_min = now - float(self.persist_sec) - WINDOW_HYST_SEC
        mask = self.time_buf >= t_min

        pts_sensor = self.xyz_buf[mask]
        dists = self.dist_buf[mask]
        intens = self.inten_buf[mask]
        times_sel = self.time_buf[mask]

        # 无点云也要刷新手动方块拾取/显示
        if pts_sensor.size == 0:
            self.scatter.setData(pos=np.empty((0, 3)), size=float(POINT_SIZE_WORLD_BASE), pxMode=SCATTER_PX_MODE)
            self.cube_mesh.setMeshData(vertexes=np.empty((0, 3), dtype=np.float32),
                                       faces=np.empty((0, 3), dtype=np.int32),
                                       faceColors=np.empty((0, 4), dtype=np.float32))
            self._screen_grid_valid = False
            self._clear_hover()

            # 仍然建立“仅手动方块”的拾取索引
            self._rebuild_cube_voxel_index_combined(
                mosaic_centers_world=np.empty((0, 3), dtype=np.float32),
                mosaic_ijk=np.empty((0, 3), dtype=np.int32),
                voxel_size_scaled=self._get_voxel_size_scaled()
            )
            self._refresh_manual_mesh()

            self._update_status(now, fps=None, shown_n=0, cubes_n=(0 if self.last_cubes_world is None else int(self.last_cubes_world.shape[0])))
            self.need_gl_update = False
            self.last_pts_world = None
            self.last_pts_sensor = None
            return

        n = int(pts_sensor.shape[0])
        max_points = int(MAX_GL_POINTS * float(self.density_factor))
        max_points = min(MAX_POINTS, max_points)
        max_points = max(8000, max_points)

        if n > max_points:
            step = int(math.ceil(n / float(max_points)))
            step = max(1, step)
            start = self._subsample_phase % step
            self._subsample_phase += 1
            idx = np.arange(start, n, step, dtype=np.int32)
            pts_sensor = pts_sensor[idx]
            dists = dists[idx]
            intens = intens[idx]
            times_sel = times_sel[idx]

        self.last_pts_sensor = pts_sensor.copy()

        pts_world = pts_sensor
        if self.calib_enabled:
            pts_world = (pts_world @ self.R_extr.T) + self.t_extr[None, :]
        if self.mirror_lr:
            pts_world = pts_world.copy()
            pts_world[:, 1] *= -1.0

        roi_mask = self.roi_cfg.apply(pts_world)
        if roi_mask is not None and roi_mask.size == pts_world.shape[0]:
            pts_world = pts_world[roi_mask]
            pts_sensor = pts_sensor[roi_mask]
            dists = dists[roi_mask]
            intens = intens[roi_mask]
            times_sel = times_sel[roi_mask]
            self.last_pts_sensor = pts_sensor.copy()

        if pts_world.size == 0:
            self.scatter.setData(pos=np.empty((0, 3)), size=float(POINT_SIZE_WORLD_BASE), pxMode=SCATTER_PX_MODE)
            self.cube_mesh.setMeshData(vertexes=np.empty((0, 3), dtype=np.float32),
                                       faces=np.empty((0, 3), dtype=np.int32),
                                       faceColors=np.empty((0, 4), dtype=np.float32))
            self._screen_grid_valid = False
            self._clear_hover()

            self._rebuild_cube_voxel_index_combined(
                mosaic_centers_world=np.empty((0, 3), dtype=np.float32),
                mosaic_ijk=np.empty((0, 3), dtype=np.int32),
                voxel_size_scaled=self._get_voxel_size_scaled()
            )
            self._refresh_manual_mesh()

            self._update_status(now, fps=self._fps_ema, shown_n=0, cubes_n=(0 if self.last_cubes_world is None else int(self.last_cubes_world.shape[0])))
            self.need_gl_update = False
            self.last_pts_world = None
            return

        shown_n = int(pts_world.shape[0])
        cubes_n = 0

        age = (now - times_sel).astype(np.float32)
        age = np.clip(age, 0.0, float(self.persist_sec))
        alpha = np.exp(-age / max(1e-6, float(FADE_TAU))).astype(np.float32)
        if abs(ALPHA_POWER - 1.0) > 1e-6:
            alpha = np.power(alpha, float(ALPHA_POWER)).astype(np.float32)
        alpha = np.clip(alpha, float(ALPHA_FLOOR), 1.0).astype(np.float32)

        voxel_size_scaled = self._get_voxel_size_scaled()

        if self.render_mode != MODE_CUBE_MOSAIC:
            colors = self._compute_colors_for_points(pts_world, dists, intens)
            if colors.shape[0] == pts_world.shape[0]:
                colors[:, 3] = alpha

            sizes = self._compute_point_sizes(dists)
            self.scatter.setData(pos=pts_world, size=sizes, color=colors, pxMode=SCATTER_PX_MODE)
            self.last_pts_world = pts_world.copy()

            self._rebuild_point_screen_index(self.last_pts_world)

            # 点云模式仍然建立“仅手动方块”的拾取索引（保证标定/测距能吸附绿色方块）
            self._rebuild_cube_voxel_index_combined(
                mosaic_centers_world=np.empty((0, 3), dtype=np.float32),
                mosaic_ijk=np.empty((0, 3), dtype=np.int32),
                voxel_size_scaled=voxel_size_scaled
            )
            self._clear_hover()

        else:
            v_center_world, v_dist, v_inten, v_time, v_count, v_ijk = voxelize_points(
                pts_world, dists, intens, times_sel, voxel_size_scaled
            )

            m = int(v_center_world.shape[0])
            if m > MAX_CUBES_RENDER:
                order = np.argsort(v_time)[-MAX_CUBES_RENDER:]
                v_center_world = v_center_world[order]
                v_dist = v_dist[order]
                v_inten = v_inten[order]
                v_time = v_time[order]
                v_count = v_count[order]
                v_ijk = v_ijk[order]
                m = int(v_center_world.shape[0])

            cubes_n = m

            if m <= 0:
                colors = np.empty((0, 4), dtype=np.float32)
            elif self.color_mode == 0:
                r = v_dist
                r_min = self.fixed_r_min if self.use_fixed_color_range else float(r.min())
                r_max = self.fixed_r_max if self.use_fixed_color_range else float(r.max())
                den = max(1e-6, float(r_max - r_min))
                norm = (r - r_min) / den
                colors = colormap_distance(norm)
            elif self.color_mode == 1:
                v = v_inten.astype(np.float32)
                v_min = float(v.min())
                v_max = float(v.max()) if float(v.max()) > v_min else v_min + 1e-3
                norm = (v - v_min) / (v_max - v_min)
                colors = colormap_blue_yellow_red_wide(norm, YELLOW_START, YELLOW_END)
            elif self.color_mode == 2:
                c = v_count.astype(np.float32)
                if HEAT_LOG_COMPRESS:
                    c = np.log1p(c)
                cur_max = float(c.max()) if float(c.max()) > 1e-6 else 1.0
                if self._heat_max_ema is None:
                    self._heat_max_ema = cur_max
                else:
                    b = HEAT_MAX_EMA_BETA
                    self._heat_max_ema = (1 - b) * self._heat_max_ema + b * cur_max
                use_max = max(1e-6, float(self._heat_max_ema))
                norm = np.clip((c / use_max), 0.0, 1.0).astype(np.float32)
                g = float(HEAT_GAMMA)
                if abs(g - 1.0) > 1e-6:
                    norm = np.power(norm, g).astype(np.float32)
                colors = colormap_blue_yellow_red_wide(norm, YELLOW_START, YELLOW_END)
            else:
                z = (v_center_world[:, 2].astype(np.float32) / SCALE)
                zmin = float(self.z_min)
                zmax = float(self.z_max)
                if zmax <= zmin + 1e-6:
                    zmax = zmin + 1.0
                norm = (z - zmin) / (zmax - zmin)
                colors = colormap_blue_yellow_red_wide(norm, YELLOW_START, YELLOW_END)

            age_v = (now - v_time).astype(np.float32)
            age_v = np.clip(age_v, 0.0, float(self.persist_sec))
            alpha_v = np.exp(-age_v / max(1e-6, float(FADE_TAU))).astype(np.float32)
            if abs(ALPHA_POWER - 1.0) > 1e-6:
                alpha_v = np.power(alpha_v, float(ALPHA_POWER)).astype(np.float32)
            alpha_v = np.clip(alpha_v, float(ALPHA_FLOOR), 1.0).astype(np.float32)
            if colors.size:
                colors[:, 3] = alpha_v

            verts, faces, face_cols = build_cube_mesh_vectorized(
                v_center_world.astype(np.float32),
                voxel_size_scaled,
                colors.astype(np.float32) if colors.size else np.empty((0, 4), dtype=np.float32)
            )
            self.cube_mesh.setMeshData(vertexes=verts, faces=faces, faceColors=face_cols)

            self.last_pts_world = v_center_world.copy()

            # 马赛克模式下：点云拾取索引不更新
            self._screen_grid_valid = False
            self._points_proj_xy = None
            self._points_proj_z = None
            self._points_proj_valid = None

            # 关键：重建“马赛克+手动”的合并拾取索引
            self._rebuild_cube_voxel_index_combined(
                mosaic_centers_world=v_center_world.copy(),
                mosaic_ijk=v_ijk.copy(),
                voxel_size_scaled=voxel_size_scaled
            )

        # 手动方块 mesh 刷新（尺寸随 voxel_size_m）
        self._refresh_manual_mesh()

        self._update_measure_line()
        self._apply_render_visibility()
        self._refresh_selected_cube_mesh()

        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 0:
                fps_inst = 1.0 / dt
                if self._fps_ema is None:
                    self._fps_ema = fps_inst
                else:
                    a = 0.15
                    self._fps_ema = (1 - a) * self._fps_ema + a * fps_inst
        self._last_frame_time = now

        cubes_total = 0 if self.last_cubes_world is None else int(self.last_cubes_world.shape[0])
        self._update_status(now, fps=self._fps_ema, shown_n=shown_n, cubes_n=cubes_total)
        self.need_gl_update = False

    # ============================================================
    # 自适应 / 状态栏
    # ============================================================

    def _adaptive_update(self, now):
        if not self.adapt_enable:
            return
        if self._fps_ema is None:
            return

        fps = float(self._fps_ema)
        q_pressure = 0.0
        try:
            q_pressure = float(self._parsed_q.qsize()) / float(PARSED_QUEUE_MAXSIZE)
        except Exception:
            q_pressure = 0.0

        target = float(ADAPT_TARGET_FPS)
        if fps < target or q_pressure > 0.65:
            self.density_factor = max(ADAPT_MIN_DENSITY, float(self.density_factor) - ADAPT_DENSITY_STEP)
            self.persist_sec = max(ADAPT_PERSIST_MIN, float(self.persist_sec) - ADAPT_STEP_SEC)
        elif fps > target + 8.0 and q_pressure < 0.25:
            self.density_factor = min(ADAPT_MAX_DENSITY, float(self.density_factor) + ADAPT_DENSITY_STEP * 0.5)
            self.persist_sec = min(ADAPT_PERSIST_MAX, float(self.persist_sec) + ADAPT_STEP_SEC * 0.5)

    def _get_val(self, v):
        try:
            return int(v.value)
        except Exception:
            return 0

    def _update_status_only(self):
        now = time.time()
        shown = 0 if self.last_pts_world is None else int(self.last_pts_world.shape[0])
        cubes_n = 0 if self.last_cubes_world is None else int(self.last_cubes_world.shape[0])
        self._update_status(now, fps=self._fps_ema, shown_n=shown, cubes_n=cubes_n)

    def _update_status(self, now, fps, shown_n, cubes_n):
        if now - self._last_stat_ts >= 1.0:
            rx_drop = self._get_val(self.stats.rx_drop)
            parse_drop = self._get_val(self.stats.parse_drop)
            rx_cnt = self._get_val(self.stats.rx_cnt)
            parse_cnt = self._get_val(self.stats.parse_cnt)

            self._rx_drop_last = rx_drop
            self._parse_drop_last = parse_drop
            self._rx_cnt_last = rx_cnt
            self._parse_cnt_last = parse_cnt
            self._last_stat_ts = now

        mode_str = "方块" if self.render_mode == MODE_CUBE_MOSAIC else "点"
        grid_str = "开" if self.chk_grid_overlay.isChecked() else "关"
        fps_txt = "-" if fps is None else f"{fps:4.1f}"

        hover_str = ""
        if self.render_mode == MODE_CUBE_MOSAIC and self.hover_cube_valid and self.hover_cube_idx is not None:
            hover_str = f" | Hover:{int(self.hover_cube_idx)}({'手动' if self.hover_is_manual else '马赛克'})"

        sel_str = ""
        if self.selected_cube_valid:
            sel_str = f" | Sel:{'手动' if self.selected_is_manual else '马赛克'}"
            if self.selected_is_manual and self.selected_manual_idx >= 0:
                sel_str += f"(idx={self.selected_manual_idx})"

        meas_str = ""
        if self.measure_segments:
            meas_str = f" | 测距段:{len(self.measure_segments)} 累计:{self.measure_total / SCALE:.3f}m"

        man_str = f" | 手动方块:{len(self.manual_cubes_world)}"
        if self.active_manual_idx >= 0:
            man_str += f"(active={self.active_manual_idx})"

        self.status.setText(
            f"实时 | 模式:{mode_str} | 网格:{grid_str} | FPS:{fps_txt} | 点:{int(shown_n)} | 方块候选:{int(cubes_n)} | "
            f"密度:{self.density_factor:.2f}x | 拖尾:{self.persist_sec:.2f}s | "
            f"RX:{self._rx_cnt_last} drop:{self._rx_drop_last} | PARSE:{self._parse_cnt_last} drop:{self._parse_drop_last}"
            + hover_str + sel_str + meas_str + man_str
        )

    # ============================================================
    # 退出
    # ============================================================

    def closeEvent(self, event):
        try:
            self._hover_timer.stop()
            self.poll_timer.stop()
            self.timer.stop()
        except Exception:
            pass

        try:
            self.stop_udp_processes()
        except Exception:
            pass

        try:
            self._drain_queue_all()
        except Exception:
            pass

        try:
            for qx in (self._raw_q, self._parsed_q):
                try:
                    qx.close()
                    qx.join_thread()
                except Exception:
                    pass
        except Exception:
            pass

        event.accept()


# ============================================================
# 15) 主入口
# ============================================================

def main():
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    if GUI_CORES:
        set_process_affinity(GUI_CORES)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    mp.freeze_support()
    main()
