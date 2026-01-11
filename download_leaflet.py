# -*- coding: utf-8 -*-
"""
一键下载 Leaflet 并解压到 web/leaflet/

用法（任选其一）：
  1) 直接双击运行（Windows）
  2) 命令行运行：
     python download_leaflet.py
     python download_leaflet.py --version 1.9.4 --out web/leaflet
"""

import argparse
import io
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


DEFAULT_VERSION = "1.9.4"


def http_get_bytes(url: str, timeout: int = 30) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LeafletDownloader/1.0"
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def try_download(urls):
    last_err = None
    for u in urls:
        try:
            data = http_get_bytes(u, timeout=40)
            if data and len(data) > 1024:
                return u, data
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
        except Exception as e:
            last_err = e
    raise RuntimeError(f"下载失败，已尝试所有源。最后错误：{last_err}")


def ensure_empty_dir(p: Path):
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)


def extract_zip_to_temp(data: bytes, tmp_dir: Path):
    zf = zipfile.ZipFile(io.BytesIO(data))
    zf.extractall(tmp_dir)


def extract_tgz_to_temp(data: bytes, tmp_dir: Path):
    tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
    tf.extractall(tmp_dir)


def find_leaflet_dist(root: Path) -> Path:
    """
    在解压目录里找 dist 目录（里面应有 leaflet.js / leaflet.css / images）。
    常见结构：
      - leaflet/ (release zip)
      - dist/   (某些压缩包)
      - package/dist/ (npm tgz)
    """
    candidates = []

    # 直接找包含 leaflet.js 和 leaflet.css 的 dist
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == "dist":
            js = p / "leaflet.js"
            css = p / "leaflet.css"
            if js.exists() and css.exists():
                candidates.append(p)

    if candidates:
        # 取最短路径的那个
        candidates.sort(key=lambda x: len(str(x)))
        return candidates[0]

    # 兜底：如果没找到 dist，就找 leaflet.js 的上级目录
    for js in root.rglob("leaflet.js"):
        parent = js.parent
        if (parent / "leaflet.css").exists():
            return parent

    raise RuntimeError("未找到 Leaflet 的 dist 目录（缺少 leaflet.js / leaflet.css）。")


def copy_dist_to_out(dist_dir: Path, out_dir: Path):
    ensure_empty_dir(out_dir)

    # 复制 leaflet.js / css / 其它可能文件（src map 等）
    for item in dist_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, out_dir / item.name)

    # images 必须在 leaflet.css 同目录下（Leaflet 官方说明）：
    img_src = dist_dir / "images"
    if img_src.exists() and img_src.is_dir():
        shutil.copytree(img_src, out_dir / "images", dirs_exist_ok=True)

    # 简单校验
    if not (out_dir / "leaflet.js").exists():
        raise RuntimeError("复制后缺少 leaflet.js")
    if not (out_dir / "leaflet.css").exists():
        raise RuntimeError("复制后缺少 leaflet.css")
    if not (out_dir / "images").exists():
        # images 有时也可能不需要（如果你不用默认 marker），但通常建议带上
        print("[WARN] 未发现 images/，如果你要用默认 Marker 图标可能会缺失。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=DEFAULT_VERSION, help="Leaflet 版本号，例如 1.9.4")
    ap.add_argument("--out", default="web/leaflet", help="输出目录，例如 web/leaflet")
    args = ap.parse_args()

    version = str(args.version).strip()
    out_dir = Path(args.out).resolve()

    # 下载源（按顺序尝试）
    # 1) GitHub release zip（通常是 leaflet.zip）
    github_zip = f"https://github.com/Leaflet/Leaflet/releases/download/v{version}/leaflet.zip"
    # 2) npm registry tgz（解压后在 package/dist）
    npm_tgz = f"https://registry.npmjs.org/leaflet/-/leaflet-{version}.tgz"

    urls = [github_zip, npm_tgz]

    print(f"[INFO] Leaflet version: {version}")
    print(f"[INFO] Target dir: {out_dir}")
    print("[INFO] Try downloading from:")
    for u in urls:
        print("       -", u)

    src_url, blob = try_download(urls)
    print(f"[OK] Downloaded from: {src_url} ({len(blob)/1024/1024:.2f} MB)")

    with tempfile.TemporaryDirectory(prefix="leaflet_dl_") as td:
        tmp = Path(td)

        # 判断格式并解压
        if src_url.endswith(".zip") or zipfile.is_zipfile(io.BytesIO(blob)):
            print("[INFO] Detected ZIP, extracting...")
            extract_zip_to_temp(blob, tmp)
        else:
            print("[INFO] Detected TGZ, extracting...")
            extract_tgz_to_temp(blob, tmp)

        dist_dir = find_leaflet_dist(tmp)
        print(f"[OK] Found dist: {dist_dir}")

        copy_dist_to_out(dist_dir, out_dir)
        print("[OK] Done. Leaflet files placed at:")
        print("     ", out_dir)
        print("      - leaflet.js")
        print("      - leaflet.css")
        print("      - images/")

    # Windows 双击时给一个停顿（可选）
    if os.name == "nt" and sys.stdout.isatty():
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", e)
        sys.exit(1)
