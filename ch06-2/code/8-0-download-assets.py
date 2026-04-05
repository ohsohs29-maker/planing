#!/usr/bin/env python3
"""
8장 실습 자산 다운로드

다운로드 대상:
- 샘플 GeoTIFF(USGS 항공영상 예시, OSGeo 샘플): o41078a5.tif
- SAM 체크포인트(vit_b): sam_vit_b_01ec64.pth

실행 예시:
  python practice/chapter08/code/8-0-download-assets.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm
import ssl

import certifi

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # practice/chapter08
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

SAMPLE_URL = "https://download.osgeo.org/geotiff/samples/usgs/o41078a5.tif"
SAMPLE_NAME = "o41078a5.tif"

SAM_VIT_B_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_VIT_B_NAME = "sam_vit_b_01ec64.pth"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx) as resp:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or "0")
        with tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar:
            with out_path.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))


def main() -> None:
    sample_path = DATA_DIR / SAMPLE_NAME
    ckpt_path = MODEL_DIR / SAM_VIT_B_NAME

    print("[다운로드] 샘플 GeoTIFF")
    print(f"- URL: {SAMPLE_URL}")
    download(SAMPLE_URL, sample_path)
    print(f"- 저장: {sample_path} ({sample_path.stat().st_size:,} bytes)")
    print(f"- SHA256: {sha256(sample_path)}")

    print("\n[다운로드] SAM 체크포인트 (vit_b)")
    print(f"- URL: {SAM_VIT_B_URL}")
    download(SAM_VIT_B_URL, ckpt_path)
    print(f"- 저장: {ckpt_path} ({ckpt_path.stat().st_size:,} bytes)")
    print(f"- SHA256: {sha256(ckpt_path)}")

    print("\n[완료] 다음 단계")
    print(
        "  python practice/chapter08/code/8-5-sam-geospatial-segmentation.py "
        f"--image {sample_path} --checkpoint {ckpt_path}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
