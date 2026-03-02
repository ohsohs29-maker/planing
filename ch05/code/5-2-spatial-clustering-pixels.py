#!/usr/bin/env python3
"""
5.2 실습(보조): Sentinel-2 픽셀 피처 공간 클러스터링 (K-means/DBSCAN)

이 스크립트는 Sentinel-2 밴드 스택(GeoTIFF)에서 NDVI/NDWI를 계산하고,
밴드 + (NDVI, NDWI) 피처로 픽셀을 샘플링해 비지도 군집화를 수행한다.

중요:
- 클러스터 ID는 "정답 라벨"이 아니다. 탐색/샘플링/타일링 전략을 돕는 용도이다.
- DBSCAN은 전체 픽셀에 대한 예측 단계가 없으므로, 샘플 결과 요약만 출력한다.

실행 예시:
  python practice/chapter05/code/5-2-spatial-clustering-pixels.py \
    --stack practice/chapter02/data/sentinel2_stack_clipped.tif \
    --out practice/chapter05/data/output
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BandIndices:
    red: int
    green: int
    nir: int


def _normalize_desc(text: str | None) -> str:
    return (text or "").strip().upper()


def infer_band_indices(src: rasterio.io.DatasetReader) -> BandIndices:
    descriptions = list(src.descriptions or [])

    def find_band(key: str) -> int | None:
        key_u = key.upper()
        for i, d in enumerate(descriptions, start=1):
            if key_u in _normalize_desc(d):
                return i
        return None

    red = find_band("B04")
    green = find_band("B03")
    nir = find_band("B08")

    if red and green and nir:
        return BandIndices(red=red, green=green, nir=nir)

    if src.count >= 4:
        return BandIndices(red=3, green=2, nir=4)

    raise ValueError("밴드 개수가 부족합니다. 최소 4개 밴드(B02,B03,B04,B08)가 필요합니다.")


def safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    denom_safe = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    return numer / denom_safe


def compute_indices(
    red: np.ndarray, green: np.ndarray, nir: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ndvi = safe_divide(nir - red, nir + red)
    ndwi = safe_divide(green - nir, green + nir)
    return ndvi, ndwi


def sample_pixels(
    stack: np.ndarray,
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    n_samples: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    valid_mask = np.isfinite(ndvi) & np.isfinite(ndwi)
    rows, cols = np.where(valid_mask)
    if rows.size == 0:
        raise ValueError("유효한 샘플이 없습니다. 입력 래스터를 확인하세요.")

    n = min(n_samples, rows.size)
    idx = rng.choice(rows.size, size=n, replace=False)
    rr = rows[idx]
    cc = cols[idx]

    pixel_bands = stack[:, rr, cc].T  # (N, bands)
    pixel_ndvi = ndvi[rr, cc].reshape(-1, 1)
    pixel_ndwi = ndwi[rr, cc].reshape(-1, 1)
    X = np.hstack([pixel_bands, pixel_ndvi, pixel_ndwi]).astype(np.float32)
    rc = np.vstack([rr, cc]).T.astype(np.int32)

    ok = np.isfinite(X).all(axis=1)
    return X[ok], rc[ok]


def write_singleband_geotiff(
    out_path: Path, values: np.ndarray, src: rasterio.io.DatasetReader, dtype: str
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = src.meta.copy()
    meta.update(count=1, dtype=dtype, compress="deflate")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(values.astype(dtype), 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sentinel-2 픽셀 클러스터링(K-means/DBSCAN)")
    p.add_argument("--stack", type=str, required=True, help="Sentinel-2 밴드 스택 GeoTIFF 경로")
    p.add_argument("--out", type=str, default="practice/chapter05/data/output", help="출력 디렉토리")

    p.add_argument("--red", type=int, default=0, help="Red 밴드 인덱스(1-based). 0이면 자동 추정")
    p.add_argument("--green", type=int, default=0, help="Green 밴드 인덱스(1-based). 0이면 자동 추정")
    p.add_argument("--nir", type=int, default=0, help="NIR 밴드 인덱스(1-based). 0이면 자동 추정")

    p.add_argument("--samples", type=int, default=30_000, help="클러스터링에 사용할 픽셀 샘플 수")
    p.add_argument("--random-seed", type=int, default=42)

    p.add_argument("--k", type=int, default=6, help="K-means 군집 수")
    p.add_argument("--dbscan-samples", type=int, default=10_000, help="DBSCAN에 사용할 샘플 수(속도 고려)")
    p.add_argument("--dbscan-eps", type=float, default=0.8, help="DBSCAN eps(표준화 후 거리)")
    p.add_argument("--dbscan-min-samples", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stack_path = Path(args.stack)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(stack_path) as src:
        bands = (
            BandIndices(red=args.red, green=args.green, nir=args.nir)
            if (args.red and args.green and args.nir)
            else infer_band_indices(src)
        )

        stack = src.read()  # (bands, H, W)
        height, width = src.height, src.width

        red = stack[bands.red - 1].astype(np.float32)
        green = stack[bands.green - 1].astype(np.float32)
        nir = stack[bands.nir - 1].astype(np.float32)
        ndvi, ndwi = compute_indices(red=red, green=green, nir=nir)

        X, rc = sample_pixels(stack=stack, ndvi=ndvi, ndwi=ndwi, n_samples=args.samples, random_seed=args.random_seed)
        print("\n[샘플] 클러스터링 입력")
        print(f"- X shape: {X.shape} (bands+NDVI+NDWI)")

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        km = KMeans(n_clusters=args.k, n_init=10, random_state=args.random_seed)
        km_labels = km.fit_predict(Xs)
        unique, counts = np.unique(km_labels, return_counts=True)
        km_counts = {int(k): int(v) for k, v in zip(unique, counts)}
        print("\n[K-means] 샘플 군집 분포")
        print(km_counts)

        # 전체 픽셀에 대한 K-means 군집 맵 생성(표준화 적용)
        flat = stack.reshape(stack.shape[0], -1).T.astype(np.float32)
        flat_ndvi = ndvi.reshape(-1, 1).astype(np.float32)
        flat_ndwi = ndwi.reshape(-1, 1).astype(np.float32)
        X_all = np.hstack([flat, flat_ndvi, flat_ndwi])
        ok = np.isfinite(X_all).all(axis=1)
        clusters = np.full((height * width,), 255, dtype=np.uint8)  # 255: NoData
        clusters[ok] = km.predict(scaler.transform(X_all[ok])).astype(np.uint8)
        clusters = clusters.reshape(height, width)

        out_km = out_dir / "kmeans_clusters.tif"
        write_singleband_geotiff(out_km, clusters, src, dtype="uint8")
        print(f"\n[저장] K-means 군집 래스터: {out_km}")

        # DBSCAN은 전체 예측이 없으므로, 샘플에 대해서만 요약 출력
        n_db = min(args.dbscan_samples, Xs.shape[0])
        Xs_db = Xs[:n_db]
        db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1)
        db_labels = db.fit_predict(Xs_db)
        unique, counts = np.unique(db_labels, return_counts=True)
        db_counts = {int(k): int(v) for k, v in zip(unique, counts)}
        noise = db_counts.get(-1, 0)
        print("\n[DBSCAN] 샘플 군집 분포(-1은 노이즈)")
        print(db_counts)
        print(f"[DBSCAN] 노이즈 비율: {noise}/{n_db} ({(noise / max(1, n_db)):.2%})")


if __name__ == "__main__":
    main()

