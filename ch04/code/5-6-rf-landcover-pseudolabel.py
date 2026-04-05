#!/usr/bin/env python3
"""
5.6 실습: Random Forest를 활용한 토지피복 분류 (의사 라벨 기반)

이 스크립트는 Sentinel-2 밴드 스택(GeoTIFF)에서 NDVI/NDWI를 계산하고,
간단한 규칙으로 의사 라벨(식생/수계/기타)을 만든 뒤 Random Forest로 분류한다.

중요:
- 의사 라벨은 교육용이며, 결과를 "실제 토지피복 지도"로 해석하면 안 된다.
- 공간 누수 방지를 위해 블록 단위로 train/test를 분할한다(간단한 block split).

실행 예시:
  python practice/chapter05/code/5-6-rf-landcover-pseudolabel.py \
    --stack practice/chapter02/data/sentinel2_stack_clipped.tif \
    --out practice/chapter05/data/output
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


@dataclass(frozen=True)
class BandIndices:
    red: int
    green: int
    nir: int


def _normalize_desc(text: str | None) -> str:
    return (text or "").strip().upper()


def infer_band_indices(src: rasterio.io.DatasetReader) -> BandIndices:
    """
    밴드 description에서 Sentinel-2 밴드 인덱스를 추정한다.

    - rasterio는 band index가 1부터 시작한다.
    - description이 없으면 흔한 스택 순서(B02,B03,B04,B08,...)를 가정하되,
      실제 파일마다 다를 수 있으므로 --red/--green/--nir로 재지정 가능하다.
    """
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

    # Fallback: 흔한 10m 밴드 스택 순서 가정
    # [1] B02, [2] B03, [3] B04, [4] B08
    if src.count >= 4:
        return BandIndices(red=3, green=2, nir=4)

    raise ValueError(
        "밴드 개수가 부족합니다. 최소 4개 밴드(B02,B03,B04,B08)가 필요합니다."
    )


def safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    denom_safe = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    return numer / denom_safe


def compute_indices(red: np.ndarray, green: np.ndarray, nir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ndvi = safe_divide(nir - red, nir + red)
    ndwi = safe_divide(green - nir, green + nir)
    return ndvi, ndwi


def make_pseudolabels(
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    ndvi_veg: float,
    ndwi_water: float,
    ndvi_water_max: float,
) -> np.ndarray:
    """
    0: 기타
    1: 식생(vegetation)
    2: 수계(water)
    """
    labels = np.zeros(ndvi.shape, dtype=np.uint8)
    vegetation = ndvi >= ndvi_veg
    water = (ndwi >= ndwi_water) & (ndvi <= ndvi_water_max)

    labels[vegetation] = 1
    labels[water] = 2
    labels[np.isnan(ndvi) | np.isnan(ndwi)] = 0
    return labels


def sample_pixels(
    stack: np.ndarray,
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    labels: np.ndarray,
    n_samples: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (bands, H, W) 스택을 픽셀 샘플로 변환한다.
    반환:
      X: (N, bands+2)  - 밴드 + (NDVI, NDWI)
      y: (N,)
      rc: (N, 2)       - (row, col) 좌표
    """
    rng = np.random.default_rng(random_seed)

    valid_mask = (labels > 0) & np.isfinite(ndvi) & np.isfinite(ndwi)
    rows, cols = np.where(valid_mask)
    if rows.size == 0:
        raise ValueError("유효한(라벨>0) 샘플이 없습니다. 임계값을 완화하거나 입력을 확인하세요.")

    n = min(n_samples, rows.size)
    idx = rng.choice(rows.size, size=n, replace=False)

    rr = rows[idx]
    cc = cols[idx]

    pixel_bands = stack[:, rr, cc].T  # (N, bands)
    pixel_ndvi = ndvi[rr, cc].reshape(-1, 1)
    pixel_ndwi = ndwi[rr, cc].reshape(-1, 1)
    X = np.hstack([pixel_bands, pixel_ndvi, pixel_ndwi]).astype(np.float32)
    y = labels[rr, cc].astype(np.uint8)
    rc = np.vstack([rr, cc]).T.astype(np.int32)

    # NaN 제거(밴드 결측 등)
    ok = np.isfinite(X).all(axis=1)
    return X[ok], y[ok], rc[ok]


def block_split(
    rc: np.ndarray,
    height: int,
    width: int,
    block_size: int,
    test_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    픽셀 좌표(rc)를 block_size 격자 블록으로 묶어, 블록 단위로 train/test 분리한다.
    """
    rng = np.random.default_rng(random_seed)

    row = rc[:, 0]
    col = rc[:, 1]

    block_r = row // block_size
    block_c = col // block_size
    n_blocks_c = int(np.ceil(width / block_size))
    block_id = block_r * n_blocks_c + block_c

    unique_blocks = np.unique(block_id)
    rng.shuffle(unique_blocks)

    n_test_blocks = max(1, int(round(unique_blocks.size * test_fraction)))
    test_blocks = set(unique_blocks[:n_test_blocks].tolist())

    test_mask = np.array([bid in test_blocks for bid in block_id], dtype=bool)
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, test_idx


def write_prediction_geotiff(
    out_path: Path,
    preds: np.ndarray,
    src: rasterio.io.DatasetReader,
    dtype: str = "uint8",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = src.meta.copy()
    meta.update(count=1, dtype=dtype, compress="deflate")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(preds.astype(dtype), 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RF 토지피복(의사 라벨) 분류 실습")
    p.add_argument("--stack", type=str, required=True, help="Sentinel-2 밴드 스택 GeoTIFF 경로")
    p.add_argument("--out", type=str, default="practice/chapter05/data/output", help="출력 디렉토리")

    p.add_argument("--red", type=int, default=0, help="Red 밴드 인덱스(1-based). 0이면 자동 추정")
    p.add_argument("--green", type=int, default=0, help="Green 밴드 인덱스(1-based). 0이면 자동 추정")
    p.add_argument("--nir", type=int, default=0, help="NIR 밴드 인덱스(1-based). 0이면 자동 추정")

    p.add_argument("--ndvi-veg", type=float, default=0.40, help="식생 의사 라벨 NDVI 임계값")
    p.add_argument(
        "--ndwi-water",
        type=float,
        default=0.10,
        help="수계 의사 라벨 NDWI 임계값(샘플 스택 기준 기본값)",
    )
    p.add_argument(
        "--ndvi-water-max",
        type=float,
        default=0.20,
        help="수계 의사 라벨 NDVI 상한(샘플 스택 기준 기본값)",
    )

    p.add_argument("--samples", type=int, default=200_000, help="학습에 사용할 픽셀 샘플 수")
    p.add_argument("--block-size", type=int, default=64, help="블록 기반 분할 블록 크기(픽셀)")
    p.add_argument("--test-frac", type=float, default=0.2, help="테스트 블록 비율(0~1)")

    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=25)
    p.add_argument("--min-samples-leaf", type=int, default=20)
    p.add_argument("--random-seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    stack_path = Path(args.stack)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(stack_path) as src:
        if args.red and args.green and args.nir:
            bands = BandIndices(red=args.red, green=args.green, nir=args.nir)
        else:
            bands = infer_band_indices(src)

        stack = src.read()  # (bands, H, W)
        height, width = src.height, src.width

        red = stack[bands.red - 1].astype(np.float32)
        green = stack[bands.green - 1].astype(np.float32)
        nir = stack[bands.nir - 1].astype(np.float32)

        ndvi, ndwi = compute_indices(red=red, green=green, nir=nir)
        labels = make_pseudolabels(
            ndvi=ndvi,
            ndwi=ndwi,
            ndvi_veg=args.ndvi_veg,
            ndwi_water=args.ndwi_water,
            ndvi_water_max=args.ndvi_water_max,
        )

        unique, counts = np.unique(labels, return_counts=True)
        label_counts = {int(k): int(v) for k, v in zip(unique, counts)}
        print("\n[의사 라벨] 픽셀 분포(전체)")
        print(
            f"- 기타(0): {label_counts.get(0, 0):,}\n"
            f"- 식생(1): {label_counts.get(1, 0):,}\n"
            f"- 수계(2): {label_counts.get(2, 0):,}"
        )

        X, y, rc = sample_pixels(
            stack=stack,
            ndvi=ndvi,
            ndwi=ndwi,
            labels=labels,
            n_samples=args.samples,
            random_seed=args.random_seed,
        )

        train_idx, test_idx = block_split(
            rc=rc,
            height=height,
            width=width,
            block_size=args.block_size,
            test_fraction=args.test_frac,
            random_seed=args.random_seed,
        )

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=-1,
            random_state=args.random_seed,
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        print("\n[평가] 블록 기반 holdout")
        eval_labels = [1, 2]
        print(confusion_matrix(y_test, y_pred, labels=eval_labels))
        print(
            classification_report(
                y_test,
                y_pred,
                labels=eval_labels,
                target_names=["식생(1)", "수계(2)"],
                digits=4,
                zero_division=0,
            )
        )

        # 전체 픽셀 예측(라벨이 0인 영역도 예측)
        # 메모리 고려: (H*W, bands+2) 구성 시 매우 커질 수 있으므로,
        # 실제 프로젝트에서는 타일 단위 처리 또는 더 작은 영역을 권장한다.
        flat = stack.reshape(stack.shape[0], -1).T.astype(np.float32)
        flat_ndvi = ndvi.reshape(-1, 1).astype(np.float32)
        flat_ndwi = ndwi.reshape(-1, 1).astype(np.float32)
        X_all = np.hstack([flat, flat_ndvi, flat_ndwi])
        ok = np.isfinite(X_all).all(axis=1)
        preds = np.zeros((height * width,), dtype=np.uint8)
        preds[ok] = rf.predict(X_all[ok]).astype(np.uint8)
        preds = preds.reshape(height, width)

        out_pred = out_dir / "rf_pseudolabel_prediction.tif"
        out_label = out_dir / "pseudolabels.tif"
        write_prediction_geotiff(out_pred, preds, src)
        write_prediction_geotiff(out_label, labels, src)

        print(f"\n[저장] 예측 래스터: {out_pred}")
        print(f"[저장] 의사 라벨: {out_label}")


if __name__ == "__main__":
    main()
