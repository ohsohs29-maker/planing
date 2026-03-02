#!/usr/bin/env python3
"""
5.5 실습(보조): 분류 지표 + 공간 성능 지도 + SHAP 요약

이 스크립트는 5.6과 동일한 입력(밴드 스택)과 의사 라벨 규칙을 사용해
Random Forest 모델을 학습/평가하고, 실무에서 자주 요구되는 산출물을 만든다.

산출물:
- confusion matrix / classification report (블록 기반 holdout)
- feature_importance.csv (Random Forest Gini 중요도)
- block_accuracy.tif (블록 단위 정확도 지도; test 샘플 기반)
- (선택) shap_importance.csv (평균 |SHAP| 기반 중요도)

실행 예시:
  python practice/chapter05/code/5-5-evaluation-and-interpretation.py \
    --stack practice/chapter02/data/sentinel2_stack_clipped.tif \
    --out practice/chapter05/data/output
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
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
    rng = np.random.default_rng(random_seed)
    valid_mask = (labels > 0) & np.isfinite(ndvi) & np.isfinite(ndwi)
    rows, cols = np.where(valid_mask)
    if rows.size == 0:
        raise ValueError("유효한(라벨>0) 샘플이 없습니다. 임계값을 완화하거나 입력을 확인하세요.")

    n = min(n_samples, rows.size)
    idx = rng.choice(rows.size, size=n, replace=False)
    rr = rows[idx]
    cc = cols[idx]

    pixel_bands = stack[:, rr, cc].T
    pixel_ndvi = ndvi[rr, cc].reshape(-1, 1)
    pixel_ndwi = ndwi[rr, cc].reshape(-1, 1)
    X = np.hstack([pixel_bands, pixel_ndvi, pixel_ndwi]).astype(np.float32)
    y = labels[rr, cc].astype(np.uint8)
    rc = np.vstack([rr, cc]).T.astype(np.int32)

    ok = np.isfinite(X).all(axis=1)
    return X[ok], y[ok], rc[ok]


def block_ids(rc: np.ndarray, width: int, block_size: int) -> np.ndarray:
    row = rc[:, 0]
    col = rc[:, 1]
    block_r = row // block_size
    block_c = col // block_size
    n_blocks_c = int(np.ceil(width / block_size))
    return (block_r * n_blocks_c + block_c).astype(np.int64)


def block_split(
    rc: np.ndarray,
    width: int,
    block_size: int,
    test_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    b = block_ids(rc=rc, width=width, block_size=block_size)
    unique_blocks = np.unique(b)
    rng.shuffle(unique_blocks)
    n_test_blocks = max(1, int(round(unique_blocks.size * test_fraction)))
    test_blocks = set(unique_blocks[:n_test_blocks].tolist())
    test_mask = np.array([bid in test_blocks for bid in b], dtype=bool)
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def write_singleband_geotiff(
    out_path: Path, values: np.ndarray, src: rasterio.io.DatasetReader, dtype: str, nodata: float | int | None = None
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = src.meta.copy()
    meta.update(count=1, dtype=dtype, compress="deflate")
    if nodata is not None:
        meta.update(nodata=nodata)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(values.astype(dtype), 1)


def feature_names(src: rasterio.io.DatasetReader) -> list[str]:
    names: list[str] = []
    if src.descriptions:
        for i, d in enumerate(src.descriptions, start=1):
            nm = (d or f"band{i}").strip()
            names.append(nm if nm else f"band{i}")
    else:
        names = [f"band{i}" for i in range(1, src.count + 1)]
    names.extend(["NDVI", "NDWI"])
    return names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RF 평가/해석 + 공간 성능 지도 + SHAP(선택)")
    p.add_argument("--stack", type=str, required=True, help="Sentinel-2 밴드 스택 GeoTIFF 경로")
    p.add_argument("--out", type=str, default="practice/chapter05/data/output", help="출력 디렉토리")

    p.add_argument("--red", type=int, default=0, help="Red 밴드 인덱스(1-based). 0이면 자동 추정")
    p.add_argument("--green", type=int, default=0, help="Green 밴드 인덱스(1-based). 0이면 자동 추정")
    p.add_argument("--nir", type=int, default=0, help="NIR 밴드 인덱스(1-based). 0이면 자동 추정")

    p.add_argument("--ndvi-veg", type=float, default=0.40)
    p.add_argument("--ndwi-water", type=float, default=0.10)
    p.add_argument("--ndvi-water-max", type=float, default=0.20)

    p.add_argument("--samples", type=int, default=80_000)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--test-frac", type=float, default=0.2)

    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=25)
    p.add_argument("--min-samples-leaf", type=int, default=20)
    p.add_argument("--random-seed", type=int, default=42)

    p.add_argument("--shap", action="store_true", help="SHAP 요약 중요도 계산(샘플링)")
    p.add_argument("--shap-samples", type=int, default=2000, help="SHAP 계산 샘플 수")
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
        names = feature_names(src)

        stack = src.read()
        height, width = src.height, src.width

        red = stack[bands.red - 1].astype(np.float32)
        green = stack[bands.green - 1].astype(np.float32)
        nir = stack[bands.nir - 1].astype(np.float32)
        ndvi, ndwi = compute_indices(red=red, green=green, nir=nir)

        labels = make_pseudolabels(
            ndvi=ndvi, ndwi=ndwi, ndvi_veg=args.ndvi_veg, ndwi_water=args.ndwi_water, ndvi_water_max=args.ndvi_water_max
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
            stack=stack, ndvi=ndvi, ndwi=ndwi, labels=labels, n_samples=args.samples, random_seed=args.random_seed
        )
        train_idx, test_idx = block_split(
            rc=rc, width=width, block_size=args.block_size, test_fraction=args.test_frac, random_seed=args.random_seed
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
        eval_labels = [1, 2]
        print("\n[평가] 블록 기반 holdout")
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

        # 변수 중요도(학습된 모델 기준)
        importances = rf.feature_importances_
        df_imp = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
        out_imp = out_dir / "feature_importance.csv"
        df_imp.to_csv(out_imp, index=False)
        print(f"[저장] 변수 중요도: {out_imp}")

        # 블록 단위 정확도 지도(test 샘플 기반)
        rc_test = rc[test_idx]
        block_test = block_ids(rc=rc_test, width=width, block_size=args.block_size)
        correct = (y_pred == y_test).astype(np.float32)

        block_acc: dict[int, float] = {}
        for bid in np.unique(block_test):
            m = block_test == bid
            block_acc[int(bid)] = float(correct[m].mean())

        n_blocks_r = int(np.ceil(height / args.block_size))
        n_blocks_c = int(np.ceil(width / args.block_size))
        acc_raster = np.full((height, width), np.nan, dtype=np.float32)
        for br in range(n_blocks_r):
            r0 = br * args.block_size
            r1 = min(height, (br + 1) * args.block_size)
            for bc in range(n_blocks_c):
                c0 = bc * args.block_size
                c1 = min(width, (bc + 1) * args.block_size)
                bid = br * n_blocks_c + bc
                if bid in block_acc:
                    acc_raster[r0:r1, c0:c1] = block_acc[bid]

        out_acc = out_dir / "block_accuracy.tif"
        write_singleband_geotiff(out_acc, acc_raster, src, dtype="float32", nodata=np.nan)
        print(f"[저장] 블록 정확도 지도: {out_acc}")

        if args.shap:
            try:
                import shap  # type: ignore
            except Exception as e:  # noqa: BLE001
                raise RuntimeError("shap이 설치되어 있지 않습니다. requirements.txt의 optional을 설치하세요.") from e

            rng = np.random.default_rng(args.random_seed)
            n = min(args.shap_samples, X_test.shape[0])
            idx = rng.choice(X_test.shape[0], size=n, replace=False)
            X_shap = X_test[idx]

            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_shap)

            # sklearn RF classifier의 출력 형태(list 또는 ndarray)를 흡수해 (n_samples, n_features)로 정규화
            if isinstance(shap_values, list):
                shap_abs = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)  # (n, p)
            else:
                sv = np.asarray(shap_values)
                if sv.ndim == 2:
                    shap_abs = np.abs(sv)  # (n, p)
                elif sv.ndim == 3:
                    shap_abs = np.mean(np.abs(sv), axis=2)  # (n, p)  (classes 축 평균)
                else:
                    raise RuntimeError(f"Unexpected SHAP shape: {sv.shape}")

            mean_abs = shap_abs.mean(axis=0)
            df_shap = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs}).sort_values(
                "mean_abs_shap", ascending=False
            )
            out_shap = out_dir / "shap_importance.csv"
            df_shap.to_csv(out_shap, index=False)
            print(f"[저장] SHAP 요약 중요도: {out_shap}")


if __name__ == "__main__":
    main()
