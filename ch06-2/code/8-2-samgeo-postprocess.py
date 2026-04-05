#!/usr/bin/env python3
"""
8.2 실습: SAM 마스크 후처리 및 벡터 변환

래스터 마스크를 정제하고 벡터(Shapefile/GeoJSON)로 변환한다.
후처리 단계:
1. 노이즈 제거 (최소 면적 필터링)
2. 홀 채우기 (모폴로지 closing)
3. 경계 스무딩 (Douglas-Peucker)

실행 예시:
  python practice/chapter08/code/8-2-samgeo-postprocess.py \
    --mask practice/chapter08/data/output/sam_mask_box.tif \
    --min-area 100 \
    --tolerance 1.0 \
    --out practice/chapter08/data/output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM 마스크 후처리 및 벡터화")
    p.add_argument("--mask", type=str, required=True, help="입력 마스크 GeoTIFF")
    p.add_argument("--min-area", type=int, default=100, help="최소 픽셀 수 (노이즈 제거)")
    p.add_argument("--tolerance", type=float, default=1.0, help="폴리곤 단순화 tolerance")
    p.add_argument("--fill-holes", action="store_true", help="홀 채우기 적용")
    p.add_argument("--out", type=str, default="practice/chapter08/data/output")
    return p.parse_args()


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """최소 크기 미만 객체 제거"""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask

    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep = np.where(np.array(sizes) >= min_size)[0] + 1
    filtered = np.isin(labeled, keep).astype(np.uint8)
    return filtered


def fill_holes_morphology(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """모폴로지 closing으로 홀 채우기"""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed


def vectorize_mask(
    mask: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.CRS,
    tolerance: float,
) -> "geopandas.GeoDataFrame":
    """래스터 마스크를 벡터 GeoDataFrame으로 변환"""
    import geopandas as gpd

    # shapes 생성
    mask_binary = (mask > 0).astype(np.uint8)
    results = list(shapes(mask_binary, mask=(mask_binary > 0), transform=transform))

    if not results:
        print("경고: 벡터화할 마스크가 없습니다.")
        return gpd.GeoDataFrame({"geometry": []}, crs=crs)

    geometries = []
    for geom, value in results:
        poly = shape(geom)
        if tolerance > 0:
            poly = poly.simplify(tolerance, preserve_topology=True)
        geometries.append(poly)

    gdf = gpd.GeoDataFrame({"geometry": geometries}, crs=crs)

    # 면적 계산
    gdf["area_crs"] = gdf.geometry.area

    return gdf


def main() -> None:
    args = parse_args()
    mask_path = Path(args.mask)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[입력] 마스크: {mask_path}")

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

    print(f"- 크기: {mask.shape}")
    print(f"- CRS: {crs}")

    original_pixels = int((mask > 0).sum())
    print(f"- 원본 마스크 픽셀: {original_pixels:,}")

    # 1. 노이즈 제거
    print(f"\n[후처리] 노이즈 제거 (min_area={args.min_area})")
    mask_clean = remove_small_objects(mask, args.min_area)
    clean_pixels = int((mask_clean > 0).sum())
    removed = original_pixels - clean_pixels
    print(f"- 제거된 픽셀: {removed:,} ({100 * removed / max(1, original_pixels):.1f}%)")

    # 2. 홀 채우기 (선택적)
    if args.fill_holes:
        print("\n[후처리] 홀 채우기 (모폴로지 closing)")
        mask_clean = fill_holes_morphology(mask_clean)
        filled_pixels = int((mask_clean > 0).sum())
        added = filled_pixels - clean_pixels
        print(f"- 채워진 픽셀: {added:,}")

    # 정제된 마스크 저장
    out_mask = out_dir / "mask_clean.tif"
    profile.update(count=1, dtype="uint8", compress="deflate")
    with rasterio.open(out_mask, "w", **profile) as dst:
        dst.write(mask_clean, 1)
    print(f"\n[저장] 정제된 마스크: {out_mask}")

    # 3. 벡터화
    print(f"\n[벡터화] tolerance={args.tolerance}")
    try:
        import geopandas as gpd

        gdf = vectorize_mask(mask_clean, transform, crs, args.tolerance)

        if len(gdf) > 0:
            out_shp = out_dir / "mask_vector.shp"
            out_geojson = out_dir / "mask_vector.geojson"

            gdf.to_file(out_shp)
            gdf.to_file(out_geojson, driver="GeoJSON")

            print(f"- 폴리곤 수: {len(gdf)}")
            print(f"- 총 면적: {gdf['area_crs'].sum():,.2f} (CRS 단위²)")
            print(f"- Shapefile: {out_shp}")
            print(f"- GeoJSON: {out_geojson}")
        else:
            print("- 벡터화할 객체가 없습니다.")

    except ImportError:
        print("geopandas가 설치되지 않아 벡터화를 건너뜁니다.")
        print("설치: pip install geopandas")

    print("\n[완료]")


if __name__ == "__main__":
    main()
