#!/usr/bin/env python3
"""
12장 실습(12.6): 위성영상 기반 산불 피해지역 자동 탐지 파이프라인
GeoAI: 지리공간 인공지능의 이론과 실제

이 스크립트는 다음을 수행합니다:
1) STAC API(Planetary Computer)로 Sentinel-2 L2A 전/후 영상 검색
2) NBR / dNBR 계산
3) dNBR 임계값 기반 피해 등급(severity class) 분류
4) 고피해(high severity) 영역 벡터화 및 면적 요약
5) GeoTIFF/GeoJSON/PNG/JSON 산출물 저장
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
import rioxarray  # noqa: F401  # rioxarray accessor 등록
from pystac_client import Client
from rasterio.features import shapes
from shapely.geometry import shape
import stackstac


@dataclass(frozen=True)
class AnalysisConfig:
    name: str
    bbox_lonlat: list[float]  # [minx, miny, maxx, maxy]
    epsg: int
    resolution_m: int
    max_cloud_cover: float
    pre_range: str
    post_range: str
    vectorize_min_severity_class: int


def open_catalog() -> Client:
    return Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )


def search_sentinel2_items(
    catalog: Client,
    bbox_lonlat: list[float],
    datetime_range: str,
    max_cloud_cover: float,
):
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox_lonlat,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    return list(search.item_collection())


def _to_datetime_str(value) -> str:
    return pd.Timestamp(value).isoformat()


def _median_composite(stack: "stackstac.DataArray") -> "stackstac.DataArray":
    if "time" not in stack.dims:
        raise ValueError("stack에 time 차원이 없습니다.")
    return stack.median(dim="time", keep_attrs=True)


def load_bands_stack(
    items,
    bbox_lonlat: list[float],
    epsg: int,
    resolution_m: int,
    bands: list[str],
):
    stack = stackstac.stack(
        items,
        assets=bands,
        epsg=epsg,
        resolution=resolution_m,
        bounds_latlon=bbox_lonlat,
        chunksize=2048,
    )
    return stack


def compute_nbr_from_stack(stack) -> tuple["stackstac.DataArray", dict]:
    """
    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    Sentinel-2: NIR=B08, SWIR2=B12 (20m)
    """
    if "band" not in stack.dims:
        raise ValueError("stack에 band 차원이 없습니다.")

    nir = stack.sel(band="B08").astype(float)
    swir2 = stack.sel(band="B12").astype(float)
    nbr = (nir - swir2) / (nir + swir2 + 1e-6)
    nbr = nbr.clip(-1, 1)

    stats = {
        "nbr_min": float(nbr.min().compute().values),
        "nbr_max": float(nbr.max().compute().values),
        "nbr_mean": float(nbr.mean().compute().values),
    }
    return nbr, stats


def classify_dnbr(dnbr: np.ndarray) -> np.ndarray:
    """
    dNBR 임계값 기반 피해 등급 (관행적 기준; 지역/센서/시기별 튜닝 필요)
    - < 0.10: 0 (Unburned/Low change)
    - 0.10–0.27: 1 (Low)
    - 0.27–0.44: 2 (Moderate-low)
    - 0.44–0.66: 3 (Moderate-high)
    - >= 0.66: 4 (High)
    """
    bins = np.array([0.10, 0.27, 0.44, 0.66])
    classes = np.digitize(dnbr, bins=bins, right=False).astype(np.uint8)
    return classes


def vectorize_high_severity(
    severity: np.ndarray,
    transform,
    crs,
    min_severity_class: int,
    min_area_m2: float = 0.0,
) -> gpd.GeoDataFrame:
    mask = severity >= min_severity_class
    if not mask.any():
        return gpd.GeoDataFrame({"severity": [], "geometry": []}, crs=crs)

    records = []
    for geom, value in shapes(severity, mask=mask, transform=transform):
        if int(value) < min_severity_class:
            continue
        geom_shape = shape(geom)
        if min_area_m2 > 0 and geom_shape.area < min_area_m2:
            continue
        records.append({"severity": int(value), "geometry": geom_shape})

    return gpd.GeoDataFrame(records, crs=crs)


def save_geotiff(dataarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataarray.rio.to_raster(str(path), driver="COG", compress="LZW")


def plot_outputs(dnbr: np.ndarray, severity: np.ndarray, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im0 = ax.imshow(dnbr, cmap="RdYlGn_r", vmin=-0.2, vmax=1.0)
    ax.set_title("dNBR (pre - post)")
    ax.set_axis_off()
    plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    cmap = plt.get_cmap("inferno", 5)
    im1 = ax.imshow(severity, cmap=cmap, vmin=0, vmax=4)
    ax.set_title("Severity class (0–4)")
    ax.set_axis_off()
    cbar = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3, 4])

    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = AnalysisConfig(
        name="maui_lahaina_2023",
        bbox_lonlat=[-156.69, 20.86, -156.60, 20.93],
        epsg=32604,
        resolution_m=20,
        max_cloud_cover=20,
        pre_range="2023-07-15/2023-08-04",
        post_range="2023-08-10/2023-08-25",
        vectorize_min_severity_class=3,
    )

    output_dir = Path(__file__).parent.parent / "data" / "output" / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("12.6 실습: Sentinel-2 기반 산불 피해(dNBR) 산출")
    print("=" * 72)
    print(f"[설정] AOI bbox(lon/lat): {config.bbox_lonlat}")
    print(f"      pre={config.pre_range}, post={config.post_range}, cloud<{config.max_cloud_cover}%")

    catalog = open_catalog()

    pre_items = search_sentinel2_items(
        catalog, config.bbox_lonlat, config.pre_range, config.max_cloud_cover
    )
    post_items = search_sentinel2_items(
        catalog, config.bbox_lonlat, config.post_range, config.max_cloud_cover
    )

    print(f"[검색] pre items: {len(pre_items)}개, post items: {len(post_items)}개")
    if not pre_items or not post_items:
        print("전/후 영상이 충분하지 않습니다. bbox/기간/구름 조건을 조정하세요.")
        return

    def item_summary(items):
        rows = []
        for it in sorted(items, key=lambda x: x.datetime):
            rows.append(
                {
                    "datetime": _to_datetime_str(it.datetime),
                    "id": it.id,
                    "cloud_cover": it.properties.get("eo:cloud_cover"),
                }
            )
        return pd.DataFrame(rows)

    pre_df = item_summary(pre_items)
    post_df = item_summary(post_items)
    pre_df.to_csv(output_dir / "pre_items.csv", index=False)
    post_df.to_csv(output_dir / "post_items.csv", index=False)

    print("[로드] B08/B12 (20m) 스택 생성...")
    bands = ["B08", "B12"]

    pre_stack = load_bands_stack(
        pre_items,
        bbox_lonlat=config.bbox_lonlat,
        epsg=config.epsg,
        resolution_m=config.resolution_m,
        bands=bands,
    )
    post_stack = load_bands_stack(
        post_items,
        bbox_lonlat=config.bbox_lonlat,
        epsg=config.epsg,
        resolution_m=config.resolution_m,
        bands=bands,
    )

    print(f"      pre stack shape: {pre_stack.shape}, dims={pre_stack.dims}")
    print(f"      post stack shape: {post_stack.shape}, dims={post_stack.dims}")

    print("[합성] 전/후 기간별 median composite...")
    pre_comp = _median_composite(pre_stack)
    post_comp = _median_composite(post_stack)

    print("[계산] NBR(pre/post) 및 dNBR...")
    pre_nbr, pre_stats = compute_nbr_from_stack(pre_comp)
    post_nbr, post_stats = compute_nbr_from_stack(post_comp)

    dnbr = (pre_nbr - post_nbr).clip(-1, 1)
    dnbr_np = dnbr.compute().values.astype(np.float32)

    print("[분류] dNBR → severity class...")
    severity_np = classify_dnbr(dnbr_np)

    pixel_area_m2 = float(config.resolution_m * config.resolution_m)
    area_by_class_km2 = {}
    for cls in range(0, 5):
        pixels = int((severity_np == cls).sum())
        area_by_class_km2[str(cls)] = float(pixels * pixel_area_m2 / 1e6)

    impact_pixels = int((severity_np >= config.vectorize_min_severity_class).sum())
    impact_area_km2 = float(impact_pixels * pixel_area_m2 / 1e6)

    print(
        f"[벡터화] severity>={config.vectorize_min_severity_class} 폴리곤 추출..."
    )
    transform = dnbr.rio.transform()
    crs = dnbr.rio.crs
    impact_gdf = vectorize_high_severity(
        severity_np,
        transform=transform,
        crs=crs,
        min_severity_class=config.vectorize_min_severity_class,
        min_area_m2=0,
    )
    impact_gdf_path = output_dir / "impact_area.geojson"
    impact_gdf.to_file(impact_gdf_path, driver="GeoJSON")

    dnbr_path = output_dir / "dnbr.tif"
    severity_path = output_dir / "severity_class.tif"

    dnbr_da = dnbr.rio.write_nodata(np.nan)
    save_geotiff(dnbr_da, dnbr_path)

    severity_da = dnbr.copy(data=severity_np).astype(np.uint8)
    severity_da = severity_da.rename("severity")
    severity_da = severity_da.rio.write_nodata(255)
    save_geotiff(severity_da, severity_path)

    fig_path = output_dir / "dnbr_severity.png"
    plot_outputs(dnbr_np, severity_np, fig_path)

    summary = {
        "config": asdict(config),
        "pre_items": len(pre_items),
        "post_items": len(post_items),
        "pre_stats": pre_stats,
        "post_stats": post_stats,
        "dnbr_min": float(np.nanmin(dnbr_np)),
        "dnbr_max": float(np.nanmax(dnbr_np)),
        "dnbr_mean": float(np.nanmean(dnbr_np)),
        "area_by_severity_class_km2": area_by_class_km2,
        "impact_pixels": impact_pixels,
        "impact_area_km2": impact_area_km2,
        "impact_polygons": int(len(impact_gdf)),
        "outputs": {
            "pre_items_csv": str(output_dir / "pre_items.csv"),
            "post_items_csv": str(output_dir / "post_items.csv"),
            "dnbr_tif": str(dnbr_path),
            "severity_tif": str(severity_path),
            "impact_area_geojson": str(impact_gdf_path),
            "png": str(fig_path),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("완료")
    print("=" * 72)
    print(f"- dNBR(mean)={summary['dnbr_mean']:.3f}, min={summary['dnbr_min']:.3f}, max={summary['dnbr_max']:.3f}")
    print(
        f"- Severity>={config.vectorize_min_severity_class} 면적(픽셀 기반): {summary['impact_area_km2']:.3f} km²"
    )
    print(f"- GeoJSON 폴리곤 수: {summary['impact_polygons']}")
    print(f"- 출력 디렉토리: {output_dir}")


if __name__ == "__main__":
    main()
