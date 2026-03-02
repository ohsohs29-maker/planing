#!/usr/bin/env python3
"""
3장 종합 실습: STAC API로 관심지역 위성영상 검색 및 NDVI 분석
GeoAI: 지리공간 인공지능의 이론과 실제

이 스크립트는 다음을 수행합니다:
1. Planetary Computer STAC API로 Sentinel-2 영상 검색
2. stackstac으로 xarray DataArray 로드
3. NDVI 시계열 계산 및 분석
4. 시각화 및 결과 저장 (COG, Zarr)
"""

from pystac_client import Client
import planetary_computer as pc
import stackstac
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def search_sentinel2(catalog, bbox, datetime_range, max_cloud_cover=10):
    """Sentinel-2 영상 검색"""
    print(f"[검색] bbox={bbox}, datetime={datetime_range}, cloud<{max_cloud_cover}%")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}}
    )

    items = list(search.item_collection())
    print(f"       {len(items)}개 영상 발견")

    return items


def load_as_xarray(items, bbox, bands=None, resolution=10):
    """STAC Item을 xarray DataArray로 로드"""
    if bands is None:
        bands = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR

    print(f"[로드] bands={bands}, resolution={resolution}m")

    stack = stackstac.stack(
        items,
        assets=bands,
        resolution=resolution,
        bounds_latlon=bbox,
        chunksize=2048
    )

    print(f"       형태: {stack.shape}")
    print(f"       차원: {stack.dims}")

    return stack


def calculate_ndvi(stack):
    """NDVI 계산: (NIR - Red) / (NIR + Red)"""
    print("[계산] NDVI...")

    nir = stack.sel(band="B08").astype(float)
    red = stack.sel(band="B04").astype(float)

    # 0으로 나누기 방지
    ndvi = (nir - red) / (nir + red + 1e-6)

    # 유효 범위 클리핑
    ndvi = ndvi.clip(-1, 1)

    return ndvi


def analyze_timeseries(ndvi):
    """NDVI 시계열 분석"""
    print("[분석] 시계열 평균...")

    # 영역 평균 계산 (실제 데이터 로드 발생)
    ndvi_mean = ndvi.mean(dim=["x", "y"]).compute()

    print("\n       시점별 평균 NDVI:")
    print("       " + "-" * 30)

    results = []
    for t, v in zip(ndvi_mean.time.values, ndvi_mean.values):
        date = np.datetime_as_string(t, unit='D')
        print(f"       {date}: {v:.3f}")
        results.append((date, float(v)))

    return results


def visualize(ndvi, ndvi_timeseries, output_path):
    """시각화: 시계열 그래프 + 공간 분포"""
    print(f"[시각화] {output_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. NDVI 시계열 그래프
    ax1 = axes[0]
    dates = [r[0] for r in ndvi_timeseries]
    values = [r[1] for r in ndvi_timeseries]

    ax1.plot(range(len(dates)), values, 'o-', color='green', linewidth=2, markersize=8)
    ax1.set_xticks(range(len(dates)))
    ax1.set_xticklabels(dates, rotation=45, ha='right')
    ax1.set_xlabel("날짜", fontsize=12)
    ax1.set_ylabel("평균 NDVI", fontsize=12)
    ax1.set_title("서울 지역 NDVI 시계열 (2024년 여름)", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.7)

    # 2. NDVI 공간 분포 (중간 시점)
    ax2 = axes[1]
    mid_idx = len(ndvi_timeseries) // 2
    ndvi_snapshot = ndvi.isel(time=mid_idx).compute()

    im = ax2.imshow(ndvi_snapshot, cmap='RdYlGn', vmin=-0.1, vmax=0.7)
    ax2.set_title(f"NDVI 공간 분포: {ndvi_timeseries[mid_idx][0]}", fontsize=14)
    ax2.set_xlabel("X (픽셀)", fontsize=12)
    ax2.set_ylabel("Y (픽셀)", fontsize=12)
    plt.colorbar(im, ax=ax2, label='NDVI', shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"       저장 완료: {output_path}")


def save_results(ndvi, output_dir):
    """결과 저장: COG (단일 시점) + Zarr (시계열)"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. COG 저장 (중간 시점)
    mid_idx = len(ndvi.time) // 2
    ndvi_snapshot = ndvi.isel(time=mid_idx)

    cog_path = output_dir / "seoul_ndvi_snapshot.tif"
    print(f"[저장] COG: {cog_path}")

    try:
        ndvi_snapshot.rio.to_raster(
            str(cog_path),
            driver="COG",
            compress="LZW"
        )
        print(f"       완료: {cog_path}")
    except Exception as e:
        print(f"       COG 저장 실패 (rioxarray 필요): {e}")

    # 2. Zarr 저장 (전체 시계열)
    zarr_path = output_dir / "seoul_ndvi_timeseries.zarr"
    print(f"[저장] Zarr: {zarr_path}")

    try:
        ndvi.to_zarr(str(zarr_path), mode="w")
        print(f"       완료: {zarr_path}")
    except Exception as e:
        print(f"       Zarr 저장 실패: {e}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("3장 종합 실습: STAC API 기반 NDVI 시계열 분석")
    print("=" * 60)

    # 설정
    seoul_bbox = [126.8, 37.4, 127.2, 37.7]
    datetime_range = "2024-06-01/2024-08-31"
    max_cloud_cover = 10
    output_dir = Path(__file__).parent.parent / "data" / "output"

    # 1. STAC 카탈로그 접속
    print("\n[1] STAC 카탈로그 접속...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )
    print(f"    접속 완료: {catalog.title}")

    # 2. Sentinel-2 검색
    print("\n[2] Sentinel-2 영상 검색...")
    items = search_sentinel2(
        catalog,
        seoul_bbox,
        datetime_range,
        max_cloud_cover
    )

    if not items:
        print("검색 결과가 없습니다.")
        return

    # 3. xarray로 로드
    print("\n[3] xarray DataArray 로드...")
    stack = load_as_xarray(items, seoul_bbox)

    # 4. NDVI 계산
    print("\n[4] NDVI 계산...")
    ndvi = calculate_ndvi(stack)

    # 5. 시계열 분석
    print("\n[5] 시계열 분석...")
    ndvi_timeseries = analyze_timeseries(ndvi)

    # 6. 시각화
    print("\n[6] 시각화...")
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize(ndvi, ndvi_timeseries, output_dir / "seoul_ndvi_analysis.png")

    # 7. 결과 저장
    print("\n[7] 결과 저장...")
    save_results(ndvi, output_dir)

    # 8. 요약
    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)

    print(f"\n결과 요약:")
    print(f"  - 분석 영상 수: {len(items)}개")
    print(f"  - NDVI 범위: {min(v for _, v in ndvi_timeseries):.3f} ~ {max(v for _, v in ndvi_timeseries):.3f}")
    print(f"  - 최대 NDVI 시점: {max(ndvi_timeseries, key=lambda x: x[1])[0]}")
    print(f"  - 출력 디렉토리: {output_dir}")


if __name__ == "__main__":
    main()
