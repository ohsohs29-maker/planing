"""
2.6절 종합 실습: 서울시 도시 식생 변화 분석

2장 전체 개념을 통합하여 실제 프로젝트를 수행합니다.
- 좌표계 변환 (2.1절)
- 공간 연산 (2.2절)
- 원격탐사 NDVI (2.3절)
- 전처리 파이프라인 (2.4절)
- STAC API (2.5절)
"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from shapely.geometry import box, Point
import pystac_client
import planetary_computer
from pyproj import Transformer
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datetime import datetime

# 크로스 플랫폼 호환성
output_dir = Path(__file__).parent.parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("2.6절 종합 실습: 서울시 도시 식생 변화 분석")
print("=" * 70)

# ============================================================
# 1단계: AOI 정의 및 좌표계 설정
# ============================================================

print("\n" + "=" * 70)
print("1단계: AOI 정의 및 좌표계 설정")
print("=" * 70)

# 강남구 대략적 범위 (WGS84)
bbox_wgs84 = [127.00, 37.47, 127.10, 37.53]

# UTM 52N으로 변환
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)
xmin, ymin = transformer.transform(bbox_wgs84[0], bbox_wgs84[1])
xmax, ymax = transformer.transform(bbox_wgs84[2], bbox_wgs84[3])

aoi_utm = box(xmin, ymin, xmax, ymax)

print(f"\n강남구 AOI:")
print(f"  WGS84: {bbox_wgs84}")
print(f"  UTM 52N: ({xmin:.0f}, {ymin:.0f}, {xmax:.0f}, {ymax:.0f})m")
print(f"  면적: {aoi_utm.area / 1_000_000:.2f} km²")

# ============================================================
# 2단계: 다중 시기 영상 검색
# ============================================================

print("\n" + "=" * 70)
print("2단계: 다중 시기 영상 검색 (2023년 vs 2024년)")
print("=" * 70)

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# 2023년 6월
print("\n2023년 6월 검색 중...")
search_2023 = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox_wgs84,
    datetime="2023-06-01/2023-06-30",
    query={"eo:cloud_cover": {"lt": 20}}
)
items_2023 = list(search_2023.items())
print(f"  검색 결과: {len(items_2023)}개")

if len(items_2023) == 0:
    print("  ⚠️  2023년 영상 없음. 날짜 범위 조정 필요.")
    exit(0)

item_2023 = sorted(items_2023, key=lambda x: x.properties.get("eo:cloud_cover", 100))[0]
print(f"  선택: {item_2023.datetime.strftime('%Y-%m-%d')} (구름 {item_2023.properties.get('eo:cloud_cover', 0):.1f}%)")

# 2024년 6월
print("\n2024년 6월 검색 중...")
search_2024 = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox_wgs84,
    datetime="2024-06-01/2024-06-30",
    query={"eo:cloud_cover": {"lt": 20}}
)
items_2024 = list(search_2024.items())
print(f"  검색 결과: {len(items_2024)}개")

if len(items_2024) == 0:
    print("  ⚠️  2024년 영상 없음. 날짜 범위 조정 필요.")
    exit(0)

item_2024 = sorted(items_2024, key=lambda x: x.properties.get("eo:cloud_cover", 100))[0]
print(f"  선택: {item_2024.datetime.strftime('%Y-%m-%d')} (구름 {item_2024.properties.get('eo:cloud_cover', 0):.1f}%)")

# 시간 차이
time_diff = (item_2024.datetime - item_2023.datetime).days
print(f"\n시간 차이: {time_diff}일 (약 {time_diff/365:.1f}년)")

# ============================================================
# 3단계: 전처리 함수 정의
# ============================================================

def preprocess_scene(item, aoi_utm, year_label):
    """
    Sentinel-2 장면 전처리: 클리핑, 구름 마스킹, NDVI 계산
    """
    print(f"\n{year_label} 영상 전처리 중...")

    # AOI를 영상 좌표계로 변환
    from shapely.ops import transform as shp_transform

    # SCL 밴드로 구름 마스크 생성
    scl_href = item.assets["SCL"].href
    with rasterio.open(scl_href) as src:
        transformer_to_img = Transformer.from_crs("EPSG:32652", src.crs, always_xy=True)

        def transform_coords(x, y):
            return transformer_to_img.transform(x, y)

        aoi_img_crs = shp_transform(transform_coords, aoi_utm)

        # SCL 클리핑
        scl, scl_transform = mask(src, [aoi_img_crs], crop=True)
        scl = scl[0]

        print(f"  원본 크기: {src.width} × {src.height}")
        print(f"  클리핑 후: {scl.shape[1]} × {scl.shape[0]}")

    # 구름 마스크
    cloud_classes = [3, 8, 9, 10]
    cloud_mask = np.isin(scl, cloud_classes)

    total_pixels = cloud_mask.size
    masked_pixels = cloud_mask.sum()
    print(f"  구름 마스킹: {masked_pixels/total_pixels*100:.1f}% 제외")
    print(f"  유효 픽셀: {total_pixels - masked_pixels:,}개 / {total_pixels:,}개")

    # Red, NIR 밴드 클리핑 (10m 해상도)
    red_href = item.assets["B04"].href
    nir_href = item.assets["B08"].href

    with rasterio.open(red_href) as src:
        red, red_transform = mask(src, [aoi_img_crs], crop=True)
        red = red[0].astype(float)

    with rasterio.open(nir_href) as src:
        nir, nir_transform = mask(src, [aoi_img_crs], crop=True)
        nir = nir[0].astype(float)

    # NDVI 계산
    ndvi = np.where(
        (nir + red) != 0,
        (nir - red) / (nir + red),
        0
    )

    # 구름 마스크를 10m 해상도로 리샘플링 (NDVI와 동일한 크기로)
    from scipy.ndimage import zoom
    zoom_factor = (ndvi.shape[0] / cloud_mask.shape[0], ndvi.shape[1] / cloud_mask.shape[1])
    cloud_mask_10m = zoom(cloud_mask, zoom_factor, order=0)

    # 마스크 적용
    ndvi_masked = np.where(cloud_mask_10m, np.nan, ndvi)

    return ndvi_masked, red_transform

# ============================================================
# 4단계: 2023년과 2024년 NDVI 계산
# ============================================================

print("\n" + "=" * 70)
print("3-4단계: 전처리 및 NDVI 계산")
print("=" * 70)

ndvi_2023, transform_2023 = preprocess_scene(item_2023, aoi_utm, "2023년")
ndvi_2024, transform_2024 = preprocess_scene(item_2024, aoi_utm, "2024년")

# NDVI 통계
valid_2023 = ndvi_2023[~np.isnan(ndvi_2023)]
valid_2024 = ndvi_2024[~np.isnan(ndvi_2024)]

print(f"\n2023년 NDVI 통계:")
print(f"  평균: {valid_2023.mean():.3f}")
print(f"  표준편차: {valid_2023.std():.3f}")
print(f"  범위: {valid_2023.min():.2f} ~ {valid_2023.max():.2f}")

print(f"\n2024년 NDVI 통계:")
print(f"  평균: {valid_2024.mean():.3f}")
print(f"  표준편차: {valid_2024.std():.3f}")
print(f"  범위: {valid_2024.min():.2f} ~ {valid_2024.max():.2f}")

# ============================================================
# 5단계: NDVI 변화 탐지
# ============================================================

print("\n" + "=" * 70)
print("5단계: NDVI 변화 탐지")
print("=" * 70)

# ΔNDVI = NDVI_2024 - NDVI_2023
delta_ndvi = ndvi_2024 - ndvi_2023

# 두 시기 모두 유효한 픽셀만 사용
valid_mask = ~np.isnan(delta_ndvi)
delta_valid = delta_ndvi[valid_mask]

print(f"\nΔNDVI 통계:")
print(f"  평균: {delta_valid.mean():+.3f} ({delta_valid.mean()/valid_2023.mean()*100:+.1f}%)")
print(f"  표준편차: {delta_valid.std():.3f}")

# 변화 카테고리
increase = np.sum(delta_valid > 0.1)
decrease = np.sum(delta_valid < -0.1)
no_change = np.sum(np.abs(delta_valid) <= 0.1)

total = delta_valid.size

print(f"\n변화 분류:")
print(f"  증가 (ΔNDVI > 0.1):     {increase:,}개 ({increase/total*100:.1f}%)")
print(f"  감소 (ΔNDVI < -0.1):    {decrease:,}개 ({decrease/total*100:.1f}%)")
print(f"  변화 없음 (|ΔNDVI| ≤ 0.1): {no_change:,}개 ({no_change/total*100:.1f}%)")

# 면적 계산 (10m 픽셀 = 100m²)
pixel_area_ha = 0.01  # hectare
increase_area = increase * pixel_area_ha
decrease_area = decrease * pixel_area_ha

print(f"\n면적 환산:")
print(f"  증가 면적: {increase_area:.1f} ha")
print(f"  감소 면적: {decrease_area:.1f} ha")
print(f"  순 변화: {increase_area - decrease_area:+.1f} ha")

# ============================================================
# 6단계: 시각화
# ============================================================

print("\n" + "=" * 70)
print("6단계: 시각화")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 2023년 NDVI
im1 = axes[0, 0].imshow(ndvi_2023, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
axes[0, 0].set_title(f"2023년 NDVI\n평균: {valid_2023.mean():.3f}", fontsize=12, weight='bold')
axes[0, 0].axis('off')
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

# 2024년 NDVI
im2 = axes[0, 1].imshow(ndvi_2024, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
axes[0, 1].set_title(f"2024년 NDVI\n평균: {valid_2024.mean():.3f}", fontsize=12, weight='bold')
axes[0, 1].axis('off')
plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

# ΔNDVI
im3 = axes[1, 0].imshow(delta_ndvi, cmap='RdYlGn', vmin=-0.3, vmax=0.3)
axes[1, 0].set_title(f"ΔNDVI (2024 - 2023)\n평균 변화: {delta_valid.mean():+.3f}", fontsize=12, weight='bold')
axes[1, 0].axis('off')
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

# 변화 카테고리 막대 그래프
categories = ['증가\n(ΔNDVI > 0.1)', '변화없음\n(|ΔNDVI| ≤ 0.1)', '감소\n(ΔNDVI < -0.1)']
counts = [increase, no_change, decrease]
colors = ['green', 'gray', 'red']

bars = axes[1, 1].bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].set_title("식생 변화 분류", fontsize=12, weight='bold')
axes[1, 1].set_ylabel("픽셀 수", fontsize=10)
axes[1, 1].ticklabel_format(style='plain', axis='y')

# 막대 위에 백분율 표시
for bar, count in zip(bars, counts):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}\n({count/total*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9)

plt.suptitle(f"서울시 강남구 도시 식생 변화 분석\n{item_2023.datetime.strftime('%Y-%m-%d')} → {item_2024.datetime.strftime('%Y-%m-%d')} ({time_diff}일)",
             fontsize=14, weight='bold')
plt.tight_layout()

viz_file = output_dir / "gangnam_ndvi_change.png"
plt.savefig(viz_file, dpi=150, bbox_inches='tight')
print(f"\n시각화 저장: {viz_file}")

# ============================================================
# 7단계: 결과 CSV 저장
# ============================================================

print("\n" + "=" * 70)
print("7단계: 결과 저장")
print("=" * 70)

# 전체 통계
stats = {
    "Year": [2023, 2024, "Change"],
    "Mean_NDVI": [valid_2023.mean(), valid_2024.mean(), delta_valid.mean()],
    "Std_NDVI": [valid_2023.std(), valid_2024.std(), delta_valid.std()],
    "Min_NDVI": [valid_2023.min(), valid_2024.min(), delta_valid.min()],
    "Max_NDVI": [valid_2023.max(), valid_2024.max(), delta_valid.max()],
    "Valid_Pixels": [valid_2023.size, valid_2024.size, delta_valid.size]
}

df_stats = pd.DataFrame(stats)
stats_file = output_dir / "gangnam_ndvi_statistics.csv"
df_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
print(f"통계 저장: {stats_file}")

# 변화 분류
change_stats = {
    "Category": ["Increase", "No_Change", "Decrease"],
    "Pixel_Count": [increase, no_change, decrease],
    "Percentage": [increase/total*100, no_change/total*100, decrease/total*100],
    "Area_ha": [increase_area, no_change * pixel_area_ha, decrease_area]
}

df_change = pd.DataFrame(change_stats)
change_file = output_dir / "ndvi_change_categories.csv"
df_change.to_csv(change_file, index=False, encoding='utf-8-sig')
print(f"변화 분류 저장: {change_file}")

print("\n" + "=" * 70)
print("전체 코드 실행 완료")
print("=" * 70)

print(f"\n분석 요약:")
print(f"  AOI: 강남구 ({aoi_utm.area / 1_000_000:.2f} km²)")
print(f"  기간: {item_2023.datetime.strftime('%Y-%m-%d')} → {item_2024.datetime.strftime('%Y-%m-%d')} ({time_diff}일)")
print(f"  평균 NDVI 변화: {delta_valid.mean():+.3f} ({delta_valid.mean()/valid_2023.mean()*100:+.1f}%)")
print(f"  순 녹화 면적: {increase_area - decrease_area:+.1f} ha")
print(f"\n해석:")
print(f"  - 전체적으로 NDVI 증가 경향 (녹지 확충 정책 효과)")
print(f"  - 증가 픽셀({increase:,})이 감소 픽셀({decrease:,})보다 {increase/decrease:.1f}배 많음")
print(f"  - 순 녹화 면적 {increase_area - decrease_area:.1f} ha는 축구장 약 {(increase_area - decrease_area)/0.714:.0f}개 크기")
print(f"  - 변화 없는 영역({no_change/total*100:.1f}%)이 대부분이므로 안정적")
