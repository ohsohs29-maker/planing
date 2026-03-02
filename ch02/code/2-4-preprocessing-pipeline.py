"""
2.4절 전처리 파이프라인 예제

Sentinel-2 영상의 전체 전처리 파이프라인을 실습합니다.
다운로드 → 구름 마스킹 → 공간 클리핑 → 스택 생성
"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import box
import pystac_client
import planetary_computer
from datetime import datetime
import matplotlib.pyplot as plt

# 크로스 플랫폼 호환성
output_dir = Path(__file__).parent.parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("2.4절 전처리 파이프라인 종합 실습")
print("=" * 70)

# ============================================================
# 1. 영상 검색 및 선택
# ============================================================

print("\n" + "=" * 70)
print("1. Sentinel-2 영상 검색")
print("=" * 70)

# 서울 강남구 영역 (UTM 좌표)
aoi_utm = box(325000, 4150000, 330000, 4155000)  # 5km × 5km

# WGS84로 변환 (검색용)
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)
aoi_bounds = aoi_utm.bounds
bbox_wgs84 = [
    *transformer.transform(aoi_bounds[0], aoi_bounds[1]),
    *transformer.transform(aoi_bounds[2], aoi_bounds[3])
]

print(f"\nAOI (관심 지역):")
print(f"  UTM 좌표: {aoi_bounds}")
print(f"  WGS84 좌표: {bbox_wgs84}")
print(f"  면적: {(aoi_bounds[2]-aoi_bounds[0]) * (aoi_bounds[3]-aoi_bounds[1]) / 1_000_000:.1f} km²")

# STAC 검색
date_range = "2024-05-01/2024-05-31"

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox_wgs84,
    datetime=date_range,
    query={"eo:cloud_cover": {"lt": 30}}  # 구름 30% 미만
)

items = list(search.items())
print(f"\n검색 결과: {len(items)}개 영상")

if len(items) == 0:
    print("\n⚠️  영상이 없습니다. 날짜 범위나 구름 기준을 조정해보세요.")
    exit(0)

# 구름 비율이 가장 낮은 영상 선택
items_sorted = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
item = items_sorted[0]

print(f"\n선택된 영상:")
print(f"  날짜: {item.datetime.strftime('%Y-%m-%d')}")
print(f"  타일: {item.id}")
print(f"  구름 비율: {item.properties.get('eo:cloud_cover', 'N/A'):.1f}%")

# ============================================================
# 2. 구름 마스크 생성
# ============================================================

print("\n" + "=" * 70)
print("2. 구름 마스크 생성")
print("=" * 70)

# SCL 밴드 읽기 및 AOI 클리핑
scl_href = item.assets["SCL"].href

with rasterio.open(scl_href) as src:
    # AOI를 영상 좌표계로 변환
    from shapely.ops import transform as shp_transform
    transformer_to_img = Transformer.from_crs("EPSG:32652", src.crs, always_xy=True)

    def transform_coords(x, y):
        return transformer_to_img.transform(x, y)

    aoi_img_crs = shp_transform(transform_coords, aoi_utm)

    # 클리핑
    scl_clipped, scl_transform = mask(src, [aoi_img_crs], crop=True)
    scl_clipped = scl_clipped[0]  # (1, H, W) → (H, W)

    print(f"\nSCL 밴드:")
    print(f"  원본 크기: {src.width} × {src.height}")
    print(f"  클리핑 후: {scl_clipped.shape[1]} × {scl_clipped.shape[0]}")
    print(f"  좌표계: {src.crs}")

# 구름 마스크 생성
cloud_classes = [3, 8, 9, 10]  # 구름, 그림자, 권운
cloud_mask = np.isin(scl_clipped, cloud_classes)

total_pixels = cloud_mask.size
masked_pixels = cloud_mask.sum()
valid_pixels = total_pixels - masked_pixels

print(f"\n구름 마스킹 결과:")
print(f"  총 픽셀: {total_pixels:,}")
print(f"  마스킹된 픽셀: {masked_pixels:,} ({masked_pixels/total_pixels*100:.2f}%)")
print(f"  유효 픽셀: {valid_pixels:,} ({valid_pixels/total_pixels*100:.2f}%)")

# ============================================================
# 3. 다중 밴드 클리핑 및 스택 생성
# ============================================================

print("\n" + "=" * 70)
print("3. 다중 밴드 클리핑 및 스택 생성")
print("=" * 70)

# 주요 밴드 선택 (10m 해상도)
bands = {
    "B02": "Blue",
    "B03": "Green",
    "B04": "Red",
    "B08": "NIR"
}

print(f"\n다운로드할 밴드: {', '.join(bands.keys())}")

# 밴드별로 클리핑하여 스택 생성
band_stack = []
band_names = []

for band_id, band_name in bands.items():
    print(f"  {band_id} ({band_name}) 처리 중...", end=" ")

    band_href = item.assets[band_id].href

    with rasterio.open(band_href) as src:
        # AOI 클리핑
        band_clipped, band_transform = mask(src, [aoi_img_crs], crop=True)
        band_clipped = band_clipped[0]

        # 스택에 추가
        band_stack.append(band_clipped)
        band_names.append(band_name)

        print(f"완료 ({band_clipped.shape})")

# NumPy 배열로 스택
stack = np.stack(band_stack, axis=0)  # (4, H, W)

print(f"\n스택 생성 완료:")
print(f"  형태: {stack.shape} (밴드, 높이, 너비)")
print(f"  밴드: {', '.join(band_names)}")
print(f"  데이터 타입: {stack.dtype}")

# ============================================================
# 4. NDVI 계산 및 마스크 적용
# ============================================================

print("\n" + "=" * 70)
print("4. NDVI 계산 및 구름 마스크 적용")
print("=" * 70)

# NDVI = (NIR - Red) / (NIR + Red)
nir = stack[3].astype(float)  # B08
red = stack[2].astype(float)  # B04

# 0으로 나누기 방지
ndvi = np.where(
    (nir + red) != 0,
    (nir - red) / (nir + red),
    0
)

print(f"\nNDVI 계산:")
print(f"  형태: {ndvi.shape}")
print(f"  범위: {ndvi.min():.3f} ~ {ndvi.max():.3f}")

# 구름 마스크 적용 (20m SCL을 10m 밴드로 리샘플링)
from scipy.ndimage import zoom
cloud_mask_resampled = zoom(cloud_mask, 2, order=0)  # Nearest neighbor

# 마스크 적용
ndvi_masked = np.where(cloud_mask_resampled, np.nan, ndvi)

valid_ndvi = ndvi_masked[~np.isnan(ndvi_masked)]
print(f"\n마스크 적용 후:")
print(f"  유효 픽셀: {valid_ndvi.size:,} / {ndvi.size:,}")
print(f"  평균 NDVI: {valid_ndvi.mean():.3f}")
print(f"  표준편차: {valid_ndvi.std():.3f}")

# ============================================================
# 5. 결과 저장
# ============================================================

print("\n" + "=" * 70)
print("5. 결과 저장")
print("=" * 70)

# 다중 밴드 스택을 GeoTIFF로 저장
stack_file = output_dir / "sentinel2_stack_clipped.tif"

with rasterio.open(
    stack_file,
    'w',
    driver='GTiff',
    height=stack.shape[1],
    width=stack.shape[2],
    count=stack.shape[0],
    dtype=stack.dtype,
    crs=src.crs,
    transform=band_transform,
    compress='lzw'
) as dst:
    for idx in range(stack.shape[0]):
        dst.write(stack[idx], idx + 1)
    dst.descriptions = tuple(band_names)

print(f"다중 밴드 스택 저장: {stack_file}")
print(f"  밴드: {', '.join(band_names)}")

# NDVI를 GeoTIFF로 저장
ndvi_file = output_dir / "ndvi_masked.tif"

with rasterio.open(
    ndvi_file,
    'w',
    driver='GTiff',
    height=ndvi_masked.shape[0],
    width=ndvi_masked.shape[1],
    count=1,
    dtype=rasterio.float32,
    crs=src.crs,
    transform=band_transform,
    nodata=np.nan,
    compress='lzw'
) as dst:
    dst.write(ndvi_masked.astype(rasterio.float32), 1)

print(f"NDVI (마스크 적용) 저장: {ndvi_file}")

# ============================================================
# 6. 시각화
# ============================================================

print("\n" + "=" * 70)
print("6. 시각화")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# RGB 합성 (정규화)
rgb = np.stack([stack[2], stack[1], stack[0]], axis=-1)  # Red, Green, Blue
rgb_normalized = rgb / 3000.0
rgb_normalized = np.clip(rgb_normalized, 0, 1)

axes[0, 0].imshow(rgb_normalized)
axes[0, 0].set_title("RGB True Color", fontsize=12, weight='bold')
axes[0, 0].axis('off')

# 구름 마스크
axes[0, 1].imshow(cloud_mask_resampled, cmap='RdYlGn_r')
axes[0, 1].set_title(f"Cloud Mask ({masked_pixels/total_pixels*100:.1f}%)", fontsize=12, weight='bold')
axes[0, 1].axis('off')

# NDVI (마스크 전)
im1 = axes[1, 0].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
axes[1, 0].set_title("NDVI (Before Masking)", fontsize=12, weight='bold')
axes[1, 0].axis('off')
plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

# NDVI (마스크 후)
im2 = axes[1, 1].imshow(ndvi_masked, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
axes[1, 1].set_title("NDVI (After Cloud Masking)", fontsize=12, weight='bold')
axes[1, 1].axis('off')
plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

plt.suptitle(f"Preprocessing Pipeline - {item.datetime.strftime('%Y-%m-%d')}", fontsize=14, weight='bold')
plt.tight_layout()

viz_file = output_dir / "preprocessing_pipeline.png"
plt.savefig(viz_file, dpi=150, bbox_inches='tight')
print(f"\n시각화 저장: {viz_file}")

print("\n" + "=" * 70)
print("전체 코드 실행 완료")
print("=" * 70)
print(f"\n파이프라인 요약:")
print(f"  1. 영상 검색: {len(items)}개 → 1개 선택 (구름 {item.properties.get('eo:cloud_cover', 0):.1f}%)")
print(f"  2. 구름 마스킹: {masked_pixels:,} / {total_pixels:,} 픽셀 제외 ({masked_pixels/total_pixels*100:.2f}%)")
print(f"  3. 공간 클리핑: {stack.shape[2]} × {stack.shape[1]} 픽셀 (AOI 5km × 5km)")
print(f"  4. 다중 밴드 스택: {len(bands)}개 밴드 (Blue, Green, Red, NIR)")
print(f"  5. NDVI 계산: 평균 {valid_ndvi.mean():.3f} (유효 픽셀 {valid_ndvi.size:,}개)")
print(f"\n해석:")
print(f"  - Level-2A 제품은 대기보정과 정사보정이 이미 적용됨")
print(f"  - 구름 마스킹으로 신뢰성 높은 픽셀만 선택")
print(f"  - AOI 클리핑으로 필요한 영역만 처리하여 효율성 향상")
print(f"  - NDVI 0.3-0.6은 건강한 식생, 0.6-0.9는 매우 건강한 식생")
