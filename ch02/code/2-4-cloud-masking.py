"""
2.4절 구름 마스킹 예제

Sentinel-2 SCL 밴드를 사용한 구름 마스킹을 실습합니다.
"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import pystac_client
import planetary_computer
from datetime import datetime

# 크로스 플랫폼 호환성
output_dir = Path(__file__).parent.parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("2.4절 구름 마스킹 실습")
print("=" * 70)

# ============================================================
# 1. Sentinel-2 Level-2A 영상 검색
# ============================================================

print("\n" + "=" * 70)
print("1. Sentinel-2 Level-2A 영상 검색")
print("=" * 70)

# 서울 중심부 영역
bbox = [126.95, 37.50, 127.05, 37.60]
date_range = "2024-06-01/2024-06-30"

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=date_range,
)

items = list(search.items())
print(f"\n검색된 영상 수: {len(items)}개")

if len(items) == 0:
    print("\n⚠️  영상이 없습니다. 날짜 범위를 확대해보세요.")
    exit(0)

# 첫 번째 영상 선택
item = items[0]
print(f"\n선택된 영상:")
print(f"  날짜: {item.datetime.strftime('%Y-%m-%d')}")
print(f"  타일: {item.id}")
print(f"  구름 비율: {item.properties.get('eo:cloud_cover', 'N/A')}%")

# ============================================================
# 2. SCL(Scene Classification Layer) 밴드 읽기
# ============================================================

print("\n" + "=" * 70)
print("2. SCL 밴드 읽기")
print("=" * 70)

# SCL 밴드 URL 가져오기
scl_href = item.assets["SCL"].href

print(f"\nSCL 밴드 다운로드 중...")
print(f"  URL: {scl_href[:80]}...")

# SCL 밴드 읽기 (메모리 효율을 위해 일부 영역만)
with rasterio.open(scl_href) as src:
    print(f"\n영상 정보:")
    print(f"  좌표계: {src.crs}")
    print(f"  크기: {src.width} × {src.height} 픽셀")
    print(f"  해상도: {src.res[0]}m")

    # 중심부 1000×1000 픽셀만 읽기
    window = rasterio.windows.Window(
        col_off=src.width // 2 - 500,
        row_off=src.height // 2 - 500,
        width=1000,
        height=1000
    )

    scl = src.read(1, window=window)
    transform = src.window_transform(window)

print(f"\n읽은 영역 크기: {scl.shape}")

# ============================================================
# 3. SCL 클래스별 통계
# ============================================================

print("\n" + "=" * 70)
print("3. SCL 클래스별 통계")
print("=" * 70)

# SCL 클래스 정의
scl_classes = {
    0: "NO_DATA",
    1: "SATURATED_DEFECTIVE",
    2: "DARK_AREA_PIXELS",
    3: "CLOUD_SHADOWS",
    4: "VEGETATION",
    5: "NOT_VEGETATED",
    6: "WATER",
    7: "UNCLASSIFIED",
    8: "CLOUD_MEDIUM_PROBABILITY",
    9: "CLOUD_HIGH_PROBABILITY",
    10: "THIN_CIRRUS",
    11: "SNOW"
}

print("\nSCL 클래스 분포:")
unique, counts = np.unique(scl, return_counts=True)
total_pixels = scl.size

for val, count in zip(unique, counts):
    class_name = scl_classes.get(val, "UNKNOWN")
    percentage = count / total_pixels * 100
    print(f"  {val:2d} {class_name:25s}: {count:7,d} 픽셀 ({percentage:5.2f}%)")

# ============================================================
# 4. 구름 마스크 생성
# ============================================================

print("\n" + "=" * 70)
print("4. 구름 마스크 생성")
print("=" * 70)

# 보수적 마스킹: 구름, 그림자, 권운
cloud_classes = [3, 8, 9, 10]
cloud_mask = np.isin(scl, cloud_classes)

# 매우 보수적 마스킹: 위 + 어두운 지역, 미분류
conservative_mask = np.isin(scl, cloud_classes + [2, 7])

print(f"\n보수적 마스킹 (구름 관련만):")
print(f"  마스킹된 픽셀: {cloud_mask.sum():,} / {total_pixels:,} ({cloud_mask.sum()/total_pixels*100:.2f}%)")
print(f"  유효 픽셀: {(~cloud_mask).sum():,} / {total_pixels:,} ({(~cloud_mask).sum()/total_pixels*100:.2f}%)")

print(f"\n매우 보수적 마스킹 (+ 어두운 지역, 미분류):")
print(f"  마스킹된 픽셀: {conservative_mask.sum():,} / {total_pixels:,} ({conservative_mask.sum()/total_pixels*100:.2f}%)")
print(f"  유효 픽셀: {(~conservative_mask).sum():,} / {total_pixels:,} ({(~conservative_mask).sum()/total_pixels*100:.2f}%)")

# ============================================================
# 5. RGB 영상에 마스크 적용
# ============================================================

print("\n" + "=" * 70)
print("5. RGB 영상에 마스크 적용")
print("=" * 70)

# RGB 밴드 읽기 (10m 해상도)
print("\nRGB 밴드 다운로드 중...")
rgb = np.zeros((3, 1000, 1000), dtype=np.uint16)

for idx, band_name in enumerate(["B04", "B03", "B02"]):  # Red, Green, Blue
    band_href = item.assets[band_name].href
    with rasterio.open(band_href) as src:
        # SCL과 동일한 영역 읽기 (20m → 10m이므로 2배 크기)
        window = rasterio.windows.Window(
            col_off=src.width // 2 - 1000,
            row_off=src.height // 2 - 1000,
            width=2000,
            height=2000
        )
        rgb[idx] = src.read(1, window=window, out_shape=(1000, 1000))

print(f"RGB 영상 크기: {rgb.shape}")

# RGB를 0-1 범위로 정규화 (시각화용)
rgb_normalized = rgb / 3000.0  # Sentinel-2 일반적인 최대값
rgb_normalized = np.clip(rgb_normalized, 0, 1)

# 마스크 적용 (마스킹된 픽셀을 빨강으로 표시)
rgb_masked = rgb_normalized.copy()
rgb_masked[0, cloud_mask] = 1.0  # Red
rgb_masked[1, cloud_mask] = 0.0  # Green
rgb_masked[2, cloud_mask] = 0.0  # Blue

# ============================================================
# 6. 시각화 및 저장
# ============================================================

print("\n" + "=" * 70)
print("6. 시각화 및 저장")
print("=" * 70)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 원본 RGB
axes[0].imshow(np.moveaxis(rgb_normalized, 0, -1))
axes[0].set_title("Original RGB", fontsize=14, weight='bold')
axes[0].axis('off')

# 구름 마스크
axes[1].imshow(cloud_mask, cmap='RdYlGn_r', vmin=0, vmax=1)
axes[1].set_title(f"Cloud Mask ({cloud_mask.sum()/total_pixels*100:.1f}% masked)", fontsize=14, weight='bold')
axes[1].axis('off')

# 마스크 오버레이
axes[2].imshow(np.moveaxis(rgb_masked, 0, -1))
axes[2].set_title("RGB + Cloud Mask (Red)", fontsize=14, weight='bold')
axes[2].axis('off')

plt.suptitle(f"Cloud Masking - {item.datetime.strftime('%Y-%m-%d')}", fontsize=16, weight='bold')
plt.tight_layout()

# 저장
output_file = output_dir / "cloud_masking_visualization.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n시각화 저장: {output_file}")

# ============================================================
# 7. 마스크 GeoTIFF 저장
# ============================================================

# 마스크를 GeoTIFF로 저장
mask_file = output_dir / "cloud_mask.tif"

with rasterio.open(
    mask_file,
    'w',
    driver='GTiff',
    height=cloud_mask.shape[0],
    width=cloud_mask.shape[1],
    count=1,
    dtype=rasterio.uint8,
    crs=src.crs,
    transform=transform,
    compress='lzw'
) as dst:
    dst.write(cloud_mask.astype(np.uint8), 1)

print(f"구름 마스크 저장: {mask_file}")

# 통계 CSV 저장
import pandas as pd

stats = []
for val, count in zip(unique, counts):
    stats.append({
        "SCL_Value": val,
        "Class_Name": scl_classes.get(val, "UNKNOWN"),
        "Pixel_Count": count,
        "Percentage": count / total_pixels * 100
    })

df = pd.DataFrame(stats)
stats_file = output_dir / "scl_statistics.csv"
df.to_csv(stats_file, index=False, encoding='utf-8-sig')
print(f"SCL 통계 저장: {stats_file}")

print("\n" + "=" * 70)
print("전체 코드 실행 완료")
print("=" * 70)
print(f"\n결과 요약:")
print(f"  총 픽셀: {total_pixels:,}")
print(f"  구름 관련 픽셀: {cloud_mask.sum():,} ({cloud_mask.sum()/total_pixels*100:.2f}%)")
print(f"  유효 픽셀: {(~cloud_mask).sum():,} ({(~cloud_mask).sum()/total_pixels*100:.2f}%)")
print(f"\n해석:")
print(f"  - SCL 밴드는 11개 클래스로 픽셀을 분류")
print(f"  - 구름 관련 클래스(3,8,9,10)를 제외하여 유효 데이터만 분석")
print(f"  - 보수적 마스킹은 데이터 손실 증가, 관대한 마스킹은 노이즈 증가")
print(f"  - 응용 목적에 따라 마스킹 전략 조정 필요")
