"""
2.1절 좌표 변환 예제

WGS84 경위도 → UTM 변환 및 역변환을 실습합니다.
"""
from pathlib import Path
from pyproj import Transformer

# 크로스 플랫폼 호환성: pathlib 사용
output_dir = Path(__file__).parent.parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 기본 좌표 변환: WGS84 → UTM 52N
# ============================================================

# WGS84(EPSG:4326) → UTM 52N(EPSG:32652) 변환
# always_xy=True: (경도, 위도) 순서 강제 (일부 좌표계는 (위도, 경도) 순서 사용)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)

# 서울시청 좌표
lon, lat = 127.0, 37.5  # 경도, 위도
x_utm, y_utm = transformer.transform(lon, lat)

print("=" * 60)
print("1. 기본 좌표 변환")
print("=" * 60)
print(f"원본 좌표 (경도, 위도): ({lon}, {lat})")
print(f"UTM 좌표: ({x_utm:.2f}m E, {y_utm:.2f}m N)")
print(f"\n해석:")
print(f"  - Easting {x_utm:.2f}m: UTM 52N 존 중심선(129°E)으로부터 동쪽 거리")
print(f"  - Northing {y_utm:.2f}m: 적도로부터 북쪽 거리")

# ============================================================
# 2. 역변환: UTM → WGS84
# ============================================================

# UTM 52N → WGS84 역변환
inv_transformer = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)
lon_back, lat_back = inv_transformer.transform(x_utm, y_utm)

print("\n" + "=" * 60)
print("2. 역변환 검증")
print("=" * 60)
print(f"역변환 좌표: ({lon_back:.6f}, {lat_back:.6f})")
print(f"변환 오차: {abs(lon - lon_back):.15f}도 (경도)")
print(f"변환 오차: {abs(lat - lat_back):.15f}도 (위도)")
print(f"\n해석:")
print(f"  - 오차는 부동소수점 연산 정밀도 한계 (10⁻¹⁰도 ≈ 1cm)")
print(f"  - 실용적으로 완벽한 가역 변환")

# ============================================================
# 3. 다중 지점 변환 (한국 주요 도시)
# ============================================================

cities = {
    "서울": (127.0, 37.5),
    "부산": (129.0, 35.1),
    "제주": (126.5, 33.5),
    "강릉": (128.9, 37.8)
}

print("\n" + "=" * 60)
print("3. 한국 주요 도시 좌표 변환")
print("=" * 60)

for city, (lon, lat) in cities.items():
    x_utm, y_utm = transformer.transform(lon, lat)
    print(f"{city:4s}: ({lon:6.1f}°E, {lat:5.1f}°N) → ({x_utm:10.2f}m E, {y_utm:10.2f}m N)")

# ============================================================
# 4. 거리 계산 비교: 경위도 vs UTM
# ============================================================

import numpy as np

# 서울-부산 거리 계산
seoul = cities["서울"]
busan = cities["부산"]

# UTM 좌표로 변환 후 유클리드 거리
seoul_utm = transformer.transform(*seoul)
busan_utm = transformer.transform(*busan)
distance_utm = np.sqrt((seoul_utm[0] - busan_utm[0])**2 +
                       (seoul_utm[1] - busan_utm[1])**2)

# 경위도 거리 (Haversine 공식)
def haversine(lon1, lat1, lon2, lat2):
    """경위도 좌표 간 거리 계산 (Haversine 공식, km)"""
    R = 6371  # 지구 반경 (km)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

distance_geo = haversine(*seoul, *busan)

print("\n" + "=" * 60)
print("4. 서울-부산 거리 계산 비교")
print("=" * 60)
print(f"UTM 좌표 유클리드 거리: {distance_utm/1000:.2f} km")
print(f"경위도 Haversine 거리:  {distance_geo:.2f} km")
print(f"차이: {abs(distance_utm/1000 - distance_geo):.2f} km ({abs(distance_utm/1000 - distance_geo)/distance_geo*100:.2f}%)")
print(f"\n해석:")
print(f"  - UTM 투영은 거리 계산이 간단 (유클리드 공식)")
print(f"  - 경위도는 구면 삼각법 필요 (Haversine)")
print(f"  - 장거리에서는 투영 왜곡으로 차이 발생")

# ============================================================
# 5. 결과 저장 (CSV)
# ============================================================

import pandas as pd

# 변환 결과를 데이터프레임으로 저장
results = []
for city, (lon, lat) in cities.items():
    x_utm, y_utm = transformer.transform(lon, lat)
    results.append({
        "도시": city,
        "경도": lon,
        "위도": lat,
        "UTM_X": x_utm,
        "UTM_Y": y_utm
    })

df = pd.DataFrame(results)
output_file = output_dir / "city_coordinates.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n" + "=" * 60)
print("5. 결과 저장")
print("=" * 60)
print(f"저장 경로: {output_file}")
print(f"데이터 샘플:")
print(df.head())

print("\n" + "=" * 60)
print("전체 코드 실행 완료")
print("=" * 60)
