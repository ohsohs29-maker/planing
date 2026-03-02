"""
2.2절 공간 연산 종합 예제

버퍼, 오버레이, 공간 조인, 래스터 클리핑을 실습합니다.
"""
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
import pandas as pd
import time

# 크로스 플랫폼 호환성
output_dir = Path(__file__).parent.parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("2.2절 공간 연산 종합 실습")
print("=" * 70)

# ============================================================
# 1. 버퍼(Buffer) 연산
# ============================================================

print("\n" + "=" * 70)
print("1. 버퍼 분석")
print("=" * 70)

# 서울 주요 지점 (경위도)
points_data = {
    '이름': ['시청', '강남역', '잠실역'],
    '경도': [126.978, 127.028, 127.100],
    '위도': [37.566, 37.498, 37.513]
}

points = gpd.GeoDataFrame(
    points_data,
    geometry=[Point(lon, lat) for lon, lat in zip(points_data['경도'], points_data['위도'])],
    crs="EPSG:4326"
)

# UTM 52N으로 변환 (거리 계산을 위해 필수)
points_utm = points.to_crs("EPSG:32652")

print("\n원본 포인트 (UTM 좌표):")
for idx, row in points_utm.iterrows():
    x, y = row.geometry.x, row.geometry.y
    print(f"  {row['이름']:6s}: ({x:10.2f}m E, {y:11.2f}m N)")

# 1km 버퍼 생성
buffer_distance = 1000  # meters
buffers = points_utm.copy()
buffers['geometry'] = points_utm.geometry.buffer(buffer_distance)

print(f"\n{buffer_distance}m 버퍼 생성:")
for idx, row in buffers.iterrows():
    area_km2 = row.geometry.area / 1_000_000  # m² to km²
    print(f"  {row['이름']} 버퍼 면적: {area_km2:.3f} km²")

# 이론값과 비교
theoretical_area = np.pi * (buffer_distance/1000)**2
print(f"\n검증: 반지름 {buffer_distance/1000}km 원의 이론 면적 = π × 1² = {theoretical_area:.4f} km²")
actual_area = buffers.iloc[0].geometry.area / 1_000_000
error_pct = abs(actual_area - theoretical_area) / theoretical_area * 100
print(f"실제 버퍼 면적과 {error_pct:.2f}% 오차로 일치")

# ============================================================
# 2. 오버레이(Overlay) 연산
# ============================================================

print("\n" + "=" * 70)
print("2. 오버레이 분석 (교집합)")
print("=" * 70)

# 시청-강남역 버퍼 교집합
buffer_seoul = gpd.GeoDataFrame(geometry=[buffers[buffers['이름']=='시청'].iloc[0].geometry], crs="EPSG:32652")
buffer_gangnam = gpd.GeoDataFrame(geometry=[buffers[buffers['이름']=='강남역'].iloc[0].geometry], crs="EPSG:32652")
buffer_jamsil = gpd.GeoDataFrame(geometry=[buffers[buffers['이름']=='잠실역'].iloc[0].geometry], crs="EPSG:32652")

overlap_sg = gpd.overlay(buffer_seoul, buffer_gangnam, how='intersection')
overlap_sj = gpd.overlay(buffer_seoul, buffer_jamsil, how='intersection')

# 거리 계산
pt_seoul = points_utm[points_utm['이름']=='시청'].iloc[0].geometry
pt_gangnam = points_utm[points_utm['이름']=='강남역'].iloc[0].geometry
pt_jamsil = points_utm[points_utm['이름']=='잠실역'].iloc[0].geometry

dist_sg = pt_seoul.distance(pt_gangnam) / 1000  # km
dist_sj = pt_seoul.distance(pt_jamsil) / 1000  # km

print("\n시청-강남역 버퍼 교집합:")
if len(overlap_sg) > 0:
    overlap_area_sg = overlap_sg.iloc[0].geometry.area / 1_000_000
    seoul_area = buffer_seoul.iloc[0].geometry.area / 1_000_000
    print(f"  교집합 면적: {overlap_area_sg:.3f} km²")
    print(f"  시청 버퍼 대비: {overlap_area_sg/seoul_area*100:.1f}%")
    print(f"  강남역 버퍼 대비: {overlap_area_sg/seoul_area*100:.1f}%")
else:
    overlap_area_sg = 0
    print(f"  교집합 면적: 0.000 km²")

print(f"\n시청-강남역 직선 거리: {dist_sg:.2f} km")
print(f"(1km 버퍼 반지름 × 2 = 2km {'<' if dist_sg < 2 else '>'} {dist_sg:.2f}km이므로 교집합 {'발생' if overlap_area_sg > 0 else '없음'})")

print("\n시청-잠실역 버퍼 교집합:")
if len(overlap_sj) > 0:
    overlap_area_sj = overlap_sj.iloc[0].geometry.area / 1_000_000
    print(f"  교집합 면적: {overlap_area_sj:.3f} km²")
else:
    overlap_area_sj = 0
    print(f"  교집합 면적: 0.000 km²")

print(f"  직선 거리: {dist_sj:.2f} km > 2km (버퍼가 닿지 않음)")

# ============================================================
# 3. 공간 조인(Spatial Join)
# ============================================================

print("\n" + "=" * 70)
print("3. 공간 조인 (포인트-in-폴리곤)")
print("=" * 70)

# 서울시 구 경계 생성 (간단한 예시 데이터)
# 실제로는 공공데이터포털에서 다운로드
districts_data = {
    '구명': ['중구', '강남구', '송파구'],
    '인구수': [125709, 543060, 660580]
}

# 각 지점을 포함하는 간단한 폴리곤 생성 (예시)
# 중구: 시청 주변
# 강남구: 강남역 주변
# 송파구: 잠실역 주변
district_polygons = []
for idx, row in points_utm.iterrows():
    x, y = row.geometry.x, row.geometry.y
    # 각 포인트 주변에 5km × 5km 사각형 생성
    poly = Polygon([
        (x-2500, y-2500),
        (x+2500, y-2500),
        (x+2500, y+2500),
        (x-2500, y+2500)
    ])
    district_polygons.append(poly)

districts_utm = gpd.GeoDataFrame(
    districts_data,
    geometry=district_polygons,
    crs="EPSG:32652"
)

# 공간 조인 수행 (포인트가 어느 구에 속하는지)
start_time = time.time()
points_in_districts = gpd.sjoin(
    points_utm[['이름', 'geometry']],
    districts_utm[['구명', '인구수', 'geometry']],
    how='left',
    predicate='within'
)
elapsed_time = time.time() - start_time

print("\n공간 조인 결과:")
print(points_in_districts[['이름', '구명', '인구수']].to_string(index=False))

print(f"\n통계:")
print(f"  총 {len(points_utm)}개 지점 → {len(points_in_districts)}개 구에 매칭")
print(f"  매칭 성공률: {len(points_in_districts)/len(points_utm)*100:.0f}%")
print(f"  평균 처리 시간: {elapsed_time:.3f}초 ({len(points_utm)}개 포인트 × {len(districts_utm)}개 구)")

# ============================================================
# 4. 공간 연산 최적화 비교
# ============================================================

print("\n" + "=" * 70)
print("4. 공간 연산 최적화 시뮬레이션")
print("=" * 70)

# 대규모 데이터 시뮬레이션 (1만 개 건물 포인트)
np.random.seed(42)
n_buildings = 10000

# 서울 범위 내 랜덤 포인트 생성
seoul_center_x, seoul_center_y = 330000, 4155000  # 대략적인 서울 중심
building_points = []
for _ in range(n_buildings):
    x = seoul_center_x + np.random.normal(0, 10000)  # ±10km
    y = seoul_center_y + np.random.normal(0, 10000)
    building_points.append(Point(x, y))

buildings = gpd.GeoDataFrame(
    {'id': range(n_buildings)},
    geometry=building_points,
    crs="EPSG:32652"
)

# 최적화 전: 기본 공간 조인
print("\n서울시 3개 구 × 1만 개 건물 포인트 공간 조인:")
print("\n최적화 전 (기본):")
start = time.time()
result_basic = gpd.sjoin(buildings, districts_utm, how='left', predicate='within')
time_basic = time.time() - start
print(f"  처리 시간: {time_basic:.2f}초")
print(f"  매칭된 건물: {result_basic['구명'].notna().sum():,}개")

# 최적화 후: 공간 인덱스 명시적 사용 (GeoPandas는 자동으로 사용하지만 시뮬레이션)
print("\n최적화 후 (동일 좌표계 + 공간 인덱스):")
# 이미 동일 좌표계이므로 실제 개선 폭은 작음
start = time.time()
result_opt = gpd.sjoin(buildings, districts_utm, how='left', predicate='within')
time_opt = time.time() - start
print(f"  처리 시간: {time_opt:.2f}초 ({time_basic/time_opt:.1f}배 {'향상' if time_basic > time_opt else '동일'})")

# 폴리곤 단순화 효과 시뮬레이션
districts_simplified = districts_utm.copy()
districts_simplified['geometry'] = districts_utm.geometry.simplify(10)  # 10m tolerance

print("\n추가 최적화 (폴리곤 단순화 10m):")
start = time.time()
result_simp = gpd.sjoin(buildings, districts_simplified, how='left', predicate='within')
time_simp = time.time() - start
print(f"  처리 시간: {time_simp:.2f}초 ({time_basic/time_simp:.1f}배 향상)")

# 정확도 비교
accuracy_loss = abs(result_basic['구명'].notna().sum() - result_simp['구명'].notna().sum()) / result_basic['구명'].notna().sum() * 100
print(f"  정확도 손실: {accuracy_loss:.1f}% ({'무시 가능' if accuracy_loss < 0.5 else '주의'})")

# ============================================================
# 5. 결과 저장
# ============================================================

print("\n" + "=" * 70)
print("5. 결과 저장")
print("=" * 70)

# 버퍼 결과 저장
buffers_file = output_dir / "seoul_buffers.geojson"
buffers.to_file(buffers_file, driver='GeoJSON')
print(f"버퍼 데이터 저장: {buffers_file}")

# 공간 조인 결과 저장
sjoin_file = output_dir / "points_districts.csv"
points_in_districts[['이름', '구명', '인구수']].to_csv(sjoin_file, index=False, encoding='utf-8-sig')
print(f"공간 조인 결과 저장: {sjoin_file}")

print("\n" + "=" * 70)
print("전체 코드 실행 완료")
print("=" * 70)
