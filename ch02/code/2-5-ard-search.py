"""
2.5절 ARD 검색 예제

Microsoft Planetary Computer에서 Sentinel-2 L2A ARD를 검색합니다.
"""
from pystac_client import Client
import planetary_computer as pc

# Microsoft Planetary Computer STAC 카탈로그 연결
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace  # 인증 토큰 자동 삽입
)

# 서울 지역 Sentinel-2 L2A ARD 검색
search = catalog.search(
    collections=["sentinel-2-l2a"],  # L2A = Surface Reflectance ARD
    bbox=[126.8, 37.4, 127.2, 37.7],  # 서울 (minx, miny, maxx, maxy)
    datetime="2024-01-01/2024-12-31"  # 2024년 전체
)

# 검색 결과 분석
items = list(search.items())
total_scenes = len(items)

print(f"검색된 ARD 장면 수: {total_scenes}")

# 클라우드 커버 통계
cloud_covers = [item.properties.get('eo:cloud_cover', 100) for item in items]
avg_cloud = sum(cloud_covers) / len(cloud_covers)
valid_scenes = [c for c in cloud_covers if c < 20]

print(f"평균 클라우드 커버: {avg_cloud:.1f}%")
print(f"유효 장면(클라우드 < 20%): {len(valid_scenes)}개")

# 월별 통계
from collections import defaultdict

monthly_count = defaultdict(int)
for item in items:
    month = item.datetime.strftime("%Y-%m")
    monthly_count[month] += 1

print("\n월별 장면 수:")
for month in sorted(monthly_count.keys()):
    print(f"  {month}: {monthly_count[month]}개")

# 유효 장면 상세 정보
print(f"\n유효 장면 ({len(valid_scenes)}개) 상세:")
valid_items = [item for item in items if item.properties.get('eo:cloud_cover', 100) < 20]
for item in valid_items[:5]:
    date = item.datetime.date()
    cloud = item.properties.get('eo:cloud_cover', 0)
    tile_id = item.properties.get('s2:mgrs_tile', 'N/A')
    print(f"  {date}: Tile {tile_id}, Cloud {cloud:.1f}%")

print(f"\n전체 코드 실행 완료")
