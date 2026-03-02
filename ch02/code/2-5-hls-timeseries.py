"""
2.5절 HLS 시계열 검색 예제

NASA CMR STAC API를 통해 HLS S30과 L30을 통합 검색하여
2-3일 주기 시계열을 구성합니다.
"""
from pystac_client import Client
from datetime import datetime

# NASA CMR STAC 카탈로그 연결
# 참고: NASA Earthdata 계정 필요 (https://urs.earthdata.nasa.gov/)
nasa_catalog = Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")

# 서울 지역 HLS S30 + L30 검색 (2024년 여름)
search = nasa_catalog.search(
    collections=["HLSS30.v2.0", "HLSL30.v2.0"],  # S30(Sentinel-2)과 L30(Landsat) 모두
    bbox=[126.8, 37.4, 127.2, 37.7],  # 서울
    datetime="2024-06-01/2024-08-31"  # 여름 3개월
)

# 검색 결과를 시간순으로 정렬
items = list(search.items())
items_sorted = sorted(items, key=lambda x: x.datetime)

print(f"총 검색된 장면 수: {len(items_sorted)}")
print("\n시간순 상위 10개:")

for item in items_sorted[:10]:
    date = item.datetime.date()
    collection = "S30" if "HLSS30" in item.collection_id else "L30"
    cloud = item.properties.get('eo:cloud_cover', -1)
    print(f"{date}: {collection}, Cloud {cloud:.1f}%")

# 통계 분석
s30_count = sum(1 for item in items if "HLSS30" in item.collection_id)
l30_count = sum(1 for item in items if "HLSL30" in item.collection_id)

print(f"\n총 장면 수: {len(items)}개 (S30: {s30_count}개, L30: {l30_count}개)")

# 시간 간격 계산
dates = sorted([item.datetime for item in items])
intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
avg_interval = sum(intervals) / len(intervals) if intervals else 0

print(f"평균 시간 간격: {avg_interval:.1f}일")

# 유효 장면 (클라우드 < 20%)
valid_items = [item for item in items if item.properties.get('eo:cloud_cover', 100) < 20]
print(f"유효 장면(클라우드 < 20%): {len(valid_items)}개")

# 조화화 검증: 동일 날짜의 S30과 L30 비교 (있는 경우)
print("\n동일 날짜 S30-L30 페어:")
from collections import defaultdict

date_groups = defaultdict(list)
for item in items:
    date_groups[item.datetime.date()].append(item)

pair_count = 0
for date, group_items in date_groups.items():
    if len(group_items) > 1:
        collections = [("S30" if "HLSS30" in item.collection_id else "L30")
                      for item in group_items]
        if "S30" in collections and "L30" in collections:
            pair_count += 1
            print(f"  {date}: S30 + L30 (조화화 검증 가능)")
            if pair_count >= 3:  # 상위 3개만 출력
                break

print(f"\n전체 코드 실행 완료")
