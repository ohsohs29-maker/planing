#!/usr/bin/env python3
"""
3장 실습: STAC API를 활용한 위성영상 검색
GeoAI: 지리공간 인공지능의 이론과 실제

이 스크립트는 Microsoft Planetary Computer의 STAC API를 사용하여
서울 지역의 Sentinel-2 위성영상을 검색하는 방법을 보여줍니다.
"""

from pystac_client import Client
import planetary_computer as pc
from datetime import datetime


def main():
    """STAC API 검색 예제"""

    print("=" * 60)
    print("STAC API 검색 예제 - Microsoft Planetary Computer")
    print("=" * 60)

    # 1. Planetary Computer STAC API 접속
    print("\n[1] STAC 카탈로그 접속...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace  # 자산 URL 서명
    )
    print(f"    카탈로그 ID: {catalog.id}")
    print(f"    카탈로그 제목: {catalog.title}")

    # 2. 사용 가능한 컬렉션 탐색
    print("\n[2] Sentinel 관련 컬렉션 목록:")
    sentinel_collections = []
    for collection in catalog.get_collections():
        if 'sentinel' in collection.id.lower():
            sentinel_collections.append(collection)
            print(f"    - {collection.id}: {collection.title}")

    # 3. 관심 지역 및 검색 조건 설정
    print("\n[3] 검색 조건 설정...")

    # 서울 지역 bbox (경도 최소, 위도 최소, 경도 최대, 위도 최대)
    seoul_bbox = [126.8, 37.4, 127.2, 37.7]
    print(f"    관심 지역 (bbox): {seoul_bbox}")

    # 시간 범위
    datetime_range = "2024-06-01/2024-08-31"
    print(f"    시간 범위: {datetime_range}")

    # 구름 비율 조건
    max_cloud_cover = 10
    print(f"    최대 구름 비율: {max_cloud_cover}%")

    # 4. 검색 수행
    print("\n[4] Sentinel-2 L2A 검색 수행...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=seoul_bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}}
    )

    # 5. 검색 결과 확인
    items = list(search.item_collection())
    print(f"\n[5] 검색 결과: {len(items)}개 영상 발견")

    if items:
        print("\n    상세 정보:")
        print("    " + "-" * 50)
        print(f"    {'날짜':<12} {'구름(%)':<10} {'타일 ID':<15} {'플랫폼'}")
        print("    " + "-" * 50)

        for item in items:
            date = item.datetime.strftime('%Y-%m-%d')
            cloud = item.properties.get('eo:cloud_cover', 'N/A')
            tile = item.properties.get('s2:mgrs_tile', 'N/A')
            platform = item.properties.get('platform', 'N/A')
            print(f"    {date:<12} {cloud:<10.1f} {tile:<15} {platform}")

        # 6. 첫 번째 아이템의 Asset 정보
        print(f"\n[6] 첫 번째 아이템의 Asset 목록:")
        first_item = items[0]
        print(f"    아이템 ID: {first_item.id}")
        print(f"    관측 시각: {first_item.datetime}")
        print(f"    공간 범위: {first_item.bbox}")
        print("\n    사용 가능한 Asset:")

        for asset_key, asset in list(first_item.assets.items())[:10]:
            asset_type = asset.media_type or "N/A"
            print(f"      - {asset_key}: {asset_type}")

        if len(first_item.assets) > 10:
            print(f"      ... 외 {len(first_item.assets) - 10}개")

    print("\n" + "=" * 60)
    print("검색 완료!")
    print("=" * 60)

    return items


if __name__ == "__main__":
    items = main()
