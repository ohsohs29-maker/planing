#!/usr/bin/env python3
"""
4.2절 실습: 공간적 특징 추출 기법
GeoAI: 지리공간 인공지능의 이론과 실제

이 스크립트는 GeoPandas를 사용하여 다양한 공간 피처를 추출합니다:
1. 형태학적 피처 (면적, 둘레, 조밀도)
2. 위상적 피처 (인접 객체 수)
3. 거리 기반 피처 (최근접 시설 거리)
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from pathlib import Path


def create_sample_buildings(n_buildings=50, seed=42):
    """샘플 건물 폴리곤 데이터 생성"""
    np.random.seed(seed)

    buildings = []
    for i in range(n_buildings):
        # 무작위 중심점
        cx = np.random.uniform(127.0, 127.1)
        cy = np.random.uniform(37.5, 37.6)

        # 무작위 크기
        width = np.random.uniform(0.0002, 0.001)
        height = np.random.uniform(0.0002, 0.001)

        # 사각형 건물
        poly = box(cx - width/2, cy - height/2,
                   cx + width/2, cy + height/2)
        buildings.append(poly)

    gdf = gpd.GeoDataFrame(
        {'building_id': range(n_buildings)},
        geometry=buildings,
        crs="EPSG:4326"
    )

    # UTM 좌표계로 변환 (면적 계산용)
    gdf = gdf.to_crs("EPSG:32652")

    return gdf


def create_sample_stations(n_stations=5, seed=42):
    """샘플 지하철역 포인트 데이터 생성"""
    np.random.seed(seed + 1)

    stations = []
    for i in range(n_stations):
        x = np.random.uniform(127.0, 127.1)
        y = np.random.uniform(37.5, 37.6)
        stations.append(Point(x, y))

    gdf = gpd.GeoDataFrame(
        {'station_name': [f'Station_{i}' for i in range(n_stations)]},
        geometry=stations,
        crs="EPSG:4326"
    )

    return gdf.to_crs("EPSG:32652")


def extract_morphological_features(gdf):
    """형태학적 피처 추출"""
    print("[1] 형태학적 피처 추출")

    # 면적 (m²)
    gdf['area_m2'] = gdf.geometry.area
    print(f"    면적 범위: {gdf['area_m2'].min():.1f} ~ {gdf['area_m2'].max():.1f} m²")

    # 둘레 (m)
    gdf['perimeter_m'] = gdf.geometry.length
    print(f"    둘레 범위: {gdf['perimeter_m'].min():.1f} ~ {gdf['perimeter_m'].max():.1f} m")

    # 조밀도 (Compactness): 4πA/P² (원형에 가까울수록 1)
    gdf['compactness'] = 4 * np.pi * gdf['area_m2'] / (gdf['perimeter_m'] ** 2)
    print(f"    조밀도 범위: {gdf['compactness'].min():.3f} ~ {gdf['compactness'].max():.3f}")

    # 종횡비 (Aspect Ratio)
    def calc_aspect_ratio(geom):
        minx, miny, maxx, maxy = geom.bounds
        width = maxx - minx
        height = maxy - miny
        return max(width, height) / min(width, height) if min(width, height) > 0 else 1

    gdf['aspect_ratio'] = gdf.geometry.apply(calc_aspect_ratio)
    print(f"    종횡비 범위: {gdf['aspect_ratio'].min():.2f} ~ {gdf['aspect_ratio'].max():.2f}")

    return gdf


def extract_topological_features(gdf):
    """위상적 피처 추출"""
    print("\n[2] 위상적 피처 추출")

    # 버퍼를 사용한 인접 건물 수 (접촉 + 근접)
    def count_neighbors(idx, gdf, buffer_dist=10):
        """버퍼 내 이웃 객체 수"""
        buffered = gdf.loc[idx, 'geometry'].buffer(buffer_dist)
        neighbors = gdf[gdf.geometry.intersects(buffered)]
        return len(neighbors) - 1  # 자기 자신 제외

    gdf['neighbors_10m'] = [count_neighbors(i, gdf, 10) for i in gdf.index]
    print(f"    10m 내 이웃 수: {gdf['neighbors_10m'].min()} ~ {gdf['neighbors_10m'].max()}")

    return gdf


def extract_distance_features(gdf, stations):
    """거리 기반 피처 추출"""
    print("\n[3] 거리 기반 피처 추출")

    # 최근접 지하철역까지 거리
    def nearest_station_distance(building_geom, stations):
        distances = stations.geometry.distance(building_geom.centroid)
        return distances.min()

    gdf['dist_to_station'] = gdf.geometry.apply(
        lambda g: nearest_station_distance(g, stations)
    )
    print(f"    지하철역 거리: {gdf['dist_to_station'].min():.0f} ~ {gdf['dist_to_station'].max():.0f} m")

    # 중심점으로부터의 거리
    centroid = gdf.unary_union.centroid
    gdf['dist_to_center'] = gdf.geometry.centroid.distance(centroid)
    print(f"    중심 거리: {gdf['dist_to_center'].min():.0f} ~ {gdf['dist_to_center'].max():.0f} m")

    return gdf


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("4.2절: 공간적 특징 추출 기법")
    print("=" * 50)

    # 샘플 데이터 생성
    print("\n[데이터 준비]")
    buildings = create_sample_buildings(n_buildings=50)
    stations = create_sample_stations(n_stations=5)
    print(f"    건물 수: {len(buildings)}")
    print(f"    지하철역 수: {len(stations)}")

    # 피처 추출
    buildings = extract_morphological_features(buildings)
    buildings = extract_topological_features(buildings)
    buildings = extract_distance_features(buildings, stations)

    # 결과 요약
    print("\n" + "=" * 50)
    print("추출된 피처 요약")
    print("=" * 50)

    feature_cols = ['area_m2', 'perimeter_m', 'compactness',
                    'aspect_ratio', 'neighbors_10m',
                    'dist_to_station', 'dist_to_center']

    print(buildings[feature_cols].describe().round(2))

    # 결과 저장
    output_dir = Path(__file__).parent.parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "buildings_with_features.gpkg"
    buildings.to_file(output_path, driver="GPKG")
    print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    main()
