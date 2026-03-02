#!/usr/bin/env python3
"""
4장 종합 실습: 공간 피처 추출과 공간 교차검증
GeoAI: 지리공간 인공지능의 이론과 실제

이 스크립트는 다음을 수행합니다:
1. 공간 피처 추출 (형태학적, 거리 기반)
2. Moran's I로 공간 자기상관 측정
3. Random CV vs Block CV 성능 비교
4. 결과 시각화
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from sklearn.model_selection import cross_val_score, GroupKFold, KFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from pathlib import Path

# PySAL 공간 자기상관 분석
try:
    from esda.moran import Moran
    from libpysal.weights import Queen, KNN
    PYSAL_AVAILABLE = True
except ImportError:
    PYSAL_AVAILABLE = False
    print("경고: PySAL이 설치되지 않았습니다. Moran's I 계산을 건너뜁니다.")


def generate_sample_data(n_points=500, seed=42):
    """
    공간적으로 자기상관된 샘플 데이터 생성

    실제 데이터 대신 시뮬레이션 데이터를 사용하여
    공간 자기상관의 효과를 명확하게 보여줍니다.
    """
    np.random.seed(seed)

    # 서울 지역 범위 (UTM 52N 좌표)
    x_min, x_max = 940000, 970000  # 약 30km
    y_min, y_max = 4130000, 4160000  # 약 30km

    # 무작위 좌표 생성
    x = np.random.uniform(x_min, x_max, n_points)
    y = np.random.uniform(y_min, y_max, n_points)

    # 공간적으로 자기상관된 타겟 변수 생성
    # (가까운 점들이 유사한 값을 가지도록)

    # 기본 공간 트렌드 (동서 그래디언트)
    spatial_trend = (x - x_min) / (x_max - x_min) * 50

    # 지역적 클러스터 효과 추가
    cluster_centers = [
        (950000, 4145000, 30),  # (x, y, intensity)
        (960000, 4135000, -20),
        (945000, 4150000, 25),
    ]

    cluster_effect = np.zeros(n_points)
    for cx, cy, intensity in cluster_centers:
        distances = np.sqrt((x - cx)**2 + (y - cy)**2)
        cluster_effect += intensity * np.exp(-distances / 5000)

    # 타겟 변수: 트렌드 + 클러스터 + 노이즈
    target = spatial_trend + cluster_effect + np.random.normal(0, 5, n_points)

    # 피처 생성 (타겟과 상관있는 피처들)
    feature1 = spatial_trend * 0.8 + np.random.normal(0, 3, n_points)
    feature2 = cluster_effect * 0.6 + np.random.normal(0, 2, n_points)
    feature3 = np.random.normal(50, 10, n_points)  # 무작위 피처

    # GeoDataFrame 생성
    geometry = [Point(xi, yi) for xi, yi in zip(x, y)]
    gdf = gpd.GeoDataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'target': target,
    }, geometry=geometry, crs="EPSG:32652")

    return gdf


def extract_spatial_features(gdf):
    """공간적 특징 추출"""
    print("[피처 추출] 공간적 특징 계산...")

    # 1. 좌표 기반 피처
    gdf['x_coord'] = gdf.geometry.x
    gdf['y_coord'] = gdf.geometry.y

    # 2. 거리 기반 피처 (중심점으로부터의 거리)
    centroid = gdf.unary_union.centroid
    gdf['dist_to_center'] = gdf.geometry.distance(centroid)

    # 3. 밀도 피처 (반경 2km 내 이웃 수)
    def count_neighbors(geom, all_geoms, radius=2000):
        buffer = geom.buffer(radius)
        return sum(1 for g in all_geoms if g.within(buffer) and g != geom)

    # 성능을 위해 샘플링
    if len(gdf) > 100:
        gdf['neighbor_count'] = 0  # 간소화
    else:
        gdf['neighbor_count'] = gdf.geometry.apply(
            lambda g: count_neighbors(g, gdf.geometry.tolist())
        )

    feature_cols = ['feature1', 'feature2', 'feature3',
                    'x_coord', 'y_coord', 'dist_to_center']

    print(f"       추출된 피처: {feature_cols}")
    return gdf, feature_cols


def calculate_morans_i(gdf, column='target'):
    """Moran's I 공간 자기상관 계산"""
    if not PYSAL_AVAILABLE:
        return None, None

    print(f"[Moran's I] '{column}' 변수의 공간 자기상관 계산...")

    # K-최근접 이웃 가중치 행렬 (Queen이 폴리곤용이므로 KNN 사용)
    w = KNN.from_dataframe(gdf, k=8)
    w.transform = 'r'  # 행 표준화

    # Moran's I 계산
    mi = Moran(gdf[column].values, w, permutations=999)

    print(f"       Moran's I: {mi.I:.4f}")
    print(f"       p-value: {mi.p_sim:.4f}")
    print(f"       해석: ", end="")

    if mi.p_sim < 0.05:
        if mi.I > 0.3:
            print("강한 양의 공간 자기상관 (클러스터링)")
        elif mi.I > 0:
            print("약한 양의 공간 자기상관")
        else:
            print("음의 공간 자기상관 (분산 패턴)")
    else:
        print("통계적으로 유의하지 않음 (무작위 분포)")

    return mi.I, mi.p_sim


def create_spatial_blocks(gdf, block_size=5000):
    """격자 기반 공간 블록 ID 생성"""
    print(f"[블록 생성] 블록 크기: {block_size}m")

    # 격자 ID 계산
    gdf['block_id'] = (
        (gdf.geometry.x // block_size).astype(int) * 10000 +
        (gdf.geometry.y // block_size).astype(int)
    )

    n_blocks = gdf['block_id'].nunique()
    print(f"       생성된 블록 수: {n_blocks}")

    return gdf


def compare_cv_methods(gdf, feature_cols, n_splits=5):
    """Random CV vs Block CV 성능 비교"""
    print("\n[교차검증 비교] Random CV vs Block CV")
    print("=" * 50)

    X = gdf[feature_cols].values
    y = gdf['target'].values

    model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                   random_state=42, n_jobs=-1)

    # 1. Random K-Fold CV
    print("\n[1] Random K-Fold CV")
    random_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    random_scores = cross_val_score(model, X, y, cv=random_cv, scoring='r2')
    print(f"    R² scores: {random_scores.round(3)}")
    print(f"    평균 R²: {random_scores.mean():.3f} ± {random_scores.std():.3f}")

    # 2. Spatial Block CV
    print("\n[2] Spatial Block CV")
    block_cv = GroupKFold(n_splits=min(n_splits, gdf['block_id'].nunique()))
    block_scores = cross_val_score(model, X, y, cv=block_cv,
                                   groups=gdf['block_id'], scoring='r2')
    print(f"    R² scores: {block_scores.round(3)}")
    print(f"    평균 R²: {block_scores.mean():.3f} ± {block_scores.std():.3f}")

    # 3. 비교 분석
    print("\n[비교 결과]")
    diff = random_scores.mean() - block_scores.mean()
    print(f"    Random CV - Block CV = {diff:.3f}")

    if diff > 0.05:
        print("    ⚠️  Random CV가 성능을 과대 추정하고 있습니다!")
        print("    → 공간 데이터에서는 Block CV 사용을 권장합니다.")
    else:
        print("    ✓ 두 방법 간 큰 차이가 없습니다.")

    return random_scores, block_scores


def visualize_results(gdf, random_scores, block_scores, output_path):
    """결과 시각화"""
    print(f"\n[시각화] {output_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 공간 분포 (타겟 변수)
    ax1 = axes[0]
    scatter = ax1.scatter(
        gdf.geometry.x, gdf.geometry.y,
        c=gdf['target'], cmap='RdYlGn', s=20, alpha=0.7
    )
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Target Variable Spatial Distribution')
    plt.colorbar(scatter, ax=ax1, label='Target')

    # 2. 블록 분포
    ax2 = axes[1]
    unique_blocks = gdf['block_id'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_blocks)))
    block_color_map = dict(zip(unique_blocks, colors))
    block_colors = [block_color_map[b] for b in gdf['block_id']]

    ax2.scatter(gdf.geometry.x, gdf.geometry.y, c=block_colors, s=20, alpha=0.7)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Spatial Block Assignment')

    # 3. CV 성능 비교
    ax3 = axes[2]
    positions = [1, 2]
    bp = ax3.boxplot([random_scores, block_scores], positions=positions,
                      widths=0.6, patch_artist=True)

    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')

    ax3.set_xticklabels(['Random CV', 'Block CV'])
    ax3.set_ylabel('R² Score')
    ax3.set_title('Cross-Validation Performance Comparison')
    ax3.axhline(y=random_scores.mean(), color='red', linestyle='--', alpha=0.5)
    ax3.axhline(y=block_scores.mean(), color='green', linestyle='--', alpha=0.5)

    # 평균값 표시
    ax3.text(1, random_scores.mean() + 0.02, f'{random_scores.mean():.3f}',
             ha='center', color='red')
    ax3.text(2, block_scores.mean() + 0.02, f'{block_scores.mean():.3f}',
             ha='center', color='green')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"       저장 완료: {output_path}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("4장 종합 실습: 공간 피처 추출과 공간 교차검증")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 샘플 데이터 생성
    print("\n[1] 샘플 데이터 생성...")
    gdf = generate_sample_data(n_points=500)
    print(f"    데이터 포인트 수: {len(gdf)}")
    print(f"    좌표계: {gdf.crs}")

    # 2. 공간 피처 추출
    print("\n[2] 공간 피처 추출...")
    gdf, feature_cols = extract_spatial_features(gdf)

    # 3. Moran's I 계산
    print("\n[3] 공간 자기상관 분석...")
    morans_i, p_value = calculate_morans_i(gdf, 'target')

    # 4. 공간 블록 생성
    print("\n[4] 공간 블록 생성...")
    gdf = create_spatial_blocks(gdf, block_size=5000)

    # 5. CV 비교
    print("\n[5] 교차검증 비교...")
    random_scores, block_scores = compare_cv_methods(gdf, feature_cols)

    # 6. 시각화
    print("\n[6] 결과 시각화...")
    visualize_results(gdf, random_scores, block_scores,
                      output_dir / "spatial_cv_comparison.png")

    # 7. 요약
    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)

    print("\n[요약]")
    print(f"  - 데이터 포인트: {len(gdf)}개")
    if morans_i is not None:
        print(f"  - Moran's I: {morans_i:.4f} (p={p_value:.4f})")
    print(f"  - Random CV R²: {random_scores.mean():.3f} ± {random_scores.std():.3f}")
    print(f"  - Block CV R²: {block_scores.mean():.3f} ± {block_scores.std():.3f}")
    print(f"  - 과대추정 정도: {(random_scores.mean() - block_scores.mean()):.3f}")

    print("\n[결론]")
    if random_scores.mean() - block_scores.mean() > 0.05:
        print("  ⚠️  공간 자기상관으로 인해 Random CV가 성능을 과대 추정합니다.")
        print("  → 공간 데이터에서는 반드시 Block CV를 사용해야 합니다.")

    print(f"\n  출력 파일: {output_dir}")


if __name__ == "__main__":
    main()
