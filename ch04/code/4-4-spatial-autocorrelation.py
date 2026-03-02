#!/usr/bin/env python3
"""
4.4절 실습: 공간 자기상관과 Moran's I
GeoAI: 지리공간 인공지능의 이론과 실제

이 스크립트는 다음을 수행합니다:
1. 공간 자기상관 개념 시연
2. Global Moran's I 계산
3. Local Moran's I (LISA) 분석
4. 핫스팟/콜드스팟 시각화
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from esda.moran import Moran, Moran_Local
    from libpysal.weights import KNN, Queen
    from splot.esda import moran_scatterplot, lisa_cluster
    PYSAL_AVAILABLE = True
except ImportError:
    PYSAL_AVAILABLE = False
    print("경고: PySAL/splot이 설치되지 않았습니다.")
    print("설치: pip install esda libpysal splot")


def generate_clustered_data(n_points=200, seed=42):
    """
    공간적으로 클러스터된 데이터 생성
    - 명확한 핫스팟과 콜드스팟 포함
    """
    np.random.seed(seed)

    points = []
    values = []

    # 핫스팟 클러스터 (높은 값)
    for _ in range(50):
        x = np.random.normal(127.05, 0.01)
        y = np.random.normal(37.55, 0.01)
        v = np.random.normal(80, 5)  # 높은 값
        points.append(Point(x, y))
        values.append(v)

    # 콜드스팟 클러스터 (낮은 값)
    for _ in range(50):
        x = np.random.normal(127.02, 0.01)
        y = np.random.normal(37.52, 0.01)
        v = np.random.normal(20, 5)  # 낮은 값
        points.append(Point(x, y))
        values.append(v)

    # 배경 (중간 값)
    for _ in range(100):
        x = np.random.uniform(127.0, 127.1)
        y = np.random.uniform(37.5, 37.6)
        v = np.random.normal(50, 10)
        points.append(Point(x, y))
        values.append(v)

    gdf = gpd.GeoDataFrame(
        {'value': values},
        geometry=points,
        crs="EPSG:4326"
    )

    return gdf


def calculate_global_morans_i(gdf, column='value'):
    """Global Moran's I 계산"""
    print("\n[Global Moran's I]")
    print("-" * 40)

    # K-최근접 이웃 가중치 행렬
    w = KNN.from_dataframe(gdf, k=8)
    w.transform = 'r'  # 행 표준화

    # Moran's I 계산
    mi = Moran(gdf[column].values, w, permutations=999)

    print(f"Moran's I 값: {mi.I:.4f}")
    print(f"기대값 (E[I]): {mi.EI:.4f}")
    print(f"Z-score: {mi.z_sim:.4f}")
    print(f"p-value: {mi.p_sim:.4f}")

    # 해석
    print("\n[해석]")
    if mi.p_sim < 0.001:
        sig = "매우 유의 (p < 0.001)"
    elif mi.p_sim < 0.01:
        sig = "유의 (p < 0.01)"
    elif mi.p_sim < 0.05:
        sig = "유의 (p < 0.05)"
    else:
        sig = "유의하지 않음 (p >= 0.05)"
    print(f"통계적 유의성: {sig}")

    if mi.I > 0.7:
        print("공간 패턴: 강한 양의 자기상관 (명확한 클러스터링)")
    elif mi.I > 0.3:
        print("공간 패턴: 중간 양의 자기상관 (클러스터 존재)")
    elif mi.I > 0:
        print("공간 패턴: 약한 양의 자기상관")
    elif mi.I > -0.3:
        print("공간 패턴: 무작위에 가까움")
    else:
        print("공간 패턴: 분산 패턴 (체스판형)")

    return mi, w


def calculate_local_morans_i(gdf, w, column='value'):
    """Local Moran's I (LISA) 계산"""
    print("\n[Local Moran's I (LISA)]")
    print("-" * 40)

    # Local Moran 계산
    lisa = Moran_Local(gdf[column].values, w, permutations=999)

    # 결과를 GeoDataFrame에 추가
    gdf['lisa_i'] = lisa.Is
    gdf['lisa_p'] = lisa.p_sim
    gdf['lisa_q'] = lisa.q  # 사분면 (1=HH, 2=LH, 3=LL, 4=HL)

    # 유의한 클러스터만 분류
    sig_threshold = 0.05
    gdf['cluster_type'] = 'Not Significant'

    sig_mask = gdf['lisa_p'] < sig_threshold
    gdf.loc[sig_mask & (gdf['lisa_q'] == 1), 'cluster_type'] = 'HH (Hot Spot)'
    gdf.loc[sig_mask & (gdf['lisa_q'] == 2), 'cluster_type'] = 'LH (Low-High)'
    gdf.loc[sig_mask & (gdf['lisa_q'] == 3), 'cluster_type'] = 'LL (Cold Spot)'
    gdf.loc[sig_mask & (gdf['lisa_q'] == 4), 'cluster_type'] = 'HL (High-Low)'

    # 클러스터 통계
    print("클러스터 유형별 개수:")
    print(gdf['cluster_type'].value_counts())

    return gdf, lisa


def visualize_results(gdf, mi, lisa, output_path):
    """결과 시각화"""
    print(f"\n[시각화] {output_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 원본 값 분포
    ax1 = axes[0]
    scatter = ax1.scatter(
        gdf.geometry.x, gdf.geometry.y,
        c=gdf['value'], cmap='RdYlBu_r', s=30, alpha=0.7
    )
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Original Values')
    plt.colorbar(scatter, ax=ax1, label='Value')

    # 2. Moran Scatter Plot
    ax2 = axes[1]
    # 표준화된 값과 공간 래그
    z = (gdf['value'] - gdf['value'].mean()) / gdf['value'].std()
    lag_z = np.array([z.iloc[list(lisa.w.neighbors[i])].mean() for i in range(len(z))])

    ax2.scatter(z, lag_z, c='steelblue', s=20, alpha=0.6)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # 회귀선 (기울기 = Moran's I)
    x_line = np.linspace(z.min(), z.max(), 100)
    y_line = mi.I * x_line
    ax2.plot(x_line, y_line, color='red', linewidth=2,
             label=f"Moran's I = {mi.I:.3f}")
    ax2.set_xlabel('Standardized Value')
    ax2.set_ylabel('Spatial Lag')
    ax2.set_title("Moran's I Scatter Plot")
    ax2.legend()

    # 사분면 레이블
    ax2.text(2, 1.5, 'HH', fontsize=12, fontweight='bold', color='red')
    ax2.text(-2, 1.5, 'LH', fontsize=12, fontweight='bold', color='purple')
    ax2.text(-2, -1.5, 'LL', fontsize=12, fontweight='bold', color='blue')
    ax2.text(2, -1.5, 'HL', fontsize=12, fontweight='bold', color='orange')

    # 3. LISA 클러스터 맵
    ax3 = axes[2]
    color_map = {
        'HH (Hot Spot)': 'red',
        'LL (Cold Spot)': 'blue',
        'LH (Low-High)': 'purple',
        'HL (High-Low)': 'orange',
        'Not Significant': 'lightgray'
    }
    colors = [color_map[ct] for ct in gdf['cluster_type']]

    ax3.scatter(gdf.geometry.x, gdf.geometry.y, c=colors, s=30, alpha=0.7)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title('LISA Cluster Map')

    # 범례
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l)
                       for l, c in color_map.items() if l in gdf['cluster_type'].values]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"저장 완료: {output_path}")


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("4.4절: 공간 자기상관과 Moran's I")
    print("=" * 50)

    if not PYSAL_AVAILABLE:
        print("\nPySAL이 필요합니다. 설치 후 다시 실행하세요.")
        print("pip install esda libpysal splot")
        return

    output_dir = Path(__file__).parent.parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 클러스터된 데이터 생성
    print("\n[1] 샘플 데이터 생성")
    gdf = generate_clustered_data(n_points=200)
    print(f"    데이터 포인트: {len(gdf)}")
    print(f"    값 범위: {gdf['value'].min():.1f} ~ {gdf['value'].max():.1f}")

    # 2. Global Moran's I
    print("\n[2] Global Moran's I 계산")
    mi, w = calculate_global_morans_i(gdf)

    # 3. Local Moran's I
    print("\n[3] Local Moran's I 계산")
    gdf, lisa = calculate_local_morans_i(gdf, w)

    # 4. 시각화
    print("\n[4] 결과 시각화")
    visualize_results(gdf, mi, lisa, output_dir / "morans_i_analysis.png")

    # 5. 요약
    print("\n" + "=" * 50)
    print("분석 완료!")
    print("=" * 50)
    print(f"\n[결론]")
    print(f"  Moran's I = {mi.I:.4f}")
    if mi.I > 0.3:
        print("  → 강한 공간 자기상관이 존재합니다.")
        print("  → ML 모델에서 Random CV 사용 시 성능 과대추정 위험!")
        print("  → Block CV 또는 Spatial CV 사용을 권장합니다.")


if __name__ == "__main__":
    main()
