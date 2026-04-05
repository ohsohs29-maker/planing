#!/usr/bin/env python3
"""
15-1-kpi-tracking.py
KPI 추적 및 선행/후행 지표 분석

이 코드는 다음을 수행한다:
1. KPI 데이터 시뮬레이션 (선행/후행 지표)
2. 지표 간 상관관계 분석
3. 목표 대비 실적 추적
4. KPI 대시보드 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)


def generate_kpi_data(months: int = 12) -> pd.DataFrame:
    """
    선행 지표와 후행 지표 데이터 생성
    선행 지표가 후행 지표에 1-2개월 선행하는 관계 모델링
    """
    dates = pd.date_range(start='2024-01-01', periods=months, freq='M')

    # 선행 지표 (Leading Indicators)
    # 고객 만족도 (NPS) - 기본 트렌드 + 노이즈
    nps_base = 45 + np.arange(months) * 0.8  # 상승 트렌드
    nps = nps_base + np.random.normal(0, 3, months)
    nps = np.clip(nps, 30, 70)

    # 영업 파이프라인 (억원)
    pipeline_base = 50 + np.arange(months) * 2
    pipeline = pipeline_base + np.random.normal(0, 8, months)
    pipeline = np.clip(pipeline, 30, 100)

    # 웹사이트 트래픽 (만 방문)
    traffic_base = 100 + np.arange(months) * 5
    traffic = traffic_base + np.random.normal(0, 15, months)
    traffic = np.clip(traffic, 80, 200)

    # 직원 몰입도 (%)
    engagement_base = 68 + np.arange(months) * 0.3
    engagement = engagement_base + np.random.normal(0, 2, months)
    engagement = np.clip(engagement, 60, 85)

    # 후행 지표 (Lagging Indicators) - 선행 지표에 1-2개월 지연 반응
    # 매출 (억원) - 파이프라인 + NPS 영향, 1개월 지연
    revenue = np.zeros(months)
    for i in range(months):
        if i == 0:
            revenue[i] = 40
        else:
            # 이전 파이프라인과 NPS가 현재 매출에 영향
            revenue[i] = (0.4 * pipeline[i-1] +
                         0.3 * nps[i-1] +
                         0.3 * revenue[i-1] +
                         np.random.normal(0, 5))
    revenue = np.clip(revenue, 30, 80)

    # 고객 이탈률 (%) - NPS의 역방향, 2개월 지연
    churn = np.zeros(months)
    for i in range(months):
        if i < 2:
            churn[i] = 5.0
        else:
            churn[i] = 12 - 0.12 * nps[i-2] + np.random.normal(0, 0.5)
    churn = np.clip(churn, 2, 10)

    # 영업이익률 (%)
    margin = np.zeros(months)
    for i in range(months):
        if i == 0:
            margin[i] = 15
        else:
            margin[i] = 10 + 0.08 * engagement[i-1] + np.random.normal(0, 1)
    margin = np.clip(margin, 12, 22)

    df = pd.DataFrame({
        '월': dates,
        # 선행 지표
        'NPS': np.round(nps, 1),
        '영업파이프라인_억원': np.round(pipeline, 1),
        '웹트래픽_만': np.round(traffic, 0).astype(int),
        '직원몰입도_%': np.round(engagement, 1),
        # 후행 지표
        '매출_억원': np.round(revenue, 1),
        '고객이탈률_%': np.round(churn, 2),
        '영업이익률_%': np.round(margin, 1)
    })

    return df


def set_kpi_targets() -> dict:
    """KPI 목표 설정"""
    return {
        # 선행 지표 목표
        'NPS': {'target': 55, 'direction': 'higher', 'weight': 0.15},
        '영업파이프라인_억원': {'target': 70, 'direction': 'higher', 'weight': 0.15},
        '웹트래픽_만': {'target': 150, 'direction': 'higher', 'weight': 0.10},
        '직원몰입도_%': {'target': 75, 'direction': 'higher', 'weight': 0.10},
        # 후행 지표 목표
        '매출_억원': {'target': 60, 'direction': 'higher', 'weight': 0.25},
        '고객이탈률_%': {'target': 4, 'direction': 'lower', 'weight': 0.10},
        '영업이익률_%': {'target': 18, 'direction': 'higher', 'weight': 0.15}
    }


def calculate_achievement(df: pd.DataFrame, targets: dict) -> pd.DataFrame:
    """목표 대비 달성률 계산"""
    latest = df.iloc[-1]

    results = []
    for kpi, config in targets.items():
        actual = latest[kpi]
        target = config['target']

        if config['direction'] == 'higher':
            achievement = (actual / target) * 100
            gap = actual - target
        else:  # lower is better
            achievement = (target / actual) * 100 if actual > 0 else 100
            gap = target - actual

        status = '달성' if achievement >= 100 else ('주의' if achievement >= 80 else '미달')

        results.append({
            'KPI': kpi,
            '목표': target,
            '실적': actual,
            '달성률_%': round(achievement, 1),
            '갭': round(gap, 1),
            '상태': status,
            '가중치': config['weight']
        })

    return pd.DataFrame(results)


def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """선행-후행 지표 간 상관관계 분석"""
    leading = ['NPS', '영업파이프라인_억원', '웹트래픽_만', '직원몰입도_%']
    lagging = ['매출_억원', '고객이탈률_%', '영업이익률_%']

    correlations = []

    for lead in leading:
        for lag in lagging:
            # 동시 상관관계
            corr_0, p_0 = stats.pearsonr(df[lead], df[lag])

            # 1개월 지연 상관관계
            if len(df) > 1:
                corr_1, p_1 = stats.pearsonr(df[lead].iloc[:-1], df[lag].iloc[1:])
            else:
                corr_1, p_1 = 0, 1

            # 2개월 지연 상관관계
            if len(df) > 2:
                corr_2, p_2 = stats.pearsonr(df[lead].iloc[:-2], df[lag].iloc[2:])
            else:
                corr_2, p_2 = 0, 1

            # 가장 강한 상관관계 선택
            correlations.append({
                '선행지표': lead,
                '후행지표': lag,
                '동시상관': round(corr_0, 3),
                '1개월지연': round(corr_1, 3),
                '2개월지연': round(corr_2, 3),
                '최대상관': round(max(abs(corr_0), abs(corr_1), abs(corr_2)), 3)
            })

    return pd.DataFrame(correlations)


def calculate_composite_score(achievement_df: pd.DataFrame) -> float:
    """가중 평균 종합 점수 계산"""
    weighted_sum = 0
    total_weight = 0

    for _, row in achievement_df.iterrows():
        weighted_sum += row['달성률_%'] * row['가중치']
        total_weight += row['가중치']

    return round(weighted_sum / total_weight, 1) if total_weight > 0 else 0


def visualize_kpi_dashboard(df: pd.DataFrame, targets: dict):
    """KPI 대시보드 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 선행 지표 추이
    ax1 = axes[0, 0]
    ax1.plot(df['월'], df['NPS'], 'b-o', label='NPS', linewidth=2)
    ax1.axhline(y=targets['NPS']['target'], color='b', linestyle='--', alpha=0.5)
    ax1.set_ylabel('NPS', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(df['월'], df['영업파이프라인_억원'], 'g-s', label='파이프라인', linewidth=2)
    ax1_twin.axhline(y=targets['영업파이프라인_억원']['target'], color='g', linestyle='--', alpha=0.5)
    ax1_twin.set_ylabel('파이프라인 (억원)', color='g')
    ax1_twin.tick_params(axis='y', labelcolor='g')

    ax1.set_title('선행 지표 추이')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    # 2. 후행 지표 추이
    ax2 = axes[0, 1]
    ax2.plot(df['월'], df['매출_억원'], 'r-o', label='매출', linewidth=2)
    ax2.axhline(y=targets['매출_억원']['target'], color='r', linestyle='--', alpha=0.5)
    ax2.fill_between(df['월'], df['매출_억원'], targets['매출_억원']['target'],
                     where=df['매출_억원'] >= targets['매출_억원']['target'],
                     alpha=0.3, color='green')
    ax2.fill_between(df['월'], df['매출_억원'], targets['매출_억원']['target'],
                     where=df['매출_억원'] < targets['매출_억원']['target'],
                     alpha=0.3, color='red')
    ax2.set_title('매출 목표 대비 실적')
    ax2.set_ylabel('매출 (억원)')
    ax2.legend()

    # 3. 목표 달성률 바 차트
    ax3 = axes[1, 0]
    achievement = calculate_achievement(df, targets)
    colors = ['green' if s == '달성' else ('orange' if s == '주의' else 'red')
              for s in achievement['상태']]
    bars = ax3.barh(achievement['KPI'], achievement['달성률_%'], color=colors)
    ax3.axvline(x=100, color='black', linestyle='--', linewidth=2)
    ax3.set_xlabel('달성률 (%)')
    ax3.set_title('KPI 목표 달성률')

    for bar, val in zip(bars, achievement['달성률_%']):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}%', va='center')

    # 4. 선행-후행 지표 산점도
    ax4 = axes[1, 1]
    ax4.scatter(df['NPS'].iloc[:-1], df['매출_억원'].iloc[1:],
               c=range(len(df)-1), cmap='viridis', s=100)
    ax4.set_xlabel('NPS (t)')
    ax4.set_ylabel('매출 (t+1)')
    ax4.set_title('NPS → 매출 선행 관계')

    # 추세선 추가
    z = np.polyfit(df['NPS'].iloc[:-1], df['매출_억원'].iloc[1:], 1)
    p = np.poly1d(z)
    ax4.plot(df['NPS'].iloc[:-1].sort_values(),
            p(df['NPS'].iloc[:-1].sort_values()),
            'r--', linewidth=2, label=f'추세선 (기울기: {z[0]:.2f})')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('../data/kpi_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("KPI 대시보드 저장: ../data/kpi_dashboard.png")


def main():
    print("=" * 60)
    print("KPI 추적 및 선행/후행 지표 분석")
    print("=" * 60)

    # 1. 데이터 생성
    print("\n[1] KPI 데이터 생성 (12개월)")
    df = generate_kpi_data(12)
    print(df.to_string(index=False))

    # 2. 목표 설정
    print("\n[2] KPI 목표 설정")
    targets = set_kpi_targets()
    for kpi, config in targets.items():
        direction = '↑' if config['direction'] == 'higher' else '↓'
        print(f"  - {kpi}: 목표 {config['target']} {direction} (가중치 {config['weight']*100:.0f}%)")

    # 3. 달성률 분석
    print("\n[3] 목표 대비 달성률 (최근 월)")
    achievement = calculate_achievement(df, targets)
    print(achievement.to_string(index=False))

    # 4. 종합 점수
    composite = calculate_composite_score(achievement)
    print(f"\n종합 KPI 점수: {composite}%")
    if composite >= 100:
        print("→ 전체 목표 달성!")
    elif composite >= 85:
        print("→ 양호 (일부 지표 보완 필요)")
    else:
        print("→ 미흡 (집중 관리 필요)")

    # 5. 상관관계 분석
    print("\n[4] 선행-후행 지표 상관관계")
    correlations = analyze_correlations(df)
    # 최대상관 기준 상위 5개
    top_corr = correlations.nlargest(5, '최대상관')
    print(top_corr.to_string(index=False))

    print("\n[인사이트]")
    for _, row in top_corr.iterrows():
        delays = [('동시', row['동시상관']),
                  ('1개월', row['1개월지연']),
                  ('2개월', row['2개월지연'])]
        best_delay = max(delays, key=lambda x: abs(x[1]))
        print(f"  - {row['선행지표']} → {row['후행지표']}: "
              f"{best_delay[0]} 지연에서 상관 {best_delay[1]:.2f}")

    # 6. 시각화
    print("\n[5] KPI 대시보드 생성")
    visualize_kpi_dashboard(df, targets)

    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
