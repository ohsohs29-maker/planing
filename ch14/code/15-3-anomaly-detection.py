#!/usr/bin/env python3
"""
15-3-anomaly-detection.py
AI 기반 이상 탐지 및 조기 경보 시스템

이 코드는 다음을 수행한다:
1. 시계열 KPI 데이터 생성
2. 통계적 이상 탐지 (Z-score, IQR)
3. 이동평균 기반 이상 탐지
4. 조기 경보 시스템 구현
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)


class AlertLevel(Enum):
    """경보 수준"""
    NORMAL = "정상"
    WATCH = "관심"
    WARNING = "경고"
    CRITICAL = "위험"


@dataclass
class Alert:
    """조기 경보"""
    timestamp: pd.Timestamp
    kpi: str
    value: float
    expected: float
    deviation: float
    level: AlertLevel
    message: str


def generate_kpi_timeseries(days: int = 180) -> pd.DataFrame:
    """
    일별 KPI 시계열 데이터 생성
    정상 패턴 + 이상치 + 트렌드 변화 포함
    """
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')

    # 기본 매출 패턴 (주간 계절성 + 월간 트렌드)
    trend = np.linspace(100, 120, days)  # 상승 트렌드
    weekly = 10 * np.sin(2 * np.pi * np.arange(days) / 7)  # 주간 패턴
    noise = np.random.normal(0, 5, days)

    daily_sales = trend + weekly + noise

    # 이상치 주입
    anomaly_indices = [30, 45, 75, 120, 150]  # 이상치 발생일
    for idx in anomaly_indices:
        if idx < days:
            daily_sales[idx] = daily_sales[idx] * (1.5 if np.random.random() > 0.5 else 0.5)

    # 트렌드 급변 (90일 이후 하락)
    if days > 100:
        daily_sales[90:100] = daily_sales[90:100] * 0.85

    # 웹 트래픽 (매출과 상관)
    web_traffic = daily_sales * 50 + np.random.normal(0, 200, days)

    # 고객 문의 수 (역상관 - 문제 발생시 증가)
    base_inquiries = 50 + np.random.normal(0, 10, days)
    # 이상치 발생일에 문의 증가
    for idx in anomaly_indices:
        if idx < days:
            base_inquiries[idx:idx+3] = base_inquiries[idx:idx+3] + 30

    # 시스템 응답 시간 (ms)
    response_time = 200 + np.random.exponential(20, days)
    # 일부 구간에서 성능 저하
    if days > 60:
        response_time[55:65] = response_time[55:65] + 100

    df = pd.DataFrame({
        '날짜': dates,
        '일매출_백만원': np.round(daily_sales, 1),
        '웹트래픽_천건': np.round(web_traffic / 1000, 1),
        '고객문의_건': np.round(base_inquiries).astype(int),
        '응답시간_ms': np.round(response_time, 0).astype(int)
    })

    return df


def detect_zscore_anomalies(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Z-score 기반 이상 탐지"""
    z_scores = np.abs(stats.zscore(series))
    return z_scores > threshold


def detect_iqr_anomalies(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """IQR 기반 이상 탐지"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return (series < lower) | (series > upper)


def detect_ma_anomalies(series: pd.Series, window: int = 7,
                        std_factor: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """이동평균 기반 이상 탐지"""
    ma = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    upper = ma + std_factor * std
    lower = ma - std_factor * std
    anomalies = (series > upper) | (series < lower)
    return anomalies, upper, lower


def detect_with_isolation_forest(df: pd.DataFrame, columns: List[str],
                                  contamination: float = 0.05) -> pd.Series:
    """Isolation Forest 기반 다변량 이상 탐지"""
    X = df[columns].values
    clf = IsolationForest(contamination=contamination, random_state=42)
    predictions = clf.fit_predict(X)
    return pd.Series(predictions == -1, index=df.index)


def generate_alerts(df: pd.DataFrame, anomalies: dict) -> List[Alert]:
    """이상 탐지 결과를 조기 경보로 변환"""
    alerts = []

    for kpi, anomaly_mask in anomalies.items():
        anomaly_dates = df.loc[anomaly_mask, '날짜']

        for date in anomaly_dates:
            idx = df[df['날짜'] == date].index[0]
            value = df.loc[idx, kpi]

            # 기대값 (이전 7일 평균)
            start_idx = max(0, idx - 7)
            expected = df.loc[start_idx:idx-1, kpi].mean() if idx > 0 else value

            # 편차
            deviation = ((value - expected) / expected * 100) if expected != 0 else 0

            # 경보 수준 결정
            abs_dev = abs(deviation)
            if abs_dev > 50:
                level = AlertLevel.CRITICAL
            elif abs_dev > 30:
                level = AlertLevel.WARNING
            elif abs_dev > 15:
                level = AlertLevel.WATCH
            else:
                level = AlertLevel.NORMAL

            # 메시지 생성
            direction = "급증" if deviation > 0 else "급감"
            message = f"{kpi} {direction} ({deviation:+.1f}%)"

            alerts.append(Alert(
                timestamp=date,
                kpi=kpi,
                value=value,
                expected=expected,
                deviation=deviation,
                level=level,
                message=message
            ))

    # 시간순 정렬
    alerts.sort(key=lambda x: x.timestamp)
    return alerts


def visualize_anomalies(df: pd.DataFrame, kpi: str, anomalies: pd.Series,
                        upper: pd.Series, lower: pd.Series):
    """이상 탐지 결과 시각화"""
    fig, ax = plt.subplots(figsize=(14, 5))

    # 정상 데이터
    normal_mask = ~anomalies
    ax.plot(df.loc[normal_mask, '날짜'], df.loc[normal_mask, kpi],
            'b-', label='정상', alpha=0.7)

    # 이상치
    ax.scatter(df.loc[anomalies, '날짜'], df.loc[anomalies, kpi],
               c='red', s=100, label='이상치', zorder=5)

    # 신뢰 구간
    ax.fill_between(df['날짜'], lower, upper, alpha=0.2, color='blue',
                    label='정상 범위 (±2σ)')

    ax.set_title(f'{kpi} 이상 탐지 결과')
    ax.set_xlabel('날짜')
    ax.set_ylabel(kpi)
    ax.legend()

    plt.tight_layout()
    filename = f'../data/anomaly_{kpi.replace("/", "_")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    return filename


def create_alert_dashboard(alerts: List[Alert]) -> pd.DataFrame:
    """경보 대시보드 생성"""
    if not alerts:
        return pd.DataFrame()

    data = []
    for alert in alerts:
        data.append({
            '일시': alert.timestamp.strftime('%Y-%m-%d'),
            'KPI': alert.kpi,
            '값': alert.value,
            '기대값': round(alert.expected, 1),
            '편차_%': round(alert.deviation, 1),
            '수준': alert.level.value,
            '메시지': alert.message
        })

    return pd.DataFrame(data)


def analyze_alert_patterns(alerts: List[Alert]) -> dict:
    """경보 패턴 분석"""
    if not alerts:
        return {}

    # 수준별 집계
    level_counts = {}
    for level in AlertLevel:
        level_counts[level.value] = sum(1 for a in alerts if a.level == level)

    # KPI별 집계
    kpi_counts = {}
    for alert in alerts:
        kpi_counts[alert.kpi] = kpi_counts.get(alert.kpi, 0) + 1

    # 주간별 집계
    weekly_counts = {}
    for alert in alerts:
        week = alert.timestamp.isocalendar()[1]
        weekly_counts[week] = weekly_counts.get(week, 0) + 1

    return {
        'total': len(alerts),
        'by_level': level_counts,
        'by_kpi': kpi_counts,
        'by_week': weekly_counts,
        'critical_ratio': level_counts.get(AlertLevel.CRITICAL.value, 0) / len(alerts) * 100
    }


def main():
    print("=" * 60)
    print("AI 기반 이상 탐지 및 조기 경보 시스템")
    print("=" * 60)

    # 1. 데이터 생성
    print("\n[1] KPI 시계열 데이터 생성 (180일)")
    df = generate_kpi_timeseries(180)
    print(f"  - 기간: {df['날짜'].min().strftime('%Y-%m-%d')} ~ {df['날짜'].max().strftime('%Y-%m-%d')}")
    print(f"  - 데이터 수: {len(df)}건")
    print("\n최근 10일 데이터:")
    print(df.tail(10).to_string(index=False))

    # 2. 이상 탐지 실행
    print("\n[2] 이상 탐지 알고리즘 적용")
    kpi = '일매출_백만원'

    # Z-score
    zscore_anomalies = detect_zscore_anomalies(df[kpi])
    print(f"  - Z-score (threshold=3.0): {zscore_anomalies.sum()}건")

    # IQR
    iqr_anomalies = detect_iqr_anomalies(df[kpi])
    print(f"  - IQR (factor=1.5): {iqr_anomalies.sum()}건")

    # 이동평균
    ma_anomalies, upper, lower = detect_ma_anomalies(df[kpi])
    print(f"  - 이동평균 (window=7, ±2σ): {ma_anomalies.sum()}건")

    # Isolation Forest (다변량)
    feature_cols = ['일매출_백만원', '웹트래픽_천건', '고객문의_건']
    if_anomalies = detect_with_isolation_forest(df, feature_cols)
    print(f"  - Isolation Forest (다변량): {if_anomalies.sum()}건")

    # 3. 이상 탐지 비교
    print("\n[3] 알고리즘별 탐지 결과 비교")
    comparison = pd.DataFrame({
        'Z-score': zscore_anomalies,
        'IQR': iqr_anomalies,
        '이동평균': ma_anomalies,
        'IsolationForest': if_anomalies
    })

    # 2개 이상 알고리즘에서 탐지된 날
    consensus = comparison.sum(axis=1) >= 2
    print(f"  - 합의된 이상치 (2개 이상 알고리즘): {consensus.sum()}건")

    # 4. 조기 경보 생성
    print("\n[4] 조기 경보 생성")
    anomalies_dict = {
        '일매출_백만원': ma_anomalies,
        '응답시간_ms': detect_ma_anomalies(df['응답시간_ms'])[0]
    }
    alerts = generate_alerts(df, anomalies_dict)
    significant_alerts = [a for a in alerts if a.level in [AlertLevel.WARNING, AlertLevel.CRITICAL]]

    print(f"  - 총 경보: {len(alerts)}건")
    print(f"  - 주요 경보 (경고+위험): {len(significant_alerts)}건")

    if significant_alerts:
        print("\n주요 경보 내역:")
        alert_df = create_alert_dashboard(significant_alerts)
        print(alert_df.to_string(index=False))

    # 5. 경보 패턴 분석
    print("\n[5] 경보 패턴 분석")
    patterns = analyze_alert_patterns(alerts)
    if patterns:
        print(f"  - 총 경보 수: {patterns['total']}건")
        print(f"  - 수준별 분포:")
        for level, count in patterns['by_level'].items():
            print(f"      {level}: {count}건")
        print(f"  - KPI별 분포:")
        for kpi_name, count in patterns['by_kpi'].items():
            print(f"      {kpi_name}: {count}건")
        print(f"  - 위험 경보 비율: {patterns['critical_ratio']:.1f}%")

    # 6. 시각화
    print("\n[6] 시각화 생성")
    filename = visualize_anomalies(df, '일매출_백만원', ma_anomalies, upper, lower)
    print(f"  저장: {filename}")

    # 7. 조기 경보 시스템 설계 가이드
    print("\n[7] 조기 경보 시스템 설계 권고")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ 권고 사항                                               │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│ 1. 다중 알고리즘 앙상블: 2개 이상 알고리즘 합의 사용    │")
    print("│ 2. 동적 임계값: 이동평균 기반 적응적 임계값 설정        │")
    print("│ 3. 맥락 고려: 시즌, 요일, 캠페인 등 맥락 변수 반영      │")
    print("│ 4. 알림 피로 방지: 유사 경보 그룹화, 우선순위화         │")
    print("│ 5. 피드백 학습: 오탐/미탐 피드백으로 모델 개선          │")
    print("└─────────────────────────────────────────────────────────┘")

    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
