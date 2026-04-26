"""
6주차 보충자료: 뉴스 트렌드 분석

이 코드는 뉴스 데이터를 분석하여 산업 트렌드를 파악한다.
시계열 트렌드, 키워드 빈도, 감성 분석을 수행한다.

Note: 실제 API 호출 대신 시뮬레이션 데이터 사용
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
import re

# 결과 저장 경로
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 시드 설정
np.random.seed(42)


def generate_simulated_news_data(
    topic: str = "전기차 배터리",
    days: int = 180,
    base_count: int = 10
) -> pd.DataFrame:
    """시뮬레이션 뉴스 데이터 생성

    실제 환경에서는 NewsAPI, Google News 등을 활용
    """
    # 날짜 생성
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(days)]
    dates.reverse()

    # 키워드별 트렌드 시뮬레이션
    keywords = {
        "리튬이온": {"trend": "stable", "base": 15},
        "전고체": {"trend": "rising", "base": 8},
        "LFP": {"trend": "rising", "base": 10},
        "나트륨이온": {"trend": "emerging", "base": 3},
        "재활용": {"trend": "rising", "base": 7},
        "공급망": {"trend": "volatile", "base": 12},
        "안전성": {"trend": "stable", "base": 8},
        "에너지밀도": {"trend": "stable", "base": 6},
    }

    news_data = []

    for i, date in enumerate(dates):
        for keyword, config in keywords.items():
            # 트렌드에 따른 기사 수 계산
            base = config["base"]
            trend = config["trend"]

            if trend == "stable":
                count = base + np.random.poisson(2)
            elif trend == "rising":
                growth = i / days * 0.5  # 50% 성장
                count = int(base * (1 + growth)) + np.random.poisson(2)
            elif trend == "emerging":
                # 특정 시점부터 급증
                if i > days * 0.6:
                    count = base * 3 + np.random.poisson(3)
                else:
                    count = base + np.random.poisson(1)
            elif trend == "volatile":
                # 변동성 높음
                count = base + np.random.randint(-5, 10)
                count = max(0, count)

            # 감성 점수 (실제로는 NLP 모델 사용)
            if keyword in ["안전성", "공급망"]:
                sentiment = np.random.choice(
                    ["positive", "neutral", "negative"],
                    p=[0.2, 0.4, 0.4]
                )
            elif keyword in ["전고체", "나트륨이온"]:
                sentiment = np.random.choice(
                    ["positive", "neutral", "negative"],
                    p=[0.6, 0.3, 0.1]
                )
            else:
                sentiment = np.random.choice(
                    ["positive", "neutral", "negative"],
                    p=[0.4, 0.4, 0.2]
                )

            news_data.append({
                "date": date,
                "keyword": keyword,
                "count": count,
                "sentiment": sentiment,
                "topic": topic
            })

    return pd.DataFrame(news_data)


def analyze_keyword_trends(df: pd.DataFrame) -> pd.DataFrame:
    """키워드별 트렌드 분석"""
    # 월별 집계
    df["month"] = df["date"].dt.to_period("M")

    monthly_trends = df.groupby(["month", "keyword"])["count"].sum().unstack(fill_value=0)

    # 성장률 계산 (첫 달 대비 마지막 달)
    growth_rates = {}
    for keyword in monthly_trends.columns:
        first_month = monthly_trends[keyword].iloc[:3].mean()  # 첫 3개월 평균
        last_month = monthly_trends[keyword].iloc[-3:].mean()  # 마지막 3개월 평균
        if first_month > 0:
            growth = (last_month - first_month) / first_month * 100
        else:
            growth = 0
        growth_rates[keyword] = growth

    growth_df = pd.DataFrame({
        "keyword": growth_rates.keys(),
        "growth_rate": growth_rates.values(),
        "total_mentions": [monthly_trends[k].sum() for k in growth_rates.keys()]
    })

    return growth_df.sort_values("growth_rate", ascending=False)


def analyze_sentiment_trends(df: pd.DataFrame) -> pd.DataFrame:
    """키워드별 감성 분석"""
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["sentiment_score"] = df["sentiment"].map(sentiment_map)

    sentiment_summary = df.groupby("keyword").agg({
        "sentiment_score": "mean",
        "count": "sum"
    }).reset_index()

    sentiment_summary.columns = ["keyword", "avg_sentiment", "total_count"]
    sentiment_summary["sentiment_label"] = sentiment_summary["avg_sentiment"].apply(
        lambda x: "긍정" if x > 0.2 else ("부정" if x < -0.2 else "중립")
    )

    return sentiment_summary.sort_values("avg_sentiment", ascending=False)


def detect_weak_signals(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """약신호 탐지: 급격한 언급 증가 패턴"""
    df["week"] = df["date"].dt.isocalendar().week
    df["year"] = df["date"].dt.year

    # 주별 집계
    weekly = df.groupby(["year", "week", "keyword"])["count"].sum().reset_index()

    weak_signals = []

    for keyword in weekly["keyword"].unique():
        kw_data = weekly[weekly["keyword"] == keyword].copy()
        kw_data = kw_data.sort_values(["year", "week"])

        if len(kw_data) < 4:
            continue

        # 이동평균 계산
        kw_data["ma4"] = kw_data["count"].rolling(4).mean()

        # 최근 주가 이동평균 대비 급증했는지 확인
        if kw_data["ma4"].iloc[-2] > 0:
            recent_ratio = kw_data["count"].iloc[-1] / kw_data["ma4"].iloc[-2]
            if recent_ratio > threshold:
                weak_signals.append({
                    "keyword": keyword,
                    "recent_count": kw_data["count"].iloc[-1],
                    "avg_count": kw_data["ma4"].iloc[-2],
                    "surge_ratio": recent_ratio,
                    "signal_strength": "강" if recent_ratio > 3 else "중"
                })

    return pd.DataFrame(weak_signals)


def visualize_trends(df: pd.DataFrame, growth_df: pd.DataFrame, filename: str):
    """트렌드 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 키워드별 시계열 트렌드
    ax1 = axes[0, 0]
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby(["month", "keyword"])["count"].sum().unstack(fill_value=0)

    # 상위 5개 키워드만 표시
    top_keywords = growth_df.nlargest(5, "total_mentions")["keyword"].tolist()
    for keyword in top_keywords:
        if keyword in monthly.columns:
            ax1.plot(range(len(monthly)), monthly[keyword], label=keyword, linewidth=2)

    ax1.set_xlabel("월")
    ax1.set_ylabel("기사 수")
    ax1.set_title("주요 키워드 월별 트렌드")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 성장률 차트
    ax2 = axes[0, 1]
    colors = ["green" if x > 0 else "red" for x in growth_df["growth_rate"]]
    ax2.barh(growth_df["keyword"], growth_df["growth_rate"], color=colors, alpha=0.7)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.set_xlabel("성장률 (%)")
    ax2.set_title("키워드별 6개월 성장률")
    ax2.grid(True, alpha=0.3, axis="x")

    # 3. 감성 분석 결과
    ax3 = axes[1, 0]
    sentiment_df = analyze_sentiment_trends(df)
    colors = ["green" if x > 0 else ("red" if x < 0 else "gray")
              for x in sentiment_df["avg_sentiment"]]
    ax3.barh(sentiment_df["keyword"], sentiment_df["avg_sentiment"], color=colors, alpha=0.7)
    ax3.axvline(x=0, color="black", linewidth=0.5)
    ax3.set_xlabel("감성 점수 (-1: 부정, +1: 긍정)")
    ax3.set_title("키워드별 감성 분석")
    ax3.set_xlim(-1, 1)
    ax3.grid(True, alpha=0.3, axis="x")

    # 4. 버블 차트: 언급량 vs 성장률 vs 감성
    ax4 = axes[1, 1]
    merged = growth_df.merge(sentiment_df, on="keyword")
    scatter = ax4.scatter(
        merged["growth_rate"],
        merged["avg_sentiment"],
        s=merged["total_mentions"] / 5,
        alpha=0.6,
        c=range(len(merged)),
        cmap="viridis"
    )
    for idx, row in merged.iterrows():
        ax4.annotate(row["keyword"], (row["growth_rate"], row["avg_sentiment"]),
                     fontsize=9, ha="center")
    ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax4.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax4.set_xlabel("성장률 (%)")
    ax4.set_ylabel("감성 점수")
    ax4.set_title("키워드 포지셔닝\n(버블 크기: 언급량)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return filepath


def generate_trend_report(df: pd.DataFrame, growth_df: pd.DataFrame,
                          sentiment_df: pd.DataFrame, weak_signals: pd.DataFrame):
    """트렌드 분석 리포트 생성"""
    print("\n" + "=" * 70)
    print("AI 기반 환경 분석 리포트: 전기차 배터리 시장")
    print("=" * 70)

    print("\n[1. 트렌드 요약]")
    print("-" * 50)
    print("\n성장 트렌드 상위 키워드:")
    for _, row in growth_df.head(3).iterrows():
        trend = "↑" if row["growth_rate"] > 0 else "↓"
        print(f"  {trend} {row['keyword']}: {row['growth_rate']:+.1f}% 성장")

    print("\n하락 트렌드 키워드:")
    for _, row in growth_df.tail(2).iterrows():
        if row["growth_rate"] < 0:
            print(f"  ↓ {row['keyword']}: {row['growth_rate']:.1f}%")

    print("\n[2. 감성 분석 결과]")
    print("-" * 50)
    positive = sentiment_df[sentiment_df["avg_sentiment"] > 0.2]
    negative = sentiment_df[sentiment_df["avg_sentiment"] < -0.2]

    print("\n긍정적 감성 키워드:")
    for _, row in positive.iterrows():
        print(f"  ✓ {row['keyword']}: {row['avg_sentiment']:.2f}")

    print("\n부정적/우려 키워드:")
    for _, row in negative.iterrows():
        print(f"  ✗ {row['keyword']}: {row['avg_sentiment']:.2f}")

    print("\n[3. 약신호 탐지]")
    print("-" * 50)
    if len(weak_signals) > 0:
        print("\n급부상 키워드 (주의 필요):")
        for _, row in weak_signals.iterrows():
            print(f"  ⚠ {row['keyword']}: {row['surge_ratio']:.1f}x 급증 "
                  f"(신호 강도: {row['signal_strength']})")
    else:
        print("  현재 탐지된 약신호 없음")

    print("\n[4. 전략적 시사점]")
    print("-" * 50)
    print("""
1. 신기술 동향:
   - 전고체 배터리, 나트륨이온 배터리에 대한 관심 급증
   - 기술 전환 가능성에 대한 모니터링 필요

2. 공급망 리스크:
   - 공급망 관련 기사의 부정적 감성 높음
   - 원자재 확보 및 공급망 다각화 전략 검토

3. 지속가능성:
   - 재활용 관련 언급 증가세
   - 순환경제 대응 전략 필요

4. 안전성:
   - 안전성 관련 부정 기사 비중 높음
   - 품질 관리 및 커뮤니케이션 강화 필요
    """)


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("6주차 보충자료: 뉴스 트렌드 분석")
    print("=" * 60)

    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 데이터 생성 (실제로는 API 호출)
    print("\n[데이터 수집 시뮬레이션]")
    df = generate_simulated_news_data(topic="전기차 배터리", days=180)
    print(f"  수집 기간: 최근 6개월")
    print(f"  총 데이터 포인트: {len(df)}")
    print(f"  분석 키워드: {df['keyword'].nunique()}개")

    # 데이터 저장
    data_path = OUTPUT_DIR / "news_trend_data.csv"
    df.to_csv(data_path, index=False)
    print(f"\n데이터 저장됨: {data_path}")

    # 2. 트렌드 분석
    print("\n[트렌드 분석]")
    growth_df = analyze_keyword_trends(df)

    # 3. 감성 분석
    print("[감성 분석]")
    sentiment_df = analyze_sentiment_trends(df)

    # 4. 약신호 탐지
    print("[약신호 탐지]")
    weak_signals = detect_weak_signals(df)

    # 5. 시각화
    print("[시각화 생성]")
    filepath = visualize_trends(df, growth_df, "news_trend_analysis.png")
    print(f"  저장됨: {filepath}")

    # 6. 리포트 생성
    generate_trend_report(df, growth_df, sentiment_df, weak_signals)


if __name__ == "__main__":
    main()
