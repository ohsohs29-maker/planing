"""
제7장 예제 7.2: 경쟁 인텔리전스 분석

이 코드는 공개 데이터를 활용한 경쟁사 분석을 수행한다.
특허 동향, 채용 트렌드, 제품 비교 분석을 포함한다.

Note: 실제 API 호출 대신 시뮬레이션 데이터 사용
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

# 결과 저장 경로
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 시드 설정
np.random.seed(42)


def generate_patent_data(years: int = 5) -> pd.DataFrame:
    """특허 데이터 시뮬레이션

    실제 환경에서는 Google Patents, USPTO API 등 활용
    """
    companies = ["CATL", "LG에너지솔루션", "파나소닉", "삼성SDI", "SK온", "BYD"]

    tech_areas = {
        "전고체 전해질": {"trend": "rising", "leaders": ["삼성SDI", "파나소닉"]},
        "리튬이온 양극재": {"trend": "stable", "leaders": ["CATL", "LG에너지솔루션"]},
        "실리콘 음극재": {"trend": "rising", "leaders": ["파나소닉", "삼성SDI"]},
        "배터리 관리시스템": {"trend": "stable", "leaders": ["CATL", "BYD"]},
        "재활용 기술": {"trend": "emerging", "leaders": ["LG에너지솔루션", "SK온"]},
        "나트륨이온": {"trend": "emerging", "leaders": ["CATL", "BYD"]},
    }

    data = []
    current_year = datetime.now().year

    for year in range(current_year - years, current_year + 1):
        for company in companies:
            for tech, config in tech_areas.items():
                # 기본 특허 수
                base = 10 if company in config["leaders"] else 5

                # 트렌드에 따른 조정
                year_idx = year - (current_year - years)
                if config["trend"] == "rising":
                    multiplier = 1 + year_idx * 0.15
                elif config["trend"] == "emerging":
                    multiplier = 0.5 + year_idx * 0.3 if year_idx > 2 else 0.3
                else:
                    multiplier = 1.0

                count = int(base * multiplier * np.random.uniform(0.7, 1.3))

                data.append({
                    "year": year,
                    "company": company,
                    "tech_area": tech,
                    "patent_count": count
                })

    return pd.DataFrame(data)


def generate_hiring_data() -> pd.DataFrame:
    """채용 공고 데이터 시뮬레이션

    채용 트렌드로 전략 방향 추론
    """
    companies = ["CATL", "LG에너지솔루션", "파나소닉", "삼성SDI", "SK온", "BYD"]

    job_categories = {
        "연구개발": {"weight": 0.3, "signal": "기술 투자"},
        "생산/제조": {"weight": 0.25, "signal": "생산 확대"},
        "영업/마케팅": {"weight": 0.15, "signal": "시장 확장"},
        "공급망관리": {"weight": 0.15, "signal": "공급망 강화"},
        "AI/데이터": {"weight": 0.1, "signal": "디지털 전환"},
        "ESG/지속가능성": {"weight": 0.05, "signal": "ESG 강화"},
    }

    data = []

    for company in companies:
        total_openings = np.random.randint(50, 200)

        for category, config in job_categories.items():
            # 회사별 특성 반영
            weight = config["weight"]
            if company in ["CATL", "BYD"] and category == "생산/제조":
                weight *= 1.5
            if company in ["삼성SDI", "파나소닉"] and category == "연구개발":
                weight *= 1.3
            if company in ["LG에너지솔루션", "SK온"] and category == "AI/데이터":
                weight *= 1.4

            count = int(total_openings * weight * np.random.uniform(0.8, 1.2))

            data.append({
                "company": company,
                "category": category,
                "openings": count,
                "strategic_signal": config["signal"]
            })

    return pd.DataFrame(data)


def generate_product_comparison() -> pd.DataFrame:
    """제품 스펙 비교 데이터"""
    products = [
        {"company": "CATL", "product": "Qilin", "type": "리튬이온",
         "energy_density": 255, "cycle_life": 5000, "charging_speed": "4C",
         "price_index": 85, "year": 2023},
        {"company": "LG에너지솔루션", "product": "NCMA", "type": "리튬이온",
         "energy_density": 240, "cycle_life": 4500, "charging_speed": "3C",
         "price_index": 90, "year": 2023},
        {"company": "삼성SDI", "product": "Gen6", "type": "리튬이온",
         "energy_density": 245, "cycle_life": 4800, "charging_speed": "3.5C",
         "price_index": 92, "year": 2023},
        {"company": "파나소닉", "product": "4680", "type": "리튬이온",
         "energy_density": 250, "cycle_life": 4000, "charging_speed": "2.5C",
         "price_index": 88, "year": 2023},
        {"company": "BYD", "product": "Blade", "type": "LFP",
         "energy_density": 180, "cycle_life": 8000, "charging_speed": "2C",
         "price_index": 70, "year": 2023},
        {"company": "CATL", "product": "Shenxing", "type": "LFP",
         "energy_density": 165, "cycle_life": 7000, "charging_speed": "4C",
         "price_index": 65, "year": 2024},
    ]

    return pd.DataFrame(products)


def analyze_patent_trends(df: pd.DataFrame) -> Dict:
    """특허 트렌드 분석"""
    # 회사별 총 특허
    company_total = df.groupby("company")["patent_count"].sum().sort_values(ascending=False)

    # 기술 영역별 성장률
    current_year = df["year"].max()
    prev_years = df[df["year"] < current_year - 1]
    recent_years = df[df["year"] >= current_year - 1]

    tech_growth = {}
    for tech in df["tech_area"].unique():
        prev = prev_years[prev_years["tech_area"] == tech]["patent_count"].mean()
        recent = recent_years[recent_years["tech_area"] == tech]["patent_count"].mean()
        if prev > 0:
            growth = (recent - prev) / prev * 100
        else:
            growth = 0
        tech_growth[tech] = growth

    # 회사별 주력 기술
    company_focus = {}
    for company in df["company"].unique():
        company_data = df[df["company"] == company]
        top_tech = company_data.groupby("tech_area")["patent_count"].sum().idxmax()
        company_focus[company] = top_tech

    return {
        "company_ranking": company_total,
        "tech_growth": pd.Series(tech_growth).sort_values(ascending=False),
        "company_focus": company_focus
    }


def analyze_hiring_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """채용 전략 분석"""
    # 회사별 채용 비중 분석
    company_profiles = []

    for company in df["company"].unique():
        company_data = df[df["company"] == company]
        total = company_data["openings"].sum()

        profile = {"company": company, "total_openings": total}
        for _, row in company_data.iterrows():
            profile[row["category"]] = row["openings"] / total * 100

        # 주요 전략 신호
        top_category = company_data.loc[company_data["openings"].idxmax(), "category"]
        profile["primary_signal"] = df[df["category"] == top_category]["strategic_signal"].iloc[0]

        company_profiles.append(profile)

    return pd.DataFrame(company_profiles)


def create_competitive_dashboard(patent_df, hiring_df, product_df, filename):
    """경쟁 인텔리전스 대시보드"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 특허 출원 현황
    ax1 = axes[0, 0]
    patent_analysis = analyze_patent_trends(patent_df)
    ranking = patent_analysis["company_ranking"]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(ranking)))
    ax1.barh(ranking.index, ranking.values, color=colors)
    ax1.set_xlabel("총 특허 수")
    ax1.set_title("경쟁사 특허 출원 현황 (최근 5년)")
    ax1.grid(True, alpha=0.3, axis="x")

    # 2. 기술 영역별 성장률
    ax2 = axes[0, 1]
    tech_growth = patent_analysis["tech_growth"]
    colors = ["green" if x > 0 else "red" for x in tech_growth.values]
    ax2.barh(tech_growth.index, tech_growth.values, color=colors, alpha=0.7)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.set_xlabel("성장률 (%)")
    ax2.set_title("기술 영역별 특허 성장률")
    ax2.grid(True, alpha=0.3, axis="x")

    # 3. 채용 전략 비교
    ax3 = axes[1, 0]
    hiring_profile = analyze_hiring_strategy(hiring_df)
    categories = ["연구개발", "생산/제조", "영업/마케팅", "AI/데이터"]
    x = np.arange(len(hiring_profile))
    width = 0.2

    for i, cat in enumerate(categories):
        if cat in hiring_profile.columns:
            ax3.bar(x + i*width, hiring_profile[cat], width, label=cat, alpha=0.8)

    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(hiring_profile["company"], rotation=45, ha="right")
    ax3.set_ylabel("비중 (%)")
    ax3.set_title("경쟁사 채용 전략 비교")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. 제품 스펙 레이더
    ax4 = axes[1, 1]
    # 에너지 밀도 vs 가격 산점도
    ax4.scatter(product_df["price_index"], product_df["energy_density"],
                s=product_df["cycle_life"]/30, alpha=0.6, c=range(len(product_df)), cmap="Set1")
    for _, row in product_df.iterrows():
        ax4.annotate(f"{row['company']}\n{row['product']}",
                     (row["price_index"], row["energy_density"]),
                     fontsize=8, ha="center")
    ax4.set_xlabel("가격 지수 (낮을수록 저렴)")
    ax4.set_ylabel("에너지 밀도 (Wh/kg)")
    ax4.set_title("제품 포지셔닝\n(버블 크기: 수명)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return filepath


def generate_competitor_profiles(patent_df, hiring_df, product_df):
    """경쟁사 프로파일 생성"""
    patent_analysis = analyze_patent_trends(patent_df)
    hiring_profile = analyze_hiring_strategy(hiring_df)

    print("\n" + "=" * 70)
    print("경쟁사 프로파일")
    print("=" * 70)

    companies = ["CATL", "LG에너지솔루션", "삼성SDI", "BYD"]

    for company in companies:
        print(f"\n[{company}]")
        print("-" * 50)

        # 특허 현황
        patent_rank = list(patent_analysis["company_ranking"].index).index(company) + 1
        patent_count = patent_analysis["company_ranking"][company]
        focus_tech = patent_analysis["company_focus"][company]
        print(f"  특허: {patent_count}건 (업계 {patent_rank}위)")
        print(f"  주력 기술: {focus_tech}")

        # 채용 전략
        if company in hiring_profile["company"].values:
            company_hiring = hiring_profile[hiring_profile["company"] == company].iloc[0]
            print(f"  채용 규모: {company_hiring['total_openings']}명")
            print(f"  전략 신호: {company_hiring['primary_signal']}")

        # 제품
        company_products = product_df[product_df["company"] == company]
        if len(company_products) > 0:
            print(f"  주요 제품:")
            for _, prod in company_products.iterrows():
                print(f"    - {prod['product']} ({prod['type']}): "
                      f"{prod['energy_density']}Wh/kg")


def generate_data_driven_swot():
    """데이터 기반 SWOT 분석 예시"""
    print("\n" + "=" * 70)
    print("데이터 기반 SWOT 분석 (C사 관점)")
    print("=" * 70)

    swot = {
        "강점 (Strengths)": [
            "전고체 배터리 특허 보유량 업계 2위 (데이터: 특허 분석)",
            "R&D 인력 비중 업계 최고 32% (데이터: 채용 분석)",
            "에너지 밀도 250Wh/kg 달성 (데이터: 제품 스펙)",
        ],
        "약점 (Weaknesses)": [
            "생산 규모 CATL 대비 40% 수준 (데이터: 생산량)",
            "LFP 배터리 라인업 부재 (데이터: 제품 분석)",
            "공급망 관리 인력 부족 (데이터: 채용 비중 8%)",
        ],
        "기회 (Opportunities)": [
            "전고체 배터리 시장 연 45% 성장 전망 (데이터: 트렌드)",
            "나트륨이온 언급량 300% 급증 (데이터: 뉴스 분석)",
            "재활용 규제 강화로 기술 수요 증가 (데이터: 정책)",
        ],
        "위협 (Threats)": [
            "중국 업체 가격 경쟁력 30% 우위 (데이터: 가격지수)",
            "공급망 리스크 부정 기사 40% 증가 (데이터: 감성분석)",
            "CATL 전고체 특허 출원 급증 (데이터: 특허 동향)",
        ],
    }

    for category, items in swot.items():
        print(f"\n{category}")
        print("-" * 50)
        for item in items:
            print(f"  • {item}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("제7장 예제 7.2: 경쟁 인텔리전스 분석")
    print("=" * 60)

    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 데이터 생성
    print("\n[데이터 수집 시뮬레이션]")
    patent_df = generate_patent_data(years=5)
    hiring_df = generate_hiring_data()
    product_df = generate_product_comparison()

    print(f"  특허 데이터: {len(patent_df)}건")
    print(f"  채용 데이터: {len(hiring_df)}건")
    print(f"  제품 데이터: {len(product_df)}건")

    # 2. 특허 트렌드 분석
    print("\n[특허 트렌드 분석]")
    patent_analysis = analyze_patent_trends(patent_df)

    print("\n기술 영역별 성장률:")
    for tech, growth in patent_analysis["tech_growth"].head(3).items():
        print(f"  ↑ {tech}: {growth:+.1f}%")

    # 3. 경쟁 대시보드 생성
    print("\n[대시보드 생성]")
    filepath = create_competitive_dashboard(
        patent_df, hiring_df, product_df,
        "competitive_intelligence_dashboard.png"
    )
    print(f"  저장됨: {filepath}")

    # 4. 경쟁사 프로파일
    generate_competitor_profiles(patent_df, hiring_df, product_df)

    # 5. 데이터 기반 SWOT
    generate_data_driven_swot()

    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
