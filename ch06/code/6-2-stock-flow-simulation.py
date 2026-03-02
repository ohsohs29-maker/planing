"""
제6장 예제 6.2: 스톡-플로우 시뮬레이션

이 코드는 시스템 다이내믹스의 스톡-플로우 모델을 구현한다.
SaaS 스타트업의 성장 모델을 시뮬레이션하고 정책 시나리오를 비교한다.

Note: PySD가 설치되지 않은 경우 순수 Python으로 시뮬레이션 수행
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# 결과 저장 경로
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 시드 설정
np.random.seed(42)


@dataclass
class SaaSModelParams:
    """SaaS 성장 모델 파라미터"""
    # 초기값
    initial_customers: float = 100
    initial_capacity: float = 200

    # 성장 관련
    word_of_mouth_rate: float = 0.02  # 구전 효과 (고객당 월 신규 유입)
    base_acquisition_rate: float = 5   # 기본 신규 고객 유입

    # 이탈 관련
    base_churn_rate: float = 0.03      # 기본 이탈률
    quality_sensitivity: float = 0.05  # 품질 민감도 (품질 저하 시 추가 이탈)

    # 서버/품질 관련
    capacity_threshold: float = 0.8    # 부하 임계치 (이 이상이면 품질 저하)
    quality_degradation_rate: float = 0.5  # 임계치 초과 시 품질 저하 속도

    # 투자 관련
    investment_trigger: float = 0.7    # 투자 트리거 (부하율)
    capacity_per_investment: float = 50  # 투자당 추가 용량
    investment_delay: int = 3          # 투자 효과 지연 (월)


def simulate_saas_growth(
    params: SaaSModelParams,
    months: int = 36,
    investment_strategy: str = "reactive"
) -> pd.DataFrame:
    """SaaS 성장 시뮬레이션

    Args:
        params: 모델 파라미터
        months: 시뮬레이션 기간 (월)
        investment_strategy: 투자 전략
            - "none": 투자 없음
            - "reactive": 반응적 투자 (부하 임계치 도달 시)
            - "proactive": 선제적 투자 (매 6개월)
            - "aggressive": 공격적 투자 (매 3개월)

    Returns:
        pd.DataFrame: 시뮬레이션 결과
    """
    # 초기화
    customers = params.initial_customers
    capacity = params.initial_capacity
    quality = 1.0  # 0~1 스케일
    pending_investments = []  # (적용 시점, 용량)

    # 기록
    history = {
        "month": [],
        "customers": [],
        "capacity": [],
        "load_ratio": [],
        "quality": [],
        "new_customers": [],
        "churned_customers": [],
        "investment": []
    }

    for month in range(months):
        # 1. 지연된 투자 효과 적용
        investments_to_apply = [inv for inv in pending_investments if inv[0] <= month]
        for inv in investments_to_apply:
            capacity += inv[1]
            pending_investments.remove(inv)

        # 2. 부하율 계산
        load_ratio = customers / capacity if capacity > 0 else 1.0

        # 3. 품질 계산 (부하율이 임계치 초과 시 저하)
        if load_ratio > params.capacity_threshold:
            quality_loss = (load_ratio - params.capacity_threshold) * params.quality_degradation_rate
            quality = max(0.1, quality - quality_loss)
        else:
            # 품질 회복
            quality = min(1.0, quality + 0.1)

        # 4. 신규 고객
        word_of_mouth = customers * params.word_of_mouth_rate * quality
        new_customers = params.base_acquisition_rate + word_of_mouth

        # 5. 이탈 고객
        effective_churn = params.base_churn_rate + params.quality_sensitivity * (1 - quality)
        churned_customers = customers * effective_churn

        # 6. 고객 수 업데이트
        customers = max(0, customers + new_customers - churned_customers)

        # 7. 투자 결정
        investment = 0
        if investment_strategy == "reactive":
            if load_ratio > params.investment_trigger:
                investment = params.capacity_per_investment
                pending_investments.append((month + params.investment_delay, investment))
        elif investment_strategy == "proactive":
            if month % 6 == 0 and month > 0:
                investment = params.capacity_per_investment
                pending_investments.append((month + params.investment_delay, investment))
        elif investment_strategy == "aggressive":
            if month % 3 == 0 and month > 0:
                investment = params.capacity_per_investment
                pending_investments.append((month + params.investment_delay, investment))

        # 8. 기록
        history["month"].append(month)
        history["customers"].append(customers)
        history["capacity"].append(capacity)
        history["load_ratio"].append(load_ratio)
        history["quality"].append(quality)
        history["new_customers"].append(new_customers)
        history["churned_customers"].append(churned_customers)
        history["investment"].append(investment)

    return pd.DataFrame(history)


def compare_investment_strategies(params: SaaSModelParams, months: int = 36):
    """투자 전략 비교 시뮬레이션"""
    strategies = ["none", "reactive", "proactive", "aggressive"]
    results = {}

    for strategy in strategies:
        df = simulate_saas_growth(params, months, strategy)
        results[strategy] = df

    return results


def visualize_simulation_results(results: Dict[str, pd.DataFrame], filename: str):
    """시뮬레이션 결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    strategies = list(results.keys())
    colors = {"none": "gray", "reactive": "blue", "proactive": "green", "aggressive": "red"}
    labels = {"none": "투자 없음", "reactive": "반응적 투자",
              "proactive": "선제적 투자", "aggressive": "공격적 투자"}

    # 1. 고객 수 추이
    ax1 = axes[0, 0]
    for strategy in strategies:
        df = results[strategy]
        ax1.plot(df["month"], df["customers"],
                 label=labels[strategy], color=colors[strategy], linewidth=2)
    ax1.set_xlabel("월")
    ax1.set_ylabel("고객 수")
    ax1.set_title("고객 수 추이")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 서비스 품질 추이
    ax2 = axes[0, 1]
    for strategy in strategies:
        df = results[strategy]
        ax2.plot(df["month"], df["quality"],
                 label=labels[strategy], color=colors[strategy], linewidth=2)
    ax2.set_xlabel("월")
    ax2.set_ylabel("서비스 품질 (0-1)")
    ax2.set_title("서비스 품질 추이")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    # 3. 부하율 추이
    ax3 = axes[1, 0]
    for strategy in strategies:
        df = results[strategy]
        ax3.plot(df["month"], df["load_ratio"],
                 label=labels[strategy], color=colors[strategy], linewidth=2)
    ax3.axhline(y=0.8, color="orange", linestyle="--", label="임계치 (0.8)")
    ax3.set_xlabel("월")
    ax3.set_ylabel("부하율")
    ax3.set_title("서버 부하율 추이")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 누적 투자 vs 최종 고객
    ax4 = axes[1, 1]
    final_data = []
    for strategy in strategies:
        df = results[strategy]
        total_investment = df["investment"].sum()
        final_customers = df["customers"].iloc[-1]
        avg_quality = df["quality"].mean()
        final_data.append({
            "strategy": labels[strategy],
            "total_investment": total_investment,
            "final_customers": final_customers,
            "avg_quality": avg_quality
        })

    final_df = pd.DataFrame(final_data)
    x = range(len(final_df))
    width = 0.35

    bars1 = ax4.bar([i - width/2 for i in x], final_df["final_customers"],
                    width, label="최종 고객 수", color="steelblue")
    ax4.set_ylabel("최종 고객 수", color="steelblue")

    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar([i + width/2 for i in x], final_df["total_investment"],
                          width, label="누적 투자", color="coral", alpha=0.7)
    ax4_twin.set_ylabel("누적 투자 (용량)", color="coral")

    ax4.set_xticks(x)
    ax4.set_xticklabels(final_df["strategy"], rotation=45, ha="right")
    ax4.set_title("전략별 최종 성과 비교")

    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return filepath


def sensitivity_analysis(base_params: SaaSModelParams, months: int = 36):
    """민감도 분석: 주요 파라미터가 결과에 미치는 영향"""
    results = []

    # 1. 구전 효과율 변화
    for wom_rate in [0.01, 0.02, 0.03, 0.04]:
        params = SaaSModelParams(word_of_mouth_rate=wom_rate)
        df = simulate_saas_growth(params, months, "reactive")
        results.append({
            "parameter": "구전 효과율",
            "value": wom_rate,
            "final_customers": df["customers"].iloc[-1],
            "avg_quality": df["quality"].mean()
        })

    # 2. 품질 민감도 변화
    for quality_sens in [0.02, 0.05, 0.08, 0.10]:
        params = SaaSModelParams(quality_sensitivity=quality_sens)
        df = simulate_saas_growth(params, months, "reactive")
        results.append({
            "parameter": "품질 민감도",
            "value": quality_sens,
            "final_customers": df["customers"].iloc[-1],
            "avg_quality": df["quality"].mean()
        })

    # 3. 투자 지연 변화
    for delay in [1, 3, 5, 7]:
        params = SaaSModelParams(investment_delay=delay)
        df = simulate_saas_growth(params, months, "reactive")
        results.append({
            "parameter": "투자 지연",
            "value": delay,
            "final_customers": df["customers"].iloc[-1],
            "avg_quality": df["quality"].mean()
        })

    return pd.DataFrame(results)


def visualize_sensitivity(sensitivity_df: pd.DataFrame, filename: str):
    """민감도 분석 결과 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    parameters = sensitivity_df["parameter"].unique()
    colors = ["steelblue", "coral", "seagreen"]

    for idx, param in enumerate(parameters):
        ax = axes[idx]
        param_data = sensitivity_df[sensitivity_df["parameter"] == param]

        ax.plot(param_data["value"], param_data["final_customers"],
                marker="o", linewidth=2, markersize=8, color=colors[idx])
        ax.set_xlabel(param)
        ax.set_ylabel("최종 고객 수")
        ax.set_title(f"{param}의 영향")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return filepath


def print_strategy_comparison(results: Dict[str, pd.DataFrame]):
    """전략별 성과 비교 출력"""
    print("\n" + "=" * 70)
    print("투자 전략별 성과 비교 (36개월 시뮬레이션)")
    print("=" * 70)

    labels = {"none": "투자 없음", "reactive": "반응적 투자",
              "proactive": "선제적 투자", "aggressive": "공격적 투자"}

    print(f"\n{'전략':<15} {'최종고객':<12} {'평균품질':<12} {'누적투자':<12} {'고객/투자':<12}")
    print("-" * 70)

    for strategy, df in results.items():
        final_customers = df["customers"].iloc[-1]
        avg_quality = df["quality"].mean()
        total_investment = df["investment"].sum()
        efficiency = final_customers / max(total_investment, 1)

        print(f"{labels[strategy]:<15} {final_customers:>10.0f} {avg_quality:>10.2f} "
              f"{total_investment:>10.0f} {efficiency:>10.1f}")

    # 최적 전략 분석
    print("\n" + "-" * 70)
    print("분석 결과:")

    best_customers = max(results.items(), key=lambda x: x[1]["customers"].iloc[-1])
    best_quality = max(results.items(), key=lambda x: x[1]["quality"].mean())
    best_efficiency = max(results.items(),
                         key=lambda x: x[1]["customers"].iloc[-1] / max(x[1]["investment"].sum(), 1))

    print(f"  - 최대 고객 확보: {labels[best_customers[0]]} "
          f"({best_customers[1]['customers'].iloc[-1]:.0f}명)")
    print(f"  - 최고 평균 품질: {labels[best_quality[0]]} "
          f"({best_quality[1]['quality'].mean():.2f})")
    print(f"  - 최고 투자 효율: {labels[best_efficiency[0]]}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("제6장 예제 6.2: 스톡-플로우 시뮬레이션")
    print("=" * 60)

    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 기본 파라미터
    params = SaaSModelParams()

    print("\n[모델 파라미터]")
    print(f"  초기 고객 수: {params.initial_customers}")
    print(f"  초기 서버 용량: {params.initial_capacity}")
    print(f"  구전 효과율: {params.word_of_mouth_rate}")
    print(f"  기본 이탈률: {params.base_churn_rate}")
    print(f"  품질 민감도: {params.quality_sensitivity}")
    print(f"  부하 임계치: {params.capacity_threshold}")
    print(f"  투자 지연: {params.investment_delay}개월")

    # 1. 투자 전략 비교
    print("\n[투자 전략 시뮬레이션]")
    results = compare_investment_strategies(params, months=36)

    # 결과 출력
    print_strategy_comparison(results)

    # 시각화
    path1 = visualize_simulation_results(results, "sd_strategy_comparison.png")
    print(f"\n시각화 저장됨: {path1}")

    # 2. 민감도 분석
    print("\n[민감도 분석]")
    sensitivity_df = sensitivity_analysis(params)

    print("\n파라미터별 영향도:")
    print("-" * 50)
    for param in sensitivity_df["parameter"].unique():
        param_data = sensitivity_df[sensitivity_df["parameter"] == param]
        min_val = param_data["final_customers"].min()
        max_val = param_data["final_customers"].max()
        impact = (max_val - min_val) / min_val * 100
        print(f"  {param}: 고객 수 변동폭 {impact:.1f}%")

    path2 = visualize_sensitivity(sensitivity_df, "sd_sensitivity_analysis.png")
    print(f"\n민감도 분석 시각화 저장됨: {path2}")

    # 3. 결론
    print("\n" + "=" * 60)
    print("핵심 시사점")
    print("=" * 60)
    print("""
1. 투자 없음 전략:
   - 초기 성장 후 급격한 쇠퇴
   - 품질 저하 → 이탈 증가의 악순환 (균형 루프 지배)

2. 반응적 투자:
   - 문제 발생 후 대응 → 품질 저하 기간 발생
   - 투자 지연이 길수록 회복 어려움

3. 선제적 투자:
   - 적정 수준의 투자로 성장과 품질 균형
   - 투자 효율성 최적화

4. 공격적 투자:
   - 가장 많은 고객 확보
   - 그러나 투자 효율성은 낮음 (과잉 투자)

결론: 시스템의 지연을 고려한 선제적 투자가 가장 효과적
    """)


if __name__ == "__main__":
    main()
