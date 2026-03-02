"""
11-1-monte-carlo-basics.py
몬테카를로 시뮬레이션 기초

이 코드는 몬테카를로 시뮬레이션의 기본 개념과
주요 확률 분포, 시뮬레이션 결과 해석 방법을 다룬다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 재현성을 위한 시드 설정
np.random.seed(42)


def point_estimate_problem():
    """
    점 추정의 한계 시연
    """
    print("=" * 60)
    print("점 추정 vs 확률적 추정")
    print("=" * 60)

    # 점 추정: 매출 1억 원 예상
    point_estimate = 100_000_000

    # 확률적 추정: 정규분포 (평균 1억, 표준편차 2천만)
    mean = 100_000_000
    std = 20_000_000

    # 시뮬레이션
    n_simulations = 10000
    simulated_sales = np.random.normal(mean, std, n_simulations)

    # 결과 분석
    print("\n[점 추정]")
    print(f"  예상 매출: {point_estimate/1e8:.1f}억 원")
    print(f"  → 단일 값만 제시, 불확실성 정보 없음")

    print("\n[확률적 추정 (10,000회 시뮬레이션)]")
    print(f"  평균: {np.mean(simulated_sales)/1e8:.2f}억 원")
    print(f"  표준편차: {np.std(simulated_sales)/1e8:.2f}억 원")
    print(f"  최소: {np.min(simulated_sales)/1e8:.2f}억 원")
    print(f"  최대: {np.max(simulated_sales)/1e8:.2f}억 원")

    # 백분위수
    percentiles = [5, 25, 50, 75, 95]
    print(f"\n[백분위수]")
    for p in percentiles:
        value = np.percentile(simulated_sales, p)
        print(f"  {p}%: {value/1e8:.2f}억 원")

    # 90% 신뢰구간
    ci_lower = np.percentile(simulated_sales, 5)
    ci_upper = np.percentile(simulated_sales, 95)
    print(f"\n[90% 신뢰구간]")
    print(f"  {ci_lower/1e8:.2f}억 원 ~ {ci_upper/1e8:.2f}억 원")

    # 목표 달성 확률
    target = 80_000_000  # 손익분기점 8천만 원
    prob_above_target = np.mean(simulated_sales >= target) * 100
    print(f"\n[목표 달성 확률]")
    print(f"  손익분기점({target/1e8:.1f}억 원) 초과 확률: {prob_above_target:.1f}%")

    return simulated_sales


def probability_distributions():
    """
    주요 확률 분포 소개
    """
    print("\n" + "=" * 60)
    print("주요 확률 분포")
    print("=" * 60)

    n_samples = 10000

    distributions = {
        "정규분포": {
            "samples": np.random.normal(100, 20, n_samples),
            "params": "평균=100, 표준편차=20",
            "use_case": "과거 데이터가 충분할 때"
        },
        "삼각분포": {
            "samples": np.random.triangular(60, 100, 150, n_samples),
            "params": "최소=60, 최빈=100, 최대=150",
            "use_case": "전문가 추정 (낙관/최빈/비관)"
        },
        "균등분포": {
            "samples": np.random.uniform(50, 150, n_samples),
            "params": "최소=50, 최대=150",
            "use_case": "정보가 거의 없을 때"
        },
        "로그정규분포": {
            "samples": np.random.lognormal(4.5, 0.3, n_samples),
            "params": "μ=4.5, σ=0.3",
            "use_case": "비용, 시간 등 양수 값"
        }
    }

    print("\n[분포별 특성]")
    print("-" * 80)
    print(f"{'분포':<15} {'평균':<12} {'표준편차':<12} {'적용 상황':<30}")
    print("-" * 80)

    for name, dist in distributions.items():
        samples = dist["samples"]
        print(f"{name:<15} {np.mean(samples):<12.1f} {np.std(samples):<12.1f} {dist['use_case']:<30}")

    print("-" * 80)

    return distributions


def simulation_convergence():
    """
    시뮬레이션 수렴 분석: 몇 번 시뮬레이션해야 하는가?
    """
    print("\n" + "=" * 60)
    print("시뮬레이션 수렴 분석")
    print("=" * 60)

    # 모수: 정규분포 (평균 100, 표준편차 20)
    true_mean = 100
    true_std = 20

    # 다양한 시뮬레이션 횟수로 평균 추정
    n_simulations = [100, 500, 1000, 5000, 10000, 50000]

    print("\n[시뮬레이션 횟수별 추정 정확도]")
    print("-" * 60)
    print(f"{'횟수':<12} {'추정 평균':<15} {'오차':<15} {'추정 오차 범위':<15}")
    print("-" * 60)

    for n in n_simulations:
        samples = np.random.normal(true_mean, true_std, n)
        est_mean = np.mean(samples)
        error = abs(est_mean - true_mean)
        std_error = true_std / np.sqrt(n)  # 표준오차

        print(f"{n:<12} {est_mean:<15.2f} {error:<15.2f} ±{1.96*std_error:.2f} (95%)")

    print("-" * 60)
    print("\n[권장사항]")
    print("  • 탐색 단계: 1,000~5,000회")
    print("  • 최종 분석: 10,000~50,000회")
    print("  • 정밀 분석: 100,000회 이상")


def result_interpretation():
    """
    시뮬레이션 결과 해석 방법
    """
    print("\n" + "=" * 60)
    print("시뮬레이션 결과 해석")
    print("=" * 60)

    # 사업 수익 시뮬레이션 (정규분포)
    n_simulations = 10000
    profits = np.random.normal(50, 30, n_simulations)  # 평균 50억, 표준편차 30억

    print("\n[기본 통계량]")
    print(f"  평균: {np.mean(profits):.1f}억 원")
    print(f"  중앙값: {np.median(profits):.1f}억 원")
    print(f"  표준편차: {np.std(profits):.1f}억 원")

    print("\n[리스크 지표]")
    # VaR (Value at Risk): 5% 분위수
    var_5 = np.percentile(profits, 5)
    print(f"  VaR(5%): {var_5:.1f}억 원")
    print(f"  → 95% 확률로 손실이 {-var_5:.1f}억 원을 초과하지 않음" if var_5 < 0
          else f"  → 95% 확률로 수익이 {var_5:.1f}억 원 이상")

    # CVaR (Conditional VaR): 5% 이하 평균
    cvar_5 = np.mean(profits[profits <= var_5])
    print(f"  CVaR(5%): {cvar_5:.1f}억 원")
    print(f"  → 최악 5% 상황의 평균 수익")

    # 손실 확률
    prob_loss = np.mean(profits < 0) * 100
    print(f"\n[손실 확률]")
    print(f"  적자 확률 (수익 < 0): {prob_loss:.1f}%")

    # 목표 달성 확률
    targets = [0, 30, 50, 80, 100]
    print(f"\n[목표 달성 확률]")
    for target in targets:
        prob = np.mean(profits >= target) * 100
        print(f"  {target}억 원 이상: {prob:.1f}%")

    return profits


def visualize_distributions(distributions, profits):
    """
    시뮬레이션 결과 시각화
    """
    print("\n" + "=" * 60)
    print("시각화 생성")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 분포 비교
    ax1 = axes[0, 0]
    for name, dist in distributions.items():
        ax1.hist(dist["samples"], bins=50, alpha=0.5, label=name, density=True)
    ax1.set_xlabel('값')
    ax1.set_ylabel('밀도')
    ax1.set_title('주요 확률 분포 비교')
    ax1.legend()

    # 2. 수익 분포 히스토그램
    ax2 = axes[0, 1]
    ax2.hist(profits, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(profits), color='red', linestyle='--', label=f'평균: {np.mean(profits):.1f}억')
    ax2.axvline(np.percentile(profits, 5), color='orange', linestyle='--', label=f'VaR(5%): {np.percentile(profits, 5):.1f}억')
    ax2.axvline(0, color='black', linestyle='-', linewidth=2, label='손익분기점')
    ax2.set_xlabel('수익 (억 원)')
    ax2.set_ylabel('빈도')
    ax2.set_title('사업 수익 시뮬레이션 분포')
    ax2.legend()

    # 3. 누적분포함수 (CDF)
    ax3 = axes[1, 0]
    sorted_profits = np.sort(profits)
    cdf = np.arange(1, len(sorted_profits) + 1) / len(sorted_profits)
    ax3.plot(sorted_profits, cdf, color='steelblue', linewidth=2)
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(0.05, color='orange', linestyle='--', alpha=0.5)
    ax3.axhline(0.95, color='orange', linestyle='--', alpha=0.5)
    ax3.set_xlabel('수익 (억 원)')
    ax3.set_ylabel('누적 확률')
    ax3.set_title('누적분포함수 (CDF)')
    ax3.grid(True, alpha=0.3)

    # 4. 목표 달성 확률
    ax4 = axes[1, 1]
    targets = np.arange(-50, 150, 10)
    probs = [np.mean(profits >= t) * 100 for t in targets]
    ax4.bar(targets, probs, width=8, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axhline(50, color='red', linestyle='--', label='50% 확률')
    ax4.set_xlabel('목표 수익 (억 원)')
    ax4.set_ylabel('달성 확률 (%)')
    ax4.set_title('목표별 달성 확률')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../data/monte_carlo_basics.png', dpi=150)
    plt.close()

    print("그래프 저장: ../data/monte_carlo_basics.png")


if __name__ == "__main__":
    # 점 추정 vs 확률적 추정
    simulated_sales = point_estimate_problem()

    # 확률 분포 소개
    distributions = probability_distributions()

    # 시뮬레이션 수렴 분석
    simulation_convergence()

    # 결과 해석 방법
    profits = result_interpretation()

    # 시각화
    visualize_distributions(distributions, profits)

    print("\n" + "=" * 60)
    print("몬테카를로 시뮬레이션 기초 완료")
    print("=" * 60)
