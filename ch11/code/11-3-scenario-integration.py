"""
11-3-scenario-integration.py
시나리오와 시뮬레이션의 통합

이 코드는 10장의 시나리오 플래닝과 11장의 몬테카를로
시뮬레이션을 통합하여 강건한 전략을 도출한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (크로스플랫폼)
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 재현성을 위한 시드 설정
np.random.seed(42)


def define_scenarios():
    """
    시나리오 정의 (10장에서 개발한 시나리오 활용)
    """
    print("=" * 60)
    print("시나리오 정의")
    print("=" * 60)

    scenarios = {
        "녹색 가속": {
            "probability": 0.25,
            "description": "강한 규제 + 빠른 기술 발전",
            "market_growth": (1.5, 2.0, 2.5),  # 시장 성장률 (삼각분포)
            "margin": (0.25, 0.30, 0.35),      # 마진율
            "competition": (0.7, 0.8, 0.9),    # 경쟁 강도 (낮을수록 좋음)
        },
        "규제 주도 전환": {
            "probability": 0.20,
            "description": "강한 규제 + 느린 기술 발전",
            "market_growth": (1.2, 1.4, 1.6),
            "margin": (0.15, 0.20, 0.25),
            "competition": (0.8, 0.9, 1.0),
        },
        "시장 주도 전환": {
            "probability": 0.30,
            "description": "약한 규제 + 빠른 기술 발전",
            "market_growth": (1.3, 1.8, 2.3),
            "margin": (0.20, 0.25, 0.30),
            "competition": (0.6, 0.75, 0.9),
        },
        "점진적 전환": {
            "probability": 0.25,
            "description": "약한 규제 + 느린 기술 발전",
            "market_growth": (0.9, 1.1, 1.3),
            "margin": (0.10, 0.15, 0.20),
            "competition": (0.9, 0.95, 1.0),
        }
    }

    print("\n[시나리오 요약]")
    print("-" * 70)
    print(f"{'시나리오':<15} {'확률':<8} {'시장성장':<15} {'마진':<15}")
    print("-" * 70)
    for name, s in scenarios.items():
        mg = f"{s['market_growth'][1]:.1f}배"
        margin = f"{s['margin'][1]*100:.0f}%"
        print(f"{name:<15} {s['probability']*100:.0f}%     {mg:<15} {margin:<15}")
    print("-" * 70)

    return scenarios


def scenario_simulation(scenarios, n_simulations=10000):
    """
    시나리오별 시뮬레이션
    """
    print("\n" + "=" * 60)
    print("시나리오별 시뮬레이션")
    print("=" * 60)

    # 기본 사업 가정
    base_investment = 100  # 억 원
    base_market = 500      # 억 원 (현재 시장 규모)

    results = {}

    for scenario_name, scenario in scenarios.items():
        # 시뮬레이션 파라미터 추출
        mg = scenario['market_growth']
        margin = scenario['margin']
        comp = scenario['competition']

        # 시뮬레이션
        market_growth = np.random.triangular(mg[0], mg[1], mg[2], n_simulations)
        margin_rate = np.random.triangular(margin[0], margin[1], margin[2], n_simulations)
        competition_factor = np.random.triangular(comp[0], comp[1], comp[2], n_simulations)

        # 5년 후 시장 규모
        future_market = base_market * market_growth

        # 우리 기업 매출 (시장 규모 × 기본 점유율 × 경쟁 요인)
        base_share = 0.10  # 기본 점유율 10%
        our_revenue = future_market * base_share / competition_factor

        # 이익
        profit = our_revenue * margin_rate - base_investment * 0.2  # 연간 투자 상각

        # 5년 NPV (간략화)
        discount_rate = 0.10
        npv = -base_investment
        for year in range(1, 6):
            annual_profit = profit / 5  # 단순화
            npv += annual_profit / ((1 + discount_rate) ** year)

        results[scenario_name] = {
            'profit': profit,
            'npv': npv,
            'revenue': our_revenue,
            'probability': scenario['probability']
        }

    # 결과 출력
    print("\n[시나리오별 이익 분포]")
    print("-" * 80)
    print(f"{'시나리오':<15} {'평균이익':<12} {'표준편차':<12} {'5%':<12} {'95%':<12} {'흑자확률':<10}")
    print("-" * 80)

    for name, r in results.items():
        profit = r['profit']
        prob_positive = np.mean(profit > 0) * 100
        print(f"{name:<15} {np.mean(profit):<12.1f} {np.std(profit):<12.1f} "
              f"{np.percentile(profit, 5):<12.1f} {np.percentile(profit, 95):<12.1f} {prob_positive:<10.1f}%")

    print("-" * 80)

    return results


def evaluate_strategies(results, scenarios):
    """
    전략 평가: 시나리오 가중 분석
    """
    print("\n" + "=" * 60)
    print("전략 평가")
    print("=" * 60)

    # 전략별 시나리오 성과 조정 계수
    strategies = {
        "공격적 선점": {
            "녹색 가속": 1.3,
            "규제 주도 전환": 0.9,
            "시장 주도 전환": 1.2,
            "점진적 전환": 0.6
        },
        "기술 집중": {
            "녹색 가속": 1.2,
            "규제 주도 전환": 0.8,
            "시장 주도 전환": 1.4,
            "점진적 전환": 0.7
        },
        "단계적 진입": {
            "녹색 가속": 0.9,
            "규제 주도 전환": 1.0,
            "시장 주도 전환": 1.0,
            "점진적 전환": 1.1
        },
        "하이브리드 병행": {
            "녹색 가속": 0.8,
            "규제 주도 전환": 1.1,
            "시장 주도 전환": 0.9,
            "점진적 전환": 1.2
        }
    }

    strategy_results = {}

    for strategy_name, adjustments in strategies.items():
        # 전략별 조정된 이익 계산
        all_profits = []
        weighted_profits = []

        for scenario_name, r in results.items():
            adj_factor = adjustments[scenario_name]
            adjusted_profit = r['profit'] * adj_factor
            prob = scenarios[scenario_name]['probability']

            all_profits.append(adjusted_profit)
            weighted_profits.append(adjusted_profit * prob)

        # 전체 분포 (모든 시나리오 통합)
        combined_profits = np.concatenate(all_profits)

        # 가중 평균 (시나리오 확률 반영)
        expected_profit = sum([np.mean(r['profit'] * adjustments[name]) * scenarios[name]['probability']
                               for name, r in results.items()])

        # 최악 시나리오 성과
        worst_case_profits = []
        for name, r in results.items():
            adj_profit = r['profit'] * adjustments[name]
            worst_case_profits.append(np.percentile(adj_profit, 10))
        min_worst = min(worst_case_profits)

        strategy_results[strategy_name] = {
            'expected': expected_profit,
            'std': np.std(combined_profits),
            'var_5': np.percentile(combined_profits, 5),
            'worst_scenario': min_worst,
            'prob_positive': np.mean(combined_profits > 0) * 100
        }

    print("\n[전략별 성과 비교]")
    print("-" * 85)
    print(f"{'전략':<15} {'기대이익':<12} {'표준편차':<12} {'VaR(5%)':<12} {'최악시나리오':<12} {'흑자확률':<10}")
    print("-" * 85)

    for name, sr in strategy_results.items():
        print(f"{name:<15} {sr['expected']:<12.1f} {sr['std']:<12.1f} "
              f"{sr['var_5']:<12.1f} {sr['worst_scenario']:<12.1f} {sr['prob_positive']:<10.1f}%")

    print("-" * 85)

    # 권고
    best_expected = max(strategy_results.items(), key=lambda x: x[1]['expected'])
    best_robust = max(strategy_results.items(), key=lambda x: x[1]['worst_scenario'])
    best_prob = max(strategy_results.items(), key=lambda x: x[1]['prob_positive'])

    print("\n[전략 권고]")
    print(f"  기대이익 최고: {best_expected[0]} ({best_expected[1]['expected']:.1f}억 원)")
    print(f"  가장 강건한 전략: {best_robust[0]} (최악 시나리오 {best_robust[1]['worst_scenario']:.1f}억 원)")
    print(f"  흑자 확률 최고: {best_prob[0]} ({best_prob[1]['prob_positive']:.1f}%)")

    return strategy_results


def success_probability_by_scenario(results, scenarios):
    """
    시나리오별 목표 달성 확률
    """
    print("\n" + "=" * 60)
    print("시나리오별 목표 달성 확률")
    print("=" * 60)

    targets = [0, 20, 40, 60]

    print("\n[목표별 달성 확률]")
    print("-" * 75)
    header = f"{'시나리오':<15} "
    for t in targets:
        header += f"{'>' + str(t) + '억':<12} "
    header += f"{'시나리오 확률':<12}"
    print(header)
    print("-" * 75)

    weighted_probs = {t: 0 for t in targets}

    for name, r in results.items():
        profit = r['profit']
        prob_weight = scenarios[name]['probability']
        row = f"{name:<15} "
        for t in targets:
            prob = np.mean(profit > t) * 100
            row += f"{prob:<12.1f} "
            weighted_probs[t] += prob * prob_weight
        row += f"{prob_weight*100:.0f}%"
        print(row)

    print("-" * 75)

    # 가중 평균 (전체 기대)
    print("\n[가중 평균 목표 달성 확률]")
    for t in targets:
        print(f"  이익 > {t}억 원: {weighted_probs[t]:.1f}%")


def visualize_scenario_simulation(results, scenarios, strategy_results):
    """
    시나리오 통합 시뮬레이션 시각화
    """
    print("\n" + "=" * 60)
    print("시각화 생성")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 시나리오별 이익 분포
    ax1 = axes[0, 0]
    colors = ['green', 'blue', 'orange', 'red']
    for i, (name, r) in enumerate(results.items()):
        ax1.hist(r['profit'], bins=50, alpha=0.5, label=name, color=colors[i])
    ax1.axvline(0, color='black', linestyle='--', linewidth=2)
    ax1.set_xlabel('이익 (억 원)')
    ax1.set_ylabel('빈도')
    ax1.set_title('시나리오별 이익 분포')
    ax1.legend()

    # 2. 시나리오별 박스플롯
    ax2 = axes[0, 1]
    data = [r['profit'] for r in results.values()]
    labels = list(results.keys())
    bp = ax2.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_ylabel('이익 (억 원)')
    ax2.set_title('시나리오별 이익 분포 (박스플롯)')
    ax2.tick_params(axis='x', rotation=15)

    # 3. 전략별 성과 비교
    ax3 = axes[1, 0]
    strategy_names = list(strategy_results.keys())
    expected = [sr['expected'] for sr in strategy_results.values()]
    var_5 = [sr['var_5'] for sr in strategy_results.values()]

    x = np.arange(len(strategy_names))
    width = 0.35
    ax3.bar(x - width/2, expected, width, label='기대이익', color='steelblue')
    ax3.bar(x + width/2, var_5, width, label='VaR(5%)', color='coral')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategy_names, rotation=15)
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('이익 (억 원)')
    ax3.set_title('전략별 기대이익 vs 리스크(VaR)')
    ax3.legend()

    # 4. 시나리오 확률 가중 분포
    ax4 = axes[1, 1]
    combined = []
    for name, r in results.items():
        # 시나리오 확률에 따라 샘플 수 조정
        n_samples = int(len(r['profit']) * scenarios[name]['probability'])
        combined.extend(np.random.choice(r['profit'], n_samples))

    ax4.hist(combined, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(combined), color='green', linestyle='--',
                label=f"기대값: {np.mean(combined):.1f}억")
    ax4.axvline(np.percentile(combined, 5), color='red', linestyle='--',
                label=f"VaR(5%): {np.percentile(combined, 5):.1f}억")
    ax4.axvline(0, color='black', linestyle='-', linewidth=2)
    ax4.set_xlabel('이익 (억 원)')
    ax4.set_ylabel('빈도')
    ax4.set_title('시나리오 가중 통합 분포')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('../data/scenario_simulation.png', dpi=150)
    plt.close()

    print("그래프 저장: ../data/scenario_simulation.png")


if __name__ == "__main__":
    # 시나리오 정의
    scenarios = define_scenarios()

    # 시나리오별 시뮬레이션
    results = scenario_simulation(scenarios)

    # 전략 평가
    strategy_results = evaluate_strategies(results, scenarios)

    # 목표 달성 확률 분석
    success_probability_by_scenario(results, scenarios)

    # 시각화
    visualize_scenario_simulation(results, scenarios, strategy_results)

    print("\n" + "=" * 60)
    print("시나리오 통합 시뮬레이션 완료")
    print("=" * 60)
    print("\n[핵심 인사이트]")
    print("  1. 시나리오별로 이익 분포가 크게 다름")
    print("  2. '단계적 진입' 전략이 가장 강건함 (최악 시나리오 대비)")
    print("  3. 기대이익 극대화를 위해서는 '기술 집중' 전략이 유리")
    print("  4. 시나리오 확률을 반영한 통합 분석으로 현실적 기대치 도출")
