"""
11-2-business-simulation.py
사업 계획 리스크 시뮬레이션

이 코드는 매출/비용 예측, NPV 분석, 민감도 분석 등
실무에서 활용하는 몬테카를로 시뮬레이션을 다룬다.
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


def sales_cost_simulation():
    """
    매출/비용 시뮬레이션
    """
    print("=" * 60)
    print("매출/비용 시뮬레이션")
    print("=" * 60)

    n_simulations = 10000

    # 입력 변수 정의 (삼각분포: 최소, 최빈, 최대)
    print("\n[입력 변수 정의]")
    print("  시장 규모: 800~1,200억 원 (최빈 1,000억)")
    print("  시장 점유율: 5~15% (최빈 10%)")
    print("  제품 단가: 8~12만 원 (최빈 10만)")
    print("  변동비율: 40~60% (최빈 50%)")
    print("  고정비: 30~50억 원 (최빈 40억)")

    # 시뮬레이션
    market_size = np.random.triangular(800, 1000, 1200, n_simulations)  # 억 원
    market_share = np.random.triangular(0.05, 0.10, 0.15, n_simulations)
    unit_price = np.random.triangular(8, 10, 12, n_simulations)  # 만 원
    variable_cost_ratio = np.random.triangular(0.40, 0.50, 0.60, n_simulations)
    fixed_cost = np.random.triangular(30, 40, 50, n_simulations)  # 억 원

    # 매출 계산 (시장규모 × 점유율)
    revenue = market_size * market_share  # 억 원

    # 비용 계산
    variable_cost = revenue * variable_cost_ratio
    total_cost = variable_cost + fixed_cost

    # 이익 계산
    profit = revenue - total_cost

    print("\n[시뮬레이션 결과 (10,000회)]")
    print("-" * 60)
    print(f"{'항목':<15} {'평균':<12} {'표준편차':<12} {'5%':<12} {'95%':<12}")
    print("-" * 60)
    print(f"{'매출':<15} {np.mean(revenue):<12.1f} {np.std(revenue):<12.1f} {np.percentile(revenue, 5):<12.1f} {np.percentile(revenue, 95):<12.1f}")
    print(f"{'비용':<15} {np.mean(total_cost):<12.1f} {np.std(total_cost):<12.1f} {np.percentile(total_cost, 5):<12.1f} {np.percentile(total_cost, 95):<12.1f}")
    print(f"{'이익':<15} {np.mean(profit):<12.1f} {np.std(profit):<12.1f} {np.percentile(profit, 5):<12.1f} {np.percentile(profit, 95):<12.1f}")
    print("-" * 60)

    # 손익분기점 분석
    prob_profit = np.mean(profit > 0) * 100
    prob_loss = np.mean(profit < 0) * 100
    print(f"\n[손익분기 분석]")
    print(f"  흑자 확률: {prob_profit:.1f}%")
    print(f"  적자 확률: {prob_loss:.1f}%")

    return {
        'revenue': revenue,
        'total_cost': total_cost,
        'profit': profit,
        'market_size': market_size,
        'market_share': market_share,
        'unit_price': unit_price,
        'variable_cost_ratio': variable_cost_ratio,
        'fixed_cost': fixed_cost
    }


def npv_simulation():
    """
    NPV(순현재가치) 리스크 시뮬레이션
    """
    print("\n" + "=" * 60)
    print("NPV 리스크 시뮬레이션")
    print("=" * 60)

    n_simulations = 10000
    years = 5
    discount_rate = 0.10  # 할인율 10%

    # 초기 투자
    initial_investment = 100  # 억 원

    print("\n[투자 조건]")
    print(f"  초기 투자: {initial_investment}억 원")
    print(f"  투자 기간: {years}년")
    print(f"  할인율: {discount_rate*100:.0f}%")

    print("\n[연간 현금흐름 가정]")
    print("  1년차: 20~40억 원 (삼각분포)")
    print("  2년차: 25~45억 원")
    print("  3년차: 30~50억 원")
    print("  4년차: 25~45억 원")
    print("  5년차: 20~40억 원")

    # 연도별 현금흐름 시뮬레이션
    cf_params = [
        (20, 30, 40),   # 1년차
        (25, 35, 45),   # 2년차
        (30, 40, 50),   # 3년차
        (25, 35, 45),   # 4년차
        (20, 30, 40),   # 5년차
    ]

    cash_flows = np.zeros((n_simulations, years))
    for year, (low, mode, high) in enumerate(cf_params):
        cash_flows[:, year] = np.random.triangular(low, mode, high, n_simulations)

    # NPV 계산
    npv_values = np.zeros(n_simulations)
    for sim in range(n_simulations):
        npv = -initial_investment
        for year in range(years):
            npv += cash_flows[sim, year] / ((1 + discount_rate) ** (year + 1))
        npv_values[sim] = npv

    print("\n[NPV 시뮬레이션 결과]")
    print(f"  평균 NPV: {np.mean(npv_values):.1f}억 원")
    print(f"  표준편차: {np.std(npv_values):.1f}억 원")
    print(f"  최소 NPV: {np.min(npv_values):.1f}억 원")
    print(f"  최대 NPV: {np.max(npv_values):.1f}억 원")

    print(f"\n[NPV 분포]")
    print(f"  5% 분위: {np.percentile(npv_values, 5):.1f}억 원")
    print(f"  25% 분위: {np.percentile(npv_values, 25):.1f}억 원")
    print(f"  50% 분위 (중앙값): {np.percentile(npv_values, 50):.1f}억 원")
    print(f"  75% 분위: {np.percentile(npv_values, 75):.1f}억 원")
    print(f"  95% 분위: {np.percentile(npv_values, 95):.1f}억 원")

    # 투자 의사결정
    prob_positive = np.mean(npv_values > 0) * 100
    print(f"\n[투자 의사결정 지원]")
    print(f"  NPV > 0 확률: {prob_positive:.1f}%")
    print(f"  VaR(5%): {np.percentile(npv_values, 5):.1f}억 원")

    if prob_positive >= 70:
        print(f"  권고: 투자 추진 (성공 확률 {prob_positive:.0f}%)")
    elif prob_positive >= 50:
        print(f"  권고: 추가 검토 필요 (성공 확률 {prob_positive:.0f}%)")
    else:
        print(f"  권고: 투자 재고 (성공 확률 {prob_positive:.0f}%)")

    return npv_values, cash_flows


def sensitivity_analysis(simulation_results):
    """
    민감도 분석: 토네이도 차트
    """
    print("\n" + "=" * 60)
    print("민감도 분석 (토네이도 차트)")
    print("=" * 60)

    # 기준값 (중앙값)
    base_revenue = np.median(simulation_results['revenue'])
    base_profit = np.median(simulation_results['profit'])

    # 각 변수의 상하위 10%에서 이익 변화 분석
    variables = {
        '시장 규모': simulation_results['market_size'],
        '시장 점유율': simulation_results['market_share'],
        '변동비율': simulation_results['variable_cost_ratio'],
        '고정비': simulation_results['fixed_cost']
    }

    profit = simulation_results['profit']

    print("\n[변수별 이익 민감도]")
    print("-" * 70)
    print(f"{'변수':<15} {'하위 10% 이익':<18} {'상위 10% 이익':<18} {'변동폭':<15}")
    print("-" * 70)

    sensitivities = []
    for var_name, var_values in variables.items():
        # 변수가 하위 10%일 때 이익
        low_mask = var_values <= np.percentile(var_values, 10)
        low_profit = np.mean(profit[low_mask])

        # 변수가 상위 10%일 때 이익
        high_mask = var_values >= np.percentile(var_values, 90)
        high_profit = np.mean(profit[high_mask])

        swing = abs(high_profit - low_profit)
        sensitivities.append({
            'name': var_name,
            'low': low_profit,
            'high': high_profit,
            'swing': swing
        })

        print(f"{var_name:<15} {low_profit:<18.1f} {high_profit:<18.1f} {swing:<15.1f}")

    print("-" * 70)

    # 민감도 순위
    sensitivities.sort(key=lambda x: x['swing'], reverse=True)
    print("\n[민감도 순위 (변동폭 기준)]")
    for i, s in enumerate(sensitivities, 1):
        print(f"  {i}. {s['name']}: {s['swing']:.1f}억 원")

    return sensitivities


def visualize_business_simulation(sim_results, npv_values, sensitivities):
    """
    사업 시뮬레이션 결과 시각화
    """
    print("\n" + "=" * 60)
    print("시각화 생성")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 이익 분포
    ax1 = axes[0, 0]
    ax1.hist(sim_results['profit'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='손익분기점')
    ax1.axvline(np.mean(sim_results['profit']), color='green', linestyle='--', label=f"평균: {np.mean(sim_results['profit']):.1f}억")
    ax1.set_xlabel('이익 (억 원)')
    ax1.set_ylabel('빈도')
    ax1.set_title('사업 이익 분포')
    ax1.legend()

    # 2. NPV 분포
    ax2 = axes[0, 1]
    ax2.hist(npv_values, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='투자 기준선')
    ax2.axvline(np.mean(npv_values), color='green', linestyle='--', label=f"평균: {np.mean(npv_values):.1f}억")
    ax2.set_xlabel('NPV (억 원)')
    ax2.set_ylabel('빈도')
    ax2.set_title('NPV 분포')
    ax2.legend()

    # 3. 토네이도 차트
    ax3 = axes[1, 0]
    sensitivities_sorted = sorted(sensitivities, key=lambda x: x['swing'])
    names = [s['name'] for s in sensitivities_sorted]
    lows = [s['low'] - np.mean(sim_results['profit']) for s in sensitivities_sorted]
    highs = [s['high'] - np.mean(sim_results['profit']) for s in sensitivities_sorted]

    y_pos = np.arange(len(names))
    ax3.barh(y_pos, lows, color='salmon', alpha=0.7, label='하위 10%')
    ax3.barh(y_pos, highs, color='lightgreen', alpha=0.7, label='상위 10%')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names)
    ax3.axvline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('이익 변화 (억 원, 평균 대비)')
    ax3.set_title('토네이도 차트: 변수별 민감도')
    ax3.legend()

    # 4. 매출-이익 산점도
    ax4 = axes[1, 1]
    scatter = ax4.scatter(sim_results['revenue'], sim_results['profit'],
                          c=sim_results['market_share'], cmap='viridis', alpha=0.3, s=5)
    ax4.axhline(0, color='red', linestyle='--', linewidth=1)
    ax4.set_xlabel('매출 (억 원)')
    ax4.set_ylabel('이익 (억 원)')
    ax4.set_title('매출-이익 관계 (색상: 시장점유율)')
    plt.colorbar(scatter, ax=ax4, label='시장점유율')

    plt.tight_layout()
    plt.savefig('../data/business_simulation.png', dpi=150)
    plt.close()

    print("그래프 저장: ../data/business_simulation.png")


if __name__ == "__main__":
    # 매출/비용 시뮬레이션
    sim_results = sales_cost_simulation()

    # NPV 시뮬레이션
    npv_values, cash_flows = npv_simulation()

    # 민감도 분석
    sensitivities = sensitivity_analysis(sim_results)

    # 시각화
    visualize_business_simulation(sim_results, npv_values, sensitivities)

    print("\n" + "=" * 60)
    print("사업 계획 리스크 시뮬레이션 완료")
    print("=" * 60)
