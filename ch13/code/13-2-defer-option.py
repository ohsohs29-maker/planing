"""
13-2-defer-option.py
연기 옵션(Option to Defer) 분석

이 코드는 투자 연기 옵션의 가치를 분석한다.
불확실성이 높은 상황에서 "기다렸다 결정할 권리"의
경제적 가치를 정량화한다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def defer_option_analysis(
    current_value: float,       # 현재 프로젝트 가치
    investment_cost: float,     # 투자 비용
    volatility: float,          # 연간 변동성
    max_defer_years: int,       # 최대 연기 기간
    risk_free_rate: float,      # 무위험 이자율
    dividend_yield: float = 0,  # 기회비용 (연기로 인한 손실)
    n_steps_per_year: int = 12  # 연간 시간 단계 수
) -> Dict:
    """
    연기 옵션 가치 분석

    Parameters:
    -----------
    current_value: 프로젝트 현재가치
    investment_cost: 필요 투자금액
    volatility: 가치의 연간 변동성
    max_defer_years: 연기 가능한 최대 기간
    risk_free_rate: 무위험 이자율
    dividend_yield: 연기에 따른 기회비용 (경쟁자 진입 등)
    n_steps_per_year: 연간 시뮬레이션 단계

    Returns:
    --------
    분석 결과 딕셔너리
    """

    n_steps = max_defer_years * n_steps_per_year
    dt = 1 / n_steps_per_year

    # 상승/하락 비율
    u = np.exp(volatility * np.sqrt(dt))
    d = 1 / u

    # 배당수익률 고려한 위험중립 확률
    p = (np.exp((risk_free_rate - dividend_yield) * dt) - d) / (u - d)

    # 할인 계수
    discount = np.exp(-risk_free_rate * dt)

    # 자산 가치 트리
    asset_tree = np.zeros((n_steps + 1, n_steps + 1))
    for i in range(n_steps + 1):
        for j in range(i + 1):
            asset_tree[j, i] = current_value * (u ** (i - j)) * (d ** j)

    # 옵션 가치 트리 (미국형 콜옵션)
    option_tree = np.zeros((n_steps + 1, n_steps + 1))
    exercise_decision = np.zeros((n_steps + 1, n_steps + 1))  # 1=행사, 0=보유

    # 만기 시점
    for j in range(n_steps + 1):
        intrinsic = max(0, asset_tree[j, n_steps] - investment_cost)
        option_tree[j, n_steps] = intrinsic
        if intrinsic > 0:
            exercise_decision[j, n_steps] = 1

    # 역방향 귀납법
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            hold_value = discount * (p * option_tree[j, i + 1] +
                                    (1 - p) * option_tree[j + 1, i + 1])
            exercise_value = max(0, asset_tree[j, i] - investment_cost)

            if exercise_value >= hold_value and exercise_value > 0:
                option_tree[j, i] = exercise_value
                exercise_decision[j, i] = 1
            else:
                option_tree[j, i] = hold_value
                exercise_decision[j, i] = 0

    # 즉시 투자 가치
    immediate_npv = current_value - investment_cost

    # 옵션 가치
    option_value = option_tree[0, 0]

    # 연기 가치 = 옵션 가치 - 즉시 투자 가치 (즉시 투자가 이익인 경우)
    defer_value = option_value - max(0, immediate_npv)

    # 최적 투자 시점 분석 (시뮬레이션)
    optimal_timing = analyze_optimal_timing(
        asset_tree, option_tree, exercise_decision,
        investment_cost, n_steps_per_year
    )

    return {
        'immediate_npv': immediate_npv,
        'option_value': option_value,
        'defer_value': defer_value,
        'asset_tree': asset_tree,
        'option_tree': option_tree,
        'exercise_decision': exercise_decision,
        'optimal_timing': optimal_timing,
        'parameters': {
            'current_value': current_value,
            'investment_cost': investment_cost,
            'volatility': volatility,
            'max_defer_years': max_defer_years,
            'risk_free_rate': risk_free_rate,
            'dividend_yield': dividend_yield
        }
    }


def analyze_optimal_timing(
    asset_tree: np.ndarray,
    option_tree: np.ndarray,
    exercise_decision: np.ndarray,
    investment_cost: float,
    n_steps_per_year: int
) -> Dict:
    """
    최적 투자 시점 분석
    """

    n_steps = asset_tree.shape[1] - 1

    # 각 시점에서 행사가 최적인 자산 가치 임계값 찾기
    thresholds = []

    for i in range(n_steps + 1):
        exercised_values = []
        for j in range(i + 1):
            if exercise_decision[j, i] == 1:
                exercised_values.append(asset_tree[j, i])

        if exercised_values:
            thresholds.append({
                'time': i / n_steps_per_year,
                'min_value': min(exercised_values),
                'threshold': min(exercised_values)
            })

    # 임계값 트렌드
    if thresholds:
        threshold_df = pd.DataFrame(thresholds)
    else:
        threshold_df = pd.DataFrame(columns=['time', 'min_value', 'threshold'])

    return {
        'thresholds': threshold_df,
        'investment_cost': investment_cost
    }


def compare_strategies(
    current_value: float,
    investment_cost: float,
    volatility: float,
    risk_free_rate: float,
    max_defer_years: int = 3
) -> pd.DataFrame:
    """
    다양한 전략 비교
    1. 즉시 투자
    2. 1년 후 결정
    3. 2년 후 결정
    4. 최대 연기 기간까지 대기
    """

    results = []

    # 즉시 투자
    immediate_npv = current_value - investment_cost
    results.append({
        '전략': '즉시 투자',
        'NPV/옵션가치': immediate_npv,
        '설명': 'NPV가 양수면 즉시 실행'
    })

    # 다양한 연기 기간
    for years in range(1, max_defer_years + 1):
        analysis = defer_option_analysis(
            current_value=current_value,
            investment_cost=investment_cost,
            volatility=volatility,
            max_defer_years=years,
            risk_free_rate=risk_free_rate,
            n_steps_per_year=12
        )

        results.append({
            '전략': f'{years}년 연기 옵션',
            'NPV/옵션가치': analysis['option_value'],
            '설명': f'최대 {years}년간 기다릴 수 있는 권리'
        })

    return pd.DataFrame(results)


def plot_defer_analysis(result: Dict, save_path: str = None):
    """
    연기 옵션 분석 결과 시각화
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    params = result['parameters']

    # 1. 가치 비교 막대 그래프
    ax1 = axes[0, 0]
    values = [result['immediate_npv'], result['option_value'], result['defer_value']]
    labels = ['즉시투자 NPV', '연기옵션 가치', '연기의 가치']
    colors = ['steelblue' if v >= 0 else 'salmon' for v in values]

    bars = ax1.bar(labels, values, color=colors, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('가치 (억 원)')
    ax1.set_title('즉시 투자 vs 연기 옵션', fontsize=12, fontweight='bold')

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.annotate(f'{value:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')

    # 2. 변동성별 연기 가치
    ax2 = axes[0, 1]
    volatilities = np.linspace(0.1, 0.6, 20)
    defer_values = []

    for vol in volatilities:
        temp_result = defer_option_analysis(
            current_value=params['current_value'],
            investment_cost=params['investment_cost'],
            volatility=vol,
            max_defer_years=params['max_defer_years'],
            risk_free_rate=params['risk_free_rate']
        )
        defer_values.append(temp_result['defer_value'])

    ax2.plot(volatilities * 100, defer_values, 'b-', linewidth=2)
    ax2.axvline(x=params['volatility'] * 100, color='red',
                linestyle='--', label=f"현재 변동성 ({params['volatility']:.0%})")
    ax2.set_xlabel('변동성 (%)')
    ax2.set_ylabel('연기 가치 (억 원)')
    ax2.set_title('변동성과 연기 가치의 관계', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 투자비용별 분석
    ax3 = axes[1, 0]
    cost_ratios = np.linspace(0.7, 1.3, 20)
    option_values = []
    npv_values = []

    for ratio in cost_ratios:
        cost = params['investment_cost'] * ratio
        temp_result = defer_option_analysis(
            current_value=params['current_value'],
            investment_cost=cost,
            volatility=params['volatility'],
            max_defer_years=params['max_defer_years'],
            risk_free_rate=params['risk_free_rate']
        )
        option_values.append(temp_result['option_value'])
        npv_values.append(params['current_value'] - cost)

    ax3.plot(cost_ratios * 100, option_values, 'b-', linewidth=2, label='옵션 가치')
    ax3.plot(cost_ratios * 100, npv_values, 'g--', linewidth=2, label='즉시투자 NPV')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(x=100, color='red', linestyle=':', label='현재 투자비용')
    ax3.set_xlabel('투자비용 비율 (%)')
    ax3.set_ylabel('가치 (억 원)')
    ax3.set_title('투자비용 변화에 따른 가치', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 연기 기간별 옵션 가치
    ax4 = axes[1, 1]
    defer_periods = range(1, 6)
    period_values = []

    for years in defer_periods:
        temp_result = defer_option_analysis(
            current_value=params['current_value'],
            investment_cost=params['investment_cost'],
            volatility=params['volatility'],
            max_defer_years=years,
            risk_free_rate=params['risk_free_rate']
        )
        period_values.append(temp_result['option_value'])

    ax4.bar(defer_periods, period_values, color='steelblue', edgecolor='black')
    ax4.axhline(y=result['immediate_npv'], color='red',
                linestyle='--', label=f"즉시투자 NPV ({result['immediate_npv']:.2f})")
    ax4.set_xlabel('최대 연기 기간 (년)')
    ax4.set_ylabel('옵션 가치 (억 원)')
    ax4.set_title('연기 기간과 옵션 가치', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")

    plt.show()


def main():
    """
    연기 옵션 분석 메인 실행

    사례: 신규 공장 건설 투자 결정
    - 공장 가동 시 현재가치: 200억 원
    - 건설 투자비용: 180억 원
    - 가치 변동성: 35% (원자재, 수요 불확실성)
    - 최대 연기 기간: 3년 (부지 매입 옵션 만료)
    - 무위험 이자율: 3%
    """

    print("=" * 60)
    print("연기 옵션(Option to Defer) 분석")
    print("=" * 60)

    # 파라미터 설정
    current_value = 200      # 억 원
    investment_cost = 180    # 억 원
    volatility = 0.35        # 35%
    max_defer_years = 3      # 3년
    risk_free_rate = 0.03    # 3%

    print("\n[사례: 신규 공장 건설 투자]")
    print(f"  프로젝트 현재가치: {current_value}억 원")
    print(f"  투자비용: {investment_cost}억 원")
    print(f"  가치 변동성: {volatility:.0%}")
    print(f"  최대 연기 기간: {max_defer_years}년")
    print(f"  무위험 이자율: {risk_free_rate:.0%}")

    # 분석 수행
    result = defer_option_analysis(
        current_value=current_value,
        investment_cost=investment_cost,
        volatility=volatility,
        max_defer_years=max_defer_years,
        risk_free_rate=risk_free_rate
    )

    print("\n[분석 결과]")
    print("-" * 50)
    print(f"즉시 투자 NPV:      {result['immediate_npv']:>12.2f}억 원")
    print(f"연기 옵션 가치:     {result['option_value']:>12.2f}억 원")
    print(f"연기의 가치:        {result['defer_value']:>12.2f}억 원")
    print("-" * 50)

    # 전략 비교
    print("\n[전략 비교]")
    strategy_df = compare_strategies(
        current_value=current_value,
        investment_cost=investment_cost,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        max_defer_years=max_defer_years
    )
    print(strategy_df.to_string(index=False))

    # 의사결정 권고
    print("\n[의사결정 권고]")
    if result['immediate_npv'] > 0:
        print(f"  즉시 투자 시 NPV = {result['immediate_npv']:.2f}억 원 (양수)")
    else:
        print(f"  즉시 투자 시 NPV = {result['immediate_npv']:.2f}억 원 (음수)")

    if result['defer_value'] > 0:
        print(f"\n  연기의 가치 = {result['defer_value']:.2f}억 원")
        print(f"  → 기다렸다가 투자하는 것이 {result['defer_value']:.2f}억 원만큼 더 가치있다.")
        print(f"\n  권고: 시장 상황 관찰 후 투자 시점 결정")
        print(f"        (최대 {max_defer_years}년 이내)")
    else:
        print(f"\n  연기의 가치가 없거나 음수임")
        print(f"  → 즉시 투자 실행 권고")

    # 기회비용 분석
    print("\n\n[기회비용 민감도 분석]")
    print("-" * 50)
    dividend_yields = [0, 0.03, 0.05, 0.08, 0.10]
    print(f"{'기회비용율':<15} {'옵션 가치':<15} {'연기 가치':<15}")
    print("-" * 50)

    for dy in dividend_yields:
        temp_result = defer_option_analysis(
            current_value=current_value,
            investment_cost=investment_cost,
            volatility=volatility,
            max_defer_years=max_defer_years,
            risk_free_rate=risk_free_rate,
            dividend_yield=dy
        )
        print(f"{dy:>10.0%}       {temp_result['option_value']:>10.2f}억 원   "
              f"{temp_result['defer_value']:>10.2f}억 원")

    print("\n[핵심 통찰]")
    print("  1. 연기에 따른 기회비용(경쟁자 진입 등)이 높으면 연기 가치가 감소한다.")
    print("  2. 변동성이 높을수록 '기다릴 권리'의 가치가 증가한다.")
    print("  3. NPV가 양수라도 연기 옵션 가치가 더 크면 기다리는 것이 최적이다.")
    print("  4. 연기 가능 기간이 길수록 옵션 가치가 높아진다.")

    # 시각화
    print("\n\n[분석 결과 시각화]")
    plot_defer_analysis(result, save_path='13-2-defer-option.png')

    return result


if __name__ == "__main__":
    result = main()
