"""
13-1-binomial-option.py
실물옵션 가치 평가: 이항 모델 기초

이 코드는 이항 모델(Binomial Model)을 사용하여
실물옵션의 가치를 평가하는 기본 프레임워크를 제공한다.
금융옵션에서 출발하여 실물옵션으로 확장하는 과정을 보여준다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def binomial_tree_option(
    S0: float,          # 현재 기초자산 가치
    K: float,           # 행사가격
    T: float,           # 만기 (년)
    r: float,           # 무위험 이자율
    sigma: float,       # 변동성
    n_steps: int,       # 시간 단계 수
    option_type: str = 'call',  # 'call' 또는 'put'
    american: bool = True       # 미국형 옵션 여부
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    이항 모델을 사용하여 옵션 가치 계산

    Parameters:
    -----------
    S0: 현재 자산 가치 (예: 프로젝트 현재가치)
    K: 행사가격 (예: 투자비용)
    T: 만기까지 기간 (년)
    r: 무위험 이자율 (연율)
    sigma: 변동성 (연간 표준편차)
    n_steps: 이항 트리 단계 수
    option_type: 'call' (투자 권리) 또는 'put' (포기 권리)
    american: True면 조기 행사 가능

    Returns:
    --------
    option_value: 옵션 가치
    asset_tree: 자산 가치 트리
    option_tree: 옵션 가치 트리
    """

    # 시간 간격
    dt = T / n_steps

    # 상승/하락 비율 계산 (Cox-Ross-Rubinstein 모델)
    u = np.exp(sigma * np.sqrt(dt))  # 상승 비율
    d = 1 / u                         # 하락 비율

    # 위험중립 확률
    p = (np.exp(r * dt) - d) / (u - d)

    # 할인 계수
    discount = np.exp(-r * dt)

    # 자산 가치 트리 생성
    asset_tree = np.zeros((n_steps + 1, n_steps + 1))
    for i in range(n_steps + 1):
        for j in range(i + 1):
            asset_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # 만기 시점의 옵션 가치 (내재가치)
    option_tree = np.zeros((n_steps + 1, n_steps + 1))

    for j in range(n_steps + 1):
        if option_type == 'call':
            option_tree[j, n_steps] = max(0, asset_tree[j, n_steps] - K)
        else:  # put
            option_tree[j, n_steps] = max(0, K - asset_tree[j, n_steps])

    # 역방향 귀납법으로 옵션 가치 계산
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            # 계속 보유 가치 (기대값의 현재가치)
            hold_value = discount * (p * option_tree[j, i + 1] +
                                    (1 - p) * option_tree[j + 1, i + 1])

            if american:
                # 조기 행사 가치
                if option_type == 'call':
                    exercise_value = max(0, asset_tree[j, i] - K)
                else:
                    exercise_value = max(0, K - asset_tree[j, i])

                option_tree[j, i] = max(hold_value, exercise_value)
            else:
                option_tree[j, i] = hold_value

    return option_tree[0, 0], asset_tree, option_tree


def npv_analysis(
    initial_investment: float,
    expected_cash_flows: List[float],
    discount_rate: float
) -> float:
    """
    전통적 NPV 계산
    """
    npv = -initial_investment
    for t, cf in enumerate(expected_cash_flows, 1):
        npv += cf / ((1 + discount_rate) ** t)
    return npv


def real_option_vs_npv(
    project_value: float,      # 프로젝트 기대 가치
    investment_cost: float,    # 투자 비용
    volatility: float,         # 가치 변동성
    time_to_decide: float,     # 의사결정 기간
    risk_free_rate: float,     # 무위험 이자율
    n_steps: int = 50
) -> dict:
    """
    전통적 NPV와 실물옵션 가치 비교

    실물옵션 관점에서 투자 결정은 콜옵션과 유사:
    - 기초자산: 프로젝트 가치
    - 행사가격: 투자 비용
    - 만기: 의사결정 기한
    """

    # 전통적 NPV
    traditional_npv = project_value - investment_cost

    # 실물옵션 가치 (투자 권리 = 콜옵션)
    option_value, asset_tree, option_tree = binomial_tree_option(
        S0=project_value,
        K=investment_cost,
        T=time_to_decide,
        r=risk_free_rate,
        sigma=volatility,
        n_steps=n_steps,
        option_type='call',
        american=True
    )

    # 확장 NPV = 전통적 NPV + 유연성 가치
    flexibility_value = option_value - max(0, traditional_npv)
    expanded_npv = traditional_npv + flexibility_value

    return {
        'traditional_npv': traditional_npv,
        'option_value': option_value,
        'flexibility_value': flexibility_value,
        'expanded_npv': expanded_npv,
        'asset_tree': asset_tree,
        'option_tree': option_tree
    }


def sensitivity_analysis(
    base_project_value: float,
    investment_cost: float,
    base_volatility: float,
    time_to_decide: float,
    risk_free_rate: float
) -> pd.DataFrame:
    """
    변동성과 프로젝트 가치에 대한 민감도 분석
    """

    volatilities = [0.1, 0.2, 0.3, 0.4, 0.5]
    value_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]  # 기준 대비 비율

    results = []

    for vol in volatilities:
        for ratio in value_ratios:
            project_value = base_project_value * ratio

            result = real_option_vs_npv(
                project_value=project_value,
                investment_cost=investment_cost,
                volatility=vol,
                time_to_decide=time_to_decide,
                risk_free_rate=risk_free_rate
            )

            results.append({
                '변동성': f'{vol:.0%}',
                '프로젝트가치비율': f'{ratio:.0%}',
                'NPV': result['traditional_npv'],
                '옵션가치': result['option_value'],
                '유연성가치': result['flexibility_value']
            })

    return pd.DataFrame(results)


def plot_binomial_tree(asset_tree: np.ndarray, option_tree: np.ndarray,
                        n_display: int = 5, save_path: str = None):
    """
    이항 트리 시각화 (처음 n_display 단계만)
    """
    n_steps = min(n_display, asset_tree.shape[1] - 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 자산 가치 트리
    ax1 = axes[0]
    for i in range(n_steps + 1):
        for j in range(i + 1):
            value = asset_tree[j, i]
            ax1.scatter(i, i - 2*j, s=300, c='steelblue', zorder=3)
            ax1.annotate(f'{value:.0f}', (i, i - 2*j),
                        ha='center', va='center', fontsize=8, color='white')

            if i < n_steps:
                # 상승 연결선
                ax1.plot([i, i+1], [i - 2*j, i + 1 - 2*j],
                        'g-', alpha=0.5, linewidth=1)
                # 하락 연결선
                ax1.plot([i, i+1], [i - 2*j, i + 1 - 2*(j+1)],
                        'r-', alpha=0.5, linewidth=1)

    ax1.set_title('자산 가치 트리 (억 원)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('시간 단계')
    ax1.set_ylabel('상태')
    ax1.grid(True, alpha=0.3)

    # 옵션 가치 트리
    ax2 = axes[1]
    for i in range(n_steps + 1):
        for j in range(i + 1):
            value = option_tree[j, i]
            color = 'darkgreen' if value > 0 else 'gray'
            ax2.scatter(i, i - 2*j, s=300, c=color, zorder=3)
            ax2.annotate(f'{value:.1f}', (i, i - 2*j),
                        ha='center', va='center', fontsize=8, color='white')

            if i < n_steps:
                ax2.plot([i, i+1], [i - 2*j, i + 1 - 2*j],
                        'g-', alpha=0.5, linewidth=1)
                ax2.plot([i, i+1], [i - 2*j, i + 1 - 2*(j+1)],
                        'r-', alpha=0.5, linewidth=1)

    ax2.set_title('옵션 가치 트리 (억 원)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('시간 단계')
    ax2.set_ylabel('상태')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")

    plt.show()


def main():
    """
    실물옵션 분석 메인 실행

    사례: 신규 시장 진출 투자 결정
    - 현재 프로젝트 가치 추정: 100억 원
    - 필요 투자비용: 90억 원
    - 의사결정 기한: 2년
    - 연간 변동성: 30%
    - 무위험 이자율: 3%
    """

    print("=" * 60)
    print("실물옵션 분석: 이항 모델")
    print("=" * 60)

    # 파라미터 설정
    project_value = 100  # 억 원
    investment_cost = 90  # 억 원
    volatility = 0.30     # 30%
    time_to_decide = 2    # 2년
    risk_free_rate = 0.03 # 3%
    n_steps = 50

    print("\n[파라미터 설정]")
    print(f"  프로젝트 현재가치: {project_value}억 원")
    print(f"  투자비용: {investment_cost}억 원")
    print(f"  변동성: {volatility:.0%}")
    print(f"  의사결정 기간: {time_to_decide}년")
    print(f"  무위험 이자율: {risk_free_rate:.0%}")

    # 분석 수행
    result = real_option_vs_npv(
        project_value=project_value,
        investment_cost=investment_cost,
        volatility=volatility,
        time_to_decide=time_to_decide,
        risk_free_rate=risk_free_rate,
        n_steps=n_steps
    )

    print("\n[분석 결과]")
    print("-" * 40)
    print(f"전통적 NPV:     {result['traditional_npv']:>10.2f}억 원")
    print(f"실물옵션 가치:  {result['option_value']:>10.2f}억 원")
    print(f"유연성 가치:    {result['flexibility_value']:>10.2f}억 원")
    print(f"확장 NPV:       {result['expanded_npv']:>10.2f}억 원")
    print("-" * 40)

    # 의사결정 권고
    print("\n[의사결정 권고]")
    if result['traditional_npv'] > 0:
        print(f"  전통적 NPV 기준: 투자 실행 권고 (NPV = {result['traditional_npv']:.2f}억 원 > 0)")
    else:
        print(f"  전통적 NPV 기준: 투자 기각 (NPV = {result['traditional_npv']:.2f}억 원 < 0)")

    print(f"\n  그러나 실물옵션 관점에서 유연성 가치 {result['flexibility_value']:.2f}억 원을 고려하면,")
    print(f"  기다렸다가 투자하는 옵션의 가치가 {result['option_value']:.2f}억 원이다.")

    if result['traditional_npv'] > 0 and result['flexibility_value'] > 0:
        print(f"\n  권고: 불확실성이 해소될 때까지 기다린 후 투자 결정")
        print(f"        기다림의 가치 = {result['flexibility_value']:.2f}억 원")

    # 민감도 분석
    print("\n\n[민감도 분석: 변동성과 프로젝트 가치]")
    print("-" * 60)

    sensitivity_df = sensitivity_analysis(
        base_project_value=project_value,
        investment_cost=investment_cost,
        base_volatility=volatility,
        time_to_decide=time_to_decide,
        risk_free_rate=risk_free_rate
    )

    # 피벗 테이블로 유연성 가치 표시
    pivot = sensitivity_df.pivot(
        index='변동성',
        columns='프로젝트가치비율',
        values='유연성가치'
    )
    print("\n유연성 가치 (억 원):")
    print(pivot.round(2).to_string())

    print("\n[핵심 통찰]")
    print("  1. 변동성이 높을수록 옵션 가치(유연성 가치)가 증가한다.")
    print("  2. NPV가 손익분기에 가까울수록 유연성의 가치가 크다.")
    print("  3. 불확실성이 높은 프로젝트에서 '기다릴 권리'의 가치가 높다.")

    # 트리 시각화
    print("\n\n[이항 트리 시각화]")
    plot_binomial_tree(
        result['asset_tree'],
        result['option_tree'],
        n_display=4,
        save_path='13-1-binomial-option.png'
    )

    return result, sensitivity_df


if __name__ == "__main__":
    result, sensitivity_df = main()
