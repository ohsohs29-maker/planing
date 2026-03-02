"""
13-3-staged-investment.py
단계적 투자 옵션(Staged Investment Options) 분석

이 코드는 R&D 프로젝트나 신규 사업처럼
여러 단계로 나누어 투자하는 경우의 옵션 가치를 분석한다.
각 단계에서 중단/계속 결정권이 갖는 가치를 정량화한다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def staged_investment_option(
    stage_investments: List[float],    # 각 단계별 투자금액
    stage_durations: List[float],      # 각 단계별 기간 (년)
    final_value: float,                # 최종 성공 시 프로젝트 가치
    volatility: float,                 # 가치 변동성
    success_probs: List[float],        # 각 단계별 기술적 성공 확률
    risk_free_rate: float,             # 무위험 이자율
    n_simulations: int = 10000         # 시뮬레이션 횟수
) -> Dict:
    """
    단계적 투자 옵션 가치 분석 (시뮬레이션 기반)

    각 단계에서:
    1. 기술적 성공 여부 결정
    2. 프로젝트 가치 업데이트 (확률적)
    3. 다음 단계 투자 여부 결정 (옵션 행사)

    Parameters:
    -----------
    stage_investments: 단계별 투자 금액 리스트
    stage_durations: 단계별 소요 기간 리스트
    final_value: 모든 단계 성공 시 예상 프로젝트 가치
    volatility: 프로젝트 가치의 연간 변동성
    success_probs: 각 단계의 기술적 성공 확률
    risk_free_rate: 무위험 이자율
    n_simulations: 몬테카를로 시뮬레이션 횟수

    Returns:
    --------
    분석 결과 딕셔너리
    """

    n_stages = len(stage_investments)
    total_investment = sum(stage_investments)
    total_duration = sum(stage_durations)

    # 전통적 NPV 계산 (모든 단계 성공 가정)
    cumulative_prob = np.prod(success_probs)
    expected_value = final_value * cumulative_prob

    traditional_npv = -total_investment
    discount_factor = np.exp(-risk_free_rate * total_duration)
    traditional_npv += expected_value * discount_factor

    # 시뮬레이션 기반 옵션 가치 계산
    simulation_results = []

    for _ in range(n_simulations):
        cumulative_time = 0
        cumulative_investment = 0
        current_value = final_value
        project_alive = True
        stage_reached = 0

        for stage in range(n_stages):
            if not project_alive:
                break

            # 단계 시작 시 투자
            cumulative_investment += stage_investments[stage]
            cumulative_time += stage_durations[stage]

            # 기술적 성공 여부
            if np.random.random() > success_probs[stage]:
                project_alive = False
                break

            # 가치 변동 (GBM)
            dt = stage_durations[stage]
            drift = (risk_free_rate - 0.5 * volatility**2) * dt
            diffusion = volatility * np.sqrt(dt) * np.random.standard_normal()
            current_value *= np.exp(drift + diffusion)

            stage_reached = stage + 1

            # 다음 단계 진행 여부 결정 (옵션)
            # 남은 투자 대비 현재 가치가 충분한가?
            remaining_investment = sum(stage_investments[stage + 1:]) if stage < n_stages - 1 else 0

            if stage < n_stages - 1:
                # 간단한 휴리스틱: 현재 가치가 남은 투자의 1.5배 미만이면 중단
                # 실제로는 더 정교한 역방향 귀납법 필요
                continue_threshold = remaining_investment * 1.2
                if current_value < continue_threshold:
                    # 옵션 행사하지 않음 (중단)
                    project_alive = False

        # 프로젝트 결과
        if project_alive and stage_reached == n_stages:
            # 성공: 최종 가치 - 총 투자 (현재가치 할인)
            final_payoff = current_value * np.exp(-risk_free_rate * total_duration)
            total_inv_pv = 0
            cum_time = 0
            for i, (inv, dur) in enumerate(zip(stage_investments, stage_durations)):
                cum_time += dur
                total_inv_pv += inv * np.exp(-risk_free_rate * cum_time)
            net_payoff = final_payoff - total_inv_pv
        else:
            # 실패: 투자금 손실 (실제 투자한 금액까지만)
            net_payoff = -cumulative_investment

        simulation_results.append({
            'net_payoff': net_payoff,
            'stage_reached': stage_reached,
            'project_success': project_alive and stage_reached == n_stages
        })

    results_df = pd.DataFrame(simulation_results)

    # 옵션 가치 = 평균 순수익 (음수 payoff는 0으로 제한하지 않음)
    option_value = results_df['net_payoff'].mean()

    # 성공률
    success_rate = results_df['project_success'].mean()

    # 단계별 도달률
    stage_reach_rates = []
    for s in range(n_stages + 1):
        rate = (results_df['stage_reached'] >= s).mean()
        stage_reach_rates.append(rate)

    # 유연성 가치
    flexibility_value = option_value - traditional_npv

    return {
        'traditional_npv': traditional_npv,
        'option_value': option_value,
        'flexibility_value': flexibility_value,
        'success_rate': success_rate,
        'stage_reach_rates': stage_reach_rates,
        'results_df': results_df,
        'parameters': {
            'stage_investments': stage_investments,
            'stage_durations': stage_durations,
            'final_value': final_value,
            'volatility': volatility,
            'success_probs': success_probs,
            'total_investment': total_investment
        }
    }


def compare_investment_approaches(
    stage_investments: List[float],
    stage_durations: List[float],
    final_value: float,
    volatility: float,
    success_probs: List[float],
    risk_free_rate: float
) -> pd.DataFrame:
    """
    단계적 투자 vs 일괄 투자 비교
    """

    total_investment = sum(stage_investments)
    total_duration = sum(stage_durations)

    # 1. 단계적 투자 (옵션 포함)
    staged_result = staged_investment_option(
        stage_investments=stage_investments,
        stage_durations=stage_durations,
        final_value=final_value,
        volatility=volatility,
        success_probs=success_probs,
        risk_free_rate=risk_free_rate
    )

    # 2. 일괄 투자 (옵션 없음)
    cumulative_prob = np.prod(success_probs)
    expected_value = final_value * cumulative_prob
    discount_factor = np.exp(-risk_free_rate * total_duration)
    lump_sum_npv = expected_value * discount_factor - total_investment

    # 3. 옵션 없는 단계적 투자 (모든 단계 의무 진행)
    # 단순 NPV와 동일

    results = [
        {
            '투자방식': '일괄 투자 (옵션 없음)',
            'NPV/가치': lump_sum_npv,
            '성공률': cumulative_prob * 100,
            '최대 손실': total_investment,
            '특징': '초기에 전액 투자, 중단 불가'
        },
        {
            '투자방식': '단계적 투자 (옵션 있음)',
            'NPV/가치': staged_result['option_value'],
            '성공률': staged_result['success_rate'] * 100,
            '최대 손실': stage_investments[0],  # 1단계만 손실 가능
            '특징': '단계별 중단 가능, 손실 제한'
        }
    ]

    return pd.DataFrame(results)


def sensitivity_by_volatility(
    stage_investments: List[float],
    stage_durations: List[float],
    final_value: float,
    success_probs: List[float],
    risk_free_rate: float
) -> pd.DataFrame:
    """
    변동성에 따른 옵션 가치 변화 분석
    """

    volatilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    results = []

    for vol in volatilities:
        result = staged_investment_option(
            stage_investments=stage_investments,
            stage_durations=stage_durations,
            final_value=final_value,
            volatility=vol,
            success_probs=success_probs,
            risk_free_rate=risk_free_rate,
            n_simulations=5000
        )

        results.append({
            '변동성': f'{vol:.0%}',
            '전통적 NPV': result['traditional_npv'],
            '옵션 가치': result['option_value'],
            '유연성 가치': result['flexibility_value']
        })

    return pd.DataFrame(results)


def plot_staged_investment(result: Dict, save_path: str = None):
    """
    단계적 투자 분석 결과 시각화
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    params = result['parameters']
    n_stages = len(params['stage_investments'])

    # 1. 가치 비교
    ax1 = axes[0, 0]
    values = [result['traditional_npv'], result['option_value'], result['flexibility_value']]
    labels = ['전통적 NPV', '단계적 투자\n옵션 가치', '유연성 가치']
    colors = ['steelblue' if v >= 0 else 'salmon' for v in values]

    bars = ax1.bar(labels, values, color=colors, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('가치 (억 원)')
    ax1.set_title('일괄 투자 vs 단계적 투자', fontsize=12, fontweight='bold')

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.annotate(f'{value:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')

    # 2. 단계별 도달률
    ax2 = axes[0, 1]
    stages = list(range(n_stages + 1))
    stage_labels = ['시작'] + [f'{i}단계 완료' for i in range(1, n_stages + 1)]

    ax2.bar(stage_labels, [r * 100 for r in result['stage_reach_rates']],
            color='steelblue', edgecolor='black')
    ax2.set_ylabel('도달률 (%)')
    ax2.set_title('단계별 프로젝트 진행률', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105)

    for i, rate in enumerate(result['stage_reach_rates']):
        ax2.annotate(f'{rate*100:.1f}%',
                    xy=(i, rate * 100 + 2),
                    ha='center', fontsize=9)

    # 3. 수익 분포
    ax3 = axes[1, 0]
    payoffs = result['results_df']['net_payoff']

    ax3.hist(payoffs, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='손익분기')
    ax3.axvline(x=payoffs.mean(), color='green', linestyle='-', linewidth=2,
                label=f'평균: {payoffs.mean():.2f}억 원')
    ax3.set_xlabel('순수익 (억 원)')
    ax3.set_ylabel('빈도')
    ax3.set_title('시뮬레이션 수익 분포', fontsize=12, fontweight='bold')
    ax3.legend()

    # 4. 단계별 투자 구조
    ax4 = axes[1, 1]

    stage_names = [f'{i+1}단계' for i in range(n_stages)]
    investments = params['stage_investments']
    probs = params['success_probs']

    x = np.arange(n_stages)
    width = 0.35

    bars1 = ax4.bar(x - width/2, investments, width, label='투자금액 (억 원)',
                   color='steelblue', edgecolor='black')

    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, [p * 100 for p in probs], width,
                        label='성공확률 (%)', color='lightgreen', edgecolor='black')

    ax4.set_xlabel('투자 단계')
    ax4.set_ylabel('투자금액 (억 원)', color='steelblue')
    ax4_twin.set_ylabel('성공확률 (%)', color='green')
    ax4.set_xticks(x)
    ax4.set_xticklabels(stage_names)
    ax4.set_title('단계별 투자 구조', fontsize=12, fontweight='bold')

    # 범례 통합
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")

    plt.show()


def main():
    """
    단계적 투자 옵션 분석 메인 실행

    사례: 신약 개발 R&D 투자
    - 1단계: 전임상 (20억 원, 2년, 성공률 70%)
    - 2단계: 임상 1상 (50억 원, 2년, 성공률 60%)
    - 3단계: 임상 2상 (100억 원, 3년, 성공률 50%)
    - 4단계: 임상 3상 (200억 원, 3년, 성공률 70%)
    - 성공 시 예상 가치: 1,000억 원
    """

    print("=" * 70)
    print("단계적 투자 옵션(Staged Investment Options) 분석")
    print("=" * 70)

    # 파라미터 설정
    stage_investments = [20, 50, 100, 200]      # 억 원
    stage_durations = [2, 2, 3, 3]              # 년
    final_value = 1000                           # 억 원
    volatility = 0.40                            # 40%
    success_probs = [0.70, 0.60, 0.50, 0.70]    # 단계별 성공률
    risk_free_rate = 0.03                        # 3%

    print("\n[사례: 신약 개발 R&D 투자]")
    print("-" * 50)

    total_investment = sum(stage_investments)
    total_duration = sum(stage_durations)
    cumulative_success = np.prod(success_probs)

    print(f"\n{'단계':<12} {'투자금액':>10} {'기간':>8} {'성공률':>10}")
    print("-" * 50)
    for i, (inv, dur, prob) in enumerate(zip(stage_investments, stage_durations, success_probs)):
        print(f"  {i+1}단계       {inv:>8}억 원  {dur:>5}년    {prob:>8.0%}")
    print("-" * 50)
    print(f"  합계        {total_investment:>8}억 원  {total_duration:>5}년    {cumulative_success:>8.1%}")

    print(f"\n  최종 성공 시 프로젝트 가치: {final_value}억 원")
    print(f"  가치 변동성: {volatility:.0%}")

    # 분석 수행
    result = staged_investment_option(
        stage_investments=stage_investments,
        stage_durations=stage_durations,
        final_value=final_value,
        volatility=volatility,
        success_probs=success_probs,
        risk_free_rate=risk_free_rate,
        n_simulations=10000
    )

    print("\n\n[분석 결과]")
    print("-" * 50)
    print(f"전통적 NPV (일괄투자):    {result['traditional_npv']:>12.2f}억 원")
    print(f"단계적 투자 옵션 가치:    {result['option_value']:>12.2f}억 원")
    print(f"유연성 가치:              {result['flexibility_value']:>12.2f}억 원")
    print(f"시뮬레이션 성공률:        {result['success_rate']*100:>12.1f}%")
    print("-" * 50)

    # 단계별 도달률
    print("\n[단계별 프로젝트 진행률]")
    for i, rate in enumerate(result['stage_reach_rates']):
        if i == 0:
            print(f"  프로젝트 시작: {rate*100:.1f}%")
        else:
            print(f"  {i}단계 완료: {rate*100:.1f}%")

    # 투자 방식 비교
    print("\n\n[투자 방식 비교]")
    print("-" * 70)
    comparison = compare_investment_approaches(
        stage_investments=stage_investments,
        stage_durations=stage_durations,
        final_value=final_value,
        volatility=volatility,
        success_probs=success_probs,
        risk_free_rate=risk_free_rate
    )
    print(comparison.to_string(index=False))

    # 변동성 민감도 분석
    print("\n\n[변동성 민감도 분석]")
    print("-" * 60)
    sensitivity = sensitivity_by_volatility(
        stage_investments=stage_investments,
        stage_durations=stage_durations,
        final_value=final_value,
        success_probs=success_probs,
        risk_free_rate=risk_free_rate
    )
    print(sensitivity.to_string(index=False))

    # 의사결정 권고
    print("\n\n[의사결정 권고]")
    print("=" * 60)

    if result['traditional_npv'] < 0 and result['option_value'] > 0:
        print(f"  전통적 NPV 기준: 투자 기각 (NPV = {result['traditional_npv']:.2f}억 원 < 0)")
        print(f"  실물옵션 기준: 투자 검토 권고 (옵션 가치 = {result['option_value']:.2f}억 원 > 0)")
        print(f"\n  → 1단계 투자만 {stage_investments[0]}억 원으로 시작하고,")
        print(f"    결과에 따라 후속 투자를 결정하는 것이 최적이다.")
        print(f"\n  유연성의 가치: {result['flexibility_value']:.2f}억 원")
        print(f"  (단계별 중단 옵션이 창출하는 추가 가치)")

    elif result['traditional_npv'] > 0 and result['option_value'] > result['traditional_npv']:
        print(f"  전통적 NPV 기준: 투자 실행 (NPV = {result['traditional_npv']:.2f}억 원 > 0)")
        print(f"  실물옵션 기준: 단계적 접근이 더 유리")
        print(f"\n  → 일괄 투자보다 단계적 투자가 {result['flexibility_value']:.2f}억 원만큼")
        print(f"    더 높은 가치를 제공한다.")

    print("\n[핵심 통찰]")
    print("  1. 단계적 투자는 각 관문에서 '중단할 권리'를 제공한다.")
    print("  2. 불확실성이 높을수록 단계적 접근의 가치가 커진다.")
    print("  3. 전통적 NPV가 음수여도 1단계 투자는 가치가 있을 수 있다.")
    print("  4. 실패 시 손실을 1단계 투자금으로 제한할 수 있다.")

    # 시각화
    print("\n\n[분석 결과 시각화]")
    plot_staged_investment(result, save_path='13-3-staged-investment.png')

    return result


if __name__ == "__main__":
    result = main()
