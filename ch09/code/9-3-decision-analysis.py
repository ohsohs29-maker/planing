"""
9-3-decision-analysis.py
베이지안 의사결정 분석: 정보의 가치와 최적 결정

이 코드는 불확실성 하에서의 의사결정과 추가 정보 수집의
경제적 가치(EVPI, EVII)를 계산하는 방법을 보여준다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def new_business_decision():
    """
    신사업 진출 의사결정 분석

    시나리오: 기업이 신규 시장 진출을 검토
    - 진출 시 투자 비용: 50억 원
    - 시장이 크면 수익 100억, 작으면 수익 20억
    - 시장 규모에 대한 불확실성 존재
    """
    print("=" * 60)
    print("신사업 진출 의사결정 분석")
    print("=" * 60)

    # 파라미터 설정
    investment = 50  # 투자 비용 (억 원)
    revenue_high = 100  # 시장이 클 때 수익
    revenue_low = 20    # 시장이 작을 때 수익

    # 사전확률: 시장이 클 확률
    prior_high = 0.40

    print(f"\n[의사결정 상황]")
    print(f"  투자 비용: {investment}억 원")
    print(f"  시장이 클 때 수익: {revenue_high}억 원")
    print(f"  시장이 작을 때 수익: {revenue_low}억 원")
    print(f"  시장이 클 확률 (사전): {prior_high:.1%}")

    # 의사결정 대안
    # 대안 1: 진출
    profit_high = revenue_high - investment
    profit_low = revenue_low - investment
    ev_enter = prior_high * profit_high + (1 - prior_high) * profit_low

    # 대안 2: 진출 안함
    ev_not_enter = 0

    print(f"\n[기대이익 분석]")
    print(f"  진출 시:")
    print(f"    시장 클 때 이익: {revenue_high} - {investment} = {profit_high}억 원")
    print(f"    시장 작을 때 이익: {revenue_low} - {investment} = {profit_low}억 원")
    print(f"    기대이익: {prior_high:.1%} × {profit_high} + {1-prior_high:.1%} × {profit_low} = {ev_enter:.1f}억 원")
    print(f"  미진출 시:")
    print(f"    기대이익: {ev_not_enter}억 원")

    optimal_decision = "진출" if ev_enter > ev_not_enter else "미진출"
    print(f"\n[현재 최적 결정: {optimal_decision}]")

    return {
        'investment': investment,
        'revenue_high': revenue_high,
        'revenue_low': revenue_low,
        'prior_high': prior_high,
        'ev_enter': ev_enter,
        'ev_not_enter': ev_not_enter
    }


def calculate_evpi(params):
    """
    완전 정보의 기대가치(EVPI) 계산
    """
    print("\n" + "=" * 60)
    print("EVPI: 완전 정보의 기대가치")
    print("=" * 60)

    investment = params['investment']
    revenue_high = params['revenue_high']
    revenue_low = params['revenue_low']
    prior_high = params['prior_high']

    profit_high = revenue_high - investment
    profit_low = revenue_low - investment

    # 현재 기대이익 (불확실성 하)
    ev_current = max(params['ev_enter'], params['ev_not_enter'])

    # 완전 정보 하의 기대이익
    # 시장이 크다는 것을 알면: 진출 (이익 50억)
    # 시장이 작다는 것을 알면: 미진출 (이익 0) - 손실 피함
    ev_with_perfect_info = prior_high * max(profit_high, 0) + (1 - prior_high) * max(profit_low, 0)

    evpi = ev_with_perfect_info - ev_current

    print(f"\n[EVPI 계산]")
    print(f"  현재 최적 기대이익: {ev_current:.1f}억 원")
    print(f"  완전 정보 하 기대이익:")
    print(f"    시장 큼(확률 {prior_high:.0%}): max({profit_high}, 0) = {max(profit_high, 0)}억")
    print(f"    시장 작음(확률 {1-prior_high:.0%}): max({profit_low}, 0) = {max(profit_low, 0)}억")
    print(f"    기대이익: {prior_high:.0%} × {max(profit_high, 0)} + {1-prior_high:.0%} × {max(profit_low, 0)} = {ev_with_perfect_info:.1f}억 원")
    print(f"\n  EVPI = {ev_with_perfect_info:.1f} - {ev_current:.1f} = {evpi:.1f}억 원")
    print(f"\n[해석]")
    print(f"  시장 규모를 완벽히 알 수 있다면 최대 {evpi:.1f}억 원을 지불할 가치가 있음")
    print(f"  시장조사 비용이 {evpi:.1f}억 원 이하라면 경제적으로 타당")

    return evpi


def calculate_evii(params, test_accuracy=0.80):
    """
    불완전 정보의 기대가치(EVII) 계산

    시장조사의 정확도가 80%인 경우
    """
    print("\n" + "=" * 60)
    print(f"EVII: 불완전 정보의 기대가치 (정확도 {test_accuracy:.0%})")
    print("=" * 60)

    investment = params['investment']
    revenue_high = params['revenue_high']
    revenue_low = params['revenue_low']
    prior_high = params['prior_high']

    profit_high = revenue_high - investment
    profit_low = revenue_low - investment

    # 현재 기대이익
    ev_current = max(params['ev_enter'], params['ev_not_enter'])

    # 시장조사 결과별 확률 (베이즈 정리)
    # P(조사결과='크다' | 실제 큼) = 정확도
    # P(조사결과='작다' | 실제 작음) = 정확도
    sensitivity = test_accuracy  # 진양성률
    specificity = test_accuracy  # 진음성률

    # P(조사='크다') = P(조사='크다'|큼)P(큼) + P(조사='크다'|작음)P(작음)
    p_test_high = sensitivity * prior_high + (1 - specificity) * (1 - prior_high)
    p_test_low = 1 - p_test_high

    # 조사 결과에 따른 사후확률
    # P(큼|조사='크다') = P(조사='크다'|큼)P(큼) / P(조사='크다')
    post_high_given_test_high = (sensitivity * prior_high) / p_test_high
    post_high_given_test_low = ((1 - sensitivity) * prior_high) / p_test_low

    print(f"\n[시장조사 정확도]")
    print(f"  시장 큼 → '크다' 예측: {sensitivity:.0%}")
    print(f"  시장 작음 → '작다' 예측: {specificity:.0%}")

    print(f"\n[조사 결과 확률]")
    print(f"  '크다' 결과 확률: {p_test_high:.1%}")
    print(f"  '작다' 결과 확률: {p_test_low:.1%}")

    print(f"\n[사후확률 (베이즈 업데이트)]")
    print(f"  '크다' 결과 시 실제로 클 확률: {post_high_given_test_high:.1%}")
    print(f"  '작다' 결과 시 실제로 클 확률: {post_high_given_test_low:.1%}")

    # 조사 결과별 최적 결정과 기대이익
    # '크다' 결과 시
    ev_enter_test_high = post_high_given_test_high * profit_high + (1 - post_high_given_test_high) * profit_low
    optimal_test_high = "진출" if ev_enter_test_high > 0 else "미진출"
    ev_optimal_test_high = max(ev_enter_test_high, 0)

    # '작다' 결과 시
    ev_enter_test_low = post_high_given_test_low * profit_high + (1 - post_high_given_test_low) * profit_low
    optimal_test_low = "진출" if ev_enter_test_low > 0 else "미진출"
    ev_optimal_test_low = max(ev_enter_test_low, 0)

    print(f"\n[조사 결과별 최적 결정]")
    print(f"  '크다' 결과 시: {optimal_test_high} (기대이익 {ev_optimal_test_high:.1f}억)")
    print(f"  '작다' 결과 시: {optimal_test_low} (기대이익 {ev_optimal_test_low:.1f}억)")

    # 조사 수행 시 기대이익
    ev_with_test = p_test_high * ev_optimal_test_high + p_test_low * ev_optimal_test_low
    evii = ev_with_test - ev_current

    print(f"\n[EVII 계산]")
    print(f"  조사 수행 시 기대이익: {p_test_high:.1%} × {ev_optimal_test_high:.1f} + {p_test_low:.1%} × {ev_optimal_test_low:.1f} = {ev_with_test:.1f}억")
    print(f"  현재 기대이익: {ev_current:.1f}억")
    print(f"  EVII = {ev_with_test:.1f} - {ev_current:.1f} = {evii:.1f}억 원")

    print(f"\n[해석]")
    print(f"  정확도 {test_accuracy:.0%}의 시장조사 가치: 최대 {evii:.1f}억 원")
    if evii > 0:
        print(f"  조사 비용이 {evii:.1f}억 원 이하라면 조사 실시 권장")
    else:
        print(f"  조사 가치 없음 - 바로 결정하는 것이 나음")

    return evii


def sensitivity_to_prior():
    """
    사전확률에 따른 의사결정 변화 분석
    """
    print("\n" + "=" * 60)
    print("사전확률 민감도 분석")
    print("=" * 60)

    investment = 50
    revenue_high = 100
    revenue_low = 20

    profit_high = revenue_high - investment
    profit_low = revenue_low - investment

    priors = np.linspace(0, 1, 100)
    ev_enters = [p * profit_high + (1-p) * profit_low for p in priors]
    ev_not_enters = [0] * len(priors)

    # 손익분기 사전확률
    # p * 50 + (1-p) * (-30) = 0
    # 50p - 30 + 30p = 0
    # 80p = 30
    # p = 30/80 = 0.375
    breakeven_prior = -profit_low / (profit_high - profit_low)

    print(f"\n[손익분기 분석]")
    print(f"  손익분기 사전확률: {breakeven_prior:.1%}")
    print(f"  시장이 클 확률이 {breakeven_prior:.1%} 이상이면 진출 결정")
    print(f"  시장이 클 확률이 {breakeven_prior:.1%} 미만이면 미진출 결정")

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(priors * 100, ev_enters, 'b-', linewidth=2, label='진출')
    ax.plot(priors * 100, ev_not_enters, 'r--', linewidth=2, label='미진출')
    ax.axvline(x=breakeven_prior * 100, color='green', linestyle=':', linewidth=2,
               label=f'손익분기: {breakeven_prior:.1%}')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax.fill_between(priors * 100, ev_enters, ev_not_enters,
                    where=[e > n for e, n in zip(ev_enters, ev_not_enters)],
                    alpha=0.3, color='blue', label='진출 유리')
    ax.fill_between(priors * 100, ev_enters, ev_not_enters,
                    where=[e <= n for e, n in zip(ev_enters, ev_not_enters)],
                    alpha=0.3, color='red', label='미진출 유리')

    ax.set_xlabel('사전확률: 시장이 클 확률 (%)', fontsize=12)
    ax.set_ylabel('기대이익 (억 원)', fontsize=12)
    ax.set_title('사전확률에 따른 최적 의사결정', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../data/decision_sensitivity.png', dpi=150)
    plt.close()

    print("\n그래프 저장: ../data/decision_sensitivity.png")

    return breakeven_prior


def decision_tree_summary():
    """
    의사결정 트리 요약
    """
    print("\n" + "=" * 60)
    print("의사결정 트리 요약")
    print("=" * 60)

    print("""
[의사결정 트리 구조]

                           ┌─ 시장 큼 (40%) ─→ 이익 50억
    ┌─ 진출 ──────────────┤
    │   EV = 2억           └─ 시장 작음 (60%) ─→ 손실 -30억
    │
시작 ┤
    │
    └─ 미진출 ──────────────────────────────→ 이익 0억
        EV = 0억

[현재 최적 결정]
  진출 (기대이익 2억 원 > 0억 원)

[정보 가치]
  EVPI = 20억 원 (완전 정보)
  EVII = 약 8억 원 (80% 정확도 조사)
    """)


if __name__ == "__main__":
    # 기본 의사결정 분석
    params = new_business_decision()

    # EVPI 계산
    evpi = calculate_evpi(params)

    # EVII 계산 (80% 정확도)
    evii_80 = calculate_evii(params, test_accuracy=0.80)

    # EVII 계산 (90% 정확도)
    evii_90 = calculate_evii(params, test_accuracy=0.90)

    # 사전확률 민감도 분석
    breakeven = sensitivity_to_prior()

    # 요약
    decision_tree_summary()

    print("\n" + "=" * 60)
    print("최종 요약")
    print("=" * 60)
    print(f"  현재 최적 결정: 진출")
    print(f"  기대이익: 2억 원")
    print(f"  EVPI: {evpi:.1f}억 원")
    print(f"  EVII (80% 정확도): {evii_80:.1f}억 원")
    print(f"  EVII (90% 정확도): {evii_90:.1f}억 원")
    print(f"  손익분기 사전확률: {breakeven:.1%}")
    print("=" * 60)
