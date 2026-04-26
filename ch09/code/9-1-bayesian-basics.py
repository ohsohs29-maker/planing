"""
9-1-bayesian-basics.py
베이지안 기초: 베이즈 정리와 직관적 이해

이 코드는 베이즈 정리의 핵심 개념을 실습하기 위한 예제를 제공한다.
의료 진단 예시를 통해 사전확률, 가능도, 사후확률의 관계를 이해한다.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (크로스플랫폼)
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def bayes_theorem(prior, likelihood, false_positive_rate):
    """
    베이즈 정리를 적용하여 사후확률 계산

    Parameters:
    -----------
    prior : float
        사전확률 P(H) - 질병 유병률
    likelihood : float
        가능도 P(D|H) - 민감도 (실제 환자를 양성으로 판정할 확률)
    false_positive_rate : float
        위양성률 P(D|~H) - 건강한 사람을 양성으로 판정할 확률

    Returns:
    --------
    posterior : float
        사후확률 P(H|D) - 양성 판정 시 실제 환자일 확률
    """
    # P(D) = P(D|H)P(H) + P(D|~H)P(~H)
    p_positive = likelihood * prior + false_positive_rate * (1 - prior)

    # P(H|D) = P(D|H)P(H) / P(D)
    posterior = (likelihood * prior) / p_positive

    return posterior


def medical_diagnosis_example():
    """
    의료 진단 예시: 희귀 질병 검사의 양성 판정 해석

    유병률 1%, 민감도 99%, 특이도 95%일 때
    양성 판정을 받았다면 실제로 아플 확률은?
    """
    print("=" * 60)
    print("의료 진단 예시: 희귀 질병 검사")
    print("=" * 60)

    # 파라미터 설정
    prevalence = 0.01  # 유병률 (사전확률)
    sensitivity = 0.99  # 민감도 (가능도)
    specificity = 0.95  # 특이도
    false_positive_rate = 1 - specificity  # 위양성률

    print(f"\n[검사 조건]")
    print(f"  유병률 (사전확률): {prevalence:.1%}")
    print(f"  민감도 (진양성률): {sensitivity:.1%}")
    print(f"  특이도: {specificity:.1%}")
    print(f"  위양성률: {false_positive_rate:.1%}")

    # 베이즈 정리 적용
    posterior = bayes_theorem(prevalence, sensitivity, false_positive_rate)

    print(f"\n[결과]")
    print(f"  양성 판정 시 실제 환자일 확률: {posterior:.1%}")
    print(f"  (대부분의 사람들이 99%라고 오해함)")

    # 1만 명 대상 시뮬레이션
    print(f"\n[1만 명 대상 시뮬레이션]")
    n = 10000
    actual_patients = int(n * prevalence)
    healthy = n - actual_patients

    true_positive = int(actual_patients * sensitivity)
    false_positive = int(healthy * false_positive_rate)
    total_positive = true_positive + false_positive

    print(f"  전체 인원: {n:,}명")
    print(f"  실제 환자: {actual_patients:,}명")
    print(f"  건강한 사람: {healthy:,}명")
    print(f"  진양성 (환자 중 양성): {true_positive:,}명")
    print(f"  위양성 (건강한데 양성): {false_positive:,}명")
    print(f"  총 양성 판정: {total_positive:,}명")
    print(f"  양성 중 실제 환자 비율: {true_positive/total_positive:.1%}")

    return posterior


def prior_sensitivity_analysis():
    """
    사전확률에 따른 사후확률 변화 분석
    """
    print("\n" + "=" * 60)
    print("사전확률 민감도 분석")
    print("=" * 60)

    # 고정 파라미터
    sensitivity = 0.99
    specificity = 0.95
    false_positive_rate = 1 - specificity

    # 사전확률 범위
    priors = np.linspace(0.001, 0.5, 100)
    posteriors = [bayes_theorem(p, sensitivity, false_positive_rate) for p in priors]

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(priors * 100, np.array(posteriors) * 100, 'b-', linewidth=2)
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50% 기준선')
    ax.axvline(x=5, color='g', linestyle='--', alpha=0.7, label='유병률 5%')

    ax.set_xlabel('사전확률 (유병률, %)', fontsize=12)
    ax.set_ylabel('사후확률 (양성 시 환자 확률, %)', fontsize=12)
    ax.set_title('사전확률에 따른 사후확률 변화\n(민감도 99%, 특이도 95%)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../data/prior_sensitivity.png', dpi=150)
    plt.close()

    print("그래프 저장: ../data/prior_sensitivity.png")

    # 주요 지점 출력
    key_priors = [0.01, 0.05, 0.10, 0.20, 0.50]
    print("\n[주요 사전확률별 사후확률]")
    print("-" * 40)
    print(f"{'사전확률':<15} {'사후확률':<15}")
    print("-" * 40)
    for p in key_priors:
        post = bayes_theorem(p, sensitivity, false_positive_rate)
        print(f"{p:>10.1%}      {post:>10.1%}")


def point_vs_probabilistic():
    """
    점 추정 vs 확률적 추정 비교
    """
    print("\n" + "=" * 60)
    print("점 추정 vs 확률적 추정 비교")
    print("=" * 60)

    # 시장 규모 추정 예시
    np.random.seed(42)

    # 점 추정: 단일 값
    point_estimate = 100  # 100억 원

    # 확률적 추정: 분포
    mean = 100
    std = 30
    samples = np.random.normal(mean, std, 10000)

    print("\n[시장 규모 추정 예시]")
    print(f"  점 추정: {point_estimate}억 원")
    print(f"  확률적 추정: 평균 {mean}억, 표준편차 {std}억")
    print(f"  50% 신뢰구간: {np.percentile(samples, 25):.0f}억 ~ {np.percentile(samples, 75):.0f}억")
    print(f"  90% 신뢰구간: {np.percentile(samples, 5):.0f}억 ~ {np.percentile(samples, 95):.0f}억")

    # 의사결정 차이
    threshold = 80  # 투자 기준
    prob_above_threshold = np.mean(samples > threshold)

    print(f"\n[의사결정 비교]")
    print(f"  투자 기준: {threshold}억 원 이상")
    print(f"  점 추정: {point_estimate}억 > {threshold}억 -> 투자 결정 (확신)")
    print(f"  확률적 추정: 기준 초과 확률 {prob_above_threshold:.1%}")
    print(f"  -> 약 {(1-prob_above_threshold):.1%} 확률로 기준 미달")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 점 추정
    ax1 = axes[0]
    ax1.axvline(x=point_estimate, color='red', linewidth=3, label=f'점 추정: {point_estimate}억')
    ax1.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'투자 기준: {threshold}억')
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('시장 규모 (억 원)', fontsize=12)
    ax1.set_title('점 추정: 불확실성 무시', fontsize=14)
    ax1.legend()

    # 확률적 추정
    ax2 = axes[1]
    ax2.hist(samples, bins=50, density=True, alpha=0.7, color='blue', label='확률 분포')
    ax2.axvline(x=mean, color='red', linewidth=2, label=f'평균: {mean}억')
    ax2.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'투자 기준: {threshold}억')
    ax2.axvspan(0, threshold, alpha=0.2, color='red', label=f'기준 미달: {(1-prob_above_threshold):.1%}')
    ax2.set_xlabel('시장 규모 (억 원)', fontsize=12)
    ax2.set_title('확률적 추정: 불확실성 명시', fontsize=14)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('../data/point_vs_probabilistic.png', dpi=150)
    plt.close()

    print("\n그래프 저장: ../data/point_vs_probabilistic.png")


if __name__ == "__main__":
    # 의료 진단 예시
    medical_diagnosis_example()

    # 사전확률 민감도 분석
    prior_sensitivity_analysis()

    # 점 추정 vs 확률적 추정
    point_vs_probabilistic()

    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)
