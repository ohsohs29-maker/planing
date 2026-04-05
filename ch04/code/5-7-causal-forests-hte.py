#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.7 인과 머신러닝: Causal Forests와 이질적 처리효과(HTE)

이 스크립트는 다음을 구현합니다:
1. 이질적 처리 효과(HTE)를 포함한 시뮬레이션 데이터 생성
2. EconML의 CausalForestDML을 사용한 CATE 추정
3. 서브그룹별 처리 효과 시각화
4. 정책 학습: 효과가 양인 집단 식별
5. K대학 계열제 사례 재분석 (개념적)

실행 환경:
- Python 3.10+
- 필요 라이브러리: econml, numpy, pandas, matplotlib, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

# 한글 폰트 설정 (macOS/Windows 호환)
matplotlib.rcParams['font.family'] = 'AppleGothic' if 'AppleGothic' in matplotlib.font_manager.get_font_names() else 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 랜덤 시드 설정
np.random.seed(42)


def generate_hte_data(n_samples=2000):
    """
    이질적 처리 효과(HTE)를 포함한 시뮬레이션 데이터 생성

    데이터 생성 모델:
    - X1, X2: 공변량 (나이, 소득 등)
    - T: 처치 (1=처치군, 0=대조군)
    - τ(X): 개인별 처치 효과 = 2 + 3*X1 - 2*X2
    - Y: 결과 = 10 + 5*X1 + 3*X2 + τ(X)*T + ε

    Returns:
        X, T, Y: 공변량, 처치, 결과 변수
    """
    print("[1단계] 이질적 처리 효과(HTE) 시뮬레이션 데이터 생성")
    print("=" * 70)

    # 공변량 생성
    X1 = np.random.uniform(0, 1, n_samples)  # 예: 정규화된 나이
    X2 = np.random.uniform(0, 1, n_samples)  # 예: 정규화된 소득
    X = np.column_stack([X1, X2])

    # 처치 할당 (무작위 배정)
    T = np.random.binomial(1, 0.5, n_samples)

    # 이질적 처치 효과 (CATE)
    # X1이 높을수록 효과 크고, X2가 높을수록 효과 작음
    true_cate = 2 + 3 * X1 - 2 * X2

    # 결과 변수 생성
    base_outcome = 10 + 5 * X1 + 3 * X2
    noise = np.random.normal(0, 0.5, n_samples)
    Y = base_outcome + true_cate * T + noise

    # 평균 처치 효과(ATE) 계산
    ate = np.mean(true_cate)

    print(f"샘플 수: {n_samples} (처치 {np.sum(T)}, 대조 {n_samples - np.sum(T)})")
    print(f"공변량: X1 (나이), X2 (소득)")
    print(f"진짜 CATE 범위: [{true_cate.min():.2f}, {true_cate.max():.2f}]")
    print(f"진짜 ATE (평균): {ate:.2f}")
    print()

    return X, T, Y, true_cate


def estimate_cate_with_causal_forest(X, T, Y):
    """
    EconML의 CausalForestDML을 사용하여 CATE 추정

    Args:
        X: 공변량
        T: 처치
        Y: 결과

    Returns:
        est: 학습된 Causal Forest 모델
        cate_pred: 추정된 CATE
    """
    print("[2단계] Causal Forest로 CATE 추정")
    print("=" * 70)

    # Train-Test Split
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.3, random_state=42
    )

    print(f"학습 샘플: {len(X_train)}, 테스트 샘플: {len(X_test)}")

    # CausalForestDML 모델 정의
    # - model_y: 결과(Y) 예측 모델
    # - model_t: 처치(T) 예측 모델 (propensity score, 이진 처치도 Regressor 사용)
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=42),
        model_t=RandomForestRegressor(n_estimators=100, random_state=42),
        n_estimators=500,  # Causal Forest의 트리 개수
        min_samples_leaf=20,  # 과적합 방지
        random_state=42
    )

    print("Causal Forest 학습 중...")
    est.fit(Y_train, T_train, X=X_train, W=None)

    # CATE 추정
    cate_pred = est.effect(X_test)

    print(f"추정 완료!")
    print(f"추정된 CATE 범위: [{cate_pred.min():.2f}, {cate_pred.max():.2f}]")
    print(f"추정된 ATE (평균 CATE): {cate_pred.mean():.2f}")
    print()

    return est, cate_pred, X_test, T_test, Y_test


def evaluate_cate_estimation(cate_pred, true_cate_test):
    """
    CATE 추정 성능 평가

    Args:
        cate_pred: 추정된 CATE
        true_cate_test: 테스트 데이터의 진짜 CATE
    """
    print("[3단계] CATE 추정 성능 평가")
    print("=" * 70)

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((cate_pred - true_cate_test) ** 2))

    # 상관계수
    correlation = np.corrcoef(cate_pred, true_cate_test)[0, 1]

    print(f"RMSE (추정 오차): {rmse:.3f}")
    print(f"상관계수: {correlation:.3f}")
    print()

    return rmse, correlation


def policy_learning(cate_pred, X_test):
    """
    정책 학습(Policy Learning): 효과가 양인 집단 식별

    Args:
        cate_pred: 추정된 CATE
        X_test: 테스트 공변량

    Returns:
        should_treat: 처치 권장 여부 (Boolean array)
    """
    print("[4단계] 정책 학습: 효과가 양인 집단 식별")
    print("=" * 70)

    # 단순 정책: CATE > 0인 집단만 처치
    should_treat = cate_pred > 0

    n_total = len(cate_pred)
    n_treat = np.sum(should_treat)
    pct_treat = n_treat / n_total * 100

    print(f"전체 인원: {n_total}명")
    print(f"처치 권장: {n_treat}명 ({pct_treat:.1f}%)")
    print(f"처치 비권장: {n_total - n_treat}명 ({100 - pct_treat:.1f}%)")

    # 타겟팅의 가치 계산
    avg_effect_all = cate_pred.mean()
    avg_effect_treated = cate_pred[should_treat].mean() if n_treat > 0 else 0

    print(f"\n[타겟팅의 가치]")
    print(f"전체 평균 효과: {avg_effect_all:.2f}")
    print(f"처치군 평균 효과: {avg_effect_treated:.2f}")

    if avg_effect_all > 0:
        efficiency_gain = (avg_effect_treated - avg_effect_all) / avg_effect_all * 100
        print(f"→ 타겟팅 시 효율성 {efficiency_gain:.1f}% 향상")

    print()

    return should_treat


def visualize_hte(X_test, cate_pred, true_cate_test, should_treat):
    """
    이질적 처치 효과 시각화

    Args:
        X_test: 테스트 공변량
        cate_pred: 추정된 CATE
        true_cate_test: 진짜 CATE
        should_treat: 처치 권장 여부
    """
    print("[5단계] 시각화")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 진짜 CATE vs 추정 CATE (산점도)
    ax1 = axes[0, 0]
    ax1.scatter(true_cate_test, cate_pred, alpha=0.5, s=10)
    ax1.plot(
        [true_cate_test.min(), true_cate_test.max()],
        [true_cate_test.min(), true_cate_test.max()],
        'r--', label='완벽한 추정'
    )
    ax1.set_xlabel('진짜 CATE')
    ax1.set_ylabel('추정 CATE')
    ax1.set_title('CATE 추정 정확도')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) 공변량 X1에 따른 CATE
    ax2 = axes[0, 1]
    x1_test = X_test[:, 0]
    ax2.scatter(x1_test, true_cate_test, alpha=0.3, s=10, label='진짜 CATE')
    ax2.scatter(x1_test, cate_pred, alpha=0.3, s=10, label='추정 CATE')
    ax2.set_xlabel('X1 (나이)')
    ax2.set_ylabel('CATE')
    ax2.set_title('X1에 따른 처치 효과 이질성')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (3) 공변량 X2에 따른 CATE
    ax3 = axes[1, 0]
    x2_test = X_test[:, 1]
    ax3.scatter(x2_test, true_cate_test, alpha=0.3, s=10, label='진짜 CATE')
    ax3.scatter(x2_test, cate_pred, alpha=0.3, s=10, label='추정 CATE')
    ax3.set_xlabel('X2 (소득)')
    ax3.set_ylabel('CATE')
    ax3.set_title('X2에 따른 처치 효과 이질성')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (4) 정책 학습 결과 (처치 권장 vs 비권장)
    ax4 = axes[1, 1]
    colors = ['red' if not treat else 'blue' for treat in should_treat]
    ax4.scatter(x1_test, x2_test, c=colors, alpha=0.5, s=10)
    ax4.set_xlabel('X1 (나이)')
    ax4.set_ylabel('X2 (소득)')
    ax4.set_title('정책 학습: 처치 권장 집단 (파란색)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../data/5-7-causal-forests-hte.png', dpi=150, bbox_inches='tight')
    print("시각화 저장 완료: ../data/5-7-causal-forests-hte.png")
    print()


def k_university_conceptual_analysis():
    """
    K대학 계열제 사례의 개념적 재분석

    실제 데이터는 없으므로, HTE 분석의 개념과 시사점만 제시
    """
    print("[6단계] K대학 계열제 사례: HTE 재분석 (개념적)")
    print("=" * 70)
    print("""
K대학 계열제 전환의 효과를 Causal Forests로 재분석한다면:

【연구 질문】
- 계열제가 모든 학생에게 동일한 효과를 주는가?
- 어떤 특성의 학생에게 더 효과적인가?

【분석 설계】
1. 공변량 (X):
   - 고교 성적 (GPA)
   - 전공 선호도 명확성
   - 진로 탐색 경험
   - 가정 배경 (부모 학력, 소득)

2. 처치 (T):
   - T=1: 계열제 입학 (2019년 이후)
   - T=0: 학과제 입학 (2019년 이전)

3. 결과 (Y):
   - 1학년 GPA
   - 전공 만족도
   - 중도탈락률

【예상 HTE 패턴】
- 고성취자 (GPA > 3.5): CATE = +0.25 ↑↑ (긍정)
  → 계열제가 더 많은 학습 기회 제공

- 중성취자 (GPA 3.0-3.5): CATE = +0.08 (미미)
  → 계열제 효과 제한적

- 저성취자 (GPA < 3.0): CATE = -0.12 ↓ (부정)
  → 계열제가 추가 부담으로 작용

【정책 함의】
1. 일률적 계열제 확대보다는 학생 특성별 맞춤형 지원
2. 저성취자를 위한 추가 지원 프로그램 필요
3. 고성취자 대상 계열제 혜택 강화

【실무 적용】
- 신입생 특성 데이터 수집 → Causal Forest 적용
- 입학 시점에 계열제 vs 학과제 권장 기준 마련
- 효과 모니터링 및 정책 개선
    """)
    print()


def main():
    """
    메인 실행 함수
    """
    print("\n" + "=" * 70)
    print("5.7 인과 머신러닝: Causal Forests와 이질적 처리효과")
    print("=" * 70)
    print()

    # 1. 데이터 생성
    X, T, Y, true_cate = generate_hte_data(n_samples=2000)

    # 2. CATE 추정
    est, cate_pred, X_test, T_test, Y_test = estimate_cate_with_causal_forest(X, T, Y)

    # 테스트 데이터의 진짜 CATE 계산
    X1_test = X_test[:, 0]
    X2_test = X_test[:, 1]
    true_cate_test = 2 + 3 * X1_test - 2 * X2_test

    # 3. 성능 평가
    rmse, correlation = evaluate_cate_estimation(cate_pred, true_cate_test)

    # 4. 정책 학습
    should_treat = policy_learning(cate_pred, X_test)

    # 5. 시각화
    visualize_hte(X_test, cate_pred, true_cate_test, should_treat)

    # 6. K대학 사례 개념적 분석
    k_university_conceptual_analysis()

    print("=" * 70)
    print("✅ 실행 완료!")
    print("=" * 70)
    print()
    print("[핵심 메시지]")
    print("1. 평균 처치 효과(ATE)만으로는 개인별/집단별 효과 차이 파악 불가")
    print("2. Causal Forests는 공변량에 따른 이질적 처치 효과(HTE) 추정")
    print("3. 정책 타겟팅으로 효율성 크게 향상 가능")
    print("4. 기획 실무: 정책/마케팅/HR에서 맞춤형 개입 설계에 활용")
    print()


if __name__ == "__main__":
    main()
