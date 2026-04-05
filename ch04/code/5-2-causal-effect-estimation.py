"""
제5장 예제 5.2: DoWhy를 활용한 인과 효과 추정

이 코드는 DoWhy 라이브러리를 활용하여 마케팅 캠페인의 인과 효과를 추정한다.
관찰 데이터에서 교란변수를 통제하여 캠페인의 순수 효과를 분리한다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# DoWhy 임포트
try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    print("Warning: DoWhy가 설치되지 않았습니다. pip install dowhy")

# 결과 저장 경로
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 시드 설정
np.random.seed(42)


def generate_marketing_data(n=5000):
    """마케팅 캠페인 데이터 생성

    시나리오: 온라인 소매업체의 이메일 마케팅 캠페인 효과 분석

    변수:
    - customer_value: 고객의 구매 성향 (교란변수)
    - recency: 마지막 구매 후 경과 일수
    - frequency: 과거 구매 빈도
    - campaign: 캠페인 노출 여부 (처치)
    - purchase: 구매 여부 (결과)

    인과 구조:
    - customer_value → campaign (성향 높은 고객에게 더 많이 노출)
    - customer_value → purchase (성향 높은 고객이 더 많이 구매)
    - recency → campaign
    - frequency → campaign
    - campaign → purchase (우리가 추정하려는 인과 효과)

    실제 인과 효과 (ATE): 0.15 (캠페인이 구매 확률을 15%p 높임)
    """
    # 교란변수: 고객 가치 (0~1)
    customer_value = np.random.beta(2, 5, n)

    # 공변량
    recency = np.random.exponential(30, n)  # 마지막 구매 후 일수
    frequency = np.random.poisson(3, n)  # 과거 구매 횟수

    # 처치 (캠페인 노출) - 교란변수의 영향을 받음
    campaign_propensity = 0.3 + 0.4 * customer_value - 0.005 * recency + 0.05 * frequency
    campaign_propensity = np.clip(campaign_propensity, 0.05, 0.95)
    campaign = np.random.binomial(1, campaign_propensity)

    # 결과 (구매) - 처치와 교란변수 모두의 영향
    TRUE_EFFECT = 0.15  # 실제 인과 효과
    purchase_prob = (
        0.1  # 기본 구매 확률
        + 0.5 * customer_value  # 고객 가치의 영향 (교란)
        + TRUE_EFFECT * campaign  # 캠페인의 인과 효과
        - 0.002 * recency  # recency 영향
        + 0.02 * frequency  # frequency 영향
        + np.random.normal(0, 0.1, n)  # 노이즈
    )
    purchase_prob = np.clip(purchase_prob, 0, 1)
    purchase = np.random.binomial(1, purchase_prob)

    df = pd.DataFrame({
        "customer_value": customer_value,
        "recency": recency,
        "frequency": frequency,
        "campaign": campaign,
        "purchase": purchase
    })

    return df, TRUE_EFFECT


def naive_estimation(df):
    """단순 비교 (편향된 추정)

    캠페인 그룹과 비캠페인 그룹의 구매율 차이를 계산.
    교란변수를 통제하지 않아 편향된 결과가 나온다.
    """
    purchase_rate_treated = df[df["campaign"] == 1]["purchase"].mean()
    purchase_rate_control = df[df["campaign"] == 0]["purchase"].mean()
    naive_effect = purchase_rate_treated - purchase_rate_control

    print("\n" + "=" * 50)
    print("1. 단순 비교 (Naive Estimation)")
    print("=" * 50)
    print(f"캠페인 그룹 구매율: {purchase_rate_treated:.3f}")
    print(f"비캠페인 그룹 구매율: {purchase_rate_control:.3f}")
    print(f"차이 (단순 추정): {naive_effect:.3f}")
    print("\n주의: 이 추정치는 교란변수를 통제하지 않아 편향됨!")

    return naive_effect


def dowhy_estimation(df, true_effect):
    """DoWhy를 활용한 인과 효과 추정"""
    if not DOWHY_AVAILABLE:
        print("\nDoWhy가 설치되지 않아 건너뜁니다.")
        return None

    print("\n" + "=" * 50)
    print("2. DoWhy 인과 효과 추정")
    print("=" * 50)

    # 인과 모델 정의
    # GML 형식으로 인과 그래프 정의
    causal_graph = """
    digraph {
        customer_value -> campaign;
        customer_value -> purchase;
        recency -> campaign;
        frequency -> campaign;
        recency -> purchase;
        frequency -> purchase;
        campaign -> purchase;
    }
    """

    model = CausalModel(
        data=df,
        treatment="campaign",
        outcome="purchase",
        graph=causal_graph
    )

    # 2.1 식별 (Identification)
    print("\n[식별 단계]")
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(f"식별된 추정량: {identified_estimand}")

    # 2.2 추정 (Estimation) - 여러 방법 비교
    results = {}

    # 방법 1: 성향점수 매칭 (Propensity Score Matching)
    print("\n[추정 방법 1: 성향점수 층화]")
    try:
        estimate_psm = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_stratification",
            target_units="ate"
        )
        results["성향점수 층화"] = estimate_psm.value
        print(f"추정된 ATE: {estimate_psm.value:.4f}")
    except Exception as e:
        print(f"성향점수 층화 실패: {e}")

    # 방법 2: 선형 회귀
    print("\n[추정 방법 2: 선형 회귀]")
    try:
        estimate_lr = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            target_units="ate"
        )
        results["선형 회귀"] = estimate_lr.value
        print(f"추정된 ATE: {estimate_lr.value:.4f}")
    except Exception as e:
        print(f"선형 회귀 실패: {e}")

    # 2.3 반박 검증 (Refutation)
    print("\n[반박 검증]")

    # 플라시보 처치 테스트
    print("\n반박 1: 플라시보 처치 테스트")
    try:
        refute_placebo = model.refute_estimate(
            identified_estimand,
            estimate_lr if "선형 회귀" in results else estimate_psm,
            method_name="placebo_treatment_refuter",
            placebo_type="permute"
        )
        print(f"플라시보 효과: {refute_placebo.new_effect:.4f}")
        print("(0에 가까워야 모델이 타당)")
    except Exception as e:
        print(f"플라시보 테스트 실패: {e}")

    # 결과 요약
    print("\n" + "=" * 50)
    print("추정 결과 요약")
    print("=" * 50)
    print(f"실제 인과 효과 (True ATE): {true_effect:.4f}")
    for method, value in results.items():
        bias = value - true_effect
        print(f"{method}: {value:.4f} (편향: {bias:+.4f})")

    return results


def visualize_confounding(df):
    """교란 효과 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. 교란변수 분포 비교
    ax1 = axes[0]
    df_treated = df[df["campaign"] == 1]
    df_control = df[df["campaign"] == 0]

    ax1.hist(df_treated["customer_value"], bins=30, alpha=0.5,
             label=f"캠페인 O (n={len(df_treated)})", density=True)
    ax1.hist(df_control["customer_value"], bins=30, alpha=0.5,
             label=f"캠페인 X (n={len(df_control)})", density=True)
    ax1.set_xlabel("고객 가치 (Customer Value)")
    ax1.set_ylabel("밀도")
    ax1.set_title("교란변수 분포 비교\n(처치 그룹 vs 통제 그룹)")
    ax1.legend()

    # 2. 교란변수와 처치의 관계
    ax2 = axes[1]
    df_binned = df.copy()
    df_binned["value_bin"] = pd.cut(df["customer_value"], bins=10)
    campaign_rate = df_binned.groupby("value_bin", observed=True)["campaign"].mean()

    ax2.bar(range(len(campaign_rate)), campaign_rate.values, color="steelblue")
    ax2.set_xlabel("고객 가치 구간")
    ax2.set_ylabel("캠페인 노출률")
    ax2.set_title("고객 가치별 캠페인 노출률\n(교란변수 → 처치)")

    # 3. 교란변수와 결과의 관계
    ax3 = axes[2]
    purchase_rate = df_binned.groupby("value_bin", observed=True)["purchase"].mean()

    ax3.bar(range(len(purchase_rate)), purchase_rate.values, color="coral")
    ax3.set_xlabel("고객 가치 구간")
    ax3.set_ylabel("구매율")
    ax3.set_title("고객 가치별 구매율\n(교란변수 → 결과)")

    plt.tight_layout()

    filepath = OUTPUT_DIR / "confounding_visualization.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\n교란 효과 시각화 저장됨: {filepath}")
    return filepath


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("제5장 예제 5.2: DoWhy를 활용한 인과 효과 추정")
    print("=" * 60)

    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 데이터 생성
    print("\n[데이터 생성]")
    df, true_effect = generate_marketing_data(n=5000)
    print(f"샘플 수: {len(df)}")
    print(f"캠페인 노출률: {df['campaign'].mean():.2%}")
    print(f"전체 구매율: {df['purchase'].mean():.2%}")
    print(f"실제 인과 효과 (True ATE): {true_effect}")

    # 데이터 저장
    data_path = OUTPUT_DIR / "marketing_campaign_data.csv"
    df.to_csv(data_path, index=False)
    print(f"\n데이터 저장됨: {data_path}")

    # 2. 단순 비교 (편향된)
    naive_effect = naive_estimation(df)

    # 3. 교란 효과 시각화
    visualize_confounding(df)

    # 4. DoWhy 추정
    results = dowhy_estimation(df, true_effect)

    # 5. 최종 비교
    print("\n" + "=" * 60)
    print("최종 결과 비교")
    print("=" * 60)
    print(f"실제 인과 효과: {true_effect:.4f}")
    print(f"단순 비교 (편향): {naive_effect:.4f} (편향: {naive_effect - true_effect:+.4f})")
    if results:
        for method, value in results.items():
            print(f"{method}: {value:.4f} (편향: {value - true_effect:+.4f})")

    print("\n결론:")
    print("- 단순 비교는 교란변수로 인해 인과 효과를 과대추정함")
    print("- DoWhy를 통해 교란변수를 통제하면 실제 효과에 가깝게 추정됨")


if __name__ == "__main__":
    main()
