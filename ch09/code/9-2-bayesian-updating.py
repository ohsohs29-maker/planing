"""
9-2-bayesian-updating.py
베이지안 업데이팅: 순차적 학습과 PyMC 활용

이 코드는 새로운 데이터가 들어올 때마다 믿음을 업데이트하는
베이지안 순차 학습을 시뮬레이션한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (크로스플랫폼)
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def conjugate_beta_binomial():
    """
    Beta-Binomial 켤레 사전분포를 이용한 순차적 업데이트
    성공률 추정 예시: 신제품 구매 전환율
    """
    print("=" * 60)
    print("Beta-Binomial 켤레 업데이트: 구매 전환율 추정")
    print("=" * 60)

    # 사전분포: Beta(2, 8) - "대략 20% 정도일 것"이라는 사전 믿음
    prior_alpha = 2
    prior_beta = 8

    print(f"\n[사전분포]")
    print(f"  Beta({prior_alpha}, {prior_beta})")
    prior_mean = prior_alpha / (prior_alpha + prior_beta)
    print(f"  사전 평균: {prior_mean:.1%}")

    # 시뮬레이션된 데이터 (실제 전환율 30%)
    np.random.seed(42)
    true_rate = 0.30
    batch_sizes = [10, 20, 50, 100]  # 각 배치 크기

    # 순차적 업데이트
    alpha = prior_alpha
    beta = prior_beta
    results = [(0, alpha, beta, alpha/(alpha+beta))]

    cumulative_n = 0
    cumulative_success = 0

    print(f"\n[순차적 업데이트 과정]")
    print("-" * 60)
    print(f"{'배치':<8} {'데이터':<15} {'누적':<15} {'사후평균':<10} {'95% CI':<20}")
    print("-" * 60)

    for i, n in enumerate(batch_sizes):
        # 새 데이터 생성
        successes = np.random.binomial(n, true_rate)
        cumulative_n += n
        cumulative_success += successes

        # Beta 업데이트 (켤레 사전분포의 장점)
        alpha += successes
        beta += (n - successes)

        # 95% 신뢰구간
        ci_low = stats.beta.ppf(0.025, alpha, beta)
        ci_high = stats.beta.ppf(0.975, alpha, beta)
        post_mean = alpha / (alpha + beta)

        results.append((cumulative_n, alpha, beta, post_mean))

        print(f"배치 {i+1:<4} {successes}/{n:<12} {cumulative_success}/{cumulative_n:<12} "
              f"{post_mean:>8.1%}    [{ci_low:.1%}, {ci_high:.1%}]")

    print("-" * 60)
    print(f"실제 전환율: {true_rate:.1%}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 사후분포 변화
    ax1 = axes[0]
    x = np.linspace(0, 1, 1000)
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(results)))

    for i, (n, a, b, _) in enumerate(results):
        label = f'n={n}' if n > 0 else '사전분포'
        ax1.plot(x, stats.beta.pdf(x, a, b), color=colors[i], linewidth=2, label=label)

    ax1.axvline(x=true_rate, color='red', linestyle='--', linewidth=2, label=f'실제값: {true_rate:.0%}')
    ax1.set_xlabel('전환율', fontsize=12)
    ax1.set_ylabel('확률밀도', fontsize=12)
    ax1.set_title('데이터 누적에 따른 사후분포 변화', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 사후평균과 신뢰구간 변화
    ax2 = axes[1]
    ns = [r[0] for r in results]
    means = [r[3] for r in results]
    ci_lows = [stats.beta.ppf(0.025, r[1], r[2]) for r in results]
    ci_highs = [stats.beta.ppf(0.975, r[1], r[2]) for r in results]

    ax2.fill_between(ns, ci_lows, ci_highs, alpha=0.3, label='95% 신뢰구간')
    ax2.plot(ns, means, 'bo-', linewidth=2, markersize=8, label='사후평균')
    ax2.axhline(y=true_rate, color='red', linestyle='--', linewidth=2, label=f'실제값: {true_rate:.0%}')
    ax2.set_xlabel('누적 표본 수', fontsize=12)
    ax2.set_ylabel('전환율', fontsize=12)
    ax2.set_title('사후평균과 신뢰구간의 수렴', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../data/sequential_updating.png', dpi=150)
    plt.close()

    print("\n그래프 저장: ../data/sequential_updating.png")

    return results


def prior_convergence():
    """
    다른 사전분포에서 시작해도 데이터가 많으면 수렴함을 보여줌
    """
    print("\n" + "=" * 60)
    print("사전분포 수렴: 다른 사전 믿음도 결국 수렴")
    print("=" * 60)

    np.random.seed(42)
    true_rate = 0.35
    n_obs = 200
    successes = np.random.binomial(n_obs, true_rate)

    # 다양한 사전분포
    priors = [
        (1, 1, "무정보 (균등)"),
        (1, 9, "비관적 (10%)"),
        (5, 5, "중립적 (50%)"),
        (9, 1, "낙관적 (90%)")
    ]

    print(f"\n[조건]")
    print(f"  실제 전환율: {true_rate:.1%}")
    print(f"  관측 데이터: {successes}/{n_obs} 성공")

    print(f"\n[사전분포별 사후분포]")
    print("-" * 60)
    print(f"{'사전분포':<20} {'사전평균':<12} {'사후평균':<12} {'95% CI':<20}")
    print("-" * 60)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 1, 1000)
    colors = ['blue', 'orange', 'green', 'purple']

    for i, (a, b, name) in enumerate(priors):
        prior_mean = a / (a + b)
        post_a = a + successes
        post_b = b + (n_obs - successes)
        post_mean = post_a / (post_a + post_b)
        ci_low = stats.beta.ppf(0.025, post_a, post_b)
        ci_high = stats.beta.ppf(0.975, post_a, post_b)

        print(f"{name:<20} {prior_mean:<12.1%} {post_mean:<12.1%} [{ci_low:.1%}, {ci_high:.1%}]")

        ax.plot(x, stats.beta.pdf(x, post_a, post_b), color=colors[i],
                linewidth=2, label=f'{name}: 사후평균 {post_mean:.1%}')

    ax.axvline(x=true_rate, color='red', linestyle='--', linewidth=2, label=f'실제값: {true_rate:.0%}')
    ax.set_xlabel('전환율', fontsize=12)
    ax.set_ylabel('확률밀도', fontsize=12)
    ax.set_title(f'다른 사전분포도 충분한 데이터로 수렴\n(n={n_obs}, 성공={successes})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../data/prior_convergence.png', dpi=150)
    plt.close()

    print("-" * 60)
    print("\n-> 충분한 데이터가 있으면 사전분포 영향이 줄어들고 수렴함")
    print("\n그래프 저장: ../data/prior_convergence.png")


def normal_updating():
    """
    정규분포 파라미터의 베이지안 업데이트
    시장 규모 추정 예시
    """
    print("\n" + "=" * 60)
    print("정규분포 업데이트: 시장 규모 추정")
    print("=" * 60)

    # 사전분포: 전문가 의견 기반
    prior_mean = 100  # 100억 원
    prior_std = 30    # 불확실성

    print(f"\n[사전분포 - 전문가 의견]")
    print(f"  평균: {prior_mean}억 원")
    print(f"  표준편차: {prior_std}억 원")
    print(f"  90% 신뢰구간: [{prior_mean - 1.645*prior_std:.0f}, {prior_mean + 1.645*prior_std:.0f}]억 원")

    # 시장조사 데이터
    np.random.seed(42)
    true_market = 120  # 실제 시장 규모
    survey_data = np.random.normal(true_market, 25, 5)  # 5개 조사 결과
    data_mean = np.mean(survey_data)
    data_std = 25  # 측정 불확실성 (알려져 있다고 가정)

    print(f"\n[시장조사 데이터]")
    print(f"  조사 결과: {[f'{x:.0f}' for x in survey_data]}억 원")
    print(f"  데이터 평균: {data_mean:.1f}억 원")

    # 정규분포 베이지안 업데이트
    n = len(survey_data)
    prior_precision = 1 / prior_std**2
    data_precision = n / data_std**2

    post_precision = prior_precision + data_precision
    post_var = 1 / post_precision
    post_std = np.sqrt(post_var)
    post_mean = (prior_precision * prior_mean + data_precision * data_mean) / post_precision

    print(f"\n[사후분포 - 업데이트된 추정]")
    print(f"  평균: {post_mean:.1f}억 원")
    print(f"  표준편차: {post_std:.1f}억 원")
    print(f"  90% 신뢰구간: [{post_mean - 1.645*post_std:.0f}, {post_mean + 1.645*post_std:.0f}]억 원")

    print(f"\n[변화 분석]")
    print(f"  평균 이동: {prior_mean}억 -> {post_mean:.1f}억 (+{post_mean-prior_mean:.1f}억)")
    print(f"  불확실성 감소: {prior_std}억 -> {post_std:.1f}억 (-{prior_std-post_std:.1f}억)")

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 200, 1000)

    ax.plot(x, stats.norm.pdf(x, prior_mean, prior_std), 'b--',
            linewidth=2, label=f'사전분포: N({prior_mean}, {prior_std})')
    ax.plot(x, stats.norm.pdf(x, data_mean, data_std/np.sqrt(n)), 'g--',
            linewidth=2, label=f'데이터: 평균 {data_mean:.0f}')
    ax.plot(x, stats.norm.pdf(x, post_mean, post_std), 'r-',
            linewidth=3, label=f'사후분포: N({post_mean:.0f}, {post_std:.0f})')
    ax.axvline(x=true_market, color='black', linestyle=':', linewidth=2, label=f'실제값: {true_market}억')

    ax.set_xlabel('시장 규모 (억 원)', fontsize=12)
    ax.set_ylabel('확률밀도', fontsize=12)
    ax.set_title('시장 규모 베이지안 추정: 사전분포 + 데이터 = 사후분포', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../data/normal_updating.png', dpi=150)
    plt.close()

    print("\n그래프 저장: ../data/normal_updating.png")

    return post_mean, post_std


def credible_interval_demo():
    """
    신용구간(Credible Interval)의 직관적 해석
    """
    print("\n" + "=" * 60)
    print("신용구간 vs 신뢰구간")
    print("=" * 60)

    print("""
[빈도주의 신뢰구간 (Confidence Interval)]
  "이 방법으로 100번 구간을 만들면, 95번은 참값을 포함한다"
  - 참값은 고정, 구간이 확률적
  - "이 특정 구간이 참값을 포함할 확률"은 말할 수 없음

[베이지안 신용구간 (Credible Interval)]
  "참값이 이 구간 안에 있을 확률이 95%이다"
  - 구간은 고정, 참값이 확률 분포를 따름
  - 직관적으로 원하는 해석
    """)

    # 예시: 전환율 추정
    alpha, beta = 35, 65  # 사후분포 Beta(35, 65)
    ci_low = stats.beta.ppf(0.025, alpha, beta)
    ci_high = stats.beta.ppf(0.975, alpha, beta)
    hdi_low, hdi_high = 0.27, 0.43  # 근사값 (실제로는 arviz 등으로 계산)

    print(f"\n[예시: 전환율 추정]")
    print(f"  사후분포: Beta({alpha}, {beta})")
    print(f"  사후평균: {alpha/(alpha+beta):.1%}")
    print(f"  95% Equal-Tailed CI: [{ci_low:.1%}, {ci_high:.1%}]")
    print(f"  -> '전환율이 {ci_low:.1%}~{ci_high:.1%} 사이일 확률이 95%'")


if __name__ == "__main__":
    # Beta-Binomial 순차 업데이트
    conjugate_beta_binomial()

    # 사전분포 수렴
    prior_convergence()

    # 정규분포 업데이트
    normal_updating()

    # 신용구간 설명
    credible_interval_demo()

    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)
