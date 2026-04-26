"""
9주차: 쉬운 베이지안 의사결정 - 바이브 코딩 실습 예제

복잡한 켤레사전분포 수학 없이, 베타 분포 한 가지로
사전→사후 업데이트와 데이터 누적 효과를 보여주는 최소 예제이다.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta


def beta_posterior(prior_alpha: float, prior_beta: float, successes: int, failures: int):
    """베타 사전분포에 이항 데이터를 결합해 베타 사후분포 파라미터를 반환한다."""
    return prior_alpha + successes, prior_beta + failures


def plot_prior_posterior(prior_alpha: float, prior_beta: float, successes: int, failures: int):
    x = np.linspace(0, 1, 200)
    post_alpha, post_beta = beta_posterior(prior_alpha, prior_beta, successes, failures)

    plt.figure(figsize=(7, 4))
    plt.plot(x, beta.pdf(x, prior_alpha, prior_beta), color="gray", linestyle="--", label="Prior")
    plt.plot(x, beta.pdf(x, post_alpha, post_beta), color="black", label="Posterior")
    plt.axvline(successes / max(successes + failures, 1), color="black", linestyle=":", alpha=0.5)

    plt.xlabel("Probability of success", color="black")
    plt.ylabel("Density", color="black")
    plt.title("Prior vs Posterior", color="black")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return post_alpha, post_beta


def update_table(prior_alpha: float, prior_beta: float, sample_sizes: list[int], success_rate: float):
    """데이터가 쌓이면 사후분포가 어떻게 변하는지 표로 만든다."""
    rows = []
    for n in sample_sizes:
        s = int(round(n * success_rate))
        f = n - s
        a, b = beta_posterior(prior_alpha, prior_beta, s, f)
        rows.append(
            {
                "sample_size": n,
                "successes": s,
                "posterior_mean": round(a / (a + b), 3),
                "posterior_lower_5%": round(beta.ppf(0.05, a, b), 3),
                "posterior_upper_95%": round(beta.ppf(0.95, a, b), 3),
            }
        )
    return pd.DataFrame(rows)


def make_decision(posterior_alpha: float, posterior_beta: float, threshold: float = 0.5):
    """사후 평균이 임계값을 넘으면 'Go', 아니면 'Hold'."""
    mean = posterior_alpha / (posterior_alpha + posterior_beta)
    prob_above = 1 - beta.cdf(threshold, posterior_alpha, posterior_beta)
    return {
        "posterior_mean": round(mean, 3),
        "P(success > threshold)": round(prob_above, 3),
        "decision": "Go" if mean > threshold else "Hold",
    }


if __name__ == "__main__":
    prior_a, prior_b = 3, 7
    print(f"\n[1] Prior Beta({prior_a}, {prior_b})  prior mean={prior_a / (prior_a + prior_b):.2f}")

    post_a, post_b = plot_prior_posterior(prior_a, prior_b, successes=7, failures=3)
    print(f"\n[2] Posterior Beta({post_a}, {post_b})  posterior mean={post_a / (post_a + post_b):.2f}")

    print("\n[3] Posterior as data accumulates (true rate = 0.7)")
    print(update_table(prior_a, prior_b, [10, 50, 200, 1000], success_rate=0.7))

    print("\n[4] Decision")
    print(make_decision(post_a, post_b, threshold=0.5))
