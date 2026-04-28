"""
11주차: 몬테카를로 시뮬레이션 - 바이브 코딩 실습 예제

복잡한 통계 모델 없이, 정규/삼각/균등/로그정규 분포로 입력을 잡고
N=10,000번 굴려서 결과 분포·손실 확률·토네이도 차트를 만드는 최소 예제이다.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def simulate_business_plan(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """청년 일자리 정책의 5년 누적 순이익 시뮬레이션."""
    rng = np.random.default_rng(seed)

    annual_revenue = rng.normal(loc=10.0, scale=2.0, size=n)            # 정규
    variable_cost_ratio = rng.triangular(left=0.4, mode=0.5, right=0.7, size=n)  # 삼각
    annual_fixed = rng.normal(loc=3.0, scale=0.5, size=n)               # 정규
    macro_shock = rng.uniform(low=-0.05, high=0.05, size=n)             # 균등 (±5%)

    annual_profit = annual_revenue * (1 - variable_cost_ratio) * (1 + macro_shock) - annual_fixed
    five_year_profit = 5 * annual_profit

    return pd.DataFrame(
        {
            "annual_revenue": annual_revenue,
            "variable_cost_ratio": variable_cost_ratio,
            "annual_fixed": annual_fixed,
            "macro_shock": macro_shock,
            "five_year_profit": five_year_profit,
        }
    )


def summarize(results: pd.DataFrame, target: str = "five_year_profit") -> dict:
    series = results[target]
    return {
        "mean": round(series.mean(), 2),
        "p05": round(np.percentile(series, 5), 2),
        "p95": round(np.percentile(series, 95), 2),
        "P(loss<0)": round((series < 0).mean(), 3),
    }


def plot_result_histogram(results: pd.DataFrame, target: str = "five_year_profit") -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(results[target], bins=40, color="white", edgecolor="black", linewidth=0.6)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel(target, color="black")
    plt.ylabel("count", color="black")
    plt.title("Result Distribution (N=10,000)", color="black")
    plt.tight_layout()
    plt.show()


def tornado(results: pd.DataFrame, target: str = "five_year_profit") -> pd.DataFrame:
    """입력 변수를 ±1 표준편차로 흔들었을 때 평균 결과 변화를 측정한다."""
    inputs = ["annual_revenue", "variable_cost_ratio", "annual_fixed", "macro_shock"]
    rows = []
    for col in inputs:
        median = results[col].median()
        std = results[col].std()
        low_mask = results[col] < (median - std)
        high_mask = results[col] > (median + std)
        low_mean = results.loc[low_mask, target].mean()
        high_mean = results.loc[high_mask, target].mean()
        rows.append(
            {"variable": col, "low_mean": round(low_mean, 2), "high_mean": round(high_mean, 2),
             "swing": round(abs(high_mean - low_mean), 2)}
        )
    return pd.DataFrame(rows).sort_values("swing", ascending=False)


def plot_tornado(tornado_df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    plt.barh(tornado_df["variable"], tornado_df["swing"], color="white", edgecolor="black")
    plt.xlabel("Swing of mean profit", color="black")
    plt.title("Tornado Chart: which input matters most?", color="black")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def scenario_simulation(n: int = 10_000) -> pd.DataFrame:
    """4시나리오 × 같은 사업 — 시나리오별 입력 평균을 다르게 잡는다."""
    rng = np.random.default_rng(0)
    rows = []
    scenarios = {
        "Green Sprint":     {"revenue_mean": 12.0, "cost_ratio": 0.45},
        "Tech Race":        {"revenue_mean": 11.0, "cost_ratio": 0.50},
        "Stuck in Traffic": {"revenue_mean":  9.0, "cost_ratio": 0.55},
        "Slow Transition":  {"revenue_mean":  8.0, "cost_ratio": 0.60},
    }
    for name, params in scenarios.items():
        revenue = rng.normal(params["revenue_mean"], 2.0, n)
        cost_ratio = rng.triangular(0.4, params["cost_ratio"], 0.7, n)
        fixed = rng.normal(3.0, 0.5, n)
        profit = 5 * (revenue * (1 - cost_ratio) - fixed)
        rows.append(
            {
                "scenario": name,
                "mean": round(profit.mean(), 2),
                "p05": round(np.percentile(profit, 5), 2),
                "p95": round(np.percentile(profit, 95), 2),
                "P(loss<0)": round((profit < 0).mean(), 3),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = simulate_business_plan()
    print("\n[1] Five-year profit summary")
    print(summarize(results))

    plot_result_histogram(results)

    print("\n[2] Tornado: which input swings the result most?")
    tornado_df = tornado(results)
    print(tornado_df)
    plot_tornado(tornado_df)

    print("\n[3] Scenario integration: same plan in 4 futures")
    print(scenario_simulation())
