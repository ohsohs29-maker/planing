"""
10주차: 시나리오 플래닝 - 바이브 코딩 실습 예제

복잡한 모델 없이, 동인 점수표·불확실성-영향도 매트릭스·2x2 시나리오·
윈드 터널링 비교를 만들어 보는 최소 예제이다.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DRIVING_FORCES = pd.DataFrame(
    {
        "force": [
            "EV battery cost decline",
            "Charging infrastructure rollout",
            "Carbon regulation tightening",
            "Renewable electricity share",
            "Aging population in cities",
            "Remote work adoption",
            "Autonomous driving readiness",
            "Used car market growth",
            "Lithium supply risk",
            "Public transit investment",
            "Shared mobility platforms",
            "Anti-EV policy backlash",
        ],
        "category": [
            "Tech", "Tech", "Political", "Ecological",
            "Social", "Social", "Tech", "Economic",
            "Economic", "Political", "Social", "Political",
        ],
        "uncertainty": [4, 7, 8, 5, 3, 6, 8, 4, 9, 6, 6, 7],
        "impact":      [8, 9, 9, 7, 5, 5, 8, 4, 6, 7, 6, 7],
    }
)


def plot_uncertainty_impact(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(
        df["uncertainty"], df["impact"],
        s=120, facecolors="white", edgecolors="black", linewidths=1.2,
    )
    for _, row in df.iterrows():
        plt.annotate(
            row["force"], (row["uncertainty"], row["impact"]),
            xytext=(5, 5), textcoords="offset points",
            color="black", fontsize=8,
        )
    plt.axhline(5.5, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(5.5, color="gray", linestyle="--", linewidth=0.8)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("Uncertainty", color="black")
    plt.ylabel("Impact", color="black")
    plt.title("Uncertainty-Impact Matrix", color="black")
    plt.tight_layout()
    plt.show()


def pick_scenario_axes(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """불확실성과 영향도 모두 높은 동인 중 상위 n개를 시나리오 축으로 추천한다."""
    df = df.copy()
    df["score"] = df["uncertainty"] * df["impact"]
    return df.sort_values("score", ascending=False).head(n)


SCENARIOS = pd.DataFrame(
    {
        "name": ["Green Sprint", "Stuck in Traffic", "Tech Race", "Slow Transition"],
        "x_axis": ["+", "-", "+", "-"],   # Carbon regulation
        "y_axis": ["+", "-", "-", "+"],   # Charging rollout
    }
)


STRATEGIES_VS_SCENARIOS = pd.DataFrame(
    [
        ["Strategy A: Aggressive EV factory",   2,  -2,  1, -1],
        ["Strategy B: Hybrid + transit invest", 1,   1,  1,  2],
        ["Strategy C: Wait & license tech",    -1,   2, -1,  1],
    ],
    columns=["strategy", "Green Sprint", "Stuck in Traffic", "Tech Race", "Slow Transition"],
)


def robustness_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["min_score"] = df.iloc[:, 1:].min(axis=1)
    df["mean_score"] = df.iloc[:, 1:5].mean(axis=1)
    return df.sort_values("min_score", ascending=False)


def early_warning_indicators() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Green Sprint",      "EV market share > 30% by 2028",        "Quarterly registrations data"],
            ["Stuck in Traffic",  "Charging stations growth < 5% per yr", "City data dashboard"],
            ["Tech Race",         "OEM autonomous pilots in 5+ cities",   "Industry news scan"],
            ["Slow Transition",   "Carbon tax delayed past 2027",         "Government bill tracker"],
        ],
        columns=["scenario", "indicator", "data_source"],
    )


if __name__ == "__main__":
    print("\n[1] Driving forces (sample)")
    print(DRIVING_FORCES.head())

    plot_uncertainty_impact(DRIVING_FORCES)

    print("\n[2] Recommended scenario axes (highest uncertainty x impact)")
    print(pick_scenario_axes(DRIVING_FORCES))

    print("\n[3] 2x2 scenarios")
    print(SCENARIOS)

    print("\n[4] Wind tunneling: strategy vs scenarios")
    print(robustness_score(STRATEGIES_VS_SCENARIOS))

    print("\n[5] Early warning indicators")
    print(early_warning_indicators())
