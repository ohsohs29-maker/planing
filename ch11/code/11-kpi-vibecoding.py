"""
13주차: 성과 측정과 KPI - 바이브 코딩 실습 예제

복잡한 시계열 모델 없이, 이동 평균과 표준편차 밴드만으로
KPI 모니터링과 이상 탐지를 보여주는 최소 예제이다.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KPIS = pd.DataFrame(
    [
        ["Pilot applications",        "Leading", "Customer",         "Apply count > 200/quarter"],
        ["Onboarding completion",     "Leading", "Internal Process", ">= 80% within 30 days"],
        ["Training hours / staff",    "Leading", "Learning & Growth", ">= 12h/year"],
        ["Spend rate vs plan",        "Leading", "Financial",        "Within +-5% of plan"],
        ["Employment rate (12 mo)",   "Lagging", "Customer",         ">= 65%"],
        ["Cost per outcome",          "Lagging", "Financial",        "<= 8M KRW per job"],
        ["Process error rate",        "Lagging", "Internal Process", "<= 2%"],
        ["NPS",                       "Lagging", "Customer",         ">= 30"],
    ],
    columns=["kpi", "type", "bsc_perspective", "smart_target"],
)


def make_synthetic_kpi_series(n_weeks: int = 52, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    week = np.arange(n_weeks)
    trend = 100 + 0.3 * week
    noise = rng.normal(0, 4, n_weeks)
    series = trend + noise
    # inject anomalies at weeks 30 and 45
    series[30] += 18
    series[45] -= 22
    return pd.DataFrame({"week": week, "value": series})


def detect_anomalies(df: pd.DataFrame, window: int = 8, k: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    df["ma"] = df["value"].rolling(window, min_periods=1).mean()
    df["sd"] = df["value"].rolling(window, min_periods=1).std().fillna(0)
    df["upper"] = df["ma"] + k * df["sd"]
    df["lower"] = df["ma"] - k * df["sd"]
    df["status"] = np.where(
        (df["value"] > df["upper"]) | (df["value"] < df["lower"]), "Alert",
        np.where(df["value"] > df["ma"] + df["sd"], "Watch", "Good"),
    )
    return df


def plot_kpi_with_band(df: pd.DataFrame, title: str = "KPI with +-2 sigma band") -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(df["week"], df["value"], color="black", linewidth=1, label="Value")
    plt.plot(df["week"], df["ma"], color="gray", linestyle="--", linewidth=0.8, label="MA")
    plt.fill_between(df["week"], df["lower"], df["upper"], color="lightgray", alpha=0.4, label="+-2 sigma")

    alerts = df[df["status"] == "Alert"]
    plt.scatter(alerts["week"], alerts["value"], s=60, facecolors="white", edgecolors="black", linewidths=1.2, label="Alert")

    plt.xlabel("Week", color="black")
    plt.ylabel("Value", color="black")
    plt.title(title, color="black")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def dashboard_status(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    return {
        "current": round(last["value"], 1),
        "ma": round(last["ma"], 1),
        "status": last["status"],
        "alerts_total": int((df["status"] == "Alert").sum()),
    }


if __name__ == "__main__":
    print("\n[1] KPI catalog")
    print(KPIS)

    series = make_synthetic_kpi_series()
    detected = detect_anomalies(series)
    print("\n[2] Last 5 weeks")
    print(detected.tail()[["week", "value", "ma", "status"]])

    plot_kpi_with_band(detected, title="Pilot applications (weekly)")

    print("\n[3] Dashboard one-liner")
    print(dashboard_status(detected))
