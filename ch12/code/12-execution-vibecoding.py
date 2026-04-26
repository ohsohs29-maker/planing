"""
12주차: 실행 계획과 리스크 관리 - 바이브 코딩 실습 예제

복잡한 PM 도구 없이, OKR·WBS·RACI·리스크 매트릭스를
표와 산점도로 보여주는 최소 예제이다.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OKR = pd.DataFrame(
    [
        ["O", "Help young residents land a stable first job in our city", "—"],
        ["KR1", "Pilot applications", "200 / quarter"],
        ["KR2", "12-month employment rate", ">= 65%"],
        ["KR3", "Cost per outcome", "<= 8M KRW per job"],
    ],
    columns=["type", "description", "target"],
)


WBS = pd.DataFrame(
    [
        ["1", "Recruit",     "1.1 Outreach campaign",       2, "Comm Lead",  "Campaign assets"],
        ["1", "Recruit",     "1.2 Online application form", 2, "IT Lead",    "Live form"],
        ["1", "Recruit",     "1.3 Document review",         3, "Ops Lead",   "Eligibility list"],
        ["2", "Train",       "2.1 Training curriculum",     3, "Edu Lead",   "Curriculum doc"],
        ["2", "Train",       "2.2 Run cohort",              8, "Edu Lead",   "Trained cohort"],
        ["3", "Match",       "3.1 Employer onboarding",     2, "Biz Lead",   "Partner list"],
        ["3", "Match",       "3.2 Interview matching",      4, "Biz Lead",   "Match results"],
        ["4", "Evaluate",    "4.1 6-month survey",          1, "Eval Lead",  "Survey data"],
        ["4", "Evaluate",    "4.2 12-month outcome report", 2, "Eval Lead",  "Outcome report"],
    ],
    columns=["L1_id", "L1_name", "L3_task", "weeks", "owner", "deliverable"],
)


RACI = pd.DataFrame(
    [
        ["1.2 Online application form",  "IT Lead",   "Director", "Comm Lead, Ops Lead", "All staff"],
        ["1.3 Document review",          "Ops Lead",  "Director", "Legal",               "Comm Lead"],
        ["2.2 Run cohort",               "Edu Lead",  "Director", "External trainers",   "All staff"],
        ["3.2 Interview matching",       "Biz Lead",  "Director", "Edu Lead",            "All staff"],
        ["4.2 12-month outcome report",  "Eval Lead", "Director", "Ops Lead",            "Board"],
    ],
    columns=["task", "Responsible", "Accountable", "Consulted", "Informed"],
)


RISKS = pd.DataFrame(
    [
        ["R1", "Application volume below target",      "Schedule",       0.5, 4, "Mitigate", "Add 2 outreach channels", "Comm Lead", "Apply < 100 by week 6"],
        ["R2", "Training drop-out",                    "Resource",       0.4, 3, "Mitigate", "Stipend + mentor",        "Edu Lead",  "Drop-out > 25%"],
        ["R3", "Employer commitment cancellation",     "External",       0.3, 5, "Transfer", "Backup partners + MOU",   "Biz Lead",  "1+ partner cancels"],
        ["R4", "Budget overrun",                       "Cost",           0.4, 4, "Mitigate", "Quarterly budget gate",   "CFO",       "Spend > +10% YTD"],
        ["R5", "Eligibility regulation change",        "External",       0.2, 5, "Avoid",    "Lobby + flexibility clause", "Director", "Bill introduced"],
        ["R6", "Data privacy incident",                "Technical",      0.2, 5, "Avoid",    "PIA + access control",    "IT Lead",   "Audit finding"],
        ["R7", "Staff turnover",                       "Organizational", 0.5, 3, "Mitigate", "Backup roles, doc",       "HR",        "1+ leader resigns"],
        ["R8", "Regional economic shock",              "External",       0.2, 4, "Accept",   "Monitor monthly indicators", "Director", "Local UR > +2pp"],
    ],
    columns=["id", "description", "category", "probability", "impact", "strategy", "response", "owner", "trigger"],
)


def plot_risk_matrix(risks: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(
        risks["probability"], risks["impact"],
        s=140, facecolors="white", edgecolors="black", linewidths=1.2,
    )
    for _, row in risks.iterrows():
        plt.annotate(
            row["id"], (row["probability"], row["impact"]),
            xytext=(5, 5), textcoords="offset points",
            color="black", fontsize=9,
        )
    plt.axhline(3, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0.4, color="gray", linestyle="--", linewidth=0.8)
    plt.xlim(0, 1)
    plt.ylim(0, 5.5)
    plt.xlabel("Probability", color="black")
    plt.ylabel("Impact (1-5)", color="black")
    plt.title("Risk Matrix", color="black")
    plt.tight_layout()
    plt.show()


def prioritized_risks(risks: pd.DataFrame) -> pd.DataFrame:
    df = risks.copy()
    df["score"] = df["probability"] * df["impact"]
    return df.sort_values("score", ascending=False)


if __name__ == "__main__":
    print("\n[1] OKR")
    print(OKR)

    print("\n[2] WBS (level 1 + level 3)")
    print(WBS)

    print("\n[3] RACI for key tasks")
    print(RACI)

    print("\n[4] Risk register, prioritized")
    print(prioritized_risks(RISKS))

    plot_risk_matrix(RISKS)
