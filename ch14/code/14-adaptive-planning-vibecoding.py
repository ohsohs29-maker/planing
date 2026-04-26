"""
14주차: 적응적 기획 - 바이브 코딩 실습 예제

복잡한 상태 머신 없이, 트리거 표·롤링 플랜·AAR 양식만 보여주는 최소 예제이다.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRIGGERS = pd.DataFrame(
    [
        ["A1", "Pilot demand strong",     "Apply count > 200/quarter",          "Apply count < 100 for 2 quarters",         "Director",  "Reduce target / extend timeline"],
        ["A2", "Onboarding works",        "Completion >= 80% in 30 days",       "Completion < 60% for 1 quarter",           "Manager",   "Redesign onboarding flow"],
        ["A3", "Cost on plan",            "Spend within +-5%",                  "Spend > +15% YTD",                         "CFO",       "Cut scope or request supplementary budget"],
        ["A4", "Outcome rate holds",      "Employment >= 65% at 12mo",          "Employment < 50% at 12mo",                 "Director",  "Pivot to a different cohort or industry"],
        ["A5", "External regulation",     "No new regulation",                  "New regulation altering eligibility",       "Director",  "Pause and reassess"],
    ],
    columns=["id", "assumption", "in_band", "trigger_breach", "owner", "next_action"],
)


ROLLING_PLAN = pd.DataFrame(
    [
        ["Quarter", "2026 Q2", "Detailed weekly schedule", "Monthly review"],
        ["Half",    "2026 H2", "Monthly milestones",       "Quarterly review"],
        ["Year",    "2027",    "Quarterly milestones",     "Half-yearly review"],
        ["Long",    "2028-30", "Vision and direction",     "Annual review"],
    ],
    columns=["horizon", "period", "detail_level", "review_cycle"],
)


AAR_TEMPLATE = pd.DataFrame(
    [
        ["1. What did we expect?", ""],
        ["2. What actually happened?", ""],
        ["3. Why was there a gap?", ""],
        ["4. What will we do differently next?", ""],
    ],
    columns=["question", "answer"],
)


def evaluate_triggers(actuals: dict, triggers: pd.DataFrame) -> pd.DataFrame:
    """현재 KPI 값과 트리거 조건을 비교해 발동 여부를 표시한다 (간이 데모)."""
    rows = []
    for _, row in triggers.iterrows():
        breached = actuals.get(row["id"], False)
        rows.append(
            {
                "id": row["id"],
                "assumption": row["assumption"],
                "trigger": row["trigger_breach"],
                "breached?": "YES" if breached else "no",
                "owner": row["owner"],
                "next_action": row["next_action"] if breached else "",
            }
        )
    return pd.DataFrame(rows)


def fill_aar(expected: str, actual: str, gap_reason: str, next_step: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": [
                "1. What did we expect?",
                "2. What actually happened?",
                "3. Why was there a gap?",
                "4. What will we do differently next?",
            ],
            "answer": [expected, actual, gap_reason, next_step],
        }
    )


if __name__ == "__main__":
    print("\n[1] Trigger table")
    print(TRIGGERS)

    print("\n[2] Rolling plan structure")
    print(ROLLING_PLAN)

    actuals = {"A1": False, "A2": True, "A3": False, "A4": False, "A5": False}
    print("\n[3] Trigger evaluation snapshot")
    print(evaluate_triggers(actuals, TRIGGERS))

    print("\n[4] AAR example")
    aar = fill_aar(
        expected="Q2 신청 200건, 완료율 80%",
        actual="신청 230건, 완료율 58%",
        gap_reason="모집은 성공했으나 등록 절차 복잡",
        next_step="등록 양식 단순화, 자동화 도입 PoC",
    )
    print(aar)
