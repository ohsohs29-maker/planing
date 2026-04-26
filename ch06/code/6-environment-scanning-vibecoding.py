"""
6주차: AI 기반 환경 스캐닝 - 바이브 코딩 실습 예제

복잡한 뉴스 API나 자연어처리 모델 없이, 짧은 뉴스 제목 목록에서
반복 키워드와 약신호 후보를 찾는 최소 예제이다.
"""

from collections import Counter
import re

import matplotlib.pyplot as plt
import pandas as pd


NEWS_TITLES = [
    "City expands youth housing support program",
    "Rising rent increases demand for public rental housing",
    "Local university opens AI job training course",
    "Small businesses ask city for digital transformation support",
    "Citizens raise privacy concerns over AI welfare screening",
    "New public data platform supports local startup growth",
    "Youth housing applications rise after rent increase",
    "AI tool helps officials review welfare documents faster",
    "Privacy groups call for rules on AI decision systems",
    "City plans more rental housing near transit stations",
    "Startup funding shifts toward AI and public data services",
    "Residents demand privacy standards for welfare AI screening",
]

STOPWORDS = {
    "for",
    "and",
    "the",
    "on",
    "over",
    "near",
    "more",
    "new",
    "city",
    "local",
    "public",
    "after",
    "call",
    "clearer",
    "faster",
    "toward",
}


def tokenize(title: str) -> list[str]:
    """뉴스 제목을 쉬운 영어 키워드 목록으로 바꾼다."""
    words = re.findall(r"[A-Za-z]+", title.lower())
    return [word for word in words if word not in STOPWORDS and (len(word) >= 3 or word == "ai")]


def count_keywords(titles: list[str]) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for title in titles:
        counter.update(tokenize(title))

    return (
        pd.DataFrame(counter.items(), columns=["keyword", "count"])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def compare_early_recent(titles: list[str]) -> pd.DataFrame:
    midpoint = len(titles) // 2
    early = count_keywords(titles[:midpoint]).rename(columns={"count": "early"})
    recent = count_keywords(titles[midpoint:]).rename(columns={"count": "recent"})

    trend = pd.merge(early, recent, on="keyword", how="outer").fillna(0)
    trend["change"] = trend["recent"] - trend["early"]
    trend["weak_signal"] = trend.apply(
        lambda row: "candidate" if row["change"] >= 1 and row["recent"] >= 2 else "",
        axis=1,
    )
    return trend.sort_values(["change", "recent"], ascending=False).reset_index(drop=True)


def make_planning_table(trend: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in trend.head(6).iterrows():
        keyword = row["keyword"]
        direction = "increasing" if row["change"] > 0 else "stable/decreasing"
        rows.append(
            {
                "keyword": keyword,
                "direction": direction,
                "opportunity": f"Plan a small response related to {keyword}.",
                "risk": f"Check whether {keyword} creates public concern or budget pressure.",
                "next_action": "Collect 5 more recent articles and ask AI to summarize them.",
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    keyword_counts = count_keywords(NEWS_TITLES)
    print("\n[1] Top keywords")
    print(keyword_counts.head(10))

    trend_table = compare_early_recent(NEWS_TITLES)
    print("\n[2] Early vs recent trend")
    print(trend_table.head(10))

    planning_table = make_planning_table(trend_table)
    print("\n[3] Planning interpretation")
    print(planning_table)

    top = keyword_counts.head(8)
    plt.figure(figsize=(8, 4))
    plt.bar(top["keyword"], top["count"])
    plt.title("Top Keywords in News Titles")
    plt.xlabel("Keyword")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
