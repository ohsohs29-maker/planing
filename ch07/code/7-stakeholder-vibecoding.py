"""
7주차: 이해관계자 분석과 정책 네트워크 - 바이브 코딩 실습 예제

복잡한 SNA 알고리즘 없이, 이해관계자 목록과 간단한 영향 관계 표만으로
영향력-관심도 매트릭스와 작은 네트워크를 그리는 최소 예제이다.
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


STAKEHOLDERS = pd.DataFrame(
    {
        "name": [
            "City Mayor",
            "Housing Dept",
            "Youth Council",
            "Local Press",
            "Construction Co.",
            "Resident Group",
            "University Lab",
            "Bank",
        ],
        "power": [9, 8, 4, 6, 7, 5, 3, 6],
        "interest": [7, 9, 9, 6, 5, 8, 6, 3],
    }
)


INFLUENCE_EDGES = [
    ("City Mayor", "Housing Dept"),
    ("Housing Dept", "Construction Co."),
    ("Housing Dept", "Resident Group"),
    ("Youth Council", "City Mayor"),
    ("Local Press", "City Mayor"),
    ("Resident Group", "Local Press"),
    ("University Lab", "Housing Dept"),
    ("Bank", "Construction Co."),
]


def label_quadrant(row: pd.Series) -> str:
    """영향력-관심도 4분면 라벨을 만든다."""
    if row["power"] >= 6 and row["interest"] >= 6:
        return "Key Player"
    if row["power"] >= 6:
        return "Manage"
    if row["interest"] >= 6:
        return "Keep Informed"
    return "Monitor"


def plot_power_interest(stakeholders: pd.DataFrame) -> None:
    df = stakeholders.copy()
    df["quadrant"] = df.apply(label_quadrant, axis=1)

    plt.figure(figsize=(7, 6))
    plt.scatter(
        df["interest"],
        df["power"],
        s=120,
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
    )
    for _, row in df.iterrows():
        plt.annotate(
            row["name"],
            (row["interest"], row["power"]),
            xytext=(6, 6),
            textcoords="offset points",
            color="black",
            fontsize=9,
        )

    plt.axhline(5.5, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(5.5, color="gray", linestyle="--", linewidth=0.8)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("Interest", color="black")
    plt.ylabel("Power", color="black")
    plt.title("Power-Interest Matrix", color="black")
    plt.tight_layout()
    plt.show()


def build_network(edges: list[tuple[str, str]]) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def centrality_table(G: nx.DiGraph) -> pd.DataFrame:
    degree = nx.degree_centrality(G)
    return (
        pd.DataFrame(
            {
                "name": list(degree.keys()),
                "degree_centrality": list(degree.values()),
            }
        )
        .sort_values("degree_centrality", ascending=False)
        .reset_index(drop=True)
    )


def plot_network(G: nx.DiGraph) -> None:
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(7, 6))
    nx.draw_networkx_nodes(
        G, pos, node_color="white", edgecolors="black", linewidths=1.2, node_size=900
    )
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True)
    nx.draw_networkx_labels(G, pos, font_color="black", font_size=8)
    plt.title("Stakeholder Influence Network", color="black")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def make_planning_table(stakeholders: pd.DataFrame) -> pd.DataFrame:
    df = stakeholders.copy()
    df["quadrant"] = df.apply(label_quadrant, axis=1)
    actions = {
        "Key Player": "Hold weekly meetings and co-design the plan.",
        "Manage": "Brief regularly so they do not block decisions.",
        "Keep Informed": "Share updates so they support the policy publicly.",
        "Monitor": "Watch only; act if their position changes.",
    }
    df["next_action"] = df["quadrant"].map(actions)
    return df[["name", "power", "interest", "quadrant", "next_action"]]


if __name__ == "__main__":
    print("\n[1] Stakeholders with quadrant")
    print(make_planning_table(STAKEHOLDERS))

    plot_power_interest(STAKEHOLDERS)

    G = build_network(INFLUENCE_EDGES)
    print("\n[2] Degree centrality")
    print(centrality_table(G))

    plot_network(G)
