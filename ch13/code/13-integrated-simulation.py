"""13주차: 통합 정책 시뮬레이션 워크숍 — 청년 주거 지원금 사례.

5개 도구(DAG·CLD·베이지안·시나리오·몬테카를로)를 한 사례에 차례대로 적용한 예시.
학생은 노트북 ch13.ipynb의 빈 코드 셀에 자기 주제로 변형하여 사용한다.
"""
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 한글 폰트
for font in ["AppleGothic", "Malgun Gothic", "맑은 고딕", "NanumGothic"]:
    if font in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams["font.family"] = font
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "black"

np.random.seed(13)


# ─────────────────────────────────────────────
# 1단계: DAG (인과 분석)
# ─────────────────────────────────────────────
def step1_dag():
    G = nx.DiGraph()
    edges = [
        ("지원금 정책", "주거 비용", "-"),
        ("주거 비용", "청년 주거 안정", "-"),
        ("주거 인프라", "청년 주거 안정", "+"),
        ("가구 소득", "주거 비용", "-"),
        ("임대 시장 가격", "주거 비용", "+"),
        ("지원금 정책", "임대 시장 가격", "+"),
        ("지역 경기", "임대 시장 가격", "+"),
        ("지역 경기", "가구 소득", "+"),
    ]
    for a, b, _ in edges:
        G.add_edge(a, b)
    pos = nx.spring_layout(G, seed=13)
    plt.figure(figsize=(9, 6))
    nx.draw(G, pos, with_labels=True, node_color="white", edgecolors="black",
            node_size=2400, font_size=9, arrowsize=15, edge_color="gray")
    edge_labels = {(a, b): s for a, b, s in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    plt.title("청년 주거 정책 DAG (예시)")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 5단계: 몬테카를로 (5년 후 안정 지수)
# ─────────────────────────────────────────────
def step5_montecarlo(n=10000, target=0.20):
    유입 = np.random.normal(0.05, 0.03, n)
    상승 = np.random.triangular(0.01, 0.04, 0.10, n)
    효과 = np.random.normal(0.6, 0.2, n)
    안정 = (효과 * (유입 - 상승) + 0.10) * 5  # 5년 누적
    print(f"평균: {안정.mean():.2%}")
    print(f"5분위: {np.percentile(안정, 5):.2%}")
    print(f"95분위: {np.percentile(안정, 95):.2%}")
    print(f"목표 {target:.0%} 달성 확률: {(안정 >= target).mean():.2%}")
    plt.figure(figsize=(8, 4))
    plt.hist(안정, bins=40, color="white", edgecolor="black")
    plt.axvline(target, color="black", linestyle="--", label=f"목표 {target:.0%}")
    plt.xlabel("5년 후 안정 지수")
    plt.ylabel("빈도")
    plt.title("몬테카를로: 정책 효과 분포 (n=10,000)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    step1_dag()
    step5_montecarlo()
