"""
제6장 예제 6.1: 인과 루프 다이어그램(CLD) 시각화

이 코드는 NetworkX를 활용하여 인과 루프 다이어그램을 시각화한다.
강화 루프(R)와 균형 루프(B)를 식별하고, 시스템 구조를 분석한다.
"""

import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

# 결과 저장 경로
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_cld_graph(edges: List[Tuple[str, str, str]]) -> nx.DiGraph:
    """인과 루프 다이어그램 그래프 생성

    Args:
        edges: (원인, 결과, 극성) 튜플 리스트
               극성: '+' (같은 방향) 또는 '-' (반대 방향)

    Returns:
        nx.DiGraph: 인과 루프 다이어그램 그래프
    """
    G = nx.DiGraph()

    for source, target, polarity in edges:
        G.add_edge(source, target, polarity=polarity)

    return G


def identify_loops(G: nx.DiGraph) -> Dict[str, List]:
    """피드백 루프 식별 및 분류

    강화 루프(R): 루프 내 음의 극성 개수가 짝수
    균형 루프(B): 루프 내 음의 극성 개수가 홀수
    """
    loops = {"reinforcing": [], "balancing": []}

    # 모든 단순 사이클 찾기
    try:
        cycles = list(nx.simple_cycles(G))
    except:
        return loops

    for cycle in cycles:
        if len(cycle) < 2:
            continue

        # 루프 내 극성 계산
        negative_count = 0
        for i in range(len(cycle)):
            source = cycle[i]
            target = cycle[(i + 1) % len(cycle)]
            if G.has_edge(source, target):
                polarity = G.edges[source, target].get("polarity", "+")
                if polarity == "-":
                    negative_count += 1

        # 분류
        if negative_count % 2 == 0:
            loops["reinforcing"].append(cycle)
        else:
            loops["balancing"].append(cycle)

    return loops


def create_saas_growth_cld():
    """SaaS 스타트업 성장 CLD 예시

    핵심 변수:
    - 고객 수
    - 구전 효과
    - 서버 부하
    - 서비스 품질
    - 고객 이탈
    - 인프라 투자

    강화 루프(R1): 고객 → 구전 → 신규고객 → 고객 (성장 엔진)
    균형 루프(B1): 고객 → 서버부하 → 품질저하 → 이탈 → 고객감소 (성장 한계)
    균형 루프(B2): 서버부하 → 투자 → 서버용량 → 부하완화 (투자 대응)
    """
    edges = [
        # 강화 루프: 성장 엔진
        ("고객 수", "구전 효과", "+"),
        ("구전 효과", "신규 고객", "+"),
        ("신규 고객", "고객 수", "+"),

        # 균형 루프: 성장 한계
        ("고객 수", "서버 부하", "+"),
        ("서버 부하", "서비스 품질", "-"),
        ("서비스 품질", "고객 만족도", "+"),
        ("고객 만족도", "고객 이탈", "-"),
        ("고객 이탈", "고객 수", "-"),

        # 균형 루프: 투자 대응
        ("서버 부하", "인프라 투자", "+"),
        ("인프라 투자", "서버 용량", "+"),
        ("서버 용량", "서버 부하", "-"),
    ]

    return create_cld_graph(edges)


def create_market_competition_cld():
    """시장 경쟁 동태 CLD

    가격 경쟁의 피드백 구조
    """
    edges = [
        # 강화 루프: 시장점유율 경쟁
        ("시장점유율", "규모의 경제", "+"),
        ("규모의 경제", "원가", "-"),
        ("원가", "가격", "+"),
        ("가격", "경쟁력", "-"),
        ("경쟁력", "시장점유율", "+"),

        # 균형 루프: 수익성 제약
        ("가격", "마진", "+"),
        ("마진", "R&D 투자", "+"),
        ("R&D 투자", "제품 품질", "+"),
        ("제품 품질", "경쟁력", "+"),

        # 경쟁사 대응
        ("시장점유율", "경쟁사 대응", "+"),
        ("경쟁사 대응", "경쟁 강도", "+"),
        ("경쟁 강도", "시장점유율", "-"),
    ]

    return create_cld_graph(edges)


def create_organizational_change_cld():
    """조직 변화 CLD

    변화 추진과 저항의 피드백 구조
    """
    edges = [
        # 강화 루프: 변화 모멘텀
        ("변화 성공", "변화 신뢰", "+"),
        ("변화 신뢰", "참여도", "+"),
        ("참여도", "변화 성공", "+"),

        # 균형 루프: 변화 저항
        ("변화 속도", "불확실성", "+"),
        ("불확실성", "저항", "+"),
        ("저항", "변화 속도", "-"),

        # 균형 루프: 피로 효과
        ("변화 속도", "업무 부담", "+"),
        ("업무 부담", "피로도", "+"),
        ("피로도", "참여도", "-"),

        # 리더십 개입
        ("저항", "리더십 개입", "+"),
        ("리더십 개입", "소통", "+"),
        ("소통", "불확실성", "-"),
    ]

    return create_cld_graph(edges)


def visualize_cld(G: nx.DiGraph, title: str, filename: str, figsize=(14, 10)):
    """CLD 시각화 및 저장"""
    plt.figure(figsize=figsize)

    # 레이아웃 설정
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

    # 노드 그리기
    nx.draw_networkx_nodes(G, pos,
                           node_color="lightblue",
                           node_size=3500,
                           alpha=0.9)

    # 노드 라벨
    nx.draw_networkx_labels(G, pos,
                            font_size=9,
                            font_weight="bold")

    # 엣지 그리기 (극성에 따라 색상 구분)
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("polarity") == "+"]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("polarity") == "-"]

    # 양의 극성: 파란색
    nx.draw_networkx_edges(G, pos,
                           edgelist=positive_edges,
                           edge_color="blue",
                           arrows=True,
                           arrowsize=20,
                           connectionstyle="arc3,rad=0.1",
                           width=2,
                           alpha=0.7)

    # 음의 극성: 빨간색
    nx.draw_networkx_edges(G, pos,
                           edgelist=negative_edges,
                           edge_color="red",
                           arrows=True,
                           arrowsize=20,
                           connectionstyle="arc3,rad=0.1",
                           width=2,
                           style="dashed",
                           alpha=0.7)

    # 극성 라벨 추가
    edge_labels = {(u, v): d.get("polarity", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=12, font_color="black")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    # 범례 추가
    plt.figtext(0.02, 0.02, "파란색(+): 같은 방향 | 빨간색(-): 반대 방향", fontsize=10)

    # 저장
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return filepath


def analyze_cld(G: nx.DiGraph, name: str):
    """CLD 구조 분석"""
    print(f"\n{'='*60}")
    print(f"CLD 분석: {name}")
    print(f"{'='*60}")

    print(f"노드 수: {G.number_of_nodes()}")
    print(f"엣지 수: {G.number_of_edges()}")

    # 극성별 엣지 수
    positive = sum(1 for _, _, d in G.edges(data=True) if d.get("polarity") == "+")
    negative = sum(1 for _, _, d in G.edges(data=True) if d.get("polarity") == "-")
    print(f"양의 극성(+): {positive}개")
    print(f"음의 극성(-): {negative}개")

    # 루프 식별
    loops = identify_loops(G)
    print(f"\n강화 루프(R) 수: {len(loops['reinforcing'])}")
    for i, loop in enumerate(loops["reinforcing"][:3], 1):  # 상위 3개만
        print(f"  R{i}: {' → '.join(loop)} → {loop[0]}")

    print(f"\n균형 루프(B) 수: {len(loops['balancing'])}")
    for i, loop in enumerate(loops["balancing"][:3], 1):  # 상위 3개만
        print(f"  B{i}: {' → '.join(loop)} → {loop[0]}")

    # 중심성 분석 (레버리지 포인트 후보)
    print("\n[레버리지 포인트 후보 (중심성 기준)]")
    degree_centrality = nx.degree_centrality(G)
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    for node, centrality in sorted_nodes[:5]:
        print(f"  {node}: {centrality:.3f}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("제6장 예제 6.1: 인과 루프 다이어그램(CLD) 시각화")
    print("=" * 60)

    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. SaaS 성장 CLD
    G_saas = create_saas_growth_cld()
    analyze_cld(G_saas, "SaaS 스타트업 성장 모델")
    path1 = visualize_cld(G_saas,
                          "SaaS 스타트업 성장 CLD\n강화 루프(R)와 균형 루프(B)의 상호작용",
                          "cld_saas_growth.png")
    print(f"\n저장됨: {path1}")

    # 2. 시장 경쟁 CLD
    G_market = create_market_competition_cld()
    analyze_cld(G_market, "시장 경쟁 동태")
    path2 = visualize_cld(G_market,
                          "시장 경쟁 동태 CLD\n가격 경쟁과 R&D 투자의 피드백",
                          "cld_market_competition.png")
    print(f"저장됨: {path2}")

    # 3. 조직 변화 CLD
    G_org = create_organizational_change_cld()
    analyze_cld(G_org, "조직 변화 동태")
    path3 = visualize_cld(G_org,
                          "조직 변화 CLD\n변화 모멘텀과 저항의 피드백",
                          "cld_organizational_change.png")
    print(f"저장됨: {path3}")

    print("\n" + "=" * 60)
    print("CLD 분석 완료")
    print("=" * 60)

    print("\n핵심 통찰:")
    print("-" * 60)
    print("1. 강화 루프는 성장이나 쇠퇴를 가속화한다")
    print("2. 균형 루프는 시스템을 목표 상태로 수렴시킨다")
    print("3. 레버리지 포인트는 많은 루프에 영향을 미치는 변수이다")
    print("4. 정책 개입은 의도치 않은 루프를 활성화할 수 있다")


if __name__ == "__main__":
    main()
