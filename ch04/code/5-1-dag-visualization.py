"""
제5장 예제 5.1: 인과 다이어그램(DAG) 시각화

이 코드는 NetworkX와 Graphviz를 활용하여 인과 다이어그램을 시각화한다.
교란변수, 매개변수, 충돌변수의 패턴을 시각적으로 이해할 수 있다.
"""

import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# 결과 저장 경로
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_confounder_dag():
    """교란변수(Confounder) 패턴 DAG 생성

    교란변수 Z가 원인 X와 결과 Y 모두에 영향을 미치는 구조.
    Z를 통제하지 않으면 X→Y의 인과 효과가 편향된다.

    예시: 마케팅 캠페인 효과 분석
    - X: 마케팅 캠페인 노출
    - Y: 구매 여부
    - Z: 고객의 구매 성향 (교란변수)
    """
    G = nx.DiGraph()

    # 노드 추가 (변수)
    G.add_node("Z", label="고객 구매성향\n(교란변수)", color="orange")
    G.add_node("X", label="캠페인 노출\n(처치)", color="lightblue")
    G.add_node("Y", label="구매 여부\n(결과)", color="lightgreen")

    # 엣지 추가 (인과 관계)
    G.add_edge("Z", "X")  # 구매 성향이 높은 고객이 캠페인을 더 많이 본다
    G.add_edge("Z", "Y")  # 구매 성향이 높은 고객이 더 많이 구매한다
    G.add_edge("X", "Y")  # 캠페인이 구매에 영향 (우리가 추정하려는 효과)

    return G


def create_mediator_dag():
    """매개변수(Mediator) 패턴 DAG 생성

    X가 M을 통해 Y에 영향을 미치는 구조.
    직접 효과와 간접 효과를 구분할 수 있다.

    예시: 교육이 소득에 미치는 영향
    - X: 교육 수준
    - M: 기술/역량 (매개변수)
    - Y: 소득
    """
    G = nx.DiGraph()

    G.add_node("X", label="교육 수준\n(원인)", color="lightblue")
    G.add_node("M", label="기술/역량\n(매개변수)", color="lightyellow")
    G.add_node("Y", label="소득\n(결과)", color="lightgreen")

    G.add_edge("X", "M")  # 교육 → 기술
    G.add_edge("M", "Y")  # 기술 → 소득 (간접 효과)
    G.add_edge("X", "Y")  # 교육 → 소득 (직접 효과: 학력 시그널링 등)

    return G


def create_collider_dag():
    """충돌변수(Collider) 패턴 DAG 생성

    X와 Y가 모두 Z에 영향을 미치는 구조.
    Z를 통제하면 오히려 X와 Y 사이에 허위 상관이 발생한다.

    예시: 재능과 노력의 역설
    - X: 재능
    - Y: 노력
    - Z: 성공 (충돌변수)

    성공한 사람만 보면 재능과 노력이 부적 상관을 보인다.
    (재능이 뛰어나면 덜 노력해도 성공, 노력을 많이 하면 재능이 덜해도 성공)
    """
    G = nx.DiGraph()

    G.add_node("X", label="재능", color="lightblue")
    G.add_node("Y", label="노력", color="lightblue")
    G.add_node("Z", label="성공\n(충돌변수)", color="salmon")

    G.add_edge("X", "Z")  # 재능 → 성공
    G.add_edge("Y", "Z")  # 노력 → 성공
    # X와 Y 사이에는 직접 인과관계 없음

    return G


def create_complex_dag():
    """복합적인 인과구조 DAG

    실제 기획 상황의 복잡한 인과구조 예시:
    디지털 전환이 기업 성과에 미치는 영향
    """
    G = nx.DiGraph()

    # 노드 추가
    nodes = {
        "기업규모": {"color": "orange"},
        "산업특성": {"color": "orange"},
        "디지털투자": {"color": "lightblue"},
        "직원역량": {"color": "lightyellow"},
        "프로세스효율": {"color": "lightyellow"},
        "고객경험": {"color": "lightyellow"},
        "매출성장": {"color": "lightgreen"},
        "비용절감": {"color": "lightgreen"},
    }

    for node, attrs in nodes.items():
        G.add_node(node, **attrs)

    # 인과 관계 추가
    edges = [
        ("기업규모", "디지털투자"),
        ("기업규모", "직원역량"),
        ("산업특성", "디지털투자"),
        ("산업특성", "매출성장"),
        ("디지털투자", "직원역량"),
        ("디지털투자", "프로세스효율"),
        ("디지털투자", "고객경험"),
        ("직원역량", "프로세스효율"),
        ("직원역량", "고객경험"),
        ("프로세스효율", "비용절감"),
        ("고객경험", "매출성장"),
    ]
    G.add_edges_from(edges)

    return G


def visualize_dag(G, title, filename, figsize=(10, 6)):
    """DAG 시각화 및 저장"""
    plt.figure(figsize=figsize)

    # 레이아웃 설정
    if len(G.nodes()) <= 4:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G)

    # 노드 색상 추출
    node_colors = [G.nodes[n].get("color", "lightgray") for n in G.nodes()]

    # 그래프 그리기
    nx.draw(G, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=3000,
            font_size=9,
            font_weight="bold",
            arrows=True,
            arrowsize=20,
            edge_color="gray",
            width=2,
            connectionstyle="arc3,rad=0.1")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    # 저장
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return filepath


def analyze_dag_structure(G, name):
    """DAG 구조 분석"""
    print(f"\n{'='*50}")
    print(f"DAG 분석: {name}")
    print(f"{'='*50}")
    print(f"노드 수: {G.number_of_nodes()}")
    print(f"엣지 수: {G.number_of_edges()}")
    print(f"노드 목록: {list(G.nodes())}")

    # 각 노드의 부모와 자식 출력
    for node in G.nodes():
        parents = list(G.predecessors(node))
        children = list(G.successors(node))
        print(f"\n  {node}:")
        print(f"    부모(원인): {parents if parents else '없음'}")
        print(f"    자식(결과): {children if children else '없음'}")

    # 경로 분석 (시작점과 끝점이 있는 경우)
    nodes = list(G.nodes())
    if len(nodes) >= 2:
        source = nodes[0]
        target = nodes[-1]
        try:
            paths = list(nx.all_simple_paths(G, source, target))
            if paths:
                print(f"\n  {source} → {target} 경로:")
                for path in paths:
                    print(f"    {' → '.join(path)}")
        except nx.NetworkXNoPath:
            print(f"\n  {source} → {target} 경로: 없음")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("제5장 예제 5.1: 인과 다이어그램(DAG) 시각화")
    print("=" * 60)

    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 교란변수 패턴
    G_conf = create_confounder_dag()
    analyze_dag_structure(G_conf, "교란변수(Confounder) 패턴")
    path1 = visualize_dag(G_conf, "교란변수(Confounder) 패턴\n마케팅 캠페인 효과 분석",
                          "dag_confounder.png")
    print(f"\n저장됨: {path1}")

    # 2. 매개변수 패턴
    G_med = create_mediator_dag()
    analyze_dag_structure(G_med, "매개변수(Mediator) 패턴")
    path2 = visualize_dag(G_med, "매개변수(Mediator) 패턴\n교육이 소득에 미치는 영향",
                          "dag_mediator.png")
    print(f"저장됨: {path2}")

    # 3. 충돌변수 패턴
    G_coll = create_collider_dag()
    analyze_dag_structure(G_coll, "충돌변수(Collider) 패턴")
    path3 = visualize_dag(G_coll, "충돌변수(Collider) 패턴\n재능과 노력의 역설",
                          "dag_collider.png")
    print(f"저장됨: {path3}")

    # 4. 복합적인 인과구조
    G_complex = create_complex_dag()
    analyze_dag_structure(G_complex, "디지털 전환 인과구조")
    path4 = visualize_dag(G_complex, "디지털 전환이 기업 성과에 미치는 영향",
                          "dag_digital_transformation.png", figsize=(12, 8))
    print(f"저장됨: {path4}")

    print("\n" + "=" * 60)
    print("모든 DAG 시각화 완료")
    print("=" * 60)

    # 요약 테이블 출력
    print("\n패턴별 특징 요약:")
    print("-" * 60)
    print(f"{'패턴':<20} {'통제 필요':<15} {'통제 시 문제'}")
    print("-" * 60)
    print(f"{'교란변수(Confounder)':<20} {'O (필수)':<15} {'없음'}")
    print(f"{'매개변수(Mediator)':<20} {'상황에 따라':<15} {'총 효과 추정 시 X'}")
    print(f"{'충돌변수(Collider)':<20} {'X (금지)':<15} {'허위 상관 발생'}")


if __name__ == "__main__":
    main()
