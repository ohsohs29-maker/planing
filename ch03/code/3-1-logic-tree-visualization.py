"""
3-1-logic-tree-visualization.py
로직 트리 시각화 예제

이 스크립트는 Why Tree와 How Tree를 Python으로 구축하고 시각화하는 방법을 보여준다.
MECE 원칙에 따른 문제 분해 구조를 그래프로 표현한다.
"""

import json
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx

# 한글 폰트 설정 (macOS/Windows 호환)
def set_korean_font():
    """시스템에 맞는 한글 폰트를 설정한다."""
    import platform
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
    elif system == 'Windows':
        font_path = 'C:/Windows/Fonts/malgun.ttf'
    else:  # Linux
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    
    try:
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    except:
        print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    
    plt.rcParams['axes.unicode_minus'] = False


class LogicTree:
    """로직 트리를 구축하고 시각화하는 클래스"""
    
    def __init__(self, root_question: str, tree_type: str = "why"):
        """
        Args:
            root_question: 트리의 루트 질문/문제
            tree_type: "why" (원인 분석) 또는 "how" (해결책 도출)
        """
        self.root = root_question
        self.tree_type = tree_type
        self.graph = nx.DiGraph()
        self.graph.add_node(root_question, level=0)
        self.node_levels = {root_question: 0}
        
    def add_branch(self, parent: str, children: list[str]) -> None:
        """부모 노드에 자식 노드들을 추가한다.
        
        Args:
            parent: 부모 노드 이름
            children: 자식 노드 이름 리스트
        """
        parent_level = self.node_levels.get(parent, 0)
        for child in children:
            self.graph.add_node(child, level=parent_level + 1)
            self.graph.add_edge(parent, child)
            self.node_levels[child] = parent_level + 1
    
    def check_mece(self) -> dict:
        """MECE 원칙 준수 여부를 간단히 검사한다.
        
        Returns:
            검사 결과 딕셔너리
        """
        result = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "max_depth": max(self.node_levels.values()) if self.node_levels else 0,
            "leaf_nodes": [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0],
            "warnings": []
        }
        
        # 단일 자식 노드 경고 (분해가 불충분할 수 있음)
        for node in self.graph.nodes():
            if self.graph.out_degree(node) == 1:
                result["warnings"].append(
                    f"'{node}'은 자식이 1개뿐입니다. MECE 분해를 검토하세요."
                )
        
        return result
    
    def to_dict(self) -> dict:
        """트리를 딕셔너리 형태로 변환한다."""
        def build_subtree(node):
            children = list(self.graph.successors(node))
            if not children:
                return {"name": node}
            return {
                "name": node,
                "children": [build_subtree(c) for c in children]
            }
        return build_subtree(self.root)
    
    def to_json(self) -> str:
        """트리를 JSON 문자열로 변환한다."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def visualize(self, figsize: tuple = (14, 10), save_path: Optional[str] = None) -> None:
        """트리를 시각화한다.
        
        Args:
            figsize: 그림 크기
            save_path: 저장 경로 (None이면 화면 출력)
        """
        set_korean_font()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 계층적 레이아웃 생성
        pos = self._hierarchical_layout()
        
        # 노드 색상 설정 (레벨별)
        max_level = max(self.node_levels.values())
        colors = plt.cm.Blues([(self.node_levels[n] + 1) / (max_level + 2) 
                               for n in self.graph.nodes()])
        
        # 그래프 그리기
        nx.draw(
            self.graph, pos,
            with_labels=True,
            node_color=colors,
            node_size=3000,
            font_size=9,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            ax=ax
        )
        
        tree_type_label = "Why Tree (원인 분석)" if self.tree_type == "why" else "How Tree (해결책 도출)"
        ax.set_title(f"{tree_type_label}\n루트: {self.root}", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"트리 이미지가 저장되었습니다: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _hierarchical_layout(self) -> dict:
        """계층적 레이아웃 좌표를 계산한다."""
        pos = {}
        levels = {}
        
        # 레벨별 노드 그룹화
        for node, level in self.node_levels.items():
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        # 각 레벨에서 노드 위치 계산
        for level, nodes in levels.items():
            n_nodes = len(nodes)
            for i, node in enumerate(nodes):
                x = (i - (n_nodes - 1) / 2) * 2  # 수평 간격
                y = -level * 1.5  # 수직 간격 (위에서 아래로)
                pos[node] = (x, y)
        
        return pos


def create_why_tree_example():
    """매출 하락 원인 분석 Why Tree 예제를 생성한다."""
    tree = LogicTree("매출이 왜 하락했는가?", tree_type="why")
    
    # 1단계: 매출 = 판매량 × 단가
    tree.add_branch("매출이 왜 하락했는가?", ["판매량 감소", "평균 단가 하락"])
    
    # 2단계: 판매량 분해
    tree.add_branch("판매량 감소", ["신규 고객 감소", "기존 고객 이탈"])
    
    # 2단계: 단가 분해
    tree.add_branch("평균 단가 하락", ["할인 증가", "저가 제품 비중 증가"])
    
    # 3단계: 세부 원인
    tree.add_branch("신규 고객 감소", ["마케팅 효과 저하", "경쟁사 진입"])
    tree.add_branch("기존 고객 이탈", ["서비스 불만족", "가격 경쟁력 상실"])
    
    return tree


def create_how_tree_example():
    """시장 점유율 확대 전략 How Tree 예제를 생성한다."""
    tree = LogicTree("시장 점유율을 어떻게 확대할 것인가?", tree_type="how")
    
    # 1단계: 핵심 전략 방향
    tree.add_branch("시장 점유율을 어떻게 확대할 것인가?", 
                    ["시장 확대", "경쟁사 점유율 뺏기"])
    
    # 2단계: 시장 확대 방안
    tree.add_branch("시장 확대", ["신규 세그먼트 진출", "지역 확장"])
    
    # 2단계: 경쟁 전략
    tree.add_branch("경쟁사 점유율 뺏기", ["가격 경쟁력 강화", "제품 차별화"])
    
    # 3단계: 세부 전략
    tree.add_branch("신규 세그먼트 진출", ["B2B 시장 진출", "프리미엄 라인 출시"])
    tree.add_branch("가격 경쟁력 강화", ["원가 절감", "번들 상품 출시"])
    tree.add_branch("제품 차별화", ["기능 혁신", "브랜드 강화"])
    
    return tree


if __name__ == "__main__":
    print("=" * 60)
    print("로직 트리 시각화 예제")
    print("=" * 60)
    
    # Why Tree 생성 및 분석
    print("\n[1] Why Tree (원인 분석) 생성")
    why_tree = create_why_tree_example()
    
    mece_check = why_tree.check_mece()
    print(f"  - 총 노드 수: {mece_check['total_nodes']}")
    print(f"  - 최대 깊이: {mece_check['max_depth']}")
    print(f"  - 리프 노드: {len(mece_check['leaf_nodes'])}개")
    if mece_check['warnings']:
        print("  - 경고:")
        for w in mece_check['warnings']:
            print(f"    * {w}")
    
    print("\n  JSON 구조:")
    print(why_tree.to_json())
    
    # Why Tree 시각화
    why_tree.visualize(save_path="../data/why_tree_example.png")
    
    # How Tree 생성 및 분석
    print("\n[2] How Tree (해결책 도출) 생성")
    how_tree = create_how_tree_example()
    
    mece_check = how_tree.check_mece()
    print(f"  - 총 노드 수: {mece_check['total_nodes']}")
    print(f"  - 최대 깊이: {mece_check['max_depth']}")
    print(f"  - 리프 노드: {len(mece_check['leaf_nodes'])}개")
    
    print("\n  JSON 구조:")
    print(how_tree.to_json())
    
    # How Tree 시각화
    how_tree.visualize(save_path="../data/how_tree_example.png")
    
    print("\n" + "=" * 60)
    print("시각화 완료. 이미지가 practice/chapter03/data/ 폴더에 저장되었습니다.")
    print("=" * 60)
