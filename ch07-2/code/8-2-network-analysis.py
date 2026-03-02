"""
8-2-network-analysis.py
이해관계자 네트워크 분석: 중심성 지표 및 커뮤니티 탐지

NetworkX를 활용한 탄소중립 도시 정책 이해관계자 네트워크 분석
- 중심성 지표: Degree, Betweenness, Closeness, Eigenvector
- 커뮤니티 탐지: Louvain Algorithm
"""

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 한글 폰트 설정
try:
    from matplotlib import font_manager, rc
    rc('font', family='AppleGothic')
except:
    pass
plt.rcParams['axes.unicode_minus'] = False


def create_stakeholder_network():
    """
    이해관계자 네트워크 데이터 생성
    
    Returns:
        G: NetworkX Graph
        node_attrs: 노드 속성 딕셔너리
    """
    # 노드 정보
    nodes = {
        'S01': {'name': '시장', 'category': '정부', 'attitude': 1},
        'S02': {'name': '환경부 장관', 'category': '정부', 'attitude': 1},
        'S03': {'name': '환경국장', 'category': '정부', 'attitude': 1},
        'S04': {'name': '재정국장', 'category': '정부', 'attitude': -1},
        'S05': {'name': '기획조정실장', 'category': '정부', 'attitude': 0},
        'S06': {'name': '환경시민단체 A', 'category': '시민사회', 'attitude': 1},
        'S07': {'name': '환경시민단체 B', 'category': '시민사회', 'attitude': 1},
        'S08': {'name': '대형 제조업체', 'category': '기업', 'attitude': -1},
        'S09': {'name': '중소기업협회', 'category': '기업', 'attitude': -1},
        'S10': {'name': '지역 언론사', 'category': '언론', 'attitude': 0},
        'S11': {'name': '환경 전문가', 'category': '전문가', 'attitude': 1},
        'S12': {'name': '대학 연구팀', 'category': '전문가', 'attitude': 1},
        'S13': {'name': '지역 상공회의소', 'category': '기업', 'attitude': -1},
        'S14': {'name': '시의회 의원', 'category': '정치', 'attitude': 0},
        'S15': {'name': '일반 시민', 'category': '시민', 'attitude': 0}
    }
    
    # 엣지 정보 (가중치: 관계 강도 1-5)
    edges = [
        # 정부 내부 네트워크
        ('S01', 'S02', 4), ('S01', 'S03', 5), ('S01', 'S04', 4), ('S01', 'S05', 5),
        ('S01', 'S14', 4), ('S02', 'S03', 4), ('S03', 'S04', 3), ('S03', 'S05', 3),
        ('S04', 'S05', 4),
        
        # 시민사회 네트워크
        ('S06', 'S07', 5), ('S06', 'S11', 4), ('S06', 'S12', 3),
        ('S07', 'S11', 3), ('S07', 'S15', 2),
        
        # 기업 네트워크
        ('S08', 'S09', 4), ('S08', 'S13', 5), ('S09', 'S13', 4),
        
        # 크로스 네트워크 (핵심 연결)
        ('S03', 'S06', 3),  # 환경국장 - 시민단체
        ('S03', 'S11', 4),  # 환경국장 - 전문가
        ('S04', 'S08', 3),  # 재정국장 - 대기업
        ('S04', 'S13', 3),  # 재정국장 - 상공회의소
        ('S10', 'S01', 3),  # 언론 - 시장
        ('S10', 'S06', 2),  # 언론 - 시민단체
        ('S10', 'S08', 2),  # 언론 - 대기업
        ('S11', 'S12', 5),  # 전문가 - 대학
        ('S14', 'S04', 3),  # 시의회 - 재정국장
        ('S14', 'S06', 2),  # 시의회 - 시민단체
        ('S05', 'S14', 3),  # 기획실 - 시의회
        ('S12', 'S03', 3),  # 대학 - 환경국장
    ]
    
    # 그래프 생성
    G = nx.Graph()
    
    # 노드 추가
    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)
    
    # 엣지 추가
    for source, target, weight in edges:
        G.add_edge(source, target, weight=weight)
    
    return G, nodes


def calculate_centrality_metrics(G):
    """
    중심성 지표 계산
    
    Args:
        G: NetworkX Graph
    
    Returns:
        pd.DataFrame: 중심성 지표 데이터프레임
    """
    # 각 중심성 지표 계산
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G, weight='weight')
    closeness_cent = nx.closeness_centrality(G)
    eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
    
    # 데이터프레임 생성
    centrality_df = pd.DataFrame({
        'node_id': list(G.nodes()),
        'name': [G.nodes[n]['name'] for n in G.nodes()],
        'category': [G.nodes[n]['category'] for n in G.nodes()],
        'attitude': [G.nodes[n]['attitude'] for n in G.nodes()],
        'degree': [degree_cent[n] for n in G.nodes()],
        'betweenness': [betweenness_cent[n] for n in G.nodes()],
        'closeness': [closeness_cent[n] for n in G.nodes()],
        'eigenvector': [eigenvector_cent[n] for n in G.nodes()]
    })
    
    # 순위 계산
    for metric in ['degree', 'betweenness', 'closeness', 'eigenvector']:
        centrality_df[f'{metric}_rank'] = centrality_df[metric].rank(ascending=False).astype(int)
    
    return centrality_df


def detect_communities(G):
    """
    Louvain 알고리즘을 사용한 커뮤니티 탐지
    
    Args:
        G: NetworkX Graph
    
    Returns:
        dict: 노드별 커뮤니티 할당
    """
    communities = louvain_communities(G, seed=42, resolution=1.0)
    
    # 커뮤니티 할당 딕셔너리 생성
    community_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            community_map[node] = idx
    
    return community_map, communities


def visualize_network(G, centrality_df, community_map):
    """
    네트워크 시각화
    
    Args:
        G: NetworkX Graph
        centrality_df: 중심성 지표 데이터프레임
        community_map: 커뮤니티 할당
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 레이아웃 계산
    pos = nx.spring_layout(G, seed=42, k=2)
    
    # --- 왼쪽: 태도 기반 시각화 ---
    ax1 = axes[0]
    attitude_colors = {1: 'green', 0: 'gray', -1: 'red'}
    node_colors_attitude = [attitude_colors[G.nodes[n]['attitude']] for n in G.nodes()]
    
    # 노드 크기 (Eigenvector 중심성 기반)
    eigenvector_values = centrality_df.set_index('node_id')['eigenvector']
    node_sizes = [2000 * eigenvector_values[n] + 300 for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors_attitude,
                          node_size=node_sizes, alpha=0.8, edgecolors='black')
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.4, width=1)
    nx.draw_networkx_labels(G, pos, ax=ax1, 
                           labels={n: G.nodes[n]['name'] for n in G.nodes()},
                           font_size=8)
    
    ax1.set_title('이해관계자 네트워크 (태도별 색상)\n노드 크기: Eigenvector 중심성', fontsize=12)
    attitude_patches = [mpatches.Patch(color='green', label='지지'),
                       mpatches.Patch(color='gray', label='중립'),
                       mpatches.Patch(color='red', label='반대')]
    ax1.legend(handles=attitude_patches, loc='upper left')
    ax1.axis('off')
    
    # --- 오른쪽: 커뮤니티 기반 시각화 ---
    ax2 = axes[1]
    community_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    node_colors_community = [community_colors[community_map[n] % len(community_colors)] 
                            for n in G.nodes()]
    
    # 노드 크기 (Betweenness 중심성 기반)
    betweenness_values = centrality_df.set_index('node_id')['betweenness']
    node_sizes_between = [3000 * betweenness_values[n] + 300 for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=node_colors_community,
                          node_size=node_sizes_between, alpha=0.8, edgecolors='black')
    nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.4, width=1)
    nx.draw_networkx_labels(G, pos, ax=ax2,
                           labels={n: G.nodes[n]['name'] for n in G.nodes()},
                           font_size=8)
    
    ax2.set_title('이해관계자 네트워크 (커뮤니티별 색상)\n노드 크기: Betweenness 중심성', fontsize=12)
    n_communities = len(set(community_map.values()))
    community_patches = [mpatches.Patch(color=community_colors[i], label=f'커뮤니티 {i+1}')
                        for i in range(n_communities)]
    ax2.legend(handles=community_patches, loc='upper left')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('../data/stakeholder_network.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("네트워크 시각화 저장: ../data/stakeholder_network.png")


def print_analysis_results(G, centrality_df, community_map, communities):
    """분석 결과 출력"""
    print("\n" + "="*70)
    print("이해관계자 네트워크 분석 결과")
    print("="*70)
    
    # 네트워크 기본 통계
    print("\n[네트워크 기본 통계]")
    print(f"  노드 수: {G.number_of_nodes()}")
    print(f"  엣지 수: {G.number_of_edges()}")
    print(f"  평균 연결 차수: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"  밀도: {nx.density(G):.3f}")
    print(f"  평균 군집 계수: {nx.average_clustering(G):.3f}")
    
    # 중심성 지표 상위 5명
    print("\n[중심성 지표 상위 5명]")
    
    for metric, label in [('degree', '연결 중심성'), 
                          ('betweenness', '매개 중심성'),
                          ('closeness', '근접 중심성'),
                          ('eigenvector', '고유벡터 중심성')]:
        print(f"\n  {label}:")
        top5 = centrality_df.nlargest(5, metric)[['name', metric]]
        for _, row in top5.iterrows():
            print(f"    - {row['name']}: {row[metric]:.3f}")
    
    # 커뮤니티 분석
    print(f"\n[커뮤니티 분석 (Louvain)]")
    print(f"  탐지된 커뮤니티 수: {len(communities)}")
    
    for idx, community in enumerate(communities):
        members = [G.nodes[n]['name'] for n in community]
        categories = [G.nodes[n]['category'] for n in community]
        attitudes = [G.nodes[n]['attitude'] for n in community]
        
        print(f"\n  커뮤니티 {idx+1} ({len(community)}명):")
        print(f"    구성원: {', '.join(members)}")
        print(f"    주요 카테고리: {max(set(categories), key=categories.count)}")
        avg_attitude = sum(attitudes) / len(attitudes)
        attitude_label = '지지 우세' if avg_attitude > 0.3 else ('반대 우세' if avg_attitude < -0.3 else '혼합')
        print(f"    태도 경향: {attitude_label} (평균: {avg_attitude:.2f})")
    
    # 브로커 식별 (구조적 공백)
    print("\n[핵심 브로커 (높은 매개 중심성)]")
    brokers = centrality_df.nlargest(3, 'betweenness')
    for _, row in brokers.iterrows():
        print(f"  - {row['name']}: betweenness={row['betweenness']:.3f}")
        print(f"    역할: 정보 흐름 통제, 서로 다른 집단 연결")


def derive_strategic_implications(G, centrality_df, community_map, communities):
    """전략적 시사점 도출"""
    print("\n" + "="*70)
    print("변화 관리 전략 권고")
    print("="*70)
    
    # 1. 핵심 영향력자 활용
    print("\n[1. 핵심 영향력자 활용]")
    top_eigenvector = centrality_df.nlargest(3, 'eigenvector')
    for _, row in top_eigenvector.iterrows():
        attitude_map = {1: '지지', 0: '중립', -1: '반대'}
        print(f"  - {row['name']} (태도: {attitude_map[row['attitude']]})")
        if row['attitude'] == 1:
            print(f"    전략: 정책 옹호자(Champion)로 활용")
        elif row['attitude'] == -1:
            print(f"    전략: 우려사항 청취 및 win-win 방안 모색")
        else:
            print(f"    전략: 정보 제공 및 지지 확보 노력")
    
    # 2. 브로커를 통한 저항 완화
    print("\n[2. 브로커를 통한 네트워크 연결]")
    top_brokers = centrality_df.nlargest(2, 'betweenness')
    for _, row in top_brokers.iterrows():
        print(f"  - {row['name']}: 서로 다른 집단을 연결하는 핵심 위치")
        print(f"    전략: 이 인물을 통해 반대 집단과의 대화 채널 구축")
    
    # 3. 커뮤니티별 차별화 전략
    print("\n[3. 커뮤니티별 맞춤 전략]")
    for idx, community in enumerate(communities):
        attitudes = [G.nodes[n]['attitude'] for n in community]
        avg_attitude = sum(attitudes) / len(attitudes)
        members = [G.nodes[n]['name'] for n in community]
        
        print(f"\n  커뮤니티 {idx+1}: {', '.join(members[:3])}{'...' if len(members) > 3 else ''}")
        if avg_attitude > 0.3:
            print(f"    전략: 지지 기반 강화, 정책 확산의 선봉 역할 부여")
        elif avg_attitude < -0.3:
            print(f"    전략: 우려사항 경청, 인센티브 설계, 점진적 참여 유도")
        else:
            print(f"    전략: 정보 제공 강화, 중립에서 지지로 전환 유도")
    
    # 4. 조기 도입자 식별
    print("\n[4. 변화 확산을 위한 조기 도입자]")
    # 지지 태도 + 높은 연결성
    supporters = centrality_df[centrality_df['attitude'] == 1].nlargest(3, 'degree')
    for _, row in supporters.iterrows():
        print(f"  - {row['name']}: 지지 + 높은 연결성")
        print(f"    역할: 정책 확산의 시작점, 동료 설득 역할")


def main():
    """메인 실행 함수"""
    # 네트워크 생성
    G, nodes = create_stakeholder_network()
    print("이해관계자 네트워크 생성 완료")
    print(f"노드: {G.number_of_nodes()}, 엣지: {G.number_of_edges()}")
    
    # 중심성 계산
    centrality_df = calculate_centrality_metrics(G)
    
    # 커뮤니티 탐지
    community_map, communities = detect_communities(G)
    
    # 시각화
    visualize_network(G, centrality_df, community_map)
    
    # 결과 출력
    print_analysis_results(G, centrality_df, community_map, communities)
    
    # 전략적 시사점
    derive_strategic_implications(G, centrality_df, community_map, communities)
    
    # 결과 저장
    centrality_df['community'] = centrality_df['node_id'].map(community_map)
    centrality_df.to_csv('../data/network_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"\n분석 결과 저장: ../data/network_analysis.csv")
    
    return G, centrality_df, communities


if __name__ == "__main__":
    G, centrality_df, communities = main()
