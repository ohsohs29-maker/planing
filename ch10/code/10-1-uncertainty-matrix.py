"""
10-1-uncertainty-matrix.py
불확실성-영향도 매트릭스: 핵심 불확실성 식별

이 코드는 시나리오 플래닝의 첫 단계인 핵심 불확실성 식별을
위한 불확실성-영향도 매트릭스를 시각화한다.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (크로스플랫폼)
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def create_uncertainty_impact_matrix():
    """
    전기차 시장 진출 시나리오를 위한 불확실성-영향도 매트릭스
    """
    print("=" * 60)
    print("불확실성-영향도 매트릭스 분석")
    print("=" * 60)
    
    # 동인(Driving Forces) 정의
    # 각 동인: (이름, 불확실성 점수 1-10, 영향도 점수 1-10)
    drivers = [
        ("정부 규제 강도", 8, 9),
        ("배터리 기술 발전", 7, 9),
        ("충전 인프라 확대", 6, 7),
        ("소비자 인식 변화", 5, 6),
        ("원자재 가격 변동", 7, 6),
        ("경쟁사 진입 속도", 6, 8),
        ("전기 요금 정책", 5, 5),
        ("경기 침체 가능성", 8, 7),
        ("자율주행 기술 발전", 9, 6),
        ("중고차 시장 형성", 4, 4),
        ("보조금 정책 지속", 7, 8),
        ("ESG 투자 트렌드", 3, 5),
    ]
    
    print("\n[동인 분석 결과]")
    print("-" * 60)
    print(f"{'동인':<20} {'불확실성':<12} {'영향도':<12} {'분류':<15}")
    print("-" * 60)
    
    # 분류 기준
    uncertainty_threshold = 6.5
    impact_threshold = 6.5
    
    critical_uncertainties = []
    trends = []
    secondary = []
    low_priority = []
    
    for name, uncertainty, impact in drivers:
        if uncertainty >= uncertainty_threshold and impact >= impact_threshold:
            category = "핵심 불확실성"
            critical_uncertainties.append((name, uncertainty, impact))
        elif uncertainty < uncertainty_threshold and impact >= impact_threshold:
            category = "확실한 트렌드"
            trends.append((name, uncertainty, impact))
        elif uncertainty >= uncertainty_threshold and impact < impact_threshold:
            category = "2차 불확실성"
            secondary.append((name, uncertainty, impact))
        else:
            category = "낮은 우선순위"
            low_priority.append((name, uncertainty, impact))
        
        print(f"{name:<20} {uncertainty:<12} {impact:<12} {category:<15}")
    
    print("-" * 60)
    
    # 분류별 요약
    print("\n[분류별 요약]")
    print(f"  핵심 불확실성 ({len(critical_uncertainties)}개): 시나리오 축 후보")
    for name, u, i in critical_uncertainties:
        print(f"    - {name} (불확실성: {u}, 영향도: {i})")
    
    print(f"\n  확실한 트렌드 ({len(trends)}개): 모든 시나리오에 포함")
    for name, u, i in trends:
        print(f"    - {name}")
    
    print(f"\n  2차 불확실성 ({len(secondary)}개): 시나리오 변형에 활용")
    for name, u, i in secondary:
        print(f"    - {name}")
    
    print(f"\n  낮은 우선순위 ({len(low_priority)}개): 모니터링")
    for name, u, i in low_priority:
        print(f"    - {name}")
    
    # 시각화
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 4분면 배경
    ax.axhline(y=impact_threshold, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=uncertainty_threshold, color='gray', linestyle='--', alpha=0.5)
    
    # 4분면 색상
    ax.fill_between([0, uncertainty_threshold], [impact_threshold, impact_threshold], 
                    [10, 10], alpha=0.2, color='blue', label='확실한 트렌드')
    ax.fill_between([uncertainty_threshold, 10], [impact_threshold, impact_threshold], 
                    [10, 10], alpha=0.2, color='red', label='핵심 불확실성')
    ax.fill_between([uncertainty_threshold, 10], [0, 0], 
                    [impact_threshold, impact_threshold], alpha=0.2, color='yellow', label='2차 불확실성')
    ax.fill_between([0, uncertainty_threshold], [0, 0], 
                    [impact_threshold, impact_threshold], alpha=0.2, color='green', label='낮은 우선순위')
    
    # 동인 플롯
    colors = []
    for name, uncertainty, impact in drivers:
        if uncertainty >= uncertainty_threshold and impact >= impact_threshold:
            colors.append('darkred')
        elif uncertainty < uncertainty_threshold and impact >= impact_threshold:
            colors.append('darkblue')
        elif uncertainty >= uncertainty_threshold and impact < impact_threshold:
            colors.append('orange')
        else:
            colors.append('darkgreen')
    
    uncertainties = [d[1] for d in drivers]
    impacts = [d[2] for d in drivers]
    names = [d[0] for d in drivers]
    
    scatter = ax.scatter(uncertainties, impacts, c=colors, s=200, alpha=0.7, edgecolors='black')
    
    # 라벨 추가
    for i, name in enumerate(names):
        ax.annotate(name, (uncertainties[i], impacts[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('불확실성 (Uncertainty)', fontsize=12)
    ax.set_ylabel('영향도 (Impact)', fontsize=12)
    ax.set_title('불확실성-영향도 매트릭스: 전기차 시장 진출 동인 분석', fontsize=14)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 4분면 라벨
    ax.text(3.25, 8.5, '확실한 트렌드\n(모든 시나리오 포함)', ha='center', fontsize=10, style='italic')
    ax.text(8.25, 8.5, '핵심 불확실성\n(시나리오 축 후보)', ha='center', fontsize=10, style='italic', color='darkred')
    ax.text(8.25, 3.25, '2차 불확실성\n(변형 시나리오)', ha='center', fontsize=10, style='italic')
    ax.text(3.25, 3.25, '낮은 우선순위\n(모니터링)', ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('../data/uncertainty_impact_matrix.png', dpi=150)
    plt.close()
    
    print("\n그래프 저장: ../data/uncertainty_impact_matrix.png")
    
    return critical_uncertainties


def select_scenario_axes(critical_uncertainties):
    """
    핵심 불확실성 중 시나리오 축 선정
    """
    print("\n" + "=" * 60)
    print("시나리오 축 선정")
    print("=" * 60)
    
    print("\n[선정 기준]")
    print("  1. 높은 불확실성: 미래 방향을 예측하기 어려움")
    print("  2. 높은 영향도: 결과에 큰 차이를 만듦")
    print("  3. 독립성: 두 축이 서로 상관없이 변동 가능")
    print("  4. 관리 가능성: 우리의 전략에 영향을 받음")
    
    # 핵심 불확실성 중 상위 선정
    print("\n[핵심 불확실성 후보]")
    for i, (name, u, i_score) in enumerate(critical_uncertainties):
        print(f"  {i+1}. {name} (불확실성: {u}, 영향도: {i_score})")
    
    # 선정된 축
    axis1 = "정부 규제 강도"
    axis2 = "배터리 기술 발전"
    
    print(f"\n[선정된 시나리오 축]")
    print(f"  축 1: {axis1}")
    print(f"    - 강한 규제: 내연기관 판매 금지, 전기차 의무 판매 비율")
    print(f"    - 약한 규제: 현행 유지, 자율적 전환")
    print(f"  축 2: {axis2}")
    print(f"    - 빠른 발전: 비용 50% 이상 하락, 주행거리 2배 증가")
    print(f"    - 느린 발전: 점진적 개선, 현 기술 고착")
    
    print("\n[독립성 검토]")
    print("  - 정부 규제와 배터리 기술은 상호 영향이 있으나 독립적 발전 가능")
    print("  - 규제가 강해도 기술이 느릴 수 있고, 규제가 약해도 기술이 빠를 수 있음")
    print("  → 독립성 충족")
    
    return axis1, axis2


if __name__ == "__main__":
    # 불확실성-영향도 매트릭스 분석
    critical = create_uncertainty_impact_matrix()
    
    # 시나리오 축 선정
    axis1, axis2 = select_scenario_axes(critical)
    
    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)
    print(f"  선정된 시나리오 축:")
    print(f"    1. {axis1}")
    print(f"    2. {axis2}")
    print("  → 2×2 시나리오 매트릭스 구축 준비 완료")
