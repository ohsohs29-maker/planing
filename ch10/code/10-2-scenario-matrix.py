"""
10-2-scenario-matrix.py
2×2 시나리오 매트릭스 구축

이 코드는 두 핵심 불확실성을 축으로 하는 4개의 시나리오를
구축하고 각 시나리오의 서사를 개발한다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def define_scenarios():
    """
    4개 시나리오 정의
    """
    print("=" * 60)
    print("2×2 시나리오 매트릭스 구축")
    print("=" * 60)
    
    scenarios = {
        "녹색 가속": {
            "axis1": "강한 규제",
            "axis2": "빠른 기술 발전",
            "quadrant": (1, 1),
            "color": "#2ecc71",
            "description": """
2030년, 전 세계적으로 탄소중립 목표가 앞당겨지면서 정부는 강력한 전기차 
정책을 시행한다. 내연기관 신차 판매는 2028년부터 금지되고, 전기차 구매 시 
대폭적인 세제 혜택이 제공된다. 동시에 고체 배터리 상용화가 예상보다 빨라져 
주행거리 800km, 충전 시간 15분의 차세대 전기차가 등장한다. 전기차 가격이 
내연기관차와 동등해지면서 소비자 수용성이 급격히 높아진다.
            """,
            "implications": [
                "선점 효과 극대화 기회",
                "초기 진입이 핵심 성공 요인",
                "대규모 생산 능력 확보 필수",
                "충전 인프라 투자 가속화"
            ],
            "probability": 0.25
        },
        "규제 주도 전환": {
            "axis1": "강한 규제",
            "axis2": "느린 기술 발전",
            "quadrant": (-1, 1),
            "color": "#3498db",
            "description": """
정부는 탄소중립 목표 달성을 위해 강력한 규제를 도입하지만, 배터리 기술은 
기대만큼 발전하지 못한다. 전기차 의무 판매 비율이 높아지면서 제조사들은 
기술 한계 속에서 양을 채우기 위해 저가형 모델에 집중한다. 주행거리와 
충전 시간에 대한 소비자 불만이 지속되지만, 규제로 인해 선택권이 제한된다.
보조금 의존도가 높아 정책 변화에 취약하다.
            """,
            "implications": [
                "규제 준수 비용 증가",
                "기술 혁신보다 정책 대응 중요",
                "저가 모델 라인업 필요",
                "정부 관계 관리 핵심"
            ],
            "probability": 0.20
        },
        "시장 주도 전환": {
            "axis1": "약한 규제",
            "axis2": "빠른 기술 발전",
            "quadrant": (1, -1),
            "color": "#f39c12",
            "description": """
정부 규제는 현행 수준을 유지하지만, 민간 주도의 기술 혁신이 시장을 변화시킨다.
배터리 가격이 kWh당 50달러 이하로 떨어지고, 에너지 밀도가 획기적으로 개선된다.
전기차가 경제성과 성능 모두에서 내연기관차를 압도하면서, 규제 없이도 자연스러운
전환이 이루어진다. 스타트업과 테크 기업의 시장 진입이 활발해진다.
            """,
            "implications": [
                "기술 경쟁력이 핵심 차별화",
                "스타트업과의 경쟁 심화",
                "R&D 투자 확대 필요",
                "소비자 선택권 확대로 마케팅 중요"
            ],
            "probability": 0.30
        },
        "점진적 전환": {
            "axis1": "약한 규제",
            "axis2": "느린 기술 발전",
            "quadrant": (-1, -1),
            "color": "#e74c3c",
            "description": """
전기차로의 전환은 예상보다 느리게 진행된다. 정부 규제가 약하고 기술 발전도
더딘 상황에서, 내연기관차와 하이브리드카가 당분간 주류로 남는다. 전기차는
도심 단거리 운행이나 프리미엄 세그먼트에서만 성장한다. 완전한 전환은 2040년
이후로 지연되며, 기존 완성차 업체들의 전통적 강점이 유지된다.
            """,
            "implications": [
                "급진적 전환보다 점진적 포트폴리오 전략",
                "하이브리드 기술 투자 유지",
                "내연기관 역량 활용",
                "장기적 관점의 전기차 투자"
            ],
            "probability": 0.25
        }
    }
    
    # 시나리오 출력
    for name, scenario in scenarios.items():
        print(f"\n[시나리오: {name}]")
        print(f"  축 설정: {scenario['axis1']} + {scenario['axis2']}")
        print(f"  발생 확률: {scenario['probability']:.0%}")
        print(f"\n  서사:")
        print(scenario['description'].strip())
        print(f"\n  전략적 함의:")
        for imp in scenario['implications']:
            print(f"    - {imp}")
    
    return scenarios


def visualize_scenario_matrix(scenarios):
    """
    2×2 시나리오 매트릭스 시각화
    """
    print("\n" + "=" * 60)
    print("시나리오 매트릭스 시각화")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 축 설정
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axvline(x=0, color='black', linewidth=2)
    
    # 4분면 색상
    for name, scenario in scenarios.items():
        x, y = scenario['quadrant']
        rect = mpatches.FancyBboxPatch(
            (0 if x > 0 else -1.5, 0 if y > 0 else -1.5),
            1.5, 1.5,
            boxstyle="round,pad=0.02",
            facecolor=scenario['color'],
            alpha=0.3,
            edgecolor='none'
        )
        ax.add_patch(rect)
    
    # 시나리오 라벨
    label_positions = {
        "녹색 가속": (0.75, 0.75),
        "규제 주도 전환": (-0.75, 0.75),
        "시장 주도 전환": (0.75, -0.75),
        "점진적 전환": (-0.75, -0.75)
    }
    
    for name, scenario in scenarios.items():
        x, y = label_positions[name]
        ax.text(x, y + 0.15, name, fontsize=16, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(x, y - 0.1, f"확률: {scenario['probability']:.0%}",
                fontsize=12, ha='center', va='center')
        
        # 핵심 함의 (첫 번째만)
        ax.text(x, y - 0.3, scenario['implications'][0],
                fontsize=10, ha='center', va='center', style='italic')
    
    # 축 라벨
    ax.text(1.4, 0.05, '빠른 기술 발전 →', fontsize=12, ha='right', fontweight='bold')
    ax.text(-1.4, 0.05, '← 느린 기술 발전', fontsize=12, ha='left', fontweight='bold')
    ax.text(0.05, 1.4, '↑ 강한 규제', fontsize=12, ha='left', fontweight='bold', rotation=0)
    ax.text(0.05, -1.4, '↓ 약한 규제', fontsize=12, ha='left', fontweight='bold', rotation=0)
    
    ax.set_title('전기차 시장 진출 시나리오 매트릭스', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../data/scenario_matrix.png', dpi=150)
    plt.close()
    
    print("그래프 저장: ../data/scenario_matrix.png")


def evaluate_strategies(scenarios):
    """
    시나리오별 전략 평가 (풍향 테스트)
    """
    print("\n" + "=" * 60)
    print("풍향 테스트: 전략별 시나리오 성과")
    print("=" * 60)
    
    strategies = {
        "공격적 선점": {
            "description": "대규모 초기 투자로 시장 선점",
            "scores": {"녹색 가속": 10, "규제 주도 전환": 6, "시장 주도 전환": 8, "점진적 전환": 3}
        },
        "기술 집중": {
            "description": "R&D 투자로 기술 우위 확보",
            "scores": {"녹색 가속": 9, "규제 주도 전환": 5, "시장 주도 전환": 10, "점진적 전환": 4}
        },
        "단계적 진입": {
            "description": "점진적 역량 확보 후 확대",
            "scores": {"녹색 가속": 6, "규제 주도 전환": 7, "시장 주도 전환": 7, "점진적 전환": 8}
        },
        "하이브리드 병행": {
            "description": "전기차와 하이브리드 동시 투자",
            "scores": {"녹색 가속": 5, "규제 주도 전환": 8, "시장 주도 전환": 6, "점진적 전환": 9}
        }
    }
    
    print("\n[전략별 시나리오 성과 (1-10점)]")
    print("-" * 80)
    header = f"{'전략':<15} "
    for scenario_name in scenarios.keys():
        header += f"{scenario_name:<12} "
    header += f"{'기대값':<10} {'최소값':<10}"
    print(header)
    print("-" * 80)
    
    results = []
    for strategy_name, strategy in strategies.items():
        scores = strategy['scores']
        expected_value = sum(scores[s] * scenarios[s]['probability'] for s in scenarios.keys())
        min_value = min(scores.values())
        
        row = f"{strategy_name:<15} "
        for scenario_name in scenarios.keys():
            row += f"{scores[scenario_name]:<12} "
        row += f"{expected_value:<10.1f} {min_value:<10}"
        print(row)
        
        results.append({
            'name': strategy_name,
            'expected': expected_value,
            'min': min_value,
            'scores': scores
        })
    
    print("-" * 80)
    
    # 분석
    print("\n[전략 분석]")
    
    # 기대값 최고 전략
    best_expected = max(results, key=lambda x: x['expected'])
    print(f"  기대값 최고: {best_expected['name']} (기대값: {best_expected['expected']:.1f})")
    
    # 최소값 최고 전략 (강건한 전략)
    best_robust = max(results, key=lambda x: x['min'])
    print(f"  강건한 전략: {best_robust['name']} (최소값: {best_robust['min']})")
    
    # 권고
    print("\n[전략 권고]")
    print(f"  • 공격적 성장 추구 시: '{best_expected['name']}' 전략")
    print(f"  • 리스크 최소화 시: '{best_robust['name']}' 전략")
    print(f"  • 권장: 초기 '{best_robust['name']}'으로 시작하여")
    print(f"    시나리오 전개에 따라 조정")
    
    return results


def define_early_warning_indicators(scenarios):
    """
    조기 경보 지표 정의
    """
    print("\n" + "=" * 60)
    print("조기 경보 지표(Early Warning Indicators)")
    print("=" * 60)
    
    indicators = {
        "녹색 가속": [
            ("EU 탄소중립 목표 앞당김 발표", "정책 동향 모니터링"),
            ("고체 배터리 상용화 발표", "기술 뉴스 추적"),
            ("전기차 월 판매량 50% 초과", "판매 데이터 분석"),
            ("주요국 내연기관 판매 금지 입법", "규제 동향 분석")
        ],
        "규제 주도 전환": [
            ("EU 탄소국경세 강화", "정책 동향 모니터링"),
            ("배터리 원자재 가격 상승 지속", "원자재 시장 분석"),
            ("전기차 리콜 증가", "품질 이슈 추적"),
            ("보조금 의존 판매 비중 80% 초과", "판매 구조 분석")
        ],
        "시장 주도 전환": [
            ("배터리 가격 kWh당 80달러 하회", "원가 분석"),
            ("테슬라 외 스타트업 IPO 증가", "경쟁 동향 분석"),
            ("전기차 잔존가치 상승", "중고차 시장 분석"),
            ("테크 기업 전기차 시장 진입 가속", "경쟁 동향 분석")
        ],
        "점진적 전환": [
            ("하이브리드 판매 비중 증가", "판매 데이터 분석"),
            ("충전 인프라 확충 지연", "인프라 현황 추적"),
            ("전기차 구매 보조금 축소", "정책 동향 모니터링"),
            ("배터리 기술 특허 출원 감소", "기술 동향 분석")
        ]
    }
    
    print("\n[시나리오별 조기 경보 지표]")
    for scenario_name, indicator_list in indicators.items():
        print(f"\n{scenario_name} 시나리오:")
        for indicator, method in indicator_list:
            print(f"  • {indicator}")
            print(f"    모니터링: {method}")
    
    return indicators


if __name__ == "__main__":
    # 시나리오 정의
    scenarios = define_scenarios()
    
    # 시나리오 매트릭스 시각화
    visualize_scenario_matrix(scenarios)
    
    # 풍향 테스트
    evaluate_strategies(scenarios)
    
    # 조기 경보 지표
    define_early_warning_indicators(scenarios)
    
    print("\n" + "=" * 60)
    print("시나리오 플래닝 완료")
    print("=" * 60)
