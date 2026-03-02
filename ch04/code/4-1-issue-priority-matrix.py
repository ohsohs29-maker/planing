"""
4-1-issue-priority-matrix.py
이슈 우선순위 매트릭스 시각화

이 코드는 영향도(Impact)와 해결가능성(Feasibility)을 기준으로
이슈들의 우선순위를 2×2 매트릭스로 시각화한다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = ['AppleGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_sample_issues():
    """
    디지털 전환 전략 수립 관련 샘플 이슈 데이터 생성
    
    Returns:
        pd.DataFrame: 이슈 목록과 영향도, 해결가능성 점수
    """
    issues = pd.DataFrame({
        'issue_id': [f'I{i:02d}' for i in range(1, 13)],
        'issue': [
            '레거시 시스템을 어떻게 현대화할 것인가?',
            '디지털 인재를 어떻게 확보할 것인가?',
            '데이터 기반 의사결정 체계를 어떻게 구축할 것인가?',
            '고객 경험을 어떻게 디지털화할 것인가?',
            '공급망을 어떻게 디지털화할 것인가?',
            '직원들의 디지털 역량을 어떻게 강화할 것인가?',
            '사이버 보안 위험을 어떻게 관리할 것인가?',
            '디지털 투자 우선순위를 어떻게 결정할 것인가?',
            '조직 문화를 어떻게 변화시킬 것인가?',
            '외부 파트너십을 어떻게 활용할 것인가?',
            'AI/ML을 어떤 업무에 먼저 적용할 것인가?',
            '디지털 전환 성과를 어떻게 측정할 것인가?'
        ],
        'impact': [8, 9, 7, 9, 6, 7, 8, 7, 6, 5, 8, 6],  # 1-10 척도
        'feasibility': [4, 5, 7, 6, 4, 8, 6, 8, 3, 7, 5, 9],  # 1-10 척도
        'stakeholder_weight': [1.2, 1.0, 1.1, 1.3, 0.8, 0.9, 1.1, 1.2, 0.7, 0.8, 1.0, 1.0]
    })
    
    # 가중 점수 계산
    issues['weighted_score'] = (
        issues['impact'] * issues['feasibility'] * issues['stakeholder_weight']
    )
    
    return issues

def classify_quadrant(impact, feasibility, threshold=5.5):
    """
    영향도와 해결가능성에 따라 사분면 분류
    
    Args:
        impact: 영향도 점수
        feasibility: 해결가능성 점수
        threshold: 분류 기준 (기본값 5.5)
        
    Returns:
        str: 사분면 분류 ('Quick Win', 'Strategic', 'Fill-in', 'Thankless')
    """
    if impact >= threshold and feasibility >= threshold:
        return 'Quick Win'  # 높은 영향도 + 높은 해결가능성
    elif impact >= threshold and feasibility < threshold:
        return 'Strategic'  # 높은 영향도 + 낮은 해결가능성 (장기 과제)
    elif impact < threshold and feasibility >= threshold:
        return 'Fill-in'  # 낮은 영향도 + 높은 해결가능성 (여유 시 수행)
    else:
        return 'Thankless'  # 낮은 영향도 + 낮은 해결가능성 (후순위)

def plot_priority_matrix(issues, save_path=None):
    """
    이슈 우선순위 매트릭스 시각화
    
    Args:
        issues: 이슈 데이터프레임
        save_path: 저장 경로 (None이면 화면에 표시)
    """
    # 사분면 분류
    issues['quadrant'] = issues.apply(
        lambda x: classify_quadrant(x['impact'], x['feasibility']), axis=1
    )
    
    # 색상 매핑
    colors = {
        'Quick Win': '#2ecc71',    # 녹색
        'Strategic': '#3498db',    # 파란색
        'Fill-in': '#f1c40f',      # 노란색
        'Thankless': '#e74c3c'     # 빨간색
    }
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 사분면 배경 색상
    ax.axhspan(5.5, 10.5, xmin=0, xmax=0.5, alpha=0.1, color='#3498db')  # Strategic
    ax.axhspan(5.5, 10.5, xmin=0.5, xmax=1, alpha=0.1, color='#2ecc71')  # Quick Win
    ax.axhspan(0, 5.5, xmin=0, xmax=0.5, alpha=0.1, color='#e74c3c')     # Thankless
    ax.axhspan(0, 5.5, xmin=0.5, xmax=1, alpha=0.1, color='#f1c40f')     # Fill-in
    
    # 이슈 포인트 그리기
    for _, row in issues.iterrows():
        ax.scatter(
            row['feasibility'], 
            row['impact'],
            s=row['weighted_score'] * 10,  # 크기는 가중 점수에 비례
            c=colors[row['quadrant']],
            alpha=0.7,
            edgecolors='white',
            linewidth=2
        )
        # 이슈 ID 레이블
        ax.annotate(
            row['issue_id'],
            (row['feasibility'], row['impact']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold'
        )
    
    # 분류선
    ax.axhline(y=5.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=5.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # 사분면 레이블
    ax.text(8, 9, 'Quick Win\n(우선 실행)', fontsize=12, ha='center', fontweight='bold', color='#27ae60')
    ax.text(3, 9, 'Strategic\n(장기 과제)', fontsize=12, ha='center', fontweight='bold', color='#2980b9')
    ax.text(8, 2, 'Fill-in\n(여유 시 수행)', fontsize=12, ha='center', fontweight='bold', color='#f39c12')
    ax.text(3, 2, 'Thankless\n(후순위)', fontsize=12, ha='center', fontweight='bold', color='#c0392b')
    
    # 축 설정
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 10.5)
    ax.set_xlabel('해결가능성 (Feasibility)', fontsize=12)
    ax.set_ylabel('영향도 (Impact)', fontsize=12)
    ax.set_title('이슈 우선순위 매트릭스: 디지털 전환 전략', fontsize=14, fontweight='bold')
    
    # 범례
    legend_elements = [
        mpatches.Patch(facecolor=colors['Quick Win'], label='Quick Win', alpha=0.7),
        mpatches.Patch(facecolor=colors['Strategic'], label='Strategic', alpha=0.7),
        mpatches.Patch(facecolor=colors['Fill-in'], label='Fill-in', alpha=0.7),
        mpatches.Patch(facecolor=colors['Thankless'], label='Thankless', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='lower left')
    
    # 그리드
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")
    else:
        plt.show()
    
    return issues

def print_priority_ranking(issues):
    """
    우선순위 순위표 출력
    
    Args:
        issues: 이슈 데이터프레임
    """
    # 가중 점수 기준 정렬
    ranked = issues.sort_values('weighted_score', ascending=False)
    
    print("\n" + "="*80)
    print("이슈 우선순위 순위표")
    print("="*80)
    print(f"{'순위':<4} {'ID':<6} {'이슈':<45} {'영향도':<6} {'해결가능성':<8} {'가중점수':<8} {'분류':<12}")
    print("-"*80)
    
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        issue_short = row['issue'][:40] + '...' if len(row['issue']) > 40 else row['issue']
        print(f"{i:<4} {row['issue_id']:<6} {issue_short:<45} {row['impact']:<6} {row['feasibility']:<8} {row['weighted_score']:<8.1f} {row['quadrant']:<12}")
    
    print("="*80)
    
    # 사분면별 요약
    print("\n사분면별 이슈 수:")
    quadrant_counts = issues['quadrant'].value_counts()
    for q, count in quadrant_counts.items():
        print(f"  - {q}: {count}개")

def main():
    """메인 실행 함수"""
    # 샘플 데이터 생성
    issues = create_sample_issues()
    
    # 저장 경로 설정
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)
    
    # 매트릭스 시각화
    issues_with_quadrant = plot_priority_matrix(
        issues, 
        save_path=output_dir / 'issue_priority_matrix.png'
    )
    
    # 순위표 출력
    print_priority_ranking(issues_with_quadrant)
    
    # CSV 저장
    issues_with_quadrant.to_csv(output_dir / 'issues_ranked.csv', index=False, encoding='utf-8-sig')
    print(f"\n데이터 저장: {output_dir / 'issues_ranked.csv'}")

if __name__ == '__main__':
    main()
