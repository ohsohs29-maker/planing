"""
8-1-stakeholder-mapping.py
이해관계자 매핑: 영향력-관심도 매트릭스 시각화

탄소중립 도시 정책 추진을 위한 이해관계자 분석 예제
Mendelow(1991)의 Power-Interest Matrix 구현
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager, rc

# 한글 폰트 설정 (macOS)
try:
    rc('font', family='AppleGothic')
except:
    pass
plt.rcParams['axes.unicode_minus'] = False


def create_stakeholder_data():
    """
    탄소중립 도시 정책 이해관계자 데이터 생성
    
    Returns:
        pd.DataFrame: 이해관계자 프로파일
    """
    stakeholders = pd.DataFrame({
        'id': ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 
               'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15'],
        'name': [
            '시장', '환경부 장관', '환경국장', '재정국장', '기획조정실장',
            '환경시민단체 A', '환경시민단체 B', '대형 제조업체', '중소기업협회',
            '지역 언론사', '환경 전문가', '대학 연구팀', '지역 상공회의소',
            '시의회 의원', '일반 시민'
        ],
        'category': [
            '정부', '정부', '정부', '정부', '정부',
            '시민사회', '시민사회', '기업', '기업',
            '언론', '전문가', '전문가', '기업',
            '정치', '시민'
        ],
        'power': [95, 85, 70, 75, 65, 40, 35, 80, 55, 60, 45, 40, 50, 70, 25],
        'interest': [80, 75, 90, 40, 50, 95, 90, 60, 45, 70, 85, 80, 55, 65, 35],
        'attitude': [1, 1, 1, -1, 0, 1, 1, -1, -1, 0, 1, 1, -1, 0, 0]  # 1: 지지, 0: 중립, -1: 반대
    })
    return stakeholders


def classify_quadrant(power, interest):
    """
    권력-관심도에 따른 4분면 분류
    
    Args:
        power: 권력 수준 (0-100)
        interest: 관심도 수준 (0-100)
    
    Returns:
        str: 분류된 4분면
    """
    if power >= 50 and interest >= 50:
        return 'Key Players (핵심 관리)'
    elif power >= 50 and interest < 50:
        return 'Keep Satisfied (만족 유지)'
    elif power < 50 and interest >= 50:
        return 'Keep Informed (정보 제공)'
    else:
        return 'Minimal Effort (최소 노력)'


def get_strategy(quadrant):
    """
    4분면별 참여 전략
    
    Args:
        quadrant: 분류된 4분면
    
    Returns:
        str: 권장 전략
    """
    strategies = {
        'Key Players (핵심 관리)': '적극적 참여와 협력, 정기적 소통, 의사결정 과정 참여',
        'Keep Satisfied (만족 유지)': '정기적 업데이트 제공, 주요 의사결정 사전 협의',
        'Keep Informed (정보 제공)': '정보 공유, 의견 청취, 필요시 참여 기회 제공',
        'Minimal Effort (최소 노력)': '기본적 정보 제공, 모니터링 유지'
    }
    return strategies.get(quadrant, '')


def plot_power_interest_matrix(df):
    """
    영향력-관심도 매트릭스 시각화
    
    Args:
        df: 이해관계자 데이터프레임
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 4분면 배경 색상
    ax.axhspan(0, 50, xmin=0, xmax=0.5, alpha=0.1, color='gray')      # 최소 노력
    ax.axhspan(0, 50, xmin=0.5, xmax=1, alpha=0.1, color='yellow')    # 만족 유지
    ax.axhspan(50, 100, xmin=0, xmax=0.5, alpha=0.1, color='blue')    # 정보 제공
    ax.axhspan(50, 100, xmin=0.5, xmax=1, alpha=0.1, color='green')   # 핵심 관리
    
    # 4분면 경계선
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=50, color='black', linestyle='--', linewidth=1)
    
    # 태도에 따른 마커 색상
    attitude_colors = {1: 'green', 0: 'gray', -1: 'red'}
    attitude_labels = {1: '지지', 0: '중립', -1: '반대'}
    
    # 카테고리에 따른 마커 모양
    category_markers = {
        '정부': 's', '시민사회': '^', '기업': 'D', 
        '언론': 'o', '전문가': 'p', '정치': 'h', '시민': '*'
    }
    
    # 이해관계자 플로팅
    for _, row in df.iterrows():
        color = attitude_colors[row['attitude']]
        marker = category_markers.get(row['category'], 'o')
        ax.scatter(row['power'], row['interest'], 
                  c=color, marker=marker, s=200, 
                  edgecolors='black', linewidths=1, alpha=0.8)
        ax.annotate(row['name'], (row['power'], row['interest']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 4분면 레이블
    ax.text(25, 75, 'Keep Informed\n(정보 제공)', ha='center', va='center', 
            fontsize=12, fontweight='bold', alpha=0.5)
    ax.text(75, 75, 'Key Players\n(핵심 관리)', ha='center', va='center', 
            fontsize=12, fontweight='bold', alpha=0.5)
    ax.text(25, 25, 'Minimal Effort\n(최소 노력)', ha='center', va='center', 
            fontsize=12, fontweight='bold', alpha=0.5)
    ax.text(75, 25, 'Keep Satisfied\n(만족 유지)', ha='center', va='center', 
            fontsize=12, fontweight='bold', alpha=0.5)
    
    # 축 설정
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('권력 (Power)', fontsize=12)
    ax.set_ylabel('관심도 (Interest)', fontsize=12)
    ax.set_title('이해관계자 영향력-관심도 매트릭스\n(탄소중립 도시 정책)', fontsize=14, fontweight='bold')
    
    # 범례
    attitude_patches = [mpatches.Patch(color=c, label=attitude_labels[a]) 
                       for a, c in attitude_colors.items()]
    ax.legend(handles=attitude_patches, loc='upper left', title='태도')
    
    plt.tight_layout()
    plt.savefig('../data/stakeholder_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("시각화 저장: ../data/stakeholder_matrix.png")


def analyze_stakeholders(df):
    """
    이해관계자 분석 및 전략 도출
    
    Args:
        df: 이해관계자 데이터프레임
    
    Returns:
        pd.DataFrame: 분석 결과
    """
    # 4분면 분류
    df['quadrant'] = df.apply(lambda x: classify_quadrant(x['power'], x['interest']), axis=1)
    df['strategy'] = df['quadrant'].apply(get_strategy)
    
    return df


def print_analysis_results(df):
    """분석 결과 출력"""
    print("\n" + "="*70)
    print("이해관계자 분석 결과")
    print("="*70)
    
    # 4분면별 집계
    quadrant_summary = df.groupby('quadrant').agg({
        'name': 'count',
        'power': 'mean',
        'interest': 'mean'
    }).round(1)
    quadrant_summary.columns = ['인원수', '평균 권력', '평균 관심도']
    
    print("\n[4분면별 요약]")
    print(quadrant_summary.to_string())
    
    # 태도별 집계
    attitude_map = {1: '지지', 0: '중립', -1: '반대'}
    df['attitude_label'] = df['attitude'].map(attitude_map)
    
    print("\n[태도별 분포]")
    attitude_summary = df.groupby('attitude_label').agg({
        'name': 'count',
        'power': 'mean'
    }).round(1)
    attitude_summary.columns = ['인원수', '평균 권력']
    print(attitude_summary.to_string())
    
    # 핵심 이해관계자 (Key Players)
    key_players = df[df['quadrant'] == 'Key Players (핵심 관리)']
    print("\n[핵심 이해관계자 (Key Players)]")
    for _, row in key_players.iterrows():
        attitude_str = attitude_map[row['attitude']]
        print(f"  - {row['name']}: 권력={row['power']}, 관심도={row['interest']}, 태도={attitude_str}")
    
    # 고위험 이해관계자 (높은 권력 + 반대 태도)
    high_risk = df[(df['power'] >= 50) & (df['attitude'] == -1)]
    if len(high_risk) > 0:
        print("\n[고위험 이해관계자 (높은 권력 + 반대)]")
        for _, row in high_risk.iterrows():
            print(f"  - {row['name']}: 권력={row['power']}, 관심도={row['interest']}")
            print(f"    권장 전략: 우선적 관계 구축 및 우려사항 해소")
    
    return df


def main():
    """메인 실행 함수"""
    # 데이터 생성
    df = create_stakeholder_data()
    print("이해관계자 데이터 생성 완료")
    print(f"총 {len(df)}명의 이해관계자")
    
    # 분석 수행
    df = analyze_stakeholders(df)
    
    # 시각화
    plot_power_interest_matrix(df)
    
    # 결과 출력
    df = print_analysis_results(df)
    
    # 결과 저장
    df.to_csv('../data/stakeholder_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"\n분석 결과 저장: ../data/stakeholder_analysis.csv")
    
    return df


if __name__ == "__main__":
    result = main()
