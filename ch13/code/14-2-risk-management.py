"""
14-2-risk-management.py
리스크 관리: 식별, 평가, 대응

이 코드는 프로젝트 리스크를 체계적으로 관리하는 도구를 제공한다.
- 리스크 식별 및 분류
- 발생 확률과 영향도 평가
- 리스크 매트릭스 시각화
- 대응 전략 우선순위화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class RiskCategory(Enum):
    """리스크 분류"""
    TECHNICAL = "기술적"
    SCHEDULE = "일정"
    COST = "비용"
    RESOURCE = "자원"
    EXTERNAL = "외부환경"
    ORGANIZATIONAL = "조직"


class ResponseStrategy(Enum):
    """리스크 대응 전략"""
    AVOID = "회피"      # 리스크 원인 제거
    MITIGATE = "완화"   # 발생 확률 또는 영향 감소
    TRANSFER = "전가"   # 제3자에게 이전 (보험, 외주 등)
    ACCEPT = "수용"     # 리스크 감수


@dataclass
class Risk:
    """리스크 정의"""
    id: str
    description: str
    category: RiskCategory
    probability: float  # 0-1 (발생 확률)
    impact: float       # 1-5 (영향도)
    response_strategy: ResponseStrategy = ResponseStrategy.ACCEPT
    response_plan: str = ""
    owner: str = ""
    trigger: str = ""   # 조기 경보 지표

    @property
    def risk_score(self) -> float:
        """리스크 점수 = 확률 × 영향"""
        return self.probability * self.impact

    @property
    def risk_level(self) -> str:
        """리스크 수준 판정"""
        score = self.risk_score
        if score >= 3.0:
            return "높음"
        elif score >= 1.5:
            return "중간"
        else:
            return "낮음"


def create_sample_risks() -> List[Risk]:
    """샘플 리스크 목록 생성: AI 서비스 출시 프로젝트"""

    risks = [
        # 기술적 리스크
        Risk(
            id="R001",
            description="AI 모델 정확도 미달",
            category=RiskCategory.TECHNICAL,
            probability=0.4,
            impact=4,
            response_strategy=ResponseStrategy.MITIGATE,
            response_plan="모델 앙상블, 추가 학습 데이터 확보",
            owner="ML Engineer",
            trigger="테스트 정확도 < 85%"
        ),
        Risk(
            id="R002",
            description="시스템 통합 지연",
            category=RiskCategory.TECHNICAL,
            probability=0.5,
            impact=3,
            response_strategy=ResponseStrategy.MITIGATE,
            response_plan="API 명세 조기 확정, 목업 서버 활용",
            owner="Architect",
            trigger="통합 테스트 실패율 > 30%"
        ),
        Risk(
            id="R003",
            description="보안 취약점 발견",
            category=RiskCategory.TECHNICAL,
            probability=0.3,
            impact=5,
            response_strategy=ResponseStrategy.MITIGATE,
            response_plan="보안 감사 조기 실시, 펜테스트",
            owner="Security",
            trigger="보안 스캔 Critical 발견"
        ),

        # 일정 리스크
        Risk(
            id="R004",
            description="개발 일정 지연",
            category=RiskCategory.SCHEDULE,
            probability=0.6,
            impact=3,
            response_strategy=ResponseStrategy.MITIGATE,
            response_plan="버퍼 일정 확보, 범위 우선순위화",
            owner="PM",
            trigger="마일스톤 1주 이상 지연"
        ),
        Risk(
            id="R005",
            description="외부 의존성 지연",
            category=RiskCategory.SCHEDULE,
            probability=0.4,
            impact=4,
            response_strategy=ResponseStrategy.TRANSFER,
            response_plan="SLA 계약, 대안 벤더 확보",
            owner="PM",
            trigger="외부 API 응답 지연"
        ),

        # 비용 리스크
        Risk(
            id="R006",
            description="클라우드 비용 초과",
            category=RiskCategory.COST,
            probability=0.5,
            impact=2,
            response_strategy=ResponseStrategy.MITIGATE,
            response_plan="비용 모니터링 대시보드, 예산 알림",
            owner="DevOps",
            trigger="월 예산 80% 소진"
        ),
        Risk(
            id="R007",
            description="추가 개발 인력 필요",
            category=RiskCategory.COST,
            probability=0.3,
            impact=3,
            response_strategy=ResponseStrategy.ACCEPT,
            response_plan="예비비 10% 확보",
            owner="PM",
            trigger="작업 백로그 증가"
        ),

        # 자원 리스크
        Risk(
            id="R008",
            description="핵심 개발자 이탈",
            category=RiskCategory.RESOURCE,
            probability=0.2,
            impact=5,
            response_strategy=ResponseStrategy.MITIGATE,
            response_plan="지식 공유 세션, 문서화 강화",
            owner="TL",
            trigger="팀원 불만 표출"
        ),
        Risk(
            id="R009",
            description="ML 전문가 부족",
            category=RiskCategory.RESOURCE,
            probability=0.4,
            impact=4,
            response_strategy=ResponseStrategy.TRANSFER,
            response_plan="외부 컨설팅 계약",
            owner="HR",
            trigger="채용 3개월 이상 미충원"
        ),

        # 외부 환경 리스크
        Risk(
            id="R010",
            description="규제 변경",
            category=RiskCategory.EXTERNAL,
            probability=0.2,
            impact=4,
            response_strategy=ResponseStrategy.ACCEPT,
            response_plan="법률 자문 확보, 유연한 설계",
            owner="Legal",
            trigger="AI 규제 법안 발의"
        ),
        Risk(
            id="R011",
            description="경쟁사 유사 서비스 출시",
            category=RiskCategory.EXTERNAL,
            probability=0.5,
            impact=3,
            response_strategy=ResponseStrategy.ACCEPT,
            response_plan="차별화 기능 강화, 출시 시점 조정",
            owner="Product",
            trigger="경쟁사 베타 테스트 공개"
        ),

        # 조직 리스크
        Risk(
            id="R012",
            description="경영진 지원 약화",
            category=RiskCategory.ORGANIZATIONAL,
            probability=0.2,
            impact=5,
            response_strategy=ResponseStrategy.MITIGATE,
            response_plan="정기 보고, 성과 가시화",
            owner="PM",
            trigger="예산 삭감 논의"
        ),
    ]

    return risks


def risks_to_dataframe(risks: List[Risk]) -> pd.DataFrame:
    """리스크를 DataFrame으로 변환"""
    rows = []
    for risk in risks:
        rows.append({
            'ID': risk.id,
            'Description': risk.description,
            'Category': risk.category.value,
            'Probability': f"{risk.probability:.0%}",
            'Impact': risk.impact,
            'Score': f"{risk.risk_score:.2f}",
            'Level': risk.risk_level,
            'Strategy': risk.response_strategy.value,
            'Owner': risk.owner
        })

    df = pd.DataFrame(rows)
    # 점수 기준 정렬 (높은 것 먼저)
    df['Score_Num'] = [r.risk_score for r in risks]
    df = df.sort_values('Score_Num', ascending=False).drop('Score_Num', axis=1)
    return df


def create_risk_register(risks: List[Risk]) -> pd.DataFrame:
    """리스크 레지스터 생성"""
    rows = []
    for risk in risks:
        rows.append({
            'ID': risk.id,
            'Description': risk.description,
            'Category': risk.category.value,
            'P': f"{risk.probability:.0%}",
            'I': risk.impact,
            'Score': f"{risk.risk_score:.2f}",
            'Level': risk.risk_level,
            'Strategy': risk.response_strategy.value,
            'Response_Plan': risk.response_plan,
            'Owner': risk.owner,
            'Trigger': risk.trigger
        })

    df = pd.DataFrame(rows)
    df['Score_Num'] = [r.risk_score for r in risks]
    df = df.sort_values('Score_Num', ascending=False).drop('Score_Num', axis=1)
    return df


def analyze_risks(risks: List[Risk]) -> Dict:
    """리스크 분석"""

    analysis = {
        'total_count': len(risks),
        'by_level': {},
        'by_category': {},
        'by_strategy': {},
        'high_priority': [],
        'total_expected_impact': 0
    }

    # 수준별 집계
    for level in ['높음', '중간', '낮음']:
        count = len([r for r in risks if r.risk_level == level])
        analysis['by_level'][level] = count

    # 카테고리별 집계
    for cat in RiskCategory:
        count = len([r for r in risks if r.category == cat])
        if count > 0:
            analysis['by_category'][cat.value] = count

    # 대응 전략별 집계
    for strategy in ResponseStrategy:
        count = len([r for r in risks if r.response_strategy == strategy])
        if count > 0:
            analysis['by_strategy'][strategy.value] = count

    # 높은 우선순위 리스크 (점수 3.0 이상)
    analysis['high_priority'] = [r.id for r in risks if r.risk_score >= 3.0]

    # 총 기대 영향 (확률 × 영향의 합)
    analysis['total_expected_impact'] = sum(r.risk_score for r in risks)

    return analysis


def plot_risk_matrix(risks: List[Risk], save_path: str = None):
    """리스크 매트릭스 시각화"""

    fig, ax = plt.subplots(figsize=(10, 8))

    # 배경 색상 매트릭스
    prob_levels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    impact_levels = [0, 1, 2, 3, 4, 5]

    # 색상 매핑 (확률 × 영향에 따른 위험도)
    colors_bg = [
        ['#c8e6c9', '#c8e6c9', '#fff9c4', '#fff9c4', '#ffcc80'],  # 낮은 확률
        ['#c8e6c9', '#fff9c4', '#fff9c4', '#ffcc80', '#ffcc80'],
        ['#fff9c4', '#fff9c4', '#ffcc80', '#ffcc80', '#ef9a9a'],
        ['#fff9c4', '#ffcc80', '#ffcc80', '#ef9a9a', '#ef9a9a'],
        ['#ffcc80', '#ffcc80', '#ef9a9a', '#ef9a9a', '#e57373'],  # 높은 확률
    ]

    # 배경 그리기
    for i in range(5):
        for j in range(5):
            rect = plt.Rectangle((j + 0.5, i * 0.2), 1, 0.2,
                                 facecolor=colors_bg[i][j], edgecolor='white', linewidth=2)
            ax.add_patch(rect)

    # 리스크 점 표시
    for risk in risks:
        x = risk.impact
        y = risk.probability

        # 색상 결정
        if risk.risk_score >= 3.0:
            color = '#c0392b'
        elif risk.risk_score >= 1.5:
            color = '#f39c12'
        else:
            color = '#27ae60'

        ax.scatter(x, y, s=200, c=color, edgecolor='black', linewidth=2, zorder=10)
        ax.annotate(risk.id, (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold')

    # 축 설정
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['1\n매우 낮음', '2\n낮음', '3\n보통', '4\n높음', '5\n매우 높음'])
    ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_yticklabels(['10%', '30%', '50%', '70%', '90%'])

    ax.set_xlabel('영향도 (Impact)', fontsize=12, fontweight='bold')
    ax.set_ylabel('발생 확률 (Probability)', fontsize=12, fontweight='bold')
    ax.set_title('리스크 매트릭스', fontsize=14, fontweight='bold')

    # 범례
    legend_elements = [
        plt.scatter([], [], s=100, c='#c0392b', edgecolor='black', label='높은 리스크 (≥3.0)'),
        plt.scatter([], [], s=100, c='#f39c12', edgecolor='black', label='중간 리스크 (1.5-3.0)'),
        plt.scatter([], [], s=100, c='#27ae60', edgecolor='black', label='낮은 리스크 (<1.5)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")

    plt.close()


def plot_risk_distribution(risks: List[Risk], save_path: str = None):
    """리스크 분포 시각화"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 수준별 분포
    ax1 = axes[0]
    levels = ['낮음', '중간', '높음']
    level_counts = [len([r for r in risks if r.risk_level == l]) for l in levels]
    colors = ['#27ae60', '#f39c12', '#c0392b']

    bars = ax1.bar(levels, level_counts, color=colors, edgecolor='black')
    ax1.set_ylabel('리스크 수')
    ax1.set_title('리스크 수준별 분포', fontsize=12, fontweight='bold')

    for bar, count in zip(bars, level_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', fontsize=12, fontweight='bold')

    # 2. 카테고리별 분포
    ax2 = axes[1]
    categories = {}
    for r in risks:
        cat = r.category.value
        categories[cat] = categories.get(cat, 0) + 1

    ax2.barh(list(categories.keys()), list(categories.values()),
             color='#3498db', edgecolor='black')
    ax2.set_xlabel('리스크 수')
    ax2.set_title('카테고리별 분포', fontsize=12, fontweight='bold')

    # 3. 대응 전략별 분포
    ax3 = axes[2]
    strategies = {}
    for r in risks:
        strat = r.response_strategy.value
        strategies[strat] = strategies.get(strat, 0) + 1

    colors_strat = ['#9b59b6', '#3498db', '#e67e22', '#95a5a6']
    ax3.pie(strategies.values(), labels=strategies.keys(), autopct='%1.0f%%',
            colors=colors_strat[:len(strategies)], startangle=90)
    ax3.set_title('대응 전략별 분포', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")

    plt.close()


def response_strategy_guide() -> pd.DataFrame:
    """대응 전략 가이드"""

    guide = {
        '전략': ['회피 (Avoid)', '완화 (Mitigate)', '전가 (Transfer)', '수용 (Accept)'],
        '정의': [
            '리스크 원인을 제거하거나 회피',
            '발생 확률 또는 영향을 감소',
            '리스크를 제3자에게 이전',
            '리스크를 감수하고 대응 준비'
        ],
        '적용 상황': [
            '높은 확률 × 높은 영향',
            '중간 수준 리스크',
            '보험, 외주 가능한 경우',
            '낮은 리스크 또는 불가피한 경우'
        ],
        '예시': [
            '고위험 기술 대신 검증된 기술 사용',
            '테스트 강화, 리뷰 프로세스 추가',
            '보험 가입, 전문 업체 외주',
            '예비비 확보, 대응 계획 수립'
        ]
    }

    return pd.DataFrame(guide)


def main():
    """리스크 관리 메인 실행"""

    print("=" * 60)
    print("리스크 관리: 식별, 평가, 대응")
    print("=" * 60)

    # 1. 리스크 식별
    print("\n[1. 리스크 식별 및 평가]")
    print("-" * 50)

    risks = create_sample_risks()
    risk_df = risks_to_dataframe(risks)
    print(risk_df[['ID', 'Description', 'Category', 'Probability', 'Impact', 'Score', 'Level']].to_string(index=False))

    # 2. 리스크 분석
    print("\n\n[2. 리스크 분석 결과]")
    print("-" * 50)

    analysis = analyze_risks(risks)
    print(f"총 리스크 수: {analysis['total_count']}개")
    print(f"기대 영향 합계: {analysis['total_expected_impact']:.2f}")

    print("\n수준별 분포:")
    for level, count in analysis['by_level'].items():
        print(f"  {level}: {count}개")

    print("\n카테고리별 분포:")
    for cat, count in analysis['by_category'].items():
        print(f"  {cat}: {count}개")

    print("\n대응 전략별 분포:")
    for strategy, count in analysis['by_strategy'].items():
        print(f"  {strategy}: {count}개")

    print(f"\n고우선순위 리스크 (점수 ≥ 3.0): {', '.join(analysis['high_priority'])}")

    # 3. 대응 전략 가이드
    print("\n\n[3. 대응 전략 가이드]")
    print("-" * 50)

    guide_df = response_strategy_guide()
    print(guide_df.to_string(index=False))

    # 4. 리스크 레지스터 (상위 5개)
    print("\n\n[4. 리스크 레지스터 (상위 5개)]")
    print("-" * 50)

    register_df = create_risk_register(risks)
    print(register_df[['ID', 'Description', 'Score', 'Strategy', 'Response_Plan', 'Trigger']].head(5).to_string(index=False))

    # 5. 시각화
    print("\n\n[5. 리스크 시각화]")
    print("-" * 50)

    plot_risk_matrix(risks, '14-2-risk-matrix.png')
    plot_risk_distribution(risks, '14-2-risk-distribution.png')

    # 6. 핵심 통찰
    print("\n[핵심 통찰]")
    print("1. 리스크 점수(확률 × 영향)로 우선순위를 결정한다.")
    print("2. 고위험 리스크(점수 ≥ 3.0)는 즉각적인 대응이 필요하다.")
    print("3. 조기 경보 지표(Trigger)로 리스크를 사전에 감지한다.")
    print("4. 대응 전략은 리스크 특성에 맞게 선택한다.")

    return risks, analysis


if __name__ == "__main__":
    risks, analysis = main()
