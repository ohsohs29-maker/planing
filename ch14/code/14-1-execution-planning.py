"""
14-1-execution-planning.py
실행 계획 도구: OKR, WBS, RACI

이 코드는 전략을 실행 계획으로 전환하는 핵심 도구를 구현한다.
- OKR (Objectives and Key Results) 구조화
- WBS (Work Breakdown Structure) 분해
- RACI 매트릭스 생성
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class KeyResult:
    """핵심 결과 (Key Result)"""
    description: str
    target: float
    current: float = 0.0
    unit: str = ""

    @property
    def progress(self) -> float:
        if self.target == 0:
            return 0.0
        return min(100, (self.current / self.target) * 100)


@dataclass
class Objective:
    """목표 (Objective)"""
    title: str
    key_results: List[KeyResult] = field(default_factory=list)

    @property
    def progress(self) -> float:
        if not self.key_results:
            return 0.0
        return np.mean([kr.progress for kr in self.key_results])


@dataclass
class OKRSystem:
    """OKR 시스템"""
    name: str
    objectives: List[Objective] = field(default_factory=list)

    def add_objective(self, objective: Objective):
        self.objectives.append(objective)

    def overall_progress(self) -> float:
        if not self.objectives:
            return 0.0
        return np.mean([obj.progress for obj in self.objectives])

    def to_dataframe(self) -> pd.DataFrame:
        """OKR을 DataFrame으로 변환"""
        rows = []
        for i, obj in enumerate(self.objectives, 1):
            for j, kr in enumerate(obj.key_results, 1):
                rows.append({
                    'Objective': f"O{i}",
                    'Objective_Title': obj.title,
                    'KR': f"KR{i}.{j}",
                    'KR_Description': kr.description,
                    'Target': f"{kr.target}{kr.unit}",
                    'Current': f"{kr.current}{kr.unit}",
                    'Progress': f"{kr.progress:.1f}%"
                })
        return pd.DataFrame(rows)


def create_sample_okr() -> OKRSystem:
    """샘플 OKR 생성: 신규 서비스 출시 프로젝트"""

    okr = OKRSystem("신규 AI 서비스 출시 프로젝트 Q1")

    # 목표 1: 제품 개발 완료
    obj1 = Objective("AI 챗봇 MVP 개발 완료")
    obj1.key_results = [
        KeyResult("핵심 기능 구현율", 100, 85, "%"),
        KeyResult("테스트 커버리지", 80, 72, "%"),
        KeyResult("버그 해결율", 95, 88, "%"),
    ]
    okr.add_objective(obj1)

    # 목표 2: 시장 검증
    obj2 = Objective("시장 검증 및 초기 사용자 확보")
    obj2.key_results = [
        KeyResult("베타 테스트 참여자", 500, 380, "명"),
        KeyResult("사용자 만족도 점수", 4.0, 4.2, "/5"),
        KeyResult("일일 활성 사용자(DAU)", 200, 145, "명"),
    ]
    okr.add_objective(obj2)

    # 목표 3: 운영 준비
    obj3 = Objective("서비스 운영 인프라 구축")
    obj3.key_results = [
        KeyResult("시스템 가용성", 99.5, 99.2, "%"),
        KeyResult("평균 응답 시간", 200, 180, "ms 이하"),
        KeyResult("운영 문서 완성도", 100, 75, "%"),
    ]
    okr.add_objective(obj3)

    return okr


@dataclass
class WBSItem:
    """WBS 작업 항목"""
    code: str
    name: str
    duration_days: int
    dependencies: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    level: int = 1


def create_sample_wbs() -> List[WBSItem]:
    """샘플 WBS 생성"""

    wbs = [
        # Level 1: 프로젝트
        WBSItem("1", "AI 서비스 출시 프로젝트", 90, level=1),

        # Level 2: 주요 단계
        WBSItem("1.1", "기획 및 설계", 20, level=2),
        WBSItem("1.2", "개발", 40, ["1.1"], level=2),
        WBSItem("1.3", "테스트", 15, ["1.2"], level=2),
        WBSItem("1.4", "출시 준비", 15, ["1.3"], level=2),

        # Level 3: 세부 작업 - 기획
        WBSItem("1.1.1", "요구사항 정의", 7, resources=["PM", "BA"], level=3),
        WBSItem("1.1.2", "시스템 설계", 8, ["1.1.1"], resources=["Architect"], level=3),
        WBSItem("1.1.3", "UI/UX 설계", 10, ["1.1.1"], resources=["Designer"], level=3),

        # Level 3: 세부 작업 - 개발
        WBSItem("1.2.1", "백엔드 개발", 30, ["1.1.2"], resources=["Backend Dev"], level=3),
        WBSItem("1.2.2", "프론트엔드 개발", 25, ["1.1.3"], resources=["Frontend Dev"], level=3),
        WBSItem("1.2.3", "AI 모델 통합", 20, ["1.2.1"], resources=["ML Engineer"], level=3),
        WBSItem("1.2.4", "API 개발", 15, ["1.2.1"], resources=["Backend Dev"], level=3),

        # Level 3: 세부 작업 - 테스트
        WBSItem("1.3.1", "단위 테스트", 5, ["1.2"], resources=["QA"], level=3),
        WBSItem("1.3.2", "통합 테스트", 5, ["1.3.1"], resources=["QA"], level=3),
        WBSItem("1.3.3", "성능 테스트", 5, ["1.3.2"], resources=["QA", "DevOps"], level=3),

        # Level 3: 세부 작업 - 출시
        WBSItem("1.4.1", "배포 환경 구성", 5, ["1.3"], resources=["DevOps"], level=3),
        WBSItem("1.4.2", "운영 매뉴얼 작성", 7, ["1.3"], resources=["Tech Writer"], level=3),
        WBSItem("1.4.3", "출시 및 모니터링", 5, ["1.4.1", "1.4.2"], resources=["PM", "DevOps"], level=3),
    ]

    return wbs


def wbs_to_dataframe(wbs: List[WBSItem]) -> pd.DataFrame:
    """WBS를 DataFrame으로 변환"""
    rows = []
    for item in wbs:
        indent = "  " * (item.level - 1)
        rows.append({
            'Code': item.code,
            'Task': f"{indent}{item.name}",
            'Duration': f"{item.duration_days}일",
            'Dependencies': ", ".join(item.dependencies) if item.dependencies else "-",
            'Resources': ", ".join(item.resources) if item.resources else "-"
        })
    return pd.DataFrame(rows)


def create_raci_matrix() -> pd.DataFrame:
    """RACI 매트릭스 생성

    R: Responsible (실행 책임)
    A: Accountable (최종 책임)
    C: Consulted (자문)
    I: Informed (통보)
    """

    tasks = [
        "요구사항 정의",
        "시스템 설계",
        "UI/UX 설계",
        "백엔드 개발",
        "프론트엔드 개발",
        "AI 모델 통합",
        "테스트 수행",
        "배포 환경 구성",
        "운영 매뉴얼 작성",
        "출시 승인"
    ]

    roles = ["PM", "Architect", "Designer", "Backend", "Frontend", "ML Eng", "QA", "DevOps"]

    # RACI 할당
    raci_data = {
        "요구사항 정의":     ["A", "C", "C", "I", "I", "I", "I", "I"],
        "시스템 설계":       ["I", "A", "C", "C", "C", "C", "I", "C"],
        "UI/UX 설계":        ["C", "C", "A", "I", "C", "I", "I", "I"],
        "백엔드 개발":       ["I", "C", "-", "A", "-", "C", "I", "I"],
        "프론트엔드 개발":   ["I", "C", "C", "-", "A", "I", "I", "I"],
        "AI 모델 통합":      ["I", "C", "-", "C", "I", "A", "I", "I"],
        "테스트 수행":       ["I", "I", "I", "C", "C", "C", "A", "C"],
        "배포 환경 구성":    ["I", "C", "-", "C", "C", "C", "I", "A"],
        "운영 매뉴얼 작성":  ["C", "C", "-", "C", "C", "C", "I", "R"],
        "출시 승인":         ["A", "C", "I", "I", "I", "I", "C", "R"],
    }

    df = pd.DataFrame(raci_data, index=roles).T
    df.index.name = "Task"

    return df


def analyze_raci(raci_df: pd.DataFrame) -> Dict:
    """RACI 매트릭스 분석"""

    analysis = {
        'tasks_per_role': {},
        'accountability_check': [],
        'workload_distribution': {}
    }

    for role in raci_df.columns:
        r_count = (raci_df[role] == 'R').sum()
        a_count = (raci_df[role] == 'A').sum()
        c_count = (raci_df[role] == 'C').sum()
        analysis['tasks_per_role'][role] = {
            'Responsible': r_count,
            'Accountable': a_count,
            'Consulted': c_count,
            'Total_Active': r_count + a_count
        }

    # 각 태스크에 Accountable이 정확히 1명인지 확인
    for task in raci_df.index:
        a_count = (raci_df.loc[task] == 'A').sum()
        if a_count != 1:
            analysis['accountability_check'].append({
                'task': task,
                'a_count': a_count,
                'issue': 'Missing A' if a_count == 0 else 'Multiple A'
            })

    return analysis


def plot_okr_progress(okr: OKRSystem, save_path: str = None):
    """OKR 진행률 시각화"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 왼쪽: 목표별 진행률
    ax1 = axes[0]
    obj_names = [f"O{i+1}: {obj.title[:15]}..." for i, obj in enumerate(okr.objectives)]
    obj_progress = [obj.progress for obj in okr.objectives]
    colors = ['#2ecc71' if p >= 70 else '#f39c12' if p >= 50 else '#e74c3c' for p in obj_progress]

    bars = ax1.barh(obj_names, obj_progress, color=colors, edgecolor='black')
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('진행률 (%)')
    ax1.set_title('목표별 OKR 진행률', fontsize=12, fontweight='bold')
    ax1.axvline(x=70, color='green', linestyle='--', alpha=0.5, label='목표(70%)')

    for bar, progress in zip(bars, obj_progress):
        ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{progress:.1f}%', va='center', fontsize=10)

    # 오른쪽: 전체 KR 진행률
    ax2 = axes[1]
    kr_labels = []
    kr_progress = []
    kr_colors = []

    for i, obj in enumerate(okr.objectives, 1):
        for j, kr in enumerate(obj.key_results, 1):
            kr_labels.append(f"KR{i}.{j}")
            kr_progress.append(kr.progress)
            color = '#2ecc71' if kr.progress >= 100 else '#3498db' if kr.progress >= 70 else '#e74c3c'
            kr_colors.append(color)

    ax2.barh(kr_labels, kr_progress, color=kr_colors, edgecolor='black')
    ax2.set_xlim(0, 120)
    ax2.set_xlabel('달성률 (%)')
    ax2.set_title('핵심 결과(KR) 달성률', fontsize=12, fontweight='bold')
    ax2.axvline(x=100, color='green', linestyle='--', alpha=0.5, label='목표(100%)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")

    plt.close()


def plot_raci_heatmap(raci_df: pd.DataFrame, save_path: str = None):
    """RACI 매트릭스 히트맵"""

    # RACI를 숫자로 변환
    raci_numeric = raci_df.replace({
        'R': 4, 'A': 3, 'C': 2, 'I': 1, '-': 0
    }).astype(int)

    fig, ax = plt.subplots(figsize=(12, 8))

    # 히트맵 생성
    cmap = plt.cm.YlOrRd
    im = ax.imshow(raci_numeric.values, cmap=cmap, aspect='auto', vmin=0, vmax=4)

    # 축 레이블
    ax.set_xticks(np.arange(len(raci_df.columns)))
    ax.set_yticks(np.arange(len(raci_df.index)))
    ax.set_xticklabels(raci_df.columns, fontsize=10)
    ax.set_yticklabels(raci_df.index, fontsize=10)

    # 셀에 텍스트 표시
    for i in range(len(raci_df.index)):
        for j in range(len(raci_df.columns)):
            text = raci_df.iloc[i, j]
            color = 'white' if raci_numeric.iloc[i, j] >= 3 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=11, fontweight='bold')

    ax.set_title('RACI 매트릭스', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('역할', fontsize=12)
    ax.set_ylabel('작업', fontsize=12)

    # 범례
    legend_text = "R: 실행책임 | A: 최종책임 | C: 자문 | I: 통보"
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=10, style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")

    plt.close()


def compare_okr_smart() -> pd.DataFrame:
    """OKR과 SMART 비교표"""

    comparison = {
        '측면': [
            '철학',
            '목표 수준',
            '주기',
            '공개 여부',
            '평가 방식',
            '적합 환경'
        ],
        'OKR': [
            '도전적 목표 추구, 70% 달성도 성공',
            '야심적 (Stretch Goals)',
            '분기별 (Quarterly)',
            '전사 공개, 정렬 중시',
            '정량적 Key Results로 측정',
            '혁신, 스타트업, 빠른 성장'
        ],
        'SMART': [
            '달성 가능한 목표 설정',
            '현실적 (Achievable)',
            '유연함 (프로젝트별)',
            '개인/팀 단위 관리',
            '단일 목표의 완료 여부',
            '안정적 환경, 전통 조직'
        ]
    }

    return pd.DataFrame(comparison)


def main():
    """실행 계획 도구 메인 실행"""

    print("=" * 60)
    print("실행 계획 도구: OKR, WBS, RACI")
    print("=" * 60)

    # 1. OKR 분석
    print("\n[1. OKR (Objectives and Key Results) 분석]")
    print("-" * 50)

    okr = create_sample_okr()
    print(f"\n프로젝트: {okr.name}")
    print(f"전체 OKR 진행률: {okr.overall_progress():.1f}%\n")

    okr_df = okr.to_dataframe()
    print(okr_df.to_string(index=False))

    # OKR 시각화
    plot_okr_progress(okr, '14-1-okr-progress.png')

    # 2. OKR vs SMART 비교
    print("\n\n[2. OKR vs SMART 비교]")
    print("-" * 50)

    comparison_df = compare_okr_smart()
    print(comparison_df.to_string(index=False))

    # 3. WBS 분석
    print("\n\n[3. WBS (Work Breakdown Structure)]")
    print("-" * 50)

    wbs = create_sample_wbs()
    wbs_df = wbs_to_dataframe(wbs)
    print(wbs_df.to_string(index=False))

    # WBS 통계
    level3_tasks = [w for w in wbs if w.level == 3]
    total_duration = sum(w.duration_days for w in level3_tasks)
    print(f"\n세부 작업(Level 3) 수: {len(level3_tasks)}개")
    print(f"총 작업일 (병렬 미고려): {total_duration}일")

    # 4. RACI 매트릭스
    print("\n\n[4. RACI 매트릭스]")
    print("-" * 50)

    raci_df = create_raci_matrix()
    print(raci_df.to_string())

    # RACI 분석
    raci_analysis = analyze_raci(raci_df)

    print("\n[역할별 책임 분포]")
    for role, counts in raci_analysis['tasks_per_role'].items():
        print(f"  {role}: R={counts['Responsible']}, A={counts['Accountable']}, "
              f"C={counts['Consulted']}, Active={counts['Total_Active']}")

    if raci_analysis['accountability_check']:
        print("\n[주의] Accountable(A) 할당 문제:")
        for issue in raci_analysis['accountability_check']:
            print(f"  - {issue['task']}: {issue['issue']}")
    else:
        print("\n[OK] 모든 작업에 Accountable(A)이 정확히 1명씩 할당됨")

    # RACI 시각화
    plot_raci_heatmap(raci_df, '14-1-raci-matrix.png')

    # 5. 핵심 통찰
    print("\n\n[핵심 통찰]")
    print("-" * 50)
    print("1. OKR은 야심적 목표(70% 달성도 성공)로 혁신을 촉진한다.")
    print("2. WBS는 복잡한 프로젝트를 관리 가능한 단위로 분해한다.")
    print("3. RACI는 역할과 책임을 명확히 하여 혼란을 방지한다.")
    print("4. 각 태스크에는 반드시 하나의 Accountable(A)이 필요하다.")

    return okr, wbs, raci_df


if __name__ == "__main__":
    okr, wbs, raci_df = main()
