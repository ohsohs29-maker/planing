"""
14-3-monte-carlo-schedule.py
몬테카를로 일정 시뮬레이션

이 코드는 프로젝트 일정의 불확실성을 정량화하고
완료 확률 분포를 분석하는 도구를 제공한다.
- 작업 기간 불확실성 모델링 (삼각분포)
- 크리티컬 패스 분석
- 완료 일정 확률 분포
- 버퍼 계산
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)


@dataclass
class Task:
    """프로젝트 작업"""
    id: str
    name: str
    optimistic: float    # 낙관적 기간 (일)
    most_likely: float   # 최빈 기간 (일)
    pessimistic: float   # 비관적 기간 (일)
    predecessors: List[str] = field(default_factory=list)

    @property
    def expected_duration(self) -> float:
        """PERT 기대 기간"""
        return (self.optimistic + 4 * self.most_likely + self.pessimistic) / 6

    @property
    def variance(self) -> float:
        """PERT 분산"""
        return ((self.pessimistic - self.optimistic) / 6) ** 2

    def sample_duration(self) -> float:
        """삼각분포에서 기간 샘플링"""
        return np.random.triangular(
            self.optimistic,
            self.most_likely,
            self.pessimistic
        )


def create_sample_project() -> List[Task]:
    """샘플 프로젝트 작업 목록 생성: AI 서비스 출시"""

    tasks = [
        # 기획 단계
        Task("T1", "요구사항 정의", 5, 7, 12, []),
        Task("T2", "시스템 설계", 6, 8, 14, ["T1"]),
        Task("T3", "UI/UX 설계", 7, 10, 15, ["T1"]),

        # 개발 단계
        Task("T4", "백엔드 개발", 20, 30, 45, ["T2"]),
        Task("T5", "프론트엔드 개발", 15, 25, 35, ["T3"]),
        Task("T6", "AI 모델 통합", 12, 20, 30, ["T4"]),
        Task("T7", "API 개발", 8, 15, 22, ["T4"]),

        # 테스트 단계
        Task("T8", "단위 테스트", 3, 5, 8, ["T5", "T6", "T7"]),
        Task("T9", "통합 테스트", 4, 5, 10, ["T8"]),
        Task("T10", "성능 테스트", 3, 5, 8, ["T9"]),

        # 출시 단계
        Task("T11", "배포 환경 구성", 3, 5, 8, ["T9"]),
        Task("T12", "운영 매뉴얼", 5, 7, 10, ["T9"]),
        Task("T13", "출시 및 모니터링", 3, 5, 7, ["T10", "T11", "T12"]),
    ]

    return tasks


def calculate_critical_path(tasks: List[Task], durations: Dict[str, float] = None) -> Tuple[List[str], float]:
    """크리티컬 패스 계산 (Forward/Backward Pass)"""

    if durations is None:
        durations = {t.id: t.expected_duration for t in tasks}

    task_dict = {t.id: t for t in tasks}

    # Forward Pass: 최조 시작/종료 시간
    early_start = {}
    early_finish = {}

    # 위상 정렬 순서로 처리
    completed = set()
    queue = [t.id for t in tasks if not t.predecessors]

    while queue:
        current = queue.pop(0)
        if current in completed:
            continue

        task = task_dict[current]

        # 선행 작업이 모두 완료되었는지 확인
        if not all(p in completed for p in task.predecessors):
            queue.append(current)
            continue

        # 최조 시작 시간 계산
        if not task.predecessors:
            early_start[current] = 0
        else:
            early_start[current] = max(early_finish[p] for p in task.predecessors)

        early_finish[current] = early_start[current] + durations[current]
        completed.add(current)

        # 후속 작업 추가
        for t in tasks:
            if current in t.predecessors and t.id not in completed:
                queue.append(t.id)

    # 프로젝트 총 기간
    project_duration = max(early_finish.values())

    # Backward Pass: 최지 시작/종료 시간
    late_finish = {}
    late_start = {}

    # 종료 작업부터 역순으로
    end_tasks = [t.id for t in tasks
                 if not any(t.id in other.predecessors for other in tasks)]

    for t_id in end_tasks:
        late_finish[t_id] = project_duration

    # 역순으로 처리
    for task in reversed(tasks):
        if task.id not in late_finish:
            # 후속 작업들의 최지 시작 중 최소값
            successors = [t for t in tasks if task.id in t.predecessors]
            if successors:
                late_finish[task.id] = min(late_start[s.id] for s in successors)
            else:
                late_finish[task.id] = project_duration

        late_start[task.id] = late_finish[task.id] - durations[task.id]

    # 크리티컬 패스 식별 (여유 시간 = 0)
    critical_path = []
    for task in tasks:
        slack = late_start[task.id] - early_start[task.id]
        if abs(slack) < 0.001:  # 부동소수점 비교
            critical_path.append(task.id)

    return critical_path, project_duration


def monte_carlo_simulation(tasks: List[Task], n_simulations: int = 10000) -> Dict:
    """몬테카를로 일정 시뮬레이션"""

    results = {
        'durations': [],
        'critical_paths': defaultdict(int),
        'task_criticality': defaultdict(int)
    }

    for _ in range(n_simulations):
        # 각 작업의 기간 샘플링
        durations = {t.id: t.sample_duration() for t in tasks}

        # 크리티컬 패스 계산
        critical_path, project_duration = calculate_critical_path(tasks, durations)

        results['durations'].append(project_duration)
        results['critical_paths'][tuple(critical_path)] += 1

        for task_id in critical_path:
            results['task_criticality'][task_id] += 1

    # 통계 계산
    durations = np.array(results['durations'])
    results['statistics'] = {
        'mean': np.mean(durations),
        'std': np.std(durations),
        'min': np.min(durations),
        'max': np.max(durations),
        'p50': np.percentile(durations, 50),
        'p75': np.percentile(durations, 75),
        'p90': np.percentile(durations, 90),
        'p95': np.percentile(durations, 95),
    }

    # 작업별 크리티컬 확률
    results['task_criticality'] = {
        k: v / n_simulations * 100
        for k, v in results['task_criticality'].items()
    }

    return results


def calculate_buffer(results: Dict, target_confidence: float = 0.9) -> Dict:
    """버퍼 계산"""

    durations = np.array(results['durations'])
    mean_duration = results['statistics']['mean']
    target_duration = np.percentile(durations, target_confidence * 100)

    buffer_info = {
        'mean_duration': mean_duration,
        'target_confidence': target_confidence,
        'target_duration': target_duration,
        'buffer_days': target_duration - mean_duration,
        'buffer_percentage': (target_duration - mean_duration) / mean_duration * 100
    }

    return buffer_info


def tasks_to_dataframe(tasks: List[Task]) -> pd.DataFrame:
    """작업을 DataFrame으로 변환"""
    rows = []
    for task in tasks:
        rows.append({
            'ID': task.id,
            'Name': task.name,
            'Optimistic': task.optimistic,
            'Most_Likely': task.most_likely,
            'Pessimistic': task.pessimistic,
            'Expected': f"{task.expected_duration:.1f}",
            'Predecessors': ', '.join(task.predecessors) if task.predecessors else '-'
        })
    return pd.DataFrame(rows)


def plot_duration_distribution(results: Dict, target_days: float = None, save_path: str = None):
    """프로젝트 기간 분포 시각화"""

    durations = np.array(results['durations'])
    stats = results['statistics']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: 히스토그램
    ax1 = axes[0]
    ax1.hist(durations, bins=50, color='steelblue', edgecolor='black', alpha=0.7, density=True)

    # 통계선 표시
    ax1.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"평균: {stats['mean']:.1f}일")
    ax1.axvline(stats['p90'], color='orange', linestyle='--', linewidth=2, label=f"90%ile: {stats['p90']:.1f}일")

    if target_days:
        ax1.axvline(target_days, color='green', linestyle='-', linewidth=2, label=f"목표: {target_days}일")

    ax1.set_xlabel('프로젝트 기간 (일)', fontsize=11)
    ax1.set_ylabel('밀도', fontsize=11)
    ax1.set_title('프로젝트 완료 기간 분포', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')

    # 오른쪽: 누적 분포
    ax2 = axes[1]
    sorted_durations = np.sort(durations)
    cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations) * 100

    ax2.plot(sorted_durations, cumulative, color='steelblue', linewidth=2)
    ax2.fill_between(sorted_durations, cumulative, alpha=0.3)

    # 백분위수 표시
    for p, label in [(50, '50%'), (75, '75%'), (90, '90%'), (95, '95%')]:
        val = np.percentile(durations, p)
        ax2.axhline(p, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(val, color='gray', linestyle=':', alpha=0.5)
        ax2.annotate(f'{label}: {val:.0f}일', (val, p), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)

    ax2.set_xlabel('프로젝트 기간 (일)', fontsize=11)
    ax2.set_ylabel('누적 확률 (%)', fontsize=11)
    ax2.set_title('완료 확률 누적 분포 (S-커브)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")

    plt.close()


def plot_task_criticality(tasks: List[Task], results: Dict, save_path: str = None):
    """작업별 크리티컬 확률 시각화"""

    task_names = [t.name for t in tasks]
    criticality = [results['task_criticality'].get(t.id, 0) for t in tasks]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#e74c3c' if c > 80 else '#f39c12' if c > 50 else '#3498db' for c in criticality]
    bars = ax.barh(task_names, criticality, color=colors, edgecolor='black')

    ax.set_xlabel('크리티컬 패스 포함 확률 (%)', fontsize=11)
    ax.set_title('작업별 크리티컬 확률', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)

    # 값 표시
    for bar, crit in zip(bars, criticality):
        if crit > 0:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{crit:.1f}%', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")

    plt.close()


def schedule_risk_analysis(tasks: List[Task], target_days: float, results: Dict) -> Dict:
    """일정 리스크 분석"""

    durations = np.array(results['durations'])

    # 목표일 내 완료 확률
    completion_prob = np.mean(durations <= target_days) * 100

    # 목표 초과 시 예상 지연
    delayed = durations[durations > target_days]
    if len(delayed) > 0:
        avg_delay = np.mean(delayed) - target_days
        max_delay = np.max(delayed) - target_days
    else:
        avg_delay = 0
        max_delay = 0

    # 상위 크리티컬 작업
    top_critical = sorted(results['task_criticality'].items(),
                         key=lambda x: x[1], reverse=True)[:5]

    analysis = {
        'target_days': target_days,
        'completion_probability': completion_prob,
        'avg_delay_if_late': avg_delay,
        'max_delay': max_delay,
        'top_critical_tasks': top_critical,
        'recommendation': []
    }

    # 권고사항 생성
    if completion_prob < 50:
        analysis['recommendation'].append("목표 달성 확률이 50% 미만입니다. 일정 재조정이 필요합니다.")
    elif completion_prob < 75:
        analysis['recommendation'].append("목표 달성 확률이 75% 미만입니다. 버퍼 확보를 권장합니다.")
    elif completion_prob < 90:
        analysis['recommendation'].append("목표 달성 확률 75-90%입니다. 리스크 모니터링을 강화하세요.")
    else:
        analysis['recommendation'].append("목표 달성 확률 90% 이상입니다. 일정이 안정적입니다.")

    if top_critical:
        critical_task = top_critical[0][0]
        analysis['recommendation'].append(f"가장 크리티컬한 작업({critical_task})에 자원을 집중하세요.")

    return analysis


def main():
    """몬테카를로 일정 시뮬레이션 메인 실행"""

    print("=" * 60)
    print("몬테카를로 일정 시뮬레이션")
    print("=" * 60)

    # 1. 프로젝트 작업 정의
    print("\n[1. 프로젝트 작업 정의]")
    print("-" * 50)

    tasks = create_sample_project()
    tasks_df = tasks_to_dataframe(tasks)
    print(tasks_df.to_string(index=False))

    # 2. 기대 기간 기준 크리티컬 패스
    print("\n\n[2. 기대 기간 기준 분석 (PERT)]")
    print("-" * 50)

    critical_path, expected_duration = calculate_critical_path(tasks)
    print(f"크리티컬 패스: {' → '.join(critical_path)}")
    print(f"기대 프로젝트 기간: {expected_duration:.1f}일")

    # 크리티컬 패스 작업 정보
    task_dict = {t.id: t for t in tasks}
    print("\n크리티컬 패스 작업:")
    for t_id in critical_path:
        task = task_dict[t_id]
        print(f"  {t_id}: {task.name} ({task.most_likely}일)")

    # 3. 몬테카를로 시뮬레이션
    print("\n\n[3. 몬테카를로 시뮬레이션 (10,000회)]")
    print("-" * 50)

    results = monte_carlo_simulation(tasks, n_simulations=10000)
    stats = results['statistics']

    print(f"평균 기간: {stats['mean']:.1f}일")
    print(f"표준편차: {stats['std']:.1f}일")
    print(f"최소: {stats['min']:.1f}일, 최대: {stats['max']:.1f}일")
    print(f"\n백분위수:")
    print(f"  50%ile (중앙값): {stats['p50']:.1f}일")
    print(f"  75%ile: {stats['p75']:.1f}일")
    print(f"  90%ile: {stats['p90']:.1f}일")
    print(f"  95%ile: {stats['p95']:.1f}일")

    # 4. 버퍼 계산
    print("\n\n[4. 버퍼 계산]")
    print("-" * 50)

    buffer_90 = calculate_buffer(results, 0.90)
    buffer_95 = calculate_buffer(results, 0.95)

    print(f"90% 신뢰수준 버퍼:")
    print(f"  목표 기간: {buffer_90['target_duration']:.1f}일")
    print(f"  버퍼: {buffer_90['buffer_days']:.1f}일 ({buffer_90['buffer_percentage']:.1f}%)")

    print(f"\n95% 신뢰수준 버퍼:")
    print(f"  목표 기간: {buffer_95['target_duration']:.1f}일")
    print(f"  버퍼: {buffer_95['buffer_days']:.1f}일 ({buffer_95['buffer_percentage']:.1f}%)")

    # 5. 작업별 크리티컬 확률
    print("\n\n[5. 작업별 크리티컬 확률 (상위 5개)]")
    print("-" * 50)

    sorted_criticality = sorted(results['task_criticality'].items(),
                               key=lambda x: x[1], reverse=True)

    for task_id, prob in sorted_criticality[:5]:
        task_name = task_dict[task_id].name
        print(f"  {task_id}: {task_name} - {prob:.1f}%")

    # 6. 일정 리스크 분석 (목표: 80일)
    print("\n\n[6. 일정 리스크 분석 (목표: 80일)]")
    print("-" * 50)

    target_days = 80
    risk_analysis = schedule_risk_analysis(tasks, target_days, results)

    print(f"목표 기간: {target_days}일")
    print(f"목표 내 완료 확률: {risk_analysis['completion_probability']:.1f}%")
    print(f"지연 시 평균 초과: {risk_analysis['avg_delay_if_late']:.1f}일")
    print(f"최대 지연 가능: {risk_analysis['max_delay']:.1f}일")

    print("\n권고사항:")
    for rec in risk_analysis['recommendation']:
        print(f"  - {rec}")

    # 7. 시각화
    print("\n\n[7. 시각화]")
    print("-" * 50)

    plot_duration_distribution(results, target_days, '14-3-duration-distribution.png')
    plot_task_criticality(tasks, results, '14-3-task-criticality.png')

    # 8. 핵심 통찰
    print("\n[핵심 통찰]")
    print("1. 몬테카를로 시뮬레이션으로 일정 불확실성을 정량화할 수 있다.")
    print("2. 90% 신뢰수준 달성에는 평균 대비 버퍼가 필요하다.")
    print("3. 크리티컬 확률이 높은 작업에 자원을 집중해야 한다.")
    print("4. 일정 목표는 달성 확률과 함께 제시해야 한다.")

    return tasks, results, risk_analysis


if __name__ == "__main__":
    tasks, results, risk_analysis = main()
