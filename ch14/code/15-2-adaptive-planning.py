#!/usr/bin/env python3
"""
15-2-adaptive-planning.py
적응적 기획 시스템

이 코드는 다음을 수행한다:
1. 환경 변화 감지 시뮬레이션
2. 계획 수정 트리거 평가
3. 적응적 재계획 로직
4. Rolling Planning 구현
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)


class TriggerLevel(Enum):
    """트리거 수준"""
    GREEN = "정상"
    YELLOW = "주의"
    ORANGE = "경고"
    RED = "위기"


@dataclass
class PlanTrigger:
    """계획 수정 트리거"""
    name: str
    category: str  # 내부/외부/성과
    current_value: float
    threshold_yellow: float
    threshold_orange: float
    threshold_red: float
    direction: str = "higher"  # higher: 높을수록 나쁨, lower: 낮을수록 나쁨

    def evaluate(self) -> TriggerLevel:
        """트리거 수준 평가"""
        val = self.current_value
        if self.direction == "higher":
            if val >= self.threshold_red:
                return TriggerLevel.RED
            elif val >= self.threshold_orange:
                return TriggerLevel.ORANGE
            elif val >= self.threshold_yellow:
                return TriggerLevel.YELLOW
            else:
                return TriggerLevel.GREEN
        else:  # lower is worse
            if val <= self.threshold_red:
                return TriggerLevel.RED
            elif val <= self.threshold_orange:
                return TriggerLevel.ORANGE
            elif val <= self.threshold_yellow:
                return TriggerLevel.YELLOW
            else:
                return TriggerLevel.GREEN


@dataclass
class AdaptivePlan:
    """적응적 계획"""
    plan_id: str
    original_target: float
    current_target: float
    actual: float
    revision_count: int = 0
    revision_history: List[dict] = field(default_factory=list)

    def revise(self, new_target: float, reason: str):
        """계획 수정"""
        self.revision_history.append({
            'from': self.current_target,
            'to': new_target,
            'reason': reason,
            'revision_num': self.revision_count + 1
        })
        self.current_target = new_target
        self.revision_count += 1


def generate_environment_data(months: int = 12) -> pd.DataFrame:
    """환경 변수 데이터 생성"""
    dates = pd.date_range(start='2024-01-01', periods=months, freq='M')

    # 외부 환경 변수
    # 시장 성장률 (%) - 변동성 있음
    market_growth = 5 + np.cumsum(np.random.normal(0, 1.5, months))
    market_growth = np.clip(market_growth, -5, 15)

    # 경쟁 강도 (1-10) - 점진적 증가
    competition = 5 + np.arange(months) * 0.2 + np.random.normal(0, 0.5, months)
    competition = np.clip(competition, 3, 9)

    # 원자재 가격 지수 (100 기준)
    material_cost = 100 + np.cumsum(np.random.normal(0.5, 3, months))
    material_cost = np.clip(material_cost, 80, 140)

    # 환율 변동 (%)
    exchange_rate = np.cumsum(np.random.normal(0, 2, months))
    exchange_rate = np.clip(exchange_rate, -15, 20)

    # 내부 성과 변수
    # 매출 달성률 (%)
    sales_achievement = 100 + np.random.normal(0, 10, months)
    # 일부 월에 큰 변동
    sales_achievement[4] = 75  # 5월 급락
    sales_achievement[8] = 115  # 9월 급등
    sales_achievement = np.clip(sales_achievement, 60, 130)

    # 비용 초과율 (%)
    cost_overrun = 5 + np.random.normal(0, 3, months)
    cost_overrun[6] = 18  # 7월 비용 급증
    cost_overrun = np.clip(cost_overrun, 0, 25)

    # 프로젝트 지연 (일)
    project_delay = np.random.exponential(5, months)
    project_delay[3] = 25  # 4월 큰 지연
    project_delay = np.clip(project_delay, 0, 40)

    df = pd.DataFrame({
        '월': dates,
        '시장성장률_%': np.round(market_growth, 1),
        '경쟁강도': np.round(competition, 1),
        '원자재지수': np.round(material_cost, 0).astype(int),
        '환율변동_%': np.round(exchange_rate, 1),
        '매출달성률_%': np.round(sales_achievement, 1),
        '비용초과율_%': np.round(cost_overrun, 1),
        '프로젝트지연_일': np.round(project_delay, 0).astype(int)
    })

    return df


def create_triggers(env_data: pd.DataFrame, month_idx: int) -> List[PlanTrigger]:
    """월별 트리거 생성"""
    row = env_data.iloc[month_idx]

    triggers = [
        # 외부 환경 트리거
        PlanTrigger(
            name="시장 성장률 급변",
            category="외부",
            current_value=abs(row['시장성장률_%'] - env_data['시장성장률_%'].iloc[max(0,month_idx-1)]),
            threshold_yellow=3, threshold_orange=5, threshold_red=8,
            direction="higher"
        ),
        PlanTrigger(
            name="경쟁 강도 상승",
            category="외부",
            current_value=row['경쟁강도'],
            threshold_yellow=6, threshold_orange=7, threshold_red=8,
            direction="higher"
        ),
        PlanTrigger(
            name="원자재 가격 상승",
            category="외부",
            current_value=row['원자재지수'],
            threshold_yellow=110, threshold_orange=120, threshold_red=130,
            direction="higher"
        ),
        PlanTrigger(
            name="환율 변동",
            category="외부",
            current_value=abs(row['환율변동_%']),
            threshold_yellow=5, threshold_orange=10, threshold_red=15,
            direction="higher"
        ),
        # 성과 트리거
        PlanTrigger(
            name="매출 미달",
            category="성과",
            current_value=row['매출달성률_%'],
            threshold_yellow=90, threshold_orange=80, threshold_red=70,
            direction="lower"
        ),
        PlanTrigger(
            name="비용 초과",
            category="성과",
            current_value=row['비용초과율_%'],
            threshold_yellow=10, threshold_orange=15, threshold_red=20,
            direction="higher"
        ),
        PlanTrigger(
            name="프로젝트 지연",
            category="내부",
            current_value=row['프로젝트지연_일'],
            threshold_yellow=10, threshold_orange=20, threshold_red=30,
            direction="higher"
        )
    ]

    return triggers


def evaluate_triggers(triggers: List[PlanTrigger]) -> dict:
    """트리거 종합 평가"""
    results = {
        'triggers': [],
        'red_count': 0,
        'orange_count': 0,
        'yellow_count': 0,
        'action_required': False,
        'replan_required': False
    }

    for trigger in triggers:
        level = trigger.evaluate()
        results['triggers'].append({
            'name': trigger.name,
            'category': trigger.category,
            'value': trigger.current_value,
            'level': level.value
        })

        if level == TriggerLevel.RED:
            results['red_count'] += 1
        elif level == TriggerLevel.ORANGE:
            results['orange_count'] += 1
        elif level == TriggerLevel.YELLOW:
            results['yellow_count'] += 1

    # 재계획 필요 여부 결정
    if results['red_count'] >= 1:
        results['replan_required'] = True
        results['action_required'] = True
    elif results['orange_count'] >= 2:
        results['replan_required'] = True
        results['action_required'] = True
    elif results['orange_count'] >= 1 or results['yellow_count'] >= 3:
        results['action_required'] = True

    return results


def simulate_adaptive_planning(env_data: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
    """적응적 기획 시뮬레이션"""
    # 초기 계획
    plan = AdaptivePlan(
        plan_id="2024-SALES",
        original_target=1000,  # 연간 매출 목표 10억
        current_target=1000,
        actual=0
    )

    monthly_results = []
    trigger_events = []

    for month_idx in range(len(env_data)):
        month = env_data.iloc[month_idx]['월']

        # 트리거 평가
        triggers = create_triggers(env_data, month_idx)
        eval_result = evaluate_triggers(triggers)

        # 월간 실적 계산 (단순화)
        monthly_target = plan.current_target / 12
        achievement_rate = env_data.iloc[month_idx]['매출달성률_%'] / 100
        monthly_actual = monthly_target * achievement_rate
        plan.actual += monthly_actual

        # 재계획 필요시
        if eval_result['replan_required']:
            # 남은 월 목표 조정
            remaining_months = 12 - month_idx - 1
            if remaining_months > 0:
                current_gap = plan.current_target - plan.actual
                adjustment_factor = 0.9 if eval_result['red_count'] > 0 else 0.95
                new_target = plan.actual + current_gap * adjustment_factor
                plan.revise(new_target, f"{month.strftime('%Y-%m')} 환경 변화 대응")

            trigger_events.append({
                'month': month,
                'red': eval_result['red_count'],
                'orange': eval_result['orange_count'],
                'action': 'REPLAN',
                'new_target': plan.current_target
            })

        monthly_results.append({
            '월': month,
            '월간목표': round(monthly_target, 1),
            '월간실적': round(monthly_actual, 1),
            '누적목표': round(plan.current_target * (month_idx + 1) / 12, 1),
            '누적실적': round(plan.actual, 1),
            '달성률_%': round((plan.actual / (plan.current_target * (month_idx + 1) / 12)) * 100, 1),
            '현재목표': plan.current_target,
            '수정횟수': plan.revision_count,
            'RED트리거': eval_result['red_count'],
            'ORANGE트리거': eval_result['orange_count']
        })

    return pd.DataFrame(monthly_results), trigger_events


def visualize_adaptive_planning(results: pd.DataFrame, trigger_events: List[dict]):
    """적응적 기획 시각화"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 1. 목표 vs 실적 추이
    ax1 = axes[0]
    ax1.plot(results['월'], results['누적목표'], 'b--', label='누적 목표', linewidth=2)
    ax1.plot(results['월'], results['누적실적'], 'g-o', label='누적 실적', linewidth=2)
    ax1.fill_between(results['월'], results['누적실적'], results['누적목표'],
                     where=results['누적실적'] >= results['누적목표'],
                     alpha=0.3, color='green', label='초과')
    ax1.fill_between(results['월'], results['누적실적'], results['누적목표'],
                     where=results['누적실적'] < results['누적목표'],
                     alpha=0.3, color='red', label='미달')

    # 재계획 시점 표시
    for event in trigger_events:
        ax1.axvline(x=event['month'], color='red', linestyle=':', alpha=0.7)
        ax1.annotate('재계획', xy=(event['month'], ax1.get_ylim()[1]*0.9),
                    fontsize=8, color='red')

    ax1.set_title('누적 목표 vs 실적 (적응적 재계획 반영)')
    ax1.set_ylabel('금액 (억원)')
    ax1.legend()

    # 2. 트리거 현황
    ax2 = axes[1]
    width = 0.35
    x = np.arange(len(results))
    ax2.bar(x - width/2, results['RED트리거'], width, label='RED', color='red', alpha=0.8)
    ax2.bar(x + width/2, results['ORANGE트리거'], width, label='ORANGE', color='orange', alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([d.strftime('%m월') for d in results['월']])
    ax2.set_title('월별 트리거 발생 현황')
    ax2.set_ylabel('트리거 수')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('../data/adaptive_planning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("적응적 기획 시각화 저장: ../data/adaptive_planning.png")


def main():
    print("=" * 60)
    print("적응적 기획 시스템 시뮬레이션")
    print("=" * 60)

    # 1. 환경 데이터 생성
    print("\n[1] 환경 변수 데이터 생성")
    env_data = generate_environment_data(12)
    print(env_data.to_string(index=False))

    # 2. 트리거 정의
    print("\n[2] 계획 수정 트리거 정의")
    print("┌─────────────────────┬──────────┬────────┬────────┬────────┐")
    print("│ 트리거              │ 카테고리 │ YELLOW │ ORANGE │  RED   │")
    print("├─────────────────────┼──────────┼────────┼────────┼────────┤")
    print("│ 시장 성장률 급변    │ 외부     │ ±3%    │ ±5%    │ ±8%    │")
    print("│ 경쟁 강도 상승      │ 외부     │ 6.0    │ 7.0    │ 8.0    │")
    print("│ 원자재 가격 상승    │ 외부     │ 110    │ 120    │ 130    │")
    print("│ 환율 변동           │ 외부     │ ±5%    │ ±10%   │ ±15%   │")
    print("│ 매출 미달           │ 성과     │ <90%   │ <80%   │ <70%   │")
    print("│ 비용 초과           │ 성과     │ >10%   │ >15%   │ >20%   │")
    print("│ 프로젝트 지연       │ 내부     │ >10일  │ >20일  │ >30일  │")
    print("└─────────────────────┴──────────┴────────┴────────┴────────┘")

    # 3. 적응적 기획 시뮬레이션
    print("\n[3] 적응적 기획 시뮬레이션 실행")
    results, trigger_events = simulate_adaptive_planning(env_data)
    print(results.to_string(index=False))

    # 4. 재계획 이벤트
    print(f"\n[4] 재계획 이벤트 ({len(trigger_events)}건)")
    for event in trigger_events:
        print(f"  - {event['month'].strftime('%Y-%m')}: "
              f"RED {event['red']}건, ORANGE {event['orange']}건 "
              f"→ 목표 조정 {event['new_target']:.0f}억")

    # 5. 최종 결과
    print("\n[5] 연간 결과 요약")
    final = results.iloc[-1]
    print(f"  - 원래 목표: 1,000억원")
    print(f"  - 최종 목표: {final['현재목표']:.0f}억원 (수정 {final['수정횟수']}회)")
    print(f"  - 실제 실적: {final['누적실적']:.0f}억원")
    print(f"  - 달성률: {final['달성률_%']:.1f}%")

    # 6. 시각화
    print("\n[6] 시각화 생성")
    visualize_adaptive_planning(results, trigger_events)

    # 7. 학습 포인트
    print("\n[7] 적응적 기획 핵심 원칙")
    print("  1. 트리거 기반 의사결정: 감에 의존하지 않고 명확한 임계값 설정")
    print("  2. 신속한 대응: RED 트리거 발생 시 즉시 재계획")
    print("  3. 점진적 조정: 급격한 목표 변경보다 단계적 조정")
    print("  4. 투명한 기록: 모든 수정 이력과 사유 문서화")

    print("\n" + "=" * 60)
    print("시뮬레이션 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
