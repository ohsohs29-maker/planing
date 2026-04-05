#!/usr/bin/env python3
"""
16-2-agent-basics.py
AI 에이전트 기초: 도구 사용, ReAct 패턴, 계획-실행 루프

이 코드는 다음을 수행한다:
1. 도구 사용 에이전트 개념 시연
2. ReAct (Reasoning + Acting) 패턴 시뮬레이션
3. 계획-실행 루프 구현
4. 에이전트 아키텍처 비교
"""

from dataclasses import dataclass, field
from typing import List, Dict, Callable, Any
from enum import Enum
import json


class ToolType(Enum):
    """도구 유형"""
    SEARCH = "검색"
    CALCULATE = "계산"
    DATABASE = "데이터베이스"
    API = "API 호출"


@dataclass
class Tool:
    """에이전트가 사용할 도구"""
    name: str
    description: str
    tool_type: ToolType
    parameters: List[str]

    def execute(self, **kwargs) -> Dict:
        """도구 실행 (시뮬레이션)"""
        # 시뮬레이션된 도구 실행 결과
        results = {
            "search_market": {
                "result": "전기차 충전 시장 규모 2조원, 연평균 성장률 25%",
                "source": "산업연구원 2024"
            },
            "calculate_roi": {
                "result": "예상 ROI: 15.3%, 회수기간: 4.2년",
                "assumptions": "초기투자 50억, 연간수익 12억"
            },
            "query_competitors": {
                "result": [
                    {"name": "한국전력", "share": "35%"},
                    {"name": "SK E&S", "share": "20%"},
                    {"name": "GS칼텍스", "share": "15%"}
                ]
            },
            "get_regulations": {
                "result": "2025년 신규 건물 충전기 의무화 시행 예정",
                "impact": "시장 확대 촉진"
            }
        }
        return results.get(self.name, {"error": "Unknown tool"})


@dataclass
class AgentStep:
    """에이전트 실행 단계"""
    step_num: int
    thought: str
    action: str = ""
    action_input: Dict = field(default_factory=dict)
    observation: str = ""


class ReActAgent:
    """ReAct 패턴 에이전트 (시뮬레이션)"""

    def __init__(self, tools: List[Tool]):
        self.tools = {t.name: t for t in tools}
        self.steps: List[AgentStep] = []
        self.max_steps = 5

    def think(self, query: str) -> List[AgentStep]:
        """ReAct 패턴으로 문제 해결 (시뮬레이션)"""
        self.steps = []

        # 시뮬레이션된 ReAct 루프
        simulated_steps = [
            AgentStep(
                step_num=1,
                thought="사용자가 전기차 충전 시장 진입 전략을 요청했다. "
                        "먼저 시장 규모와 성장성을 파악해야 한다.",
                action="search_market",
                action_input={"query": "전기차 충전 인프라 시장 규모"},
                observation="전기차 충전 시장 규모 2조원, 연평균 성장률 25%"
            ),
            AgentStep(
                step_num=2,
                thought="시장이 빠르게 성장 중이다. 경쟁 현황을 파악해야 한다.",
                action="query_competitors",
                action_input={"industry": "전기차 충전"},
                observation="한국전력 35%, SK E&S 20%, GS칼텍스 15% 점유"
            ),
            AgentStep(
                step_num=3,
                thought="상위 3개사가 70%를 점유. 규제 환경을 확인해야 한다.",
                action="get_regulations",
                action_input={"sector": "전기차 충전"},
                observation="2025년 신규 건물 충전기 의무화로 시장 확대 예상"
            ),
            AgentStep(
                step_num=4,
                thought="규제가 시장 성장을 지원한다. 투자 수익성을 계산해보자.",
                action="calculate_roi",
                action_input={"investment": 50, "revenue": 12},
                observation="ROI 15.3%, 회수기간 4.2년으로 양호"
            ),
            AgentStep(
                step_num=5,
                thought="모든 정보를 종합하면, 시장 진입이 유망하다. "
                        "차별화 전략으로 틈새 시장 공략이 적합하다.",
                action="final_answer",
                action_input={},
                observation="분석 완료"
            )
        ]

        self.steps = simulated_steps
        return self.steps

    def format_trace(self) -> str:
        """실행 추적 포맷팅"""
        output = []
        for step in self.steps:
            output.append(f"\n[Step {step.step_num}]")
            output.append(f"Thought: {step.thought}")
            if step.action != "final_answer":
                output.append(f"Action: {step.action}")
                output.append(f"Action Input: {json.dumps(step.action_input, ensure_ascii=False)}")
                output.append(f"Observation: {step.observation}")
        return "\n".join(output)


@dataclass
class PlanStep:
    """계획 단계"""
    step_id: int
    task: str
    dependencies: List[int] = field(default_factory=list)
    status: str = "pending"
    result: str = ""


class PlanAndExecuteAgent:
    """계획-실행 에이전트"""

    def __init__(self):
        self.plan: List[PlanStep] = []

    def create_plan(self, goal: str) -> List[PlanStep]:
        """목표에 대한 계획 수립"""
        # 시뮬레이션된 계획
        self.plan = [
            PlanStep(1, "현재 시장 규모 및 성장률 조사", []),
            PlanStep(2, "주요 경쟁사 분석", [1]),
            PlanStep(3, "규제 환경 및 정책 동향 파악", [1]),
            PlanStep(4, "기술 트렌드 및 진입 장벽 분석", [2]),
            PlanStep(5, "투자 규모 및 수익성 분석", [2, 3]),
            PlanStep(6, "진입 전략 수립 및 권고안 작성", [4, 5])
        ]
        return self.plan

    def execute_plan(self) -> List[PlanStep]:
        """계획 실행"""
        simulated_results = {
            1: "시장 규모 2조원, CAGR 25%, 2027년 5조원 전망",
            2: "한국전력(35%), SK(20%), GS(15%) 등 대기업 중심, 스타트업 진입 활발",
            3: "2025년 의무화 시행, 보조금 지속, 전기요금 특례 유지",
            4: "급속충전 기술 발전, 표준화 진행, 설치 비용 하락 추세",
            5: "50억 투자 시 ROI 15%, 회수 4년, 손익분기 3년",
            6: "틈새시장(아파트/물류센터) 특화, 차별화된 앱 서비스, 단계적 확장"
        }

        for step in self.plan:
            # 의존성 확인
            deps_complete = all(
                self.plan[d-1].status == "completed"
                for d in step.dependencies
            )

            if deps_complete:
                step.status = "completed"
                step.result = simulated_results.get(step.step_id, "")

        return self.plan

    def format_plan(self) -> str:
        """계획 포맷팅"""
        output = ["[실행 계획]"]
        for step in self.plan:
            deps = f" (의존: {step.dependencies})" if step.dependencies else ""
            status_icon = "✅" if step.status == "completed" else "⏳"
            output.append(f"{status_icon} {step.step_id}. {step.task}{deps}")
            if step.result:
                output.append(f"   → {step.result}")
        return "\n".join(output)


def demonstrate_tool_use():
    """도구 사용 에이전트 시연"""
    print("\n" + "=" * 60)
    print("도구 사용 (Tool Use) 에이전트")
    print("=" * 60)

    tools = [
        Tool("search_market", "시장 정보 검색", ToolType.SEARCH, ["query"]),
        Tool("calculate_roi", "투자 수익률 계산", ToolType.CALCULATE, ["investment", "revenue"]),
        Tool("query_competitors", "경쟁사 정보 조회", ToolType.DATABASE, ["industry"]),
        Tool("get_regulations", "규제 정보 조회", ToolType.API, ["sector"])
    ]

    print("\n[사용 가능한 도구]")
    print("-" * 40)
    for tool in tools:
        print(f"  {tool.name}: {tool.description}")
        print(f"    유형: {tool.tool_type.value}, 매개변수: {tool.parameters}")

    print("\n[도구 실행 예시]")
    print("-" * 40)
    for tool in tools:
        result = tool.execute()
        print(f"\n  {tool.name} 실행 결과:")
        print(f"    {json.dumps(result, ensure_ascii=False, indent=4)}")

    return tools


def demonstrate_react():
    """ReAct 패턴 시연"""
    print("\n" + "=" * 60)
    print("ReAct (Reasoning + Acting) 패턴")
    print("=" * 60)

    tools = [
        Tool("search_market", "시장 검색", ToolType.SEARCH, ["query"]),
        Tool("query_competitors", "경쟁사 조회", ToolType.DATABASE, ["industry"]),
        Tool("get_regulations", "규제 조회", ToolType.API, ["sector"]),
        Tool("calculate_roi", "ROI 계산", ToolType.CALCULATE, ["investment", "revenue"])
    ]

    agent = ReActAgent(tools)

    query = "전기차 충전 인프라 시장에 신규 진입하려고 합니다. 시장 분석과 진입 전략을 제시해주세요."

    print(f"\n[사용자 질의]")
    print(f"  {query}")

    print("\n[ReAct 실행 추적]")
    print("-" * 40)
    steps = agent.think(query)
    print(agent.format_trace())

    print("\n[ReAct 패턴 특징]")
    print("-" * 40)
    print("""
  1. Thought: 현재 상황에 대한 추론
  2. Action: 수행할 도구/행동 선택
  3. Action Input: 도구에 전달할 입력
  4. Observation: 도구 실행 결과 관찰
  → 반복하여 최종 답변 도출
""")

    return steps


def demonstrate_plan_execute():
    """계획-실행 에이전트 시연"""
    print("\n" + "=" * 60)
    print("계획-실행 (Plan-and-Execute) 패턴")
    print("=" * 60)

    agent = PlanAndExecuteAgent()

    goal = "전기차 충전 인프라 시장 진입 전략 수립"

    print(f"\n[목표]")
    print(f"  {goal}")

    print("\n[1단계: 계획 수립]")
    print("-" * 40)
    plan = agent.create_plan(goal)
    for step in plan:
        deps = f" (선행: {step.dependencies})" if step.dependencies else ""
        print(f"  {step.step_id}. {step.task}{deps}")

    print("\n[2단계: 계획 실행]")
    print("-" * 40)
    executed = agent.execute_plan()
    print(agent.format_plan())

    print("\n[계획-실행 패턴 특징]")
    print("-" * 40)
    print("""
  1. 먼저 전체 계획을 수립 (의존성 고려)
  2. 계획에 따라 순차/병렬 실행
  3. 각 단계 결과를 다음 단계에 반영
  4. 필요시 계획 수정 (Re-planning)
""")

    return executed


def compare_agent_patterns():
    """에이전트 패턴 비교"""
    print("\n" + "=" * 60)
    print("에이전트 아키텍처 비교")
    print("=" * 60)

    patterns = {
        "ReAct": {
            "approach": "단계별 추론-행동 반복",
            "planning": "암시적 (각 단계에서)",
            "flexibility": "높음",
            "use_case": "탐색적 질의, 다양한 도구 조합"
        },
        "Plan-Execute": {
            "approach": "먼저 계획, 후 실행",
            "planning": "명시적 (사전 계획)",
            "flexibility": "중간",
            "use_case": "구조화된 작업, 복잡한 의존성"
        },
        "Reflexion": {
            "approach": "실행-반성-개선 반복",
            "planning": "반복적 개선",
            "flexibility": "높음",
            "use_case": "학습 필요, 품질 개선"
        },
        "Multi-Agent": {
            "approach": "역할별 에이전트 협업",
            "planning": "분산 협업",
            "flexibility": "매우 높음",
            "use_case": "복잡한 프로젝트, 다양한 전문성"
        }
    }

    print("\n[패턴별 특징 비교]")
    print("-" * 70)
    print(f"{'패턴':<15} {'접근법':<25} {'계획 방식':<15} {'유연성':<10}")
    print("-" * 70)
    for name, info in patterns.items():
        print(f"{name:<15} {info['approach']:<25} {info['planning']:<15} {info['flexibility']:<10}")

    print("\n[패턴별 적합 사용처]")
    print("-" * 40)
    for name, info in patterns.items():
        print(f"  {name}: {info['use_case']}")

    return patterns


def main():
    print("=" * 60)
    print("AI 에이전트 기초: 도구 사용과 패턴")
    print("=" * 60)

    # 1. 도구 사용 시연
    print("\n[1] 도구 사용 (Tool Use)")
    tools = demonstrate_tool_use()

    # 2. ReAct 패턴 시연
    print("\n[2] ReAct 패턴")
    react_steps = demonstrate_react()

    # 3. 계획-실행 패턴 시연
    print("\n[3] 계획-실행 패턴")
    plan_steps = demonstrate_plan_execute()

    # 4. 패턴 비교
    print("\n[4] 패턴 비교")
    patterns = compare_agent_patterns()

    # 5. 요약
    print("\n" + "=" * 60)
    print("요약: AI 에이전트 핵심 개념")
    print("=" * 60)
    print("""
┌─────────────────────────────────────────────────────────┐
│ 1. 도구 사용: 에이전트가 외부 기능을 호출하여 능력 확장  │
│ 2. ReAct: 추론(Thought)과 행동(Action)의 교차 반복      │
│ 3. Plan-Execute: 계획 수립 후 순차 실행                 │
│ 4. Reflexion: 실행 결과 반성하여 지속 개선              │
│ 5. Multi-Agent: 역할별 에이전트가 협업하여 복잡 문제 해결│
└─────────────────────────────────────────────────────────┘
""")

    # 결과 요약
    summary = {
        'tools_count': len(tools),
        'react_steps': len(react_steps),
        'plan_steps': len(plan_steps),
        'patterns_compared': len(patterns)
    }

    print(f"\n실행 결과 요약:")
    print(f"  - 시연된 도구: {summary['tools_count']}개")
    print(f"  - ReAct 단계: {summary['react_steps']}단계")
    print(f"  - 계획 단계: {summary['plan_steps']}단계")
    print(f"  - 비교 패턴: {summary['patterns_compared']}개")

    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
