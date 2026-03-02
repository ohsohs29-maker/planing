#!/usr/bin/env python3
"""
16-3-planning-workflow.py
기획 워크플로우 자동화: 멀티 에이전트 협업 시스템

이 코드는 다음을 수행한다:
1. 기획 전용 에이전트 역할 정의
2. 멀티 에이전트 워크플로우 시뮬레이션
3. 환경분석 → 시나리오 → 리스크 → 보고서 파이프라인
4. Human-in-the-Loop 체크포인트
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime
import json


class AgentRole(Enum):
    """기획 에이전트 역할"""
    ENVIRONMENT_ANALYST = "환경분석가"
    SCENARIO_PLANNER = "시나리오기획자"
    RISK_ASSESSOR = "리스크평가자"
    REPORT_WRITER = "보고서작성자"
    COORDINATOR = "코디네이터"


class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "대기"
    IN_PROGRESS = "진행중"
    AWAITING_REVIEW = "검토대기"
    APPROVED = "승인됨"
    COMPLETED = "완료"


@dataclass
class AgentOutput:
    """에이전트 출력"""
    agent_role: AgentRole
    task_name: str
    output: Dict
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    status: TaskStatus = TaskStatus.COMPLETED


@dataclass
class PlanningAgent:
    """기획 에이전트"""
    role: AgentRole
    name: str
    description: str
    capabilities: List[str]

    def execute(self, input_data: Dict) -> AgentOutput:
        """작업 실행 (시뮬레이션)"""
        outputs = {
            AgentRole.ENVIRONMENT_ANALYST: self._analyze_environment,
            AgentRole.SCENARIO_PLANNER: self._generate_scenarios,
            AgentRole.RISK_ASSESSOR: self._assess_risks,
            AgentRole.REPORT_WRITER: self._write_report
        }

        executor = outputs.get(self.role)
        if executor:
            return executor(input_data)
        return AgentOutput(self.role, "unknown", {}, 0.0)

    def _analyze_environment(self, input_data: Dict) -> AgentOutput:
        """환경 분석"""
        output = {
            "market_analysis": {
                "size": "2.1조원 (2024)",
                "growth_rate": "25% CAGR",
                "forecast_2027": "5.2조원"
            },
            "competitive_landscape": {
                "market_leaders": ["한국전력(35%)", "SK E&S(20%)", "GS칼텍스(15%)"],
                "new_entrants": ["스타트업 급증", "자동차 OEM 진출"],
                "entry_barriers": "중간 (자본, 부지 확보)"
            },
            "external_factors": {
                "regulations": "2025년 신축건물 충전기 의무화",
                "subsidies": "설치비 50% 보조금 지속",
                "technology": "급속충전 기술 발전, 표준화 진행"
            },
            "key_insights": [
                "시장 고성장 지속 전망",
                "대기업 중심이나 틈새 기회 존재",
                "규제가 시장 성장 촉진"
            ]
        }
        return AgentOutput(
            self.role, "환경분석", output, 0.92,
            status=TaskStatus.AWAITING_REVIEW
        )

    def _generate_scenarios(self, input_data: Dict) -> AgentOutput:
        """시나리오 생성"""
        output = {
            "base_case": {
                "name": "기준 시나리오",
                "probability": 0.50,
                "description": "현재 추세 지속, 연 25% 성장",
                "market_2027": "5.2조원",
                "our_share": "3%",
                "revenue": "156억원"
            },
            "optimistic_case": {
                "name": "낙관 시나리오",
                "probability": 0.25,
                "description": "EV 보급 가속화, 규제 강화",
                "market_2027": "7.0조원",
                "our_share": "5%",
                "revenue": "350억원"
            },
            "pessimistic_case": {
                "name": "비관 시나리오",
                "probability": 0.25,
                "description": "경기 침체, 보조금 축소",
                "market_2027": "3.5조원",
                "our_share": "2%",
                "revenue": "70억원"
            },
            "expected_value": {
                "revenue": "183억원",
                "calculation": "156×0.5 + 350×0.25 + 70×0.25"
            }
        }
        return AgentOutput(
            self.role, "시나리오분석", output, 0.88,
            status=TaskStatus.AWAITING_REVIEW
        )

    def _assess_risks(self, input_data: Dict) -> AgentOutput:
        """리스크 평가"""
        output = {
            "risk_matrix": [
                {
                    "id": "R1",
                    "category": "시장",
                    "description": "대기업의 공격적 가격 경쟁",
                    "probability": 0.7,
                    "impact": 4,
                    "risk_score": 2.8,
                    "mitigation": "차별화 서비스, 틈새 시장 집중"
                },
                {
                    "id": "R2",
                    "category": "규제",
                    "description": "보조금 정책 변경",
                    "probability": 0.3,
                    "impact": 3,
                    "risk_score": 0.9,
                    "mitigation": "자체 수익모델 강화, 로비 활동"
                },
                {
                    "id": "R3",
                    "category": "기술",
                    "description": "충전 표준 변경",
                    "probability": 0.2,
                    "impact": 4,
                    "risk_score": 0.8,
                    "mitigation": "모듈형 설계, 업그레이드 대비"
                },
                {
                    "id": "R4",
                    "category": "운영",
                    "description": "부지 확보 어려움",
                    "probability": 0.5,
                    "impact": 3,
                    "risk_score": 1.5,
                    "mitigation": "제휴 모델 (아파트, 마트 등)"
                },
                {
                    "id": "R5",
                    "category": "재무",
                    "description": "초기 손실 장기화",
                    "probability": 0.4,
                    "impact": 4,
                    "risk_score": 1.6,
                    "mitigation": "단계적 투자, 손익분기 모니터링"
                }
            ],
            "total_risk_score": 7.6,
            "risk_level": "중간",
            "top_risks": ["R1: 가격 경쟁", "R5: 초기 손실"],
            "recommendation": "진입 가능하나 리스크 관리 필수"
        }
        return AgentOutput(
            self.role, "리스크평가", output, 0.90,
            status=TaskStatus.AWAITING_REVIEW
        )

    def _write_report(self, input_data: Dict) -> AgentOutput:
        """보고서 작성"""
        output = {
            "title": "전기차 충전 인프라 시장 진입 전략 보고서",
            "executive_summary": """
전기차 충전 인프라 시장은 2024년 2.1조원에서 2027년 5.2조원으로
연평균 25% 성장이 예상되는 고성장 시장입니다.

대기업 중심의 경쟁 구도이나, 아파트/물류센터 등 틈새 시장에서
신규 진입 기회가 존재합니다.

50억원 투자 시 기준 시나리오에서 ROI 15%, 4년 회수가 예상되며,
가격 경쟁과 초기 손실 리스크에 대한 관리가 필요합니다.

단계적 진입과 차별화 서비스 전략을 권고합니다.
""",
            "sections": [
                "1. 시장 현황 및 전망",
                "2. 경쟁 환경 분석",
                "3. 시나리오별 재무 전망",
                "4. 리스크 분석 및 대응 방안",
                "5. 진입 전략 권고안",
                "6. 실행 로드맵"
            ],
            "recommendations": [
                "1단계: 제휴 기반 파일럿 (0-12개월)",
                "2단계: 틈새시장 확대 (12-24개월)",
                "3단계: 본격 스케일업 (24-36개월)"
            ],
            "next_steps": [
                "경영진 의사결정 회의",
                "파일럿 파트너 협의",
                "상세 사업계획서 수립"
            ]
        }
        return AgentOutput(
            self.role, "보고서작성", output, 0.95,
            status=TaskStatus.AWAITING_REVIEW
        )


class PlanningWorkflow:
    """기획 워크플로우 오케스트레이터"""

    def __init__(self):
        self.agents: Dict[AgentRole, PlanningAgent] = {}
        self.outputs: List[AgentOutput] = []
        self.checkpoints: List[Dict] = []
        self._initialize_agents()

    def _initialize_agents(self):
        """에이전트 초기화"""
        agents_config = [
            (AgentRole.ENVIRONMENT_ANALYST, "EnvBot",
             "시장, 경쟁, 규제 환경 분석",
             ["시장조사", "경쟁분석", "PEST분석"]),
            (AgentRole.SCENARIO_PLANNER, "ScenarioBot",
             "시나리오 기획 및 재무 모델링",
             ["시나리오설계", "확률추정", "재무모델링"]),
            (AgentRole.RISK_ASSESSOR, "RiskBot",
             "리스크 식별 및 평가",
             ["리스크식별", "영향분석", "대응전략"]),
            (AgentRole.REPORT_WRITER, "ReportBot",
             "종합 보고서 작성",
             ["문서작성", "요약", "시각화"])
        ]

        for role, name, desc, caps in agents_config:
            self.agents[role] = PlanningAgent(role, name, desc, caps)

    def run_workflow(self, project_brief: Dict) -> List[AgentOutput]:
        """워크플로우 실행"""
        print("\n[워크플로우 시작]")
        print(f"프로젝트: {project_brief.get('title', 'Unknown')}")
        print("-" * 50)

        # 1. 환경 분석
        print("\n▶ 1단계: 환경 분석")
        env_output = self.agents[AgentRole.ENVIRONMENT_ANALYST].execute(project_brief)
        self.outputs.append(env_output)
        self._add_checkpoint("환경분석", env_output)
        print(f"  ✓ 완료 (신뢰도: {env_output.confidence:.0%})")

        # 2. 시나리오 분석
        print("\n▶ 2단계: 시나리오 분석")
        scenario_output = self.agents[AgentRole.SCENARIO_PLANNER].execute(
            {"environment": env_output.output}
        )
        self.outputs.append(scenario_output)
        self._add_checkpoint("시나리오분석", scenario_output)
        print(f"  ✓ 완료 (신뢰도: {scenario_output.confidence:.0%})")

        # 3. 리스크 평가
        print("\n▶ 3단계: 리스크 평가")
        risk_output = self.agents[AgentRole.RISK_ASSESSOR].execute(
            {"environment": env_output.output, "scenarios": scenario_output.output}
        )
        self.outputs.append(risk_output)
        self._add_checkpoint("리스크평가", risk_output)
        print(f"  ✓ 완료 (신뢰도: {risk_output.confidence:.0%})")

        # 4. 보고서 작성
        print("\n▶ 4단계: 보고서 작성")
        report_output = self.agents[AgentRole.REPORT_WRITER].execute(
            {
                "environment": env_output.output,
                "scenarios": scenario_output.output,
                "risks": risk_output.output
            }
        )
        self.outputs.append(report_output)
        self._add_checkpoint("보고서작성", report_output)
        print(f"  ✓ 완료 (신뢰도: {report_output.confidence:.0%})")

        return self.outputs

    def _add_checkpoint(self, stage: str, output: AgentOutput):
        """Human-in-the-Loop 체크포인트 추가"""
        self.checkpoints.append({
            "stage": stage,
            "status": output.status.value,
            "requires_review": output.status == TaskStatus.AWAITING_REVIEW,
            "confidence": output.confidence,
            "timestamp": output.timestamp
        })

    def get_summary(self) -> Dict:
        """워크플로우 요약"""
        return {
            "total_stages": len(self.outputs),
            "avg_confidence": sum(o.confidence for o in self.outputs) / len(self.outputs),
            "pending_reviews": sum(1 for c in self.checkpoints if c["requires_review"]),
            "checkpoints": self.checkpoints
        }


def demonstrate_workflow():
    """워크플로우 시연"""
    print("=" * 60)
    print("기획 워크플로우 자동화 시스템")
    print("=" * 60)

    # 프로젝트 브리프
    project_brief = {
        "title": "전기차 충전 인프라 시장 진입 전략",
        "objective": "신규 사업 진출 타당성 분석 및 전략 수립",
        "timeline": "3개월",
        "budget": "컨설팅 예산 5천만원, 투자 예산 50억원"
    }

    print("\n[프로젝트 브리프]")
    for key, value in project_brief.items():
        print(f"  {key}: {value}")

    # 워크플로우 실행
    workflow = PlanningWorkflow()
    outputs = workflow.run_workflow(project_brief)

    # 결과 출력
    print("\n" + "=" * 60)
    print("워크플로우 결과")
    print("=" * 60)

    for output in outputs:
        print(f"\n[{output.agent_role.value}] {output.task_name}")
        print(f"  상태: {output.status.value}")
        print(f"  신뢰도: {output.confidence:.0%}")

        # 주요 내용 출력
        if output.agent_role == AgentRole.ENVIRONMENT_ANALYST:
            insights = output.output.get("key_insights", [])
            print("  핵심 인사이트:")
            for insight in insights:
                print(f"    - {insight}")

        elif output.agent_role == AgentRole.SCENARIO_PLANNER:
            ev = output.output.get("expected_value", {})
            print(f"  기대 수익: {ev.get('revenue', 'N/A')}")

        elif output.agent_role == AgentRole.RISK_ASSESSOR:
            top_risks = output.output.get("top_risks", [])
            print(f"  상위 리스크: {', '.join(top_risks)}")

        elif output.agent_role == AgentRole.REPORT_WRITER:
            recs = output.output.get("recommendations", [])
            print("  권고안:")
            for rec in recs:
                print(f"    - {rec}")

    # 요약
    summary = workflow.get_summary()
    print("\n" + "=" * 60)
    print("워크플로우 요약")
    print("=" * 60)
    print(f"  총 단계: {summary['total_stages']}")
    print(f"  평균 신뢰도: {summary['avg_confidence']:.0%}")
    print(f"  검토 대기: {summary['pending_reviews']}건")

    return workflow, outputs


def demonstrate_human_in_loop():
    """Human-in-the-Loop 시연"""
    print("\n" + "=" * 60)
    print("Human-in-the-Loop 체크포인트")
    print("=" * 60)

    checkpoints = [
        {"stage": "환경분석", "action": "검토 및 승인", "reviewer": "전략팀장"},
        {"stage": "시나리오분석", "action": "가정 검증", "reviewer": "재무팀"},
        {"stage": "리스크평가", "action": "리스크 추가/수정", "reviewer": "리스크관리팀"},
        {"stage": "최종보고서", "action": "경영진 승인", "reviewer": "CEO/CSO"}
    ]

    print("\n[Human-in-the-Loop 체크포인트]")
    print("-" * 50)
    for i, cp in enumerate(checkpoints, 1):
        print(f"  {i}. [{cp['stage']}]")
        print(f"     액션: {cp['action']}")
        print(f"     검토자: {cp['reviewer']}")

    print("\n[협업 모델 유형]")
    print("-" * 50)
    models = {
        "AI 주도형": "AI가 초안, 인간이 검토/승인",
        "인간 주도형": "인간이 방향 설정, AI가 실행",
        "협업형": "각 단계에서 AI-인간 상호작용",
        "감독형": "AI 자율 실행, 인간이 모니터링"
    }
    for model, desc in models.items():
        print(f"  {model}: {desc}")

    return checkpoints


def main():
    print("=" * 60)
    print("AI 에이전트 기반 기획 워크플로우")
    print("=" * 60)

    # 1. 에이전트 역할 설명
    print("\n[1] 기획 에이전트 역할")
    print("-" * 40)
    roles = {
        "환경분석가": "시장, 경쟁, 규제 환경 분석",
        "시나리오기획자": "시나리오 설계 및 재무 모델링",
        "리스크평가자": "리스크 식별, 평가, 대응 전략",
        "보고서작성자": "분석 결과 종합 및 보고서 작성",
        "코디네이터": "워크플로우 조율 및 품질 관리"
    }
    for role, desc in roles.items():
        print(f"  {role}: {desc}")

    # 2. 워크플로우 시연
    print("\n[2] 워크플로우 실행")
    workflow, outputs = demonstrate_workflow()

    # 3. Human-in-the-Loop
    print("\n[3] Human-in-the-Loop 설계")
    checkpoints = demonstrate_human_in_loop()

    # 4. 요약
    print("\n" + "=" * 60)
    print("요약: AI 에이전트 기획 시스템")
    print("=" * 60)
    print("""
┌─────────────────────────────────────────────────────────┐
│ 워크플로우 파이프라인                                   │
│                                                         │
│  환경분석 → 시나리오 → 리스크평가 → 보고서              │
│     ↓         ↓           ↓          ↓                  │
│  [검토]    [검증]      [보완]     [승인]                │
│                                                         │
│ Human-in-the-Loop: 각 단계에서 인간 검토 및 피드백      │
│                                                         │
│ 핵심 가치:                                              │
│  - AI: 분석 속도, 일관성, 24/7 가용성                   │
│  - 인간: 판단력, 맥락 이해, 최종 책임                   │
└─────────────────────────────────────────────────────────┘
""")

    # 결과 통계
    summary = workflow.get_summary()
    print(f"\n실행 결과:")
    print(f"  - 실행된 에이전트: 4개")
    print(f"  - 평균 신뢰도: {summary['avg_confidence']:.1%}")
    print(f"  - 검토 대기 항목: {summary['pending_reviews']}건")
    print(f"  - 체크포인트: {len(checkpoints)}개")

    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
