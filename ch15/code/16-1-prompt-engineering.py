#!/usr/bin/env python3
"""
16-1-prompt-engineering.py
프롬프트 엔지니어링 기초 및 고급 기법 시연

이 코드는 다음을 수행한다:
1. 기본 프롬프트 vs 구조화된 프롬프트 비교
2. Chain-of-Thought (CoT) 프롬프팅 시연
3. Few-Shot Learning 예제
4. 프롬프트 품질 평가 프레임워크
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
import json


class PromptComponent(Enum):
    """프롬프트 구성 요소"""
    ROLE = "역할"
    CONTEXT = "맥락"
    INSTRUCTION = "지시사항"
    FORMAT = "출력형식"
    CONSTRAINTS = "제약조건"
    EXAMPLES = "예시"


@dataclass
class Prompt:
    """구조화된 프롬프트"""
    role: str = ""
    context: str = ""
    instruction: str = ""
    output_format: str = ""
    constraints: List[str] = None
    examples: List[Dict] = None

    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
        if self.examples is None:
            self.examples = []

    def to_text(self) -> str:
        """프롬프트를 텍스트로 변환"""
        parts = []

        if self.role:
            parts.append(f"# 역할\n{self.role}")

        if self.context:
            parts.append(f"# 맥락\n{self.context}")

        if self.instruction:
            parts.append(f"# 지시사항\n{self.instruction}")

        if self.constraints:
            constraints_text = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"# 제약조건\n{constraints_text}")

        if self.examples:
            examples_text = "\n\n".join(
                f"입력: {ex['input']}\n출력: {ex['output']}"
                for ex in self.examples
            )
            parts.append(f"# 예시\n{examples_text}")

        if self.output_format:
            parts.append(f"# 출력 형식\n{self.output_format}")

        return "\n\n".join(parts)

    def component_count(self) -> int:
        """포함된 구성 요소 수"""
        count = 0
        if self.role: count += 1
        if self.context: count += 1
        if self.instruction: count += 1
        if self.output_format: count += 1
        if self.constraints: count += 1
        if self.examples: count += 1
        return count


def compare_prompt_quality():
    """기본 프롬프트 vs 구조화된 프롬프트 비교"""
    print("\n" + "=" * 60)
    print("기본 프롬프트 vs 구조화된 프롬프트 비교")
    print("=" * 60)

    # 기본 프롬프트 (비구조화)
    basic_prompt = "시장 분석해줘"

    # 구조화된 프롬프트
    structured_prompt = Prompt(
        role="당신은 10년 경력의 전략 컨설턴트입니다.",
        context="""
현재 분석 대상: 국내 전기차 충전 인프라 시장
시장 규모: 2024년 기준 약 2조원
주요 플레이어: 한국전력, SK E&S, GS칼텍스, 현대차그룹
""",
        instruction="""
다음 관점에서 시장을 분석하세요:
1. 시장 성장 동인과 저해 요인
2. 경쟁 구도 분석
3. 신규 진입자를 위한 기회 영역
""",
        output_format="""
JSON 형식으로 출력하세요:
{
    "growth_drivers": [...],
    "barriers": [...],
    "competitive_landscape": {...},
    "opportunities": [...]
}
""",
        constraints=[
            "데이터 기반 분석을 우선하세요",
            "불확실한 정보는 가정임을 명시하세요",
            "3개 이상의 기회 영역을 도출하세요"
        ]
    )

    print("\n[기본 프롬프트]")
    print("-" * 40)
    print(basic_prompt)
    print(f"\n구성 요소: 1개 (지시사항만)")
    print("예상 문제: 모호함, 맥락 부재, 출력 형식 불명확")

    print("\n[구조화된 프롬프트]")
    print("-" * 40)
    print(structured_prompt.to_text())
    print(f"\n구성 요소: {structured_prompt.component_count()}개")
    print("장점: 명확한 역할, 충분한 맥락, 구체적 지시, 형식 지정")

    return {
        'basic': {'components': 1, 'clarity': 'low'},
        'structured': {'components': structured_prompt.component_count(), 'clarity': 'high'}
    }


def demonstrate_cot():
    """Chain-of-Thought 프롬프팅 시연"""
    print("\n" + "=" * 60)
    print("Chain-of-Thought (CoT) 프롬프팅")
    print("=" * 60)

    # 일반 프롬프트
    standard_prompt = """
질문: 회사 A의 매출이 100억에서 150억으로 증가했고,
비용은 80억에서 100억으로 증가했습니다.
수익성이 개선되었나요?
"""

    # CoT 프롬프트
    cot_prompt = """
질문: 회사 A의 매출이 100억에서 150억으로 증가했고,
비용은 80억에서 100억으로 증가했습니다.
수익성이 개선되었나요?

단계별로 분석해 보겠습니다:

1단계: 이전 이익 계산
- 이전 이익 = 이전 매출 - 이전 비용
- 이전 이익 = 100억 - 80억 = 20억

2단계: 현재 이익 계산
- 현재 이익 = 현재 매출 - 현재 비용
- 현재 이익 = 150억 - 100억 = 50억

3단계: 이익률 비교
- 이전 이익률 = 20억 / 100억 = 20%
- 현재 이익률 = 50억 / 150억 = 33.3%

4단계: 결론
- 이익이 20억에서 50억으로 150% 증가
- 이익률이 20%에서 33.3%로 개선
- 따라서 수익성이 개선되었습니다.
"""

    print("\n[일반 프롬프트]")
    print("-" * 40)
    print(standard_prompt)
    print("→ LLM이 바로 결론을 내릴 수 있으나, 추론 과정이 불투명")

    print("\n[Chain-of-Thought 프롬프트]")
    print("-" * 40)
    print(cot_prompt)
    print("→ 단계별 추론으로 정확도 향상, 검증 가능")

    # CoT 결과 정리
    result = {
        'previous_profit': 20,
        'current_profit': 50,
        'previous_margin': 20.0,
        'current_margin': 33.3,
        'improvement': True
    }

    print("\n[CoT 분석 결과]")
    print(f"  이전 이익: {result['previous_profit']}억원")
    print(f"  현재 이익: {result['current_profit']}억원")
    print(f"  이익률 변화: {result['previous_margin']}% → {result['current_margin']}%")
    print(f"  수익성 개선: {'예' if result['improvement'] else '아니오'}")

    return result


def demonstrate_few_shot():
    """Few-Shot Learning 프롬프팅 시연"""
    print("\n" + "=" * 60)
    print("Few-Shot Learning 프롬프팅")
    print("=" * 60)

    few_shot_prompt = Prompt(
        role="당신은 기업 리스크 분류 전문가입니다.",
        instruction="주어진 리스크 설명을 읽고 적절한 카테고리로 분류하세요.",
        examples=[
            {
                "input": "핵심 개발자 3명이 경쟁사로 이직했습니다.",
                "output": "카테고리: 인적자원 리스크\n심각도: 높음\n이유: 핵심 역량 유출 및 프로젝트 지연 우려"
            },
            {
                "input": "주요 원자재 가격이 30% 상승했습니다.",
                "output": "카테고리: 재무/운영 리스크\n심각도: 중간\n이유: 원가 상승으로 마진 압박, 가격 전가 필요"
            },
            {
                "input": "새로운 환경 규제가 내년부터 시행됩니다.",
                "output": "카테고리: 규제/컴플라이언스 리스크\n심각도: 중간\n이유: 준비 기간 있음, 설비 투자 필요 가능"
            }
        ],
        output_format="카테고리: [분류]\n심각도: [높음/중간/낮음]\n이유: [설명]"
    )

    print("\n[Few-Shot 프롬프트]")
    print("-" * 40)
    print(few_shot_prompt.to_text())

    # 새로운 입력에 대한 예상 분류
    new_risks = [
        "주요 고객사가 계약 해지를 통보했습니다.",
        "서버 장애로 서비스가 4시간 중단되었습니다.",
        "경쟁사가 파괴적 가격 정책을 발표했습니다."
    ]

    expected_classifications = [
        {"category": "고객/매출 리스크", "severity": "높음",
         "reason": "매출 직접 영향, 다른 고객 확보 필요"},
        {"category": "운영/기술 리스크", "severity": "중간",
         "reason": "서비스 신뢰도 저하, 보상 및 재발 방지 필요"},
        {"category": "시장/경쟁 리스크", "severity": "높음",
         "reason": "가격 경쟁 심화, 차별화 전략 필요"}
    ]

    print("\n[새로운 입력에 대한 예상 분류]")
    for i, (risk, classification) in enumerate(zip(new_risks, expected_classifications), 1):
        print(f"\n입력 {i}: {risk}")
        print(f"  카테고리: {classification['category']}")
        print(f"  심각도: {classification['severity']}")
        print(f"  이유: {classification['reason']}")

    return expected_classifications


def evaluate_prompt_quality():
    """프롬프트 품질 평가 프레임워크"""
    print("\n" + "=" * 60)
    print("프롬프트 품질 평가 프레임워크")
    print("=" * 60)

    criteria = {
        "명확성": {
            "description": "지시사항이 모호하지 않고 구체적인가?",
            "weight": 0.25
        },
        "완전성": {
            "description": "필요한 모든 맥락과 정보가 포함되어 있는가?",
            "weight": 0.20
        },
        "구조화": {
            "description": "역할, 맥락, 지시, 형식이 체계적으로 구성되어 있는가?",
            "weight": 0.20
        },
        "제약조건": {
            "description": "출력 범위와 제한사항이 명시되어 있는가?",
            "weight": 0.15
        },
        "예시 제공": {
            "description": "Few-shot 예시로 기대 출력을 보여주는가?",
            "weight": 0.10
        },
        "검증 가능성": {
            "description": "출력 결과를 검증할 기준이 있는가?",
            "weight": 0.10
        }
    }

    # 두 프롬프트 평가
    basic_scores = {
        "명확성": 2, "완전성": 1, "구조화": 1,
        "제약조건": 1, "예시 제공": 1, "검증 가능성": 1
    }

    structured_scores = {
        "명확성": 5, "완전성": 5, "구조화": 5,
        "제약조건": 4, "예시 제공": 4, "검증 가능성": 4
    }

    def calculate_weighted_score(scores):
        total = 0
        for criterion, score in scores.items():
            total += score * criteria[criterion]["weight"]
        return total

    basic_total = calculate_weighted_score(basic_scores)
    structured_total = calculate_weighted_score(structured_scores)

    print("\n[평가 기준]")
    print("-" * 40)
    for criterion, info in criteria.items():
        print(f"  {criterion} (가중치 {info['weight']*100:.0f}%): {info['description']}")

    print("\n[평가 결과] (5점 만점)")
    print("-" * 40)
    print(f"{'기준':<12} {'기본 프롬프트':>12} {'구조화 프롬프트':>14}")
    print("-" * 40)
    for criterion in criteria:
        print(f"{criterion:<12} {basic_scores[criterion]:>12} {structured_scores[criterion]:>14}")
    print("-" * 40)
    print(f"{'가중 평균':<12} {basic_total:>12.2f} {structured_total:>14.2f}")

    improvement = (structured_total - basic_total) / basic_total * 100
    print(f"\n구조화 프롬프트가 기본 대비 {improvement:.1f}% 높은 점수")

    return {
        'basic_score': basic_total,
        'structured_score': structured_total,
        'improvement': improvement
    }


def main():
    print("=" * 60)
    print("프롬프트 엔지니어링 기초 및 고급 기법")
    print("=" * 60)

    # 1. 기본 vs 구조화 프롬프트 비교
    print("\n[1] 프롬프트 구조 비교")
    compare_result = compare_prompt_quality()

    # 2. Chain-of-Thought 시연
    print("\n[2] Chain-of-Thought (CoT) 시연")
    cot_result = demonstrate_cot()

    # 3. Few-Shot Learning 시연
    print("\n[3] Few-Shot Learning 시연")
    few_shot_result = demonstrate_few_shot()

    # 4. 품질 평가 프레임워크
    print("\n[4] 프롬프트 품질 평가")
    quality_result = evaluate_prompt_quality()

    # 5. 요약
    print("\n" + "=" * 60)
    print("요약: 프롬프트 엔지니어링 핵심 원칙")
    print("=" * 60)
    print("""
┌─────────────────────────────────────────────────────────┐
│ 1. 역할 부여: "당신은 ~입니다" 로 전문성 맥락 설정      │
│ 2. 맥락 제공: 배경 정보와 현재 상황 명시               │
│ 3. 구체적 지시: 무엇을, 어떻게, 왜를 명확히            │
│ 4. 출력 형식: JSON, 표, 목록 등 원하는 형태 지정       │
│ 5. 예시 제공: Few-shot으로 기대 출력 시연              │
│ 6. 단계적 사고: CoT로 복잡한 문제 분해                 │
└─────────────────────────────────────────────────────────┘
""")

    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
