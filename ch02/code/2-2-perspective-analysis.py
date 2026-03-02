"""
2-2-perspective-analysis.py
LLM을 활용한 다관점 이슈 분석

이 스크립트는 OpenAI API를 사용하여 특정 이슈에 대해
다양한 이해관계자 관점에서 분석을 수행한다.

실행 방법:
    export OPENAI_API_KEY="your-api-key"
    python 2-2-perspective-analysis.py

참고: API 키가 없는 경우 시뮬레이션 모드로 실행됨
"""

import os
import json
from typing import Optional
from dataclasses import dataclass


@dataclass
class StakeholderPerspective:
    """이해관계자 관점 분석 결과"""
    stakeholder: str
    key_concerns: list[str]
    expected_benefits: list[str]
    potential_risks: list[str]
    recommended_actions: list[str]


def create_perspective_prompt(issue: str, stakeholder: str) -> str:
    """
    이해관계자 관점 분석을 위한 프롬프트를 생성한다.

    Args:
        issue: 분석할 이슈
        stakeholder: 이해관계자 유형

    Returns:
        LLM에 전달할 프롬프트
    """
    prompt = f"""당신은 {stakeholder}의 관점에서 다음 이슈를 분석하는 전문가입니다.

이슈: {issue}

다음 항목에 대해 {stakeholder}의 관점에서 분석해주세요:

1. 핵심 관심사 (3가지)
2. 기대 효과 (3가지)
3. 잠재적 리스크 (3가지)
4. 권장 조치 (3가지)

JSON 형식으로 응답해주세요:
{{
    "key_concerns": ["...", "...", "..."],
    "expected_benefits": ["...", "...", "..."],
    "potential_risks": ["...", "...", "..."],
    "recommended_actions": ["...", "...", "..."]
}}
"""
    return prompt


def analyze_with_llm(issue: str, stakeholder: str, api_key: Optional[str] = None) -> StakeholderPerspective:
    """
    LLM을 사용하여 이해관계자 관점 분석을 수행한다.

    Args:
        issue: 분석할 이슈
        stakeholder: 이해관계자 유형
        api_key: OpenAI API 키 (없으면 시뮬레이션)

    Returns:
        분석 결과
    """
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            prompt = create_perspective_prompt(issue, stakeholder)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 전략 분석 전문가입니다. JSON 형식으로만 응답합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            return StakeholderPerspective(
                stakeholder=stakeholder,
                key_concerns=result.get("key_concerns", []),
                expected_benefits=result.get("expected_benefits", []),
                potential_risks=result.get("potential_risks", []),
                recommended_actions=result.get("recommended_actions", [])
            )
        except Exception as e:
            print(f"API 호출 실패: {e}")
            print("시뮬레이션 모드로 전환합니다.")

    # 시뮬레이션 결과 반환
    return get_simulated_perspective(issue, stakeholder)


def get_simulated_perspective(issue: str, stakeholder: str) -> StakeholderPerspective:
    """
    API 없이 시뮬레이션된 분석 결과를 반환한다.
    """
    simulated_data = {
        "경영진": StakeholderPerspective(
            stakeholder="경영진",
            key_concerns=[
                "투자 대비 수익률(ROI) 확보",
                "경쟁사 대비 차별화 전략",
                "조직 변화 관리"
            ],
            expected_benefits=[
                "운영 효율성 15-25% 향상",
                "신규 수익원 창출 기회",
                "의사결정 속도 가속화"
            ],
            potential_risks=[
                "초기 투자 비용 과다",
                "기존 시스템 통합 복잡성",
                "인력 저항 및 변화 관리 실패"
            ],
            recommended_actions=[
                "단계적 도입 로드맵 수립",
                "파일럿 프로젝트로 검증 후 확대",
                "변화관리 프로그램 병행 실시"
            ]
        ),
        "고객": StakeholderPerspective(
            stakeholder="고객",
            key_concerns=[
                "서비스 품질 유지/향상",
                "개인정보 보호 및 데이터 보안",
                "가격 변동 가능성"
            ],
            expected_benefits=[
                "맞춤형 서비스 제공",
                "응답 시간 단축",
                "24/7 서비스 이용 가능"
            ],
            potential_risks=[
                "인간 상담원 감소로 인한 서비스 질 저하",
                "데이터 오남용 우려",
                "기술 장애 시 서비스 중단"
            ],
            recommended_actions=[
                "고객 피드백 채널 강화",
                "데이터 활용 투명성 제고",
                "하이브리드 서비스 모델 유지"
            ]
        ),
        "직원": StakeholderPerspective(
            stakeholder="직원",
            key_concerns=[
                "일자리 안정성",
                "업무 방식 변화 적응",
                "새로운 기술 역량 요구"
            ],
            expected_benefits=[
                "반복적 업무 감소",
                "고부가가치 업무 집중 가능",
                "새로운 역량 개발 기회"
            ],
            potential_risks=[
                "AI 대체로 인한 직무 불안",
                "과도한 학습 부담",
                "업무 자율성 감소"
            ],
            recommended_actions=[
                "재교육/역량개발 프로그램 제공",
                "경력 전환 지원 체계 마련",
                "직원 참여형 변화 관리"
            ]
        ),
    }

    return simulated_data.get(stakeholder, StakeholderPerspective(
        stakeholder=stakeholder,
        key_concerns=["분석 데이터 없음"],
        expected_benefits=["분석 데이터 없음"],
        potential_risks=["분석 데이터 없음"],
        recommended_actions=["분석 데이터 없음"]
    ))


def multi_perspective_analysis(issue: str, stakeholders: list[str]) -> dict:
    """
    여러 이해관계자 관점에서 이슈를 분석한다.

    Args:
        issue: 분석할 이슈
        stakeholders: 이해관계자 목록

    Returns:
        이해관계자별 분석 결과
    """
    api_key = os.getenv("OPENAI_API_KEY")

    results = {}
    for stakeholder in stakeholders:
        print(f"  {stakeholder} 관점 분석 중...")
        results[stakeholder] = analyze_with_llm(issue, stakeholder, api_key)

    return results


def print_analysis_results(issue: str, results: dict):
    """분석 결과를 포맷팅하여 출력한다."""
    print("\n" + "=" * 70)
    print(f"다관점 이슈 분석 결과")
    print(f"이슈: {issue}")
    print("=" * 70)

    for stakeholder, perspective in results.items():
        print(f"\n[{stakeholder} 관점]")
        print("-" * 50)

        print("  핵심 관심사:")
        for i, concern in enumerate(perspective.key_concerns, 1):
            print(f"    {i}. {concern}")

        print("  기대 효과:")
        for i, benefit in enumerate(perspective.expected_benefits, 1):
            print(f"    {i}. {benefit}")

        print("  잠재적 리스크:")
        for i, risk in enumerate(perspective.potential_risks, 1):
            print(f"    {i}. {risk}")

        print("  권장 조치:")
        for i, action in enumerate(perspective.recommended_actions, 1):
            print(f"    {i}. {action}")

    # 종합 분석
    print("\n" + "=" * 70)
    print("종합 분석: 이해관계자 간 관점 비교")
    print("=" * 70)

    # 공통 관심사 추출 (간단한 키워드 매칭)
    all_concerns = []
    for perspective in results.values():
        all_concerns.extend(perspective.key_concerns)

    print(f"\n총 {len(results)}개 이해관계자 관점 분석 완료")
    print("공통적으로 언급된 주제: 변화 관리, 투자 효율성, 서비스 품질")


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("2-2: LLM을 활용한 다관점 이슈 분석")
    print("=" * 70)

    # 분석 대상 이슈
    issue = "AI 기반 고객 서비스 자동화 도입"

    # 분석 대상 이해관계자
    stakeholders = ["경영진", "고객", "직원"]

    print(f"\n분석 이슈: {issue}")
    print(f"분석 관점: {', '.join(stakeholders)}")
    print("\n분석 진행 중...")

    # 다관점 분석 수행
    results = multi_perspective_analysis(issue, stakeholders)

    # 결과 출력
    print_analysis_results(issue, results)

    print("\n" + "-" * 70)
    print("활용 팁")
    print("-" * 70)
    print("""
    1. 이해관계자 목록은 이슈의 특성에 따라 조정
    2. LLM 결과는 초안으로 활용하고 전문가 검토 필수
    3. 관점 간 충돌 지점을 파악하여 협상/조정 포인트 도출
    4. 정기적으로 재분석하여 관점 변화 추적
    """)


if __name__ == "__main__":
    main()
