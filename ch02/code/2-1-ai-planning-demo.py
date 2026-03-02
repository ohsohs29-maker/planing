"""
2-1-ai-planning-demo.py
AI 증강 기획 개요 데모

이 스크립트는 AI가 기획 프로세스의 각 단계에서 어떻게 활용될 수 있는지
간단한 예시를 통해 보여준다.

실행 방법:
    python 2-1-ai-planning-demo.py

필요 환경변수:
    OPENAI_API_KEY: OpenAI API 키
"""

import os
from typing import Optional

# API 키 설정 (실제 실행 시 환경변수 사용)
# from dotenv import load_dotenv
# load_dotenv()


def demonstrate_planning_stages():
    """
    기획 프로세스의 4단계와 각 단계에서의 AI 활용 방식을 보여준다.
    """
    planning_stages = {
        "1. 문제 정의": {
            "ai_역할": "이슈 탐지, 다관점 분석, 유사 사례 검색",
            "활용_예시": "뉴스/보고서에서 약신호 탐지, 이해관계자별 관점 분석",
            "주요_도구": ["LLM (GPT, Claude)", "NLP 분류 모델", "검색 엔진"],
        },
        "2. 인과 분석": {
            "ai_역할": "데이터 기반 인과 발견, 가설 생성, 교란변수 식별",
            "활용_예시": "변수 간 인과관계 자동 추론, DAG 자동 제안",
            "주요_도구": ["DoWhy", "CausalML", "PC Algorithm"],
        },
        "3. 미래 탐색": {
            "ai_역할": "시나리오 생성, 시뮬레이션, 확률 분포 추정",
            "활용_예시": "4-시나리오 매트릭스 자동 생성, 몬테카를로 시뮬레이션",
            "주요_도구": ["LLM", "NumPy/SciPy", "PyMC"],
        },
        "4. 의사결정": {
            "ai_역할": "민감도 분석, 편향 탐지, 대안 비교",
            "활용_예시": "변수별 민감도 자동 계산, 인지 편향 경고",
            "주요_도구": ["다기준 분석 도구", "LLM 검증"],
        },
    }

    print("=" * 60)
    print("AI 증강 기획: 단계별 AI 활용 맵")
    print("=" * 60)

    for stage, details in planning_stages.items():
        print(f"\n[{stage}]")
        print(f"  AI 역할: {details['ai_역할']}")
        print(f"  활용 예시: {details['활용_예시']}")
        print(f"  주요 도구: {', '.join(details['주요_도구'])}")

    return planning_stages


def simulate_multi_perspective_analysis(issue: str) -> dict:
    """
    이슈에 대한 다관점 분석을 시뮬레이션한다.

    실제 구현에서는 LLM API를 호출하여 각 이해관계자 관점의
    분석을 생성한다.

    Args:
        issue: 분석할 이슈/문제

    Returns:
        이해관계자별 관점 분석 결과
    """
    # 이해관계자 정의
    stakeholders = ["경영진", "고객", "직원", "투자자", "규제기관"]

    # 시뮬레이션 결과 (실제로는 LLM 호출)
    # 아래는 "디지털 전환 가속화" 이슈에 대한 예시 결과
    perspectives = {
        "경영진": {
            "핵심_관심사": "ROI, 경쟁우위, 실행 리스크",
            "기대_효과": "운영 효율성 20% 향상, 신규 수익원 창출",
            "우려_사항": "초기 투자 비용, 기존 시스템 통합 복잡성",
        },
        "고객": {
            "핵심_관심사": "서비스 품질, 개인정보 보호, 가격",
            "기대_효과": "맞춤형 서비스, 빠른 응답 시간",
            "우려_사항": "데이터 활용에 대한 불안, 인간 상담원 감소",
        },
        "직원": {
            "핵심_관심사": "일자리 안정성, 업무 변화, 역량 개발",
            "기대_효과": "반복 업무 감소, 고부가가치 업무 집중",
            "우려_사항": "AI 대체 가능성, 새로운 기술 학습 부담",
        },
        "투자자": {
            "핵심_관심사": "수익성, 성장 잠재력, 리스크",
            "기대_효과": "장기 경쟁력 확보, 시장 가치 상승",
            "우려_사항": "투자 회수 기간, 기술 불확실성",
        },
        "규제기관": {
            "핵심_관심사": "공정성, 투명성, 소비자 보호",
            "기대_효과": "산업 혁신 촉진, 국가 경쟁력 강화",
            "우려_사항": "알고리즘 편향, 데이터 독점, 일자리 영향",
        },
    }

    print(f"\n{'=' * 60}")
    print(f"이슈: {issue}")
    print(f"{'=' * 60}")

    for stakeholder, analysis in perspectives.items():
        print(f"\n[{stakeholder} 관점]")
        for key, value in analysis.items():
            print(f"  {key}: {value}")

    return perspectives


def demonstrate_ai_limitations():
    """
    AI 활용의 한계와 Human-in-the-Loop 원칙을 설명한다.
    """
    limitations = {
        "환각(Hallucination)": {
            "설명": "그럴듯하지만 사실이 아닌 정보 생성",
            "완화_전략": "RAG, Chain-of-Verification, 사실 검증",
            "완화_효과": "GPT-4o 환각률 53% → 23% 감소 (2025 연구)",
        },
        "학습_데이터_편향": {
            "설명": "학습 데이터의 편향이 출력에 반영됨",
            "완화_전략": "다양한 소스 교차 검증, 편향 탐지 도구",
            "완화_효과": "편향 유형 식별 및 경고 가능",
        },
        "맥락_이해_한계": {
            "설명": "조직 문화, 정치적 맥락 등 암묵지 이해 부족",
            "완화_전략": "Human-in-the-Loop, 전문가 검토",
            "완화_효과": "최종 판단의 적절성 확보",
        },
        "인과관계_추론_한계": {
            "설명": "상관관계를 인과관계로 오인할 수 있음",
            "완화_전략": "도메인 전문가 개입, 인과추론 프레임워크 활용",
            "완화_효과": "잘못된 인과 추론 방지",
        },
    }

    print("\n" + "=" * 60)
    print("AI 활용의 한계와 완화 전략")
    print("=" * 60)

    for limitation, details in limitations.items():
        print(f"\n[{limitation}]")
        for key, value in details.items():
            print(f"  {key}: {value}")

    # HITL 원칙 출력
    print("\n" + "-" * 60)
    print("Human-in-the-Loop 핵심 원칙")
    print("-" * 60)
    hitl_principles = [
        "1. AI 출력은 제안이지 결정이 아니다",
        "2. 최종 판단과 책임은 인간에게 있다",
        "3. AI의 한계를 인식하고 검증 프로세스를 갖춘다",
        "4. 중요한 결정일수록 인간 검토 단계를 강화한다",
        "5. AI와 인간의 역할 경계를 명확히 정의한다",
    ]
    for principle in hitl_principles:
        print(f"  {principle}")

    return limitations


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 60)
    print("제2장: AI 증강 기획(AI-Augmented Planning) 데모")
    print("=" * 60)

    # 1. 기획 단계별 AI 활용 맵
    planning_stages = demonstrate_planning_stages()

    # 2. 다관점 분석 시뮬레이션
    issue = "디지털 전환 가속화 전략"
    perspectives = simulate_multi_perspective_analysis(issue)

    # 3. AI 한계와 HITL 원칙
    limitations = demonstrate_ai_limitations()

    print("\n" + "=" * 60)
    print("핵심 메시지")
    print("=" * 60)
    print("""
    AI는 기획자를 대체하는 것이 아니라 증강(Augment)하는 도구이다.

    - 잘하는 것: 대규모 정보 처리, 패턴 인식, 시나리오 생성
    - 못하는 것: 맥락 이해, 가치 판단, 창의적 도약

    최종 판단은 인간의 몫이다.
    """)


if __name__ == "__main__":
    main()
