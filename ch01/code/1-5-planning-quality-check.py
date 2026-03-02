"""
제1장 실습: 기획 품질 자가 진단 도구

이 스크립트는 기획의 품질을 세 가지 조건(인과적 타당성, 불확실성 인식, 적응성)에
따라 진단하는 자가 진단 도구를 구현한다.
"""

import json
from typing import Dict, List


def get_checklist() -> Dict[str, List[str]]:
    """기획 품질 진단을 위한 체크리스트를 반환한다."""

    checklist = {
        "인과적 타당성": [
            "핵심 가정이 명확히 식별되어 있는가?",
            "가정을 뒷받침하는 데이터나 증거가 있는가?",
            "인과관계가 논리적으로 연결되어 있는가?",
            "교란변수(제3의 요인)가 고려되었는가?",
            "변화 이론(Theory of Change)이 명시되어 있는가?"
        ],
        "불확실성 인식": [
            "주요 불확실성 요인이 식별되어 있는가?",
            "결과가 범위로 표현되어 있는가(점 추정 대신)?",
            "시나리오 분석이 수행되었는가?",
            "민감도 분석이 수행되었는가?",
            "리스크 대응 방안이 마련되어 있는가?"
        ],
        "적응성": [
            "환경 변화 모니터링 체계가 있는가?",
            "계획 수정 트리거가 정의되어 있는가?",
            "단계적 접근(옵션 확보)이 설계되어 있는가?",
            "학습 및 피드백 프로세스가 있는가?",
            "비상 계획(contingency plan)이 있는가?"
        ]
    }

    return checklist


def run_diagnosis(checklist: Dict[str, List[str]]) -> Dict[str, Dict]:
    """사용자 입력을 받아 진단을 수행한다."""

    results = {}

    print("\n" + "=" * 60)
    print("기획 품질 자가 진단 도구")
    print("=" * 60)
    print("\n각 질문에 대해 '예(y)' 또는 '아니오(n)'로 답해주세요.\n")

    for category, questions in checklist.items():
        print(f"\n[{category}]")
        print("-" * 40)

        yes_count = 0
        responses = []

        for i, question in enumerate(questions, 1):
            while True:
                response = input(f"{i}. {question} (y/n): ").strip().lower()
                if response in ['y', 'n', 'yes', 'no']:
                    is_yes = response in ['y', 'yes']
                    responses.append(is_yes)
                    if is_yes:
                        yes_count += 1
                    break
                print("   'y' 또는 'n'으로 답해주세요.")

        score = yes_count / len(questions) * 100
        results[category] = {
            "score": score,
            "yes_count": yes_count,
            "total": len(questions),
            "responses": responses
        }

    return results


def display_results(results: Dict[str, Dict]) -> None:
    """진단 결과를 표시한다."""

    print("\n" + "=" * 60)
    print("진단 결과")
    print("=" * 60)

    total_score = 0

    for category, data in results.items():
        score = data["score"]
        yes_count = data["yes_count"]
        total = data["total"]
        total_score += score

        # 등급 결정
        if score >= 80:
            grade = "우수"
            emoji = "★★★"
        elif score >= 60:
            grade = "양호"
            emoji = "★★☆"
        elif score >= 40:
            grade = "보통"
            emoji = "★☆☆"
        else:
            grade = "미흡"
            emoji = "☆☆☆"

        print(f"\n[{category}]")
        print(f"  점수: {score:.0f}점 ({yes_count}/{total})")
        print(f"  등급: {grade} {emoji}")

    # 종합 점수
    overall = total_score / len(results)
    print("\n" + "-" * 60)
    print(f"종합 점수: {overall:.0f}점")

    if overall >= 80:
        print("평가: 기획의 품질이 우수합니다.")
    elif overall >= 60:
        print("평가: 기획의 품질이 양호하나, 일부 보완이 필요합니다.")
    elif overall >= 40:
        print("평가: 기획의 품질 개선이 필요합니다.")
    else:
        print("평가: 기획의 근본적인 재검토가 필요합니다.")


def get_improvement_suggestions(results: Dict[str, Dict]) -> List[str]:
    """개선 제안을 생성한다."""

    suggestions = []

    for category, data in results.items():
        if data["score"] < 60:
            if category == "인과적 타당성":
                suggestions.append(
                    "- 인과적 타당성 개선: 5장(인과추론) 학습을 권장합니다. "
                    "인과 다이어그램(DAG)을 활용하여 핵심 가정을 명시하고 검증하세요."
                )
            elif category == "불확실성 인식":
                suggestions.append(
                    "- 불확실성 인식 개선: 9-11장 학습을 권장합니다. "
                    "베이지안 추론, 시나리오 플래닝, 몬테카를로 시뮬레이션을 "
                    "활용하여 불확실성을 명시적으로 다루세요."
                )
            elif category == "적응성":
                suggestions.append(
                    "- 적응성 개선: 13장(실물옵션), 15장(적응적 기획) 학습을 권장합니다. "
                    "유연성을 확보하고 환경 변화에 대응하는 체계를 구축하세요."
                )

    return suggestions


def display_suggestions(suggestions: List[str]) -> None:
    """개선 제안을 표시한다."""

    if suggestions:
        print("\n" + "=" * 60)
        print("개선 제안")
        print("=" * 60)
        for suggestion in suggestions:
            print(f"\n{suggestion}")
    else:
        print("\n모든 영역에서 양호한 점수를 받았습니다!")


def main():
    """메인 실행 함수"""

    # 체크리스트 가져오기
    checklist = get_checklist()

    # 진단 수행
    results = run_diagnosis(checklist)

    # 결과 표시
    display_results(results)

    # 개선 제안 생성 및 표시
    suggestions = get_improvement_suggestions(results)
    display_suggestions(suggestions)

    # 결과 저장 (JSON)
    print("\n" + "-" * 60)
    save = input("결과를 파일로 저장하시겠습니까? (y/n): ").strip().lower()
    if save in ['y', 'yes']:
        with open("diagnosis_result.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("결과가 'diagnosis_result.json'에 저장되었습니다.")


if __name__ == "__main__":
    main()
