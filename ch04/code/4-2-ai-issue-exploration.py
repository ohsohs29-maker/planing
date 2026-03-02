"""
4-2-ai-issue-exploration.py
AI를 활용한 다관점 이슈 탐색

이 코드는 LLM API를 활용하여 다양한 이해관계자 관점에서
이슈를 도출하고, MECE 검증 및 누락 탐지를 수행한다.
"""

import json
import os
from typing import Optional
from dataclasses import dataclass, field, asdict

# API 클라이언트는 실제 환경에서 임포트
# from openai import OpenAI
# from anthropic import Anthropic

@dataclass
class Issue:
    """이슈 데이터 클래스"""
    id: str
    question: str
    perspective: str
    category: str
    priority: str = 'medium'
    sub_issues: list = field(default_factory=list)

@dataclass
class IssueAnalysisResult:
    """이슈 분석 결과"""
    issues: list
    mece_check: dict
    missing_perspectives: list
    recommendations: list

def generate_multiperspective_issues(
    problem_statement: str,
    stakeholders: list[str],
    context: str,
    api_key: Optional[str] = None
) -> list[Issue]:
    """
    다관점 이슈 도출
    
    Args:
        problem_statement: 문제 진술
        stakeholders: 이해관계자 목록
        context: 배경 정보
        api_key: API 키 (환경변수에서 가져올 수도 있음)
        
    Returns:
        list[Issue]: 도출된 이슈 목록
    """
    # 프롬프트 구성
    prompt = f"""당신은 전략 컨설턴트입니다. 다음 문제 상황에서 각 이해관계자 관점의 핵심 이슈를 도출해주세요.

## 문제 상황
{problem_statement}

## 배경
{context}

## 이해관계자
{', '.join(stakeholders)}

## 지시사항
1. 각 이해관계자 관점에서 2-3개의 핵심 이슈를 질문 형태로 도출하세요.
2. 이슈는 구체적이고 검증 가능해야 합니다.
3. "~해야 하는가?", "어떻게 ~할 것인가?", "왜 ~인가?" 형태를 사용하세요.
4. 각 이슈의 카테고리(전략/운영/조직/재무/기술)를 명시하세요.

## 출력 형식 (JSON)
{{
    "issues": [
        {{
            "id": "I01",
            "question": "질문 형태의 이슈",
            "perspective": "이해관계자명",
            "category": "카테고리",
            "priority": "high/medium/low",
            "sub_issues": ["하위 이슈 1", "하위 이슈 2"]
        }}
    ]
}}

JSON만 출력하세요."""

    # 실제 API 호출 (데모용 시뮬레이션)
    # 실제 환경에서는 아래 코드 사용:
    # client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}],
    #     response_format={"type": "json_object"}
    # )
    # result = json.loads(response.choices[0].message.content)
    
    # 데모용 시뮬레이션 결과
    simulated_result = {
        "issues": [
            {
                "id": "I01",
                "question": "디지털 전환 투자에 대한 ROI를 어떻게 보장할 수 있는가?",
                "perspective": "CEO",
                "category": "전략",
                "priority": "high",
                "sub_issues": [
                    "단계별 투자 규모를 어떻게 결정할 것인가?",
                    "경쟁사 대비 적정 투자 수준은 얼마인가?"
                ]
            },
            {
                "id": "I02",
                "question": "기존 생산 라인을 중단하지 않으면서 어떻게 스마트 팩토리를 구현할 것인가?",
                "perspective": "생산부문",
                "category": "운영",
                "priority": "high",
                "sub_issues": [
                    "시범 라인 선정 기준은 무엇인가?",
                    "생산성 저하 없이 전환하는 방법은?"
                ]
            },
            {
                "id": "I03",
                "question": "레거시 시스템과 신규 시스템의 통합을 어떻게 진행할 것인가?",
                "perspective": "IT부문",
                "category": "기술",
                "priority": "high",
                "sub_issues": [
                    "데이터 마이그레이션 전략은?",
                    "시스템 다운타임을 최소화하는 방법은?"
                ]
            },
            {
                "id": "I04",
                "question": "디지털 채널 전환 시 기존 영업 조직의 역할을 어떻게 재정의할 것인가?",
                "perspective": "영업부문",
                "category": "조직",
                "priority": "medium",
                "sub_issues": [
                    "온-오프라인 채널 갈등을 어떻게 해소할 것인가?",
                    "영업 인센티브 체계를 어떻게 변경할 것인가?"
                ]
            },
            {
                "id": "I05",
                "question": "자동화로 인한 고용 불안을 어떻게 해소할 것인가?",
                "perspective": "노조",
                "category": "조직",
                "priority": "high",
                "sub_issues": [
                    "재배치 대상 인원에 대한 재교육 방안은?",
                    "고용 보장 기간은 얼마로 할 것인가?"
                ]
            }
        ]
    }
    
    issues = [Issue(**issue_data) for issue_data in simulated_result['issues']]
    return issues

def check_mece(issues: list[Issue]) -> dict:
    """
    이슈의 MECE 검증
    
    Args:
        issues: 이슈 목록
        
    Returns:
        dict: MECE 검증 결과
    """
    prompt = f"""다음 이슈들이 MECE(Mutually Exclusive, Collectively Exhaustive) 원칙을 충족하는지 검증하세요.

## 이슈 목록
{json.dumps([asdict(i) for i in issues], ensure_ascii=False, indent=2)}

## 검증 항목
1. 상호 배타성(ME): 이슈 간 중복이 있는가?
2. 전체 포괄성(CE): 누락된 중요 영역이 있는가?

## 출력 형식 (JSON)
{{
    "is_mece": true/false,
    "overlapping_issues": [["I01", "I03", "중복 설명"]],
    "missing_areas": ["누락된 영역 1", "누락된 영역 2"],
    "suggestions": ["개선 제안 1", "개선 제안 2"]
}}

JSON만 출력하세요."""

    # 시뮬레이션 결과
    mece_result = {
        "is_mece": False,
        "overlapping_issues": [
            ["I03", "I02", "시스템 통합과 스마트 팩토리 구현이 기술적으로 연결됨"]
        ],
        "missing_areas": [
            "재무: 디지털 전환 자금 조달 방안",
            "법무/컴플라이언스: 데이터 보안 및 규제 준수",
            "외부 이해관계자: 고객 경험 관점의 이슈"
        ],
        "suggestions": [
            "I02와 I03를 '기술 인프라' 범주로 통합하고 하위 이슈로 분리",
            "재무 관점의 이슈 추가 (자금 조달, 비용 통제)",
            "고객 관점의 이슈 추가 (디지털 서비스 기대)"
        ]
    }
    
    return mece_result

def identify_so_what(issue: Issue) -> dict:
    """
    "So What?" 테스트 수행
    
    Args:
        issue: 검증할 이슈
        
    Returns:
        dict: So What 테스트 결과
    """
    prompt = f""""So What?" 테스트를 수행하세요.

## 이슈
{issue.question}

## 검증 질문
1. 이 이슈가 해결되면 어떤 결과가 나오는가?
2. 이 결과가 왜 중요한가?
3. 의사결정에 어떤 영향을 미치는가?

## 출력 형식 (JSON)
{{
    "passes_test": true/false,
    "consequence": "이슈 해결 시 예상 결과",
    "importance": "중요한 이유",
    "decision_impact": "의사결정 영향",
    "recommendation": "통과/보완 필요/재정의 필요"
}}"""

    # 시뮬레이션 (issue I01 기준)
    so_what_result = {
        "passes_test": True,
        "consequence": "투자 대비 수익 예측이 가능해져 이사회 승인 확보 가능",
        "importance": "대규모 자본 지출의 정당성 확보 및 주주 신뢰 유지",
        "decision_impact": "투자 규모와 시기 결정에 직접 영향",
        "recommendation": "통과"
    }
    
    return so_what_result

def generate_issue_tree(issues: list[Issue], root_problem: str) -> dict:
    """
    이슈 트리 생성
    
    Args:
        issues: 이슈 목록
        root_problem: 루트 문제
        
    Returns:
        dict: 이슈 트리 구조
    """
    # 카테고리별 그룹화
    categories = {}
    for issue in issues:
        if issue.category not in categories:
            categories[issue.category] = []
        categories[issue.category].append(asdict(issue))
    
    issue_tree = {
        "root": root_problem,
        "branches": [
            {
                "category": cat,
                "issues": cat_issues
            }
            for cat, cat_issues in categories.items()
        ],
        "total_issues": len(issues),
        "categories": list(categories.keys())
    }
    
    return issue_tree

def export_results(
    issues: list[Issue],
    mece_result: dict,
    issue_tree: dict,
    output_path: str
):
    """
    분석 결과 JSON 내보내기
    
    Args:
        issues: 이슈 목록
        mece_result: MECE 검증 결과
        issue_tree: 이슈 트리
        output_path: 출력 파일 경로
    """
    result = {
        "analysis_summary": {
            "total_issues": len(issues),
            "by_priority": {
                "high": len([i for i in issues if i.priority == 'high']),
                "medium": len([i for i in issues if i.priority == 'medium']),
                "low": len([i for i in issues if i.priority == 'low'])
            },
            "by_category": {
                cat: len([i for i in issues if i.category == cat])
                for cat in set(i.category for i in issues)
            }
        },
        "issues": [asdict(i) for i in issues],
        "mece_check": mece_result,
        "issue_tree": issue_tree
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"분석 결과 저장: {output_path}")

def main():
    """메인 실행 함수"""
    # 문제 상황 정의
    problem_statement = "중견 제조기업 B사의 디지털 전환 전략 수립"
    context = """
    - 매출 5,000억원 규모의 자동차 부품 제조기업
    - 최근 3년간 매출 정체 (연평균 성장률 1.2%)
    - 경쟁사 대비 디지털 역량 부족 (디지털 성숙도 2.1/5.0)
    - 평균 근속연수 18년, 50대 이상 비율 35%
    - 주요 고객사의 디지털 협업 요구 증가
    """
    stakeholders = ["CEO", "생산부문", "IT부문", "영업부문", "노조"]
    
    print("="*60)
    print("AI 기반 다관점 이슈 탐색")
    print("="*60)
    print(f"\n문제 상황: {problem_statement}")
    print(f"이해관계자: {', '.join(stakeholders)}")
    
    # 1. 다관점 이슈 도출
    print("\n[1단계] 다관점 이슈 도출 중...")
    issues = generate_multiperspective_issues(
        problem_statement, stakeholders, context
    )
    print(f"  → {len(issues)}개 이슈 도출 완료")
    
    for issue in issues:
        print(f"\n  {issue.id} [{issue.perspective}] {issue.category}")
        print(f"  Q: {issue.question}")
        if issue.sub_issues:
            for sub in issue.sub_issues:
                print(f"    - {sub}")
    
    # 2. MECE 검증
    print("\n[2단계] MECE 검증 중...")
    mece_result = check_mece(issues)
    print(f"  → MECE 충족: {mece_result['is_mece']}")
    if mece_result['overlapping_issues']:
        print("  → 중복 이슈:")
        for overlap in mece_result['overlapping_issues']:
            print(f"    - {overlap[0]}, {overlap[1]}: {overlap[2]}")
    if mece_result['missing_areas']:
        print("  → 누락 영역:")
        for missing in mece_result['missing_areas']:
            print(f"    - {missing}")
    
    # 3. So What 테스트 (첫 번째 이슈만 데모)
    print("\n[3단계] 'So What?' 테스트 (I01 예시)...")
    so_what_result = identify_so_what(issues[0])
    print(f"  → 테스트 통과: {so_what_result['passes_test']}")
    print(f"  → 결과: {so_what_result['consequence']}")
    print(f"  → 중요성: {so_what_result['importance']}")
    
    # 4. 이슈 트리 생성
    print("\n[4단계] 이슈 트리 생성...")
    issue_tree = generate_issue_tree(issues, problem_statement)
    print(f"  → 카테고리: {', '.join(issue_tree['categories'])}")
    
    # 5. 결과 저장
    from pathlib import Path
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)
    
    export_results(
        issues, 
        mece_result, 
        issue_tree,
        str(output_dir / 'issue_analysis_result.json')
    )
    
    print("\n" + "="*60)
    print("분석 완료")
    print("="*60)

if __name__ == '__main__':
    main()
