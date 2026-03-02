"""
3-2-ai-logic-tree-generation.py
AI를 활용한 로직 트리 생성

이 스크립트는 LLM(Claude/GPT)을 활용하여 문제를 MECE 원칙에 따라
구조화된 로직 트리로 분해하는 방법을 보여준다.
"""

import json
import os
from typing import Optional

# API 키 설정 (환경 변수에서 읽기)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def create_logic_tree_prompt(problem: str, tree_type: str = "why") -> str:
    """로직 트리 생성을 위한 프롬프트를 작성한다.
    
    Args:
        problem: 분석할 문제/질문
        tree_type: "why" (원인 분석) 또는 "how" (해결책 도출)
    
    Returns:
        구조화된 프롬프트 문자열
    """
    type_desc = "원인 분석 (Why Tree)" if tree_type == "why" else "해결책 도출 (How Tree)"
    question_prefix = "왜" if tree_type == "why" else "어떻게"
    
    prompt = f"""당신은 McKinsey 출신 전략 컨설턴트입니다. 
다음 문제를 MECE(Mutually Exclusive, Collectively Exhaustive) 원칙에 따라 
{type_desc}로 분해해 주세요.

## 문제
{problem}

## 요구사항
1. 각 분기(branch)는 상호 배타적(Mutually Exclusive)이어야 합니다
2. 모든 분기를 합치면 전체를 포괄(Collectively Exhaustive)해야 합니다
3. 3단계 깊이까지 분해하세요
4. 각 분기에는 간결하고 명확한 이름을 붙이세요
5. 리프 노드는 실행 가능한 수준이어야 합니다

## 출력 형식
다음 JSON 형식으로 출력하세요:

```json
{{
  "problem": "원본 문제",
  "tree_type": "{tree_type}",
  "root": {{
    "name": "핵심 질문",
    "children": [
      {{
        "name": "1단계 분기 1",
        "children": [
          {{"name": "2단계 분기 1-1", "children": [...]}},
          {{"name": "2단계 분기 1-2", "children": [...]}}
        ]
      }},
      {{
        "name": "1단계 분기 2",
        "children": [...]
      }}
    ]
  }},
  "mece_check": {{
    "mutually_exclusive": true/false,
    "collectively_exhaustive": true/false,
    "notes": "MECE 검증 설명"
  }}
}}
```

JSON만 출력하고 다른 설명은 추가하지 마세요."""

    return prompt


def generate_with_anthropic(prompt: str, model: str = "claude-3-5-sonnet-20241022") -> Optional[dict]:
    """Anthropic Claude API를 사용하여 로직 트리를 생성한다.
    
    Args:
        prompt: 생성 프롬프트
        model: 사용할 Claude 모델
    
    Returns:
        파싱된 JSON 딕셔너리 또는 None
    """
    if not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        return None
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        message = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = message.content[0].text
        
        # JSON 파싱 시도
        # ```json ... ``` 형식 처리
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()
        
        return json.loads(json_str)
        
    except ImportError:
        print("anthropic 패키지가 설치되지 않았습니다: pip install anthropic")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        print(f"응답 원문: {response_text[:500]}...")
        return None
    except Exception as e:
        print(f"API 호출 오류: {e}")
        return None


def generate_with_openai(prompt: str, model: str = "gpt-4o") -> Optional[dict]:
    """OpenAI GPT API를 사용하여 로직 트리를 생성한다.
    
    Args:
        prompt: 생성 프롬프트
        model: 사용할 GPT 모델
    
    Returns:
        파싱된 JSON 딕셔너리 또는 None
    """
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY가 설정되지 않았습니다.")
        return None
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert strategy consultant."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        return json.loads(response_text)
        
    except ImportError:
        print("openai 패키지가 설치되지 않았습니다: pip install openai")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return None
    except Exception as e:
        print(f"API 호출 오류: {e}")
        return None


def validate_mece(tree: dict) -> dict:
    """생성된 트리의 MECE 원칙 준수 여부를 검증한다.
    
    Args:
        tree: 로직 트리 딕셔너리
    
    Returns:
        검증 결과 딕셔너리
    """
    def count_nodes(node, depth=0):
        count = 1
        max_depth = depth
        if "children" in node and node["children"]:
            for child in node["children"]:
                child_count, child_depth = count_nodes(child, depth + 1)
                count += child_count
                max_depth = max(max_depth, child_depth)
        return count, max_depth
    
    def find_leaves(node, leaves=None):
        if leaves is None:
            leaves = []
        if "children" not in node or not node["children"]:
            leaves.append(node["name"])
        else:
            for child in node["children"]:
                find_leaves(child, leaves)
        return leaves
    
    root = tree.get("root", tree)
    total_nodes, max_depth = count_nodes(root)
    leaves = find_leaves(root)
    
    return {
        "total_nodes": total_nodes,
        "max_depth": max_depth,
        "leaf_count": len(leaves),
        "leaves": leaves,
        "ai_mece_check": tree.get("mece_check", {})
    }


def generate_logic_tree(
    problem: str,
    tree_type: str = "why",
    provider: str = "anthropic"
) -> Optional[dict]:
    """문제에 대한 로직 트리를 생성한다.
    
    Args:
        problem: 분석할 문제/질문
        tree_type: "why" 또는 "how"
        provider: "anthropic" 또는 "openai"
    
    Returns:
        생성된 로직 트리 딕셔너리
    """
    prompt = create_logic_tree_prompt(problem, tree_type)
    
    if provider == "anthropic":
        result = generate_with_anthropic(prompt)
    elif provider == "openai":
        result = generate_with_openai(prompt)
    else:
        print(f"지원하지 않는 provider: {provider}")
        return None
    
    if result:
        validation = validate_mece(result)
        result["validation"] = validation
    
    return result


# 데모용 샘플 트리 (API 키 없이도 실행 가능)
SAMPLE_WHY_TREE = {
    "problem": "스타트업 A사의 성장 둔화 원인 분석",
    "tree_type": "why",
    "root": {
        "name": "왜 성장이 둔화되었는가?",
        "children": [
            {
                "name": "매출 성장 둔화",
                "children": [
                    {
                        "name": "신규 고객 획득 감소",
                        "children": [
                            {"name": "마케팅 ROI 하락"},
                            {"name": "경쟁 심화"}
                        ]
                    },
                    {
                        "name": "기존 고객 매출 정체",
                        "children": [
                            {"name": "업셀링 부진"},
                            {"name": "고객 이탈 증가"}
                        ]
                    }
                ]
            },
            {
                "name": "비용 구조 악화",
                "children": [
                    {
                        "name": "고정비 증가",
                        "children": [
                            {"name": "인건비 상승"},
                            {"name": "임대료 증가"}
                        ]
                    },
                    {
                        "name": "변동비 증가",
                        "children": [
                            {"name": "원자재비 상승"},
                            {"name": "물류비 증가"}
                        ]
                    }
                ]
            }
        ]
    },
    "mece_check": {
        "mutually_exclusive": True,
        "collectively_exhaustive": True,
        "notes": "매출과 비용으로 성장 둔화를 MECE하게 분해. 각 하위 분기도 중복 없이 전체를 커버."
    }
}


if __name__ == "__main__":
    print("=" * 60)
    print("AI 기반 로직 트리 생성")
    print("=" * 60)
    
    # 분석할 문제 정의
    problem = "스타트업 A사의 성장이 둔화되고 있다. 지난 2분기 연속 매출 성장률이 전년 대비 절반으로 떨어졌다."
    
    print(f"\n[문제] {problem}")
    
    # API 키 확인
    if ANTHROPIC_API_KEY or OPENAI_API_KEY:
        provider = "anthropic" if ANTHROPIC_API_KEY else "openai"
        print(f"\n[1] {provider.upper()} API를 사용하여 Why Tree 생성 중...")
        
        result = generate_logic_tree(problem, tree_type="why", provider=provider)
        
        if result:
            print("\n[생성된 로직 트리]")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            # 결과 저장
            output_path = "../data/ai_generated_why_tree.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n결과가 {output_path}에 저장되었습니다.")
    else:
        print("\n[참고] API 키가 설정되지 않아 샘플 트리를 사용합니다.")
        print("API를 사용하려면 ANTHROPIC_API_KEY 또는 OPENAI_API_KEY 환경 변수를 설정하세요.")
        
        print("\n[샘플 Why Tree]")
        print(json.dumps(SAMPLE_WHY_TREE, ensure_ascii=False, indent=2))
        
        # 검증
        validation = validate_mece(SAMPLE_WHY_TREE)
        print("\n[MECE 검증 결과]")
        print(f"  - 총 노드 수: {validation['total_nodes']}")
        print(f"  - 최대 깊이: {validation['max_depth']}")
        print(f"  - 리프 노드 수: {validation['leaf_count']}")
        print(f"  - 리프 노드: {validation['leaves']}")
        print(f"  - AI MECE 평가: {validation['ai_mece_check']}")
        
        # 샘플 결과 저장
        output_path = "../data/sample_why_tree.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(SAMPLE_WHY_TREE, f, ensure_ascii=False, indent=2)
        print(f"\n샘플 트리가 {output_path}에 저장되었습니다.")
    
    print("\n" + "=" * 60)
    print("완료")
    print("=" * 60)
