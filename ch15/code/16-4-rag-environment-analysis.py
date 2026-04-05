#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16.4.1 환경 분석 에이전트와 RAG 시스템

이 스크립트는 다음을 구현합니다:
1. 기획 문서 로딩 및 전처리 (Markdown, PDF 지원)
2. LangChain을 사용한 문서 청크 분할
3. OpenAI Embeddings + FAISS 벡터 데이터베이스 구축
4. Semantic Search를 통한 검색
5. RAG 기반 질의응답 (LLM + 검색 결과 통합)
6. 신뢰도 평가 및 출처 추적

실행 환경:
- Python 3.10+
- 필요 라이브러리: langchain, langchain-openai, faiss-cpu, tiktoken
- 환경 변수: OPENAI_API_KEY 필요

주의:
- OpenAI API 키가 없으면 일부 기능만 시연 (embedding/LLM 제외)
- 샘플 데이터로 기획 문서 3개 제공
"""

import os
import numpy as np
from pathlib import Path

try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_classic.chains import RetrievalQA
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"⚠️  LangChain 라이브러리 import 실패: {e}")
    print("설치: pip install langchain langchain-openai langchain-community faiss-cpu tiktoken")


def check_api_key():
    """
    OpenAI API 키 확인

    Returns:
        bool: API 키 존재 여부
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  OpenAI API 키가 설정되지 않았습니다.")
        print("환경 변수 OPENAI_API_KEY를 설정하거나 .env 파일을 사용하세요.")
        print()
        print("예시:")
        print("  export OPENAI_API_KEY='your-api-key-here'  # macOS/Linux")
        print("  set OPENAI_API_KEY=your-api-key-here       # Windows")
        print()
        return False
    return True


def load_documents(data_path="../data/sample_strategy_docs"):
    """
    기획 문서 로딩

    Args:
        data_path: 문서가 저장된 경로

    Returns:
        documents: LangChain Document 객체 리스트
    """
    print("[1단계] 기획 문서 로딩")
    print("=" * 70)

    if not Path(data_path).exists():
        print(f"⚠️  경로가 존재하지 않습니다: {data_path}")
        return []

    # Markdown 파일 로더
    loader = DirectoryLoader(
        data_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()
    print(f"총 문서: {len(documents)}개")

    for i, doc in enumerate(documents):
        filename = Path(doc.metadata["source"]).name
        doc_length = len(doc.page_content)
        print(f"  [{i+1}] {filename} ({doc_length:,} 문자)")

    print()
    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    문서를 청크(Chunk)로 분할

    Args:
        documents: LangChain Document 객체 리스트
        chunk_size: 청크당 토큰 수 (대략 500자 = 125 토큰)
        chunk_overlap: 청크 간 중복 토큰 수

    Returns:
        chunks: 분할된 Document 청크 리스트
    """
    print("[2단계] 문서 청크 분할")
    print("=" * 70)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    total_chars = sum(len(doc.page_content) for doc in documents)
    avg_chunk_size = np.mean([len(chunk.page_content) for chunk in chunks])

    print(f"청크 수: {len(chunks)}개")
    print(f"평균 청크 크기: {avg_chunk_size:.0f}자")
    print(f"전체 문자 수: {total_chars:,}자")
    print()

    return chunks


def build_vectorstore(chunks, embeddings_model=None):
    """
    FAISS 벡터 데이터베이스 구축

    Args:
        chunks: 문서 청크 리스트
        embeddings_model: OpenAI Embeddings 모델

    Returns:
        vectorstore: FAISS 벡터 저장소
    """
    print("[3단계] 벡터 데이터베이스 구축")
    print("=" * 70)

    if embeddings_model is None:
        print("⚠️  Embeddings 모델이 없어 벡터 DB를 구축할 수 없습니다.")
        return None

    print("벡터 임베딩 생성 중... (OpenAI API 호출)")
    vectorstore = FAISS.from_documents(chunks, embeddings_model)

    print(f"✅ FAISS 벡터 DB 구축 완료")
    print(f"  - 벡터 개수: {len(chunks)}개")
    print(f"  - 임베딩 차원: {vectorstore.index.d}차원")
    print()

    return vectorstore


def search_documents(vectorstore, query, top_k=5):
    """
    Semantic Search 수행

    Args:
        vectorstore: FAISS 벡터 저장소
        query: 검색 질의
        top_k: 반환할 문서 개수

    Returns:
        results: 검색된 문서 리스트 (Document, score 튜플)
    """
    print(f"[4단계] Semantic Search: '{query}'")
    print("=" * 70)

    if vectorstore is None:
        print("⚠️  벡터 DB가 없어 검색을 수행할 수 없습니다.")
        return []

    # Similarity search with scores
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    print(f"검색 결과: {len(results)}개 문서")
    for i, (doc, score) in enumerate(results):
        filename = Path(doc.metadata["source"]).name
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{i+1}] {filename} (유사도: {1-score:.3f})")
        print(f"      {preview}...")

    print()
    return results


def rag_query(vectorstore, llm, query):
    """
    RAG 기반 질의응답

    Args:
        vectorstore: FAISS 벡터 저장소
        llm: LLM 모델 (ChatOpenAI)
        query: 사용자 질문

    Returns:
        answer: LLM 생성 답변
        source_docs: 참조한 문서들
    """
    print(f"[5단계] RAG 질의응답: '{query}'")
    print("=" * 70)

    if vectorstore is None or llm is None:
        print("⚠️  벡터 DB 또는 LLM이 없어 RAG를 수행할 수 없습니다.")
        return None, []

    # Retrieval QA Chain 설정
    template = """다음 기획 문서를 참고하여 질문에 답변하세요.
답변은 한국어로 작성하고, 참조한 문서의 내용을 명시하세요.
문서에 없는 내용은 "문서에 해당 정보가 없습니다"라고 답하세요.

맥락:
{context}

질문: {question}

답변:"""

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # 질의 실행
    print("LLM 답변 생성 중... (OpenAI API 호출)")
    result = qa_chain.invoke({"query": query})

    answer = result["result"]
    source_docs = result["source_documents"]

    print(f"\n【답변】")
    print(answer)
    print(f"\n【참조 문서】")
    for i, doc in enumerate(source_docs):
        filename = Path(doc.metadata["source"]).name
        print(f"  [{i+1}] {filename}")

    print()
    return answer, source_docs


def evaluate_confidence(answer, source_docs):
    """
    신뢰도 평가 (간단한 휴리스틱)

    Args:
        answer: LLM 답변
        source_docs: 참조 문서

    Returns:
        confidence: 신뢰도 점수 (0-5)
    """
    print("[6단계] 신뢰도 평가")
    print("=" * 70)

    # 휴리스틱 기반 신뢰도 계산
    confidence = 5.0  # 초기값

    # 1. 참조 문서 개수
    if len(source_docs) == 0:
        confidence -= 2.0

    # 2. 답변에 "없습니다", "모르겠습니다" 포함 시
    if any(phrase in answer for phrase in ["없습니다", "모르겠습니다", "확인할 수 없습니다"]):
        confidence -= 1.5

    # 3. 답변 길이
    if len(answer) < 50:
        confidence -= 0.5

    confidence = max(0.0, min(5.0, confidence))

    print(f"신뢰도 점수: {confidence:.1f}/5.0")

    if confidence >= 4.0:
        print("→ 높은 신뢰도: 참조 문서에 명확한 근거 있음")
    elif confidence >= 2.5:
        print("→ 중간 신뢰도: 일부 근거 있으나 불확실성 존재")
    else:
        print("→ 낮은 신뢰도: 참조 문서에 명확한 정보 부족")

    print()
    return confidence


def demo_without_api():
    """
    OpenAI API 없이 RAG 개념 시연 (임베딩/LLM 제외)
    """
    print("\n" + "=" * 70)
    print("16.4.1 환경 분석 에이전트와 RAG 시스템 (데모 모드)")
    print("=" * 70)
    print()
    print("⚠️  OpenAI API 키가 없어 임베딩/LLM 기능은 건너뜁니다.")
    print("실제 구현에서는 다음 단계가 추가됩니다:")
    print()
    print("[데모 불가 단계]")
    print("  - 벡터 임베딩 생성 (OpenAI text-embedding-3-small)")
    print("  - FAISS 벡터 DB 구축")
    print("  - Semantic Search")
    print("  - LLM 기반 답변 생성 (GPT-4)")
    print()
    print("[RAG 워크플로우 개념]")
    print("  1. 문서 로딩 → 청크 분할")
    print("  2. 임베딩 → 벡터 DB 저장")
    print("  3. 질문 임베딩 → 유사 문서 검색")
    print("  4. 검색 결과 + 질문 → LLM → 답변 생성")
    print("  5. 출처 추적 및 신뢰도 평가")
    print()


def main():
    """
    메인 실행 함수
    """
    if not LANGCHAIN_AVAILABLE:
        print("\n❌ LangChain이 설치되지 않아 실행할 수 없습니다.")
        return

    print("\n" + "=" * 70)
    print("16.4.1 환경 분석 에이전트와 RAG 시스템")
    print("=" * 70)
    print()

    # API 키 확인
    has_api_key = check_api_key()

    if not has_api_key:
        demo_without_api()
        return

    # 1. 문서 로딩
    documents = load_documents("../data/sample_strategy_docs")

    if not documents:
        print("❌ 문서를 로딩할 수 없습니다. 샘플 데이터를 생성하세요.")
        return

    # 2. 청크 분할
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)

    # 3. 벡터 DB 구축
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = build_vectorstore(chunks, embeddings)

    # 4. Semantic Search 예시
    query1 = "경쟁사 디지털 전환 전략은 무엇인가?"
    results1 = search_documents(vectorstore, query1, top_k=5)

    # 5. RAG 질의응답 예시
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    query2 = "프로젝트 A의 주요 위험 요인은 무엇인가?"
    answer2, source_docs2 = rag_query(vectorstore, llm, query2)

    # 6. 신뢰도 평가
    if answer2:
        confidence = evaluate_confidence(answer2, source_docs2)

    print("=" * 70)
    print("✅ 실행 완료!")
    print("=" * 70)
    print()
    print("[핵심 메시지]")
    print("1. RAG는 LLM의 환각을 방지하고 신뢰성을 높임")
    print("2. 기획 업무: 과거 문서 검색 → 유사 사례 참고 → 의사결정 지원")
    print("3. 출처 추적으로 투명성 확보 (규제 산업에서 중요)")
    print("4. 벡터 DB (FAISS) + LangChain으로 빠른 프로토타이핑 가능")
    print()


if __name__ == "__main__":
    main()
