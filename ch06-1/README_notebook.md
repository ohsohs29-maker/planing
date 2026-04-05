# 제7-1장: AI 기반 환경 스캐닝 (Jupyter Notebook)

## 파일 정보
- **파일명**: `ch07-1_lecture.ipynb`
- **크기**: 37KB
- **총 셀**: 31개 (Markdown 13개, Code 18개)
- **대상**: 3시간 강의 (Part 1~5)
- **시작일**: 2026-02-25

---

## 강의 구조 (3시간)

| 시간 | 구분 | 내용 | 셀 |
|------|------|------|-----|
| 0:00-0:40 | Part 1 | 환경분석의 목적과 전통적 프레임워크의 한계 | 0-6 |
| 0:40-1:15 | Part 2 | AI 기반 트렌드 탐지: 약신호, 감성 분석, LLM | 7-15 |
| 1:15-1:30 | 휴식 | | 16 |
| 1:30-2:00 | Part 3 | 경쟁 인텔리전스: 특허, 채용, 제품 분석 | 17-22 |
| 2:00-2:30 | Part 4 | 실습: 뉴스 트렌드 분석 & 약신호 탐지 | 23-25 |
| 2:30-3:00 | Part 5 | 실습: 경쟁사 프로파일링 & 데이터 기반 SWOT | 26-30 |

---

## 주요 학습 목표

1. ✅ 전통적 환경분석 프레임워크(PEST, SWOT, 5 Forces)의 역할과 한계 이해
2. ✅ AI 기반 트렌드 탐지와 약신호(Weak Signal) 포착 원리 습득
3. ✅ 뉴스 분석, 감성 분석, 이상 탐지 코드 구현
4. ✅ 경쟁 인텔리전스(특허, 채용, 제품) 자동화 방법 학습
5. ✅ 데이터 기반 SWOT 작성 및 기회/위협 정량화

---

## 셀별 세부 구성

### Part 1: 환경분석의 기초 (셀 0-6)

**이론 콘텐츠:**
- 환경분석의 세 가지 목적 (기회/위협 식별, 의사결정 지원, 미래 변화 대비)
- 전통적 프레임워크 비교 (PEST, 5 Forces, SWOT, Value Chain)
- 전통적 접근의 4가지 한계 (정적, 주관, 정보처리, 정량화)

**시각화:**
- 레이더 차트: Traditional vs AI-Based Environmental Analysis
- 패러다임 전환 테이블

**과제:**
- 이론 과제 7-1-1: PEST 상호작용, 5 Forces 한계, Ansoff의 약신호
- 제출란 7-1-1

---

### Part 2: AI 기반 트렌드 탐지 (셀 7-15)

**이론 콘텐츠:**
- 약신호(Weak Signal) 정의 및 중요성 (Ansoff 1975)
- AI가 약신호 탐지에 강한 이유 (대량 처리, 패턴 탐지, 편향 제거)
- 감성 분석(Sentiment Analysis) 접근법 (사전 기반, 기계학습, LLM)
- LLM 활용 시 환각(Hallucination) 문제 및 해결책

**코드 시뮬레이션:**
- **셀 8**: 배터리 산업 키워드 트렌드 분석 (180일 시뮬레이션)
  - 6개 키워드: Lithium-ion, Solid-state, LFP, Sodium-ion, Recycling, Supply Chain
  - 트렌드 타입: stable, rising, emerging, volatile
  - 주별 집계 및 라인 차트 시각화

- **셀 9**: 약신호 탐지 알고리즘
  - 이동평균 대비 비율 계산
  - 성장률 계산 (초기 vs 최근)
  - 약신호 판별 (threshold > 2.0 or growth > 100%)
  - 산점도 + 경계선 시각화

- **셀 11**: 감성 분석 시뮬레이션
  - 8개 키워드별 감성 점수 (-1~1)
  - 월별 감성 변동
  - 히트맵 시각화 (RdYlGn 컬러맵)

**과제:**
- 이론 과제 7-1-2: 약신호 사례, 감성 분석 활용, LLM 환각 위험
- 제출란 7-1-2

---

### Part 3: 경쟁 인텔리전스 (셀 17-22)

**이론 콘텐츠:**
- 경쟁 인텔리전스(CI) 정의
- 자동 추적 가능한 공개 정보원 6가지
- 채용 공고가 전략의 선행 지표인 이유

**코드 분석:**

- **셀 17**: 특허 동향 분석
  - 6개 기업: CATL, LG Energy, Samsung SDI, BYD, SK On, Panasonic
  - 6개 기술 분야: Solid-state, Li-ion Cathode, Si Anode, BMS, Recycling, Sodium-ion
  - 5년 시뮬레이션 (2021-2026)
  - 2개 시각화:
    1. 기업별 연간 특허 출원 추이 (라인 차트)
    2. 기술 분야별 최근 성장률 (수평 막대 차트)

- **셀 18**: 채용 공고 분석
  - 5개 직무 카테고리: R&D, Manufacturing, Sales/Marketing, Software/AI, Supply Chain
  - 기업별 채용 비중 분석
  - 기업별 AI/SW 비중 계산
  - 누적 막대 차트 + 전략 추론

**과제:**
- 이론 과제 7-1-3: 채용 공고 → 전략 예측, 특허 분석 → CI
- 제출란 7-1-3

---

### Part 4: 실습 1 - 트렌드 분석 (셀 23-25)

**실습 7-1-4: 트렌드 분석 함수 구현**
- TODO 1: 성장률 계산 함수 작성 (초기 4주 vs 최근 4주)
- TODO 2: 모든 키워드 성장률 계산
- TODO 3: 성장률 기반 막대 차트 시각화
- TODO 4: 결과 해석 및 인사이트 도출

**실습 7-1-5: 감성 분석 시각화**
- TODO 1: signal_df + avg_sentiment merge
- TODO 2: 성장률 × 감성 산점도 (버블 차트)
  - 사분면 해석: 기회/리스크/안정/위험
- TODO 3: 사분면 경계선 추가 (x=50, y=0)
- TODO 4: 핵심 기회, 주요 리스크 도출

---

### Part 5: 실습 2 - 경쟁 분석 (셀 26-30)

**셀 27: 경쟁사 기술 포트폴리오 분석**
- 최근 2년(2025-2026) 특허 기반 분석
- 4개 주요 기업별 기술 포트폴리오
- 레이더 차트 시각화 (6개 기술 분야)

**실습 7-1-6 & 7-1-7: 데이터 기반 SWOT**
- C사 배경 설정: 정밀화학 → 배터리 시장 진출 검토
- TODO 1: SWOT 매트릭스 정의 (데이터 근거 포함)
- TODO 2: Plotly Table로 SWOT 시각화
- TODO 3: 기회/위협 정량화 (영향도 × 확률 매트릭스)
- TODO 4: 전략적 시사점 도출
  - 타겟 시장 설정
  - 진출 방식 검토 (자체/JV/인수)
  - 차별화 포인트 선정

---

## 주요 특징

### 1. 이론과 실습의 균형
- 부분: 이론 50% + 코드 시각화 50%
- 학생: 조사 과제(이론) + 실습 과제(코드) 병행

### 2. 실제 데이터 기반 시뮬레이션
- 배터리 산업을 사례로 한 현실적 시나리오
- 6개월 뉴스 트렌드, 5년 특허 데이터, 채용 공고 분석
- 학생이 실제 업계 데이터 분석 경험

### 3. 크로스플랫폼 코드
- pathlib 활용 (경로 호환성)
- Plotly 사용 (한글 폰트 문제 없음)
- 모든 그래프 텍스트 영어 (한글 주석은 print 사용)

### 4. 조사 과제와 실습 과제의 명확한 구분
- **이론 과제**: 외부 리소스 검색 후 직접 타이핑 (이해도 증진)
- **실습 과제**: TODO 주석으로 단계별 구현 (GitHub Copilot 활용 가능)

### 5. 조기경보 체계 내재화
- 약신호 탐지 알고리즘 (이상 탐지)
- 감성 분석을 통한 여론 모니터링
- 경쟁사 움직임의 조기 포착

---

## 실행 방법

### VSCode에서 열기
```bash
cd /sessions/practical-brave-faraday/mnt/strategy/practice/chapter07-1
code ch07-1_lecture.ipynb
```

### Jupyter Notebook 서버 실행
```bash
jupyter notebook ch07-1_lecture.ipynb
```

### 주요 라이브러리
```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
```

---

## 예상 학습 시간

| 구성 | 예상 시간 |
|------|----------|
| Part 1 이론 | 40분 |
| Part 2 이론 + 코드 | 35분 |
| 휴식 | 15분 |
| Part 3 이론 + 코드 | 30분 |
| Part 4 실습 | 30분 |
| Part 5 실습 | 30분 |
| **합계** | **3시간** |

---

## 다음 단계

**제7-2장: 이해관계자와 네트워크 분석**
- 환경 분석에서 "누가 관련되어 있는가" 파악
- 이해관계자 맵핑
- 권력-이해 매트릭스
- 사회적 네트워크 분석(SNA)

---

## 참고 자료

### 필수 문헌
- Ansoff, H.I. (1975). Managing Strategic Surprise by Response to Weak Signals. *California Management Review*, 18(2), 21-33.
- Porter, M.E. (1985). *Competitive Advantage*. Free Press.
- Day, G.S. & Schoemaker, P.J. (2019). *See Sooner, Act Faster*. MIT Press.
- Fleisher, C.S. & Bensoussan, B.E. (2015). *Business and Competitive Analysis*. FT Press.

### Python 도구
- NewsAPI (뉴스 수집)
- VADER (감성분석)
- BERTopic (토픽 모델링)

### 특허 검색
- Google Patents (patents.google.com)
- USPTO (uspto.gov)
- KIPRIS (kipris.kipo.go.kr)

---

**생성일**: 2026-02-25
**버전**: 1.0
**상태**: 완성 (31개 셀, 37KB)
