# 정책분석과 기획 (Policy Analysis and Strategic Planning)

> **한신대학교 공공인재빅데이터융합전공**  
> AI 증강 기획과 데이터 기반 정책분석 교육 자료

## 📚 과목 소개

이 저장소는 전통적인 기획 방법론과 최신 AI/데이터 분석 기법을 융합한 정책분석 및 전략기획 과목의 강의 자료를 포함합니다. 각 챕터는 이론 강의와 Python 기반 실습을 통해 실무 적용 능력을 함양합니다.

### 주요 특징
- ✅ **AI 증강 기획**: LLM과 데이터 분석을 활용한 현대적 기획 방법론
- ✅ **실습 중심**: Jupyter Notebook을 통한 인터랙티브 학습
- ✅ **공간정보 분석**: 위성영상, GIS 데이터를 활용한 정책 분석 사례
- ✅ **정량적 의사결정**: 몬테카를로 시뮬레이션, 베이지안 분석 등 통계적 기법

---

## 📂 저장소 구조

```
planing/
├── ch01/          # 기획의 정의와 프로세스
├── ch02/          # AI 증강 기획
├── ch03/          # 구조적 사고 (MECE, 로직 트리)
├── ch04/          # 이슈 정의와 프레이밍
├── ch05/          # 인과추론 기초
├── ch06/          # 시스템 다이내믹스
├── ch07-1/        # AI 기반 환경 스캐닝
├── ch07-2/        # 이해관계자 분석과 네트워크
├── ch08/          # 베이지안 실험설계와 정책 평가
├── ch09/          # 베이지안 의사결정
├── ch10/          # 시나리오 플래닝
├── ch11/          # 몬테카를로 시뮬레이션
├── ch12/          # 다기준 의사결정
├── ch13/          # 실물옵션 분석
├── ch14/          # 실행 계획과 리스크 관리
├── ch15/          # 모니터링과 적응적 기획
├── ch16/          # AI 에이전트와 기획 워크플로우
└── tools/         # 참고 자료
```

각 챕터 폴더는 다음 구조를 따릅니다:
```
ch**/
├── ch**.ipynb           # 강의 노트북 (3시간 분량)
├── code/                # Python 실습 코드
│   ├── *-*.py          # 개별 실습 스크립트
│   └── requirements.txt # 필요 패키지 목록
└── data/                # 실습 데이터 (해당 챕터만)
```

---

## 📖 커리큘럼

### Part 1: 기획의 기초 (1-4장)

| 챕터 | 제목 | 주요 내용 |
|------|------|----------|
| **01** | **기획이란 무엇인가?** | 기획의 정의, 6단계 프로세스, 전통적 프레임워크의 한계 |
| **02** | **AI 증강 기획** | AI가 기획을 증강하는 4가지 방식, Human-in-the-Loop 원칙 |
| **03** | **구조적 사고** | MECE 원칙, Why Tree/How Tree, 피라미드 원칙 |
| **04** | **이슈 정의와 프레이밍** | Problem vs Issue vs Task, SCQA 프레임워크, 우선순위화 |

### Part 2: 인과 분석과 시스템 사고 (5-6장)

| 챕터 | 제목 | 주요 내용 |
|------|------|----------|
| **05** | **인과추론 기초** | 상관관계 vs 인과관계, DAG, do-연산자, 준실험 설계 |
| **06** | **시스템 다이내믹스** | 피드백 루프, 스톡-플로우, CLD, 레버리지 포인트 |

### Part 3: 환경 분석과 이해관계자 (7장)

| 챕터 | 제목 | 주요 내용 |
|------|------|----------|
| **07-1** | **AI 기반 환경 스캐닝** | 트렌드 탐지, 약신호 포착, 경쟁 인텔리전스, 위성영상 분석 |
| **07-2** | **이해관계자 네트워크 분석** | 네트워크 분석, SAM 모델, YOLO+SAM 파이프라인 |

### Part 4: 정책 평가와 의사결정 (8-9장)

| 챕터 | 제목 | 주요 내용 |
|------|------|----------|
| **08** | **베이지안 실험설계와 정책 평가** | 사전-사후분포, Thompson Sampling, 베이즈 팩터 |
| **09** | **베이지안 의사결정** | 베이즈 정리, 순차적 업데이트, EVPI/EVII |

### Part 5: 미래 탐색과 불확실성 (10-11장)

| 챕터 | 제목 | 주요 내용 |
|------|------|----------|
| **10** | **시나리오 플래닝** | STEEP 프레임워크, 2×2 시나리오 매트릭스, 윈드 터널링 |
| **11** | **몬테카를로 시뮬레이션** | 확률 분포, 리스크 정량화, 민감도 분석 |

### Part 6: 의사결정과 옵션 (12-13장)

| 챕터 | 제목 | 주요 내용 |
|------|------|----------|
| **12** | **다기준 의사결정** | 가중 점수법, AHP, 의사결정 나무, 위성영상 기반 재해 분석 |
| **13** | **실물옵션 분석** | 유연성의 가치, 이항 모델, 단계적 투자, GEE 활용 |

### Part 7: 실행과 모니터링 (14-16장)

| 챕터 | 제목 | 주요 내용 |
|------|------|----------|
| **14** | **실행 계획과 리스크 관리** | OKR, SMART, WBS, RACI, 리스크 레지스터 |
| **15** | **모니터링과 적응적 기획** | 선행/후행 지표, Balanced Scorecard, 이상 탐지, AAR |
| **16** | **AI 에이전트와 기획 워크플로우** | 프롬프트 엔지니어링, ReAct 패턴, RAG |

---

## 🚀 시작하기

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/LeeSeogMin/planing.git
cd planing

# 가상환경 생성 및 활성화
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 각 챕터별 패키지 설치 (예: ch01)
pip install -r ch01/code/requirements.txt
```

### 2. Jupyter 커널 등록

```bash
pip install ipykernel
python -m ipykernel install --user --name=strategy-lecture --display-name="AI 기획 강의 (Python 3)"
```

### 3. Jupyter Notebook 실행

```bash
# Jupyter Lab 실행
jupyter lab

# 또는 VS Code에서 .ipynb 파일 직접 열기
code ch01/ch01.ipynb
```

**VS Code 사용 시:**
- 우측 상단에서 커널을 `AI 기획 강의 (Python 3)` 또는 `.venv` 로 선택
- 각 셀을 순차적으로 실행

---

## 📦 주요 의존성

### 기본 라이브러리
- **Python 3.10+**
- `numpy`, `pandas` - 데이터 분석
- `matplotlib`, `plotly` - 시각화
- `scipy` - 통계 분석
- `networkx` - 네트워크 분석

### 공간정보 분석 (ch02, ch07-1, ch07-2, ch12, ch13)
- `geopandas` - 공간 데이터 처리
- `rasterio`, `earthengine-api` - 위성영상 분석
- `pystac-client` - STAC API 검색
- `leafmap`, `localtileserver` - 지도 시각화

### AI/머신러닝 (ch07-1, ch07-2, ch16)
- `openai`, `anthropic` - LLM API
- `langchain` - AI 에이전트 프레임워크
- `chromadb` - 벡터 데이터베이스 (RAG)
- `ultralytics` - YOLO 객체 탐지
- `segment-anything` - SAM 모델

### 통계/시뮬레이션 (ch08, ch09, ch11, ch13)
- `pymc` - 베이지안 추론
- `dowhy` - 인과추론
- `scikit-learn` - 머신러닝

### 네트워크/시스템 (ch06, ch07-2)
- `networkx` - 네트워크 분석
- `simpy` - 이산사건 시뮬레이션

---

## 💡 학습 방법

### 각 챕터의 학습 흐름 (3시간)
```
Part 1 (0:00-0:40)  이론 학습 + 조사 과제
Part 2 (0:40-1:15)  심화 이론 + 조사 과제
휴식    (1:15-1:30)  15분 휴식
Part 3 (1:30-2:00)  응용 이론 + 조사 과제
Part 4 (2:00-2:30)  코드 실습 1
Part 5 (2:30-3:00)  코드 실습 2 (종합 실습)
```

### 실습 파일 활용
- **Jupyter Notebook** (`ch**.ipynb`): 강의 전체 내용 (이론 + 실습)
- **Python 스크립트** (`code/*.py`): 독립 실행 가능한 실습 코드
- **데이터** (`data/`): 실습용 샘플 데이터

---

## 🎯 핵심 실습 프로젝트

| 챕터 | 프로젝트 | 기술 스택 |
|------|---------|-----------|
| **ch02** | 서울시 녹지 변화 분석 | Sentinel-2, NDVI, 공간 연산 |
| **ch03** | 스타트업 성장 둔화 로직 트리 분석 | MECE, Why/How Tree |
| **ch04** | 도시 이슈 우선순위 매트릭스 | 공간 자기상관, 군집 분석 |
| **ch05** | 녹지와 대기질 인과 분석 | DAG, 교란변수 통제 |
| **ch06** | 재고 관리 시스템 시뮬레이션 | 스톡-플로우, CLD |
| **ch07-1** | 뉴스 트렌드와 약신호 탐지 | UNET, 위성영상 분류 |
| **ch07-2** | 이해관계자 네트워크 분석 | YOLO+SAM, 객체 탐지·분할 |
| **ch08** | K대학 장학금 정책 효과 추정 | 베이지안 추론, Thompson Sampling |
| **ch09** | 시장 진입 의사결정 분석 | EVPI, EVII |
| **ch10** | 2030 모빌리티 시나리오 | 윈드 터널링, 전략 평가 |
| **ch11** | 신규 사업 NPV 리스크 분석 | 몬테카를로, 민감도 분석 |
| **ch12** | 마우이 산불 피해 분석 | Sentinel-2, 번번 강도 분석 |
| **ch13** | 서울시 NDVI 시계열 분석 | Google Earth Engine, 월별 식생 지수 |
| **ch14** | 프로젝트 일정 리스크 시뮬레이션 | PERT, 몬테카를로 |
| **ch15** | KPI 대시보드와 이상 탐지 | 시계열 분석, Z-score |
| **ch16** | RAG 기반 전략 문서 분석 | LangChain, ChromaDB |

---

## 🌍 공간정보 활용 사례

본 과정은 정책분석에서 공간정보의 중요성을 강조하며, 다음과 같은 실제 사례를 다룹니다:

### 위성영상 분석
- **Sentinel-2 활용**: NDVI 시계열 분석, 토지피복 분류, 재해 탐지
- **HLS(Harmonized Landsat Sentinel-2)**: 고빈도 시계열 데이터
- **Google Earth Engine**: 대규모 지리공간 데이터 처리

### 공간 분석 기법
- 버퍼 분석, 공간 조인, 좌표 변환
- 공간 자기상관 (Moran's I, LISA)
- 래스터 연산 (밴드 연산, 마스킹, 타일링)

### AI 기반 객체 탐지/분할
- **YOLO**: 위성영상/항공사진에서 객체 탐지
- **SAM (Segment Anything Model)**: 자동 분할
- **U-NET**: 토지피복 시맨틱 세그멘테이션

---

## 📚 참고 자료

### 강의 관련 도구
- [GitHub Copilot 가이드](tools/github_copilot_guide.md)

### 추천 도서
- *Strategic Thinking* (Rich Horwath)
- *The McKinsey Way* (Ethan Rasiel)
- *Scenario Planning* (Woody Wade)
- *Real Options Analysis* (Tom Copeland)
- *Thinking in Systems* (Donella Meadows)

### 온라인 리소스
- [Google Earth Engine](https://earthengine.google.com/)
- [STAC (SpatioTemporal Asset Catalog)](https://stacspec.org/)
- [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
- [LangChain Documentation](https://python.langchain.com/)

---

## 🤝 기여 및 문의

### 교수자
- 한신대학교 공공인재빅데이터융합전공

### 문의사항
- GitHub Issues를 통해 버그 리포트 및 개선 제안
- 실습 데이터 또는 코드 관련 질문은 각 챕터의 `code/` 폴더 내 코드 주석 참고

---

## 📄 라이선스

본 교육 자료는 학습 목적으로 제공됩니다. 상업적 사용 시 저작권자의 사전 동의가 필요합니다.

---

## 🔄 업데이트 이력

- **2026-03-03**: 초기 저장소 구성
  - 폴더명 변경: `chapter*` → `ch*`
  - 노트북 파일명 정리: `ch*_lecture.ipynb` → `ch*.ipynb`
  - GitHub 저장소 연동

---

**© 2026 한신대학교 공공인재빅데이터융합전공. All rights reserved.**
