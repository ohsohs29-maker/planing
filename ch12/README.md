# Chapter 12: 다기침 의사결정 (Multi-Criteria Decision Making)

## 파일 위치
- **Notebook**: `ch12_lecture.ipynb`
- **이 파일**: `README.md`

## 강의 개요

### 학습 목표
1. 인지 편향과 구조화된 의사결정의 필요성 이해
2. 가중 점수법(Weighted Scoring)으로 대안 평가
3. AHP(Analytic Hierarchy Process)로 체계적 가중치 도출
4. 의사결정 나무와 정보의 가치(EVPI, EVII) 계산
5. 민감도 분석으로 의사결정 강건성 확보

### 강의 구조 (3시간)

| 파트 | 시간 | 내용 |
|------|------|------|
| Part 1 | 0:00-0:40 | 의사결정의 본질과 인지 편향 |
| Part 2 | 0:40-1:15 | 가중 점수법과 AHP |
| 휴식 | 1:15-1:30 | - |
| Part 3 | 1:30-2:00 | 불확실성하 의사결정과 정보의 가치 |
| Part 4 | 2:00-2:30 | 실습: 가중 점수법 (Smart City 프로젝트) |
| Part 5 | 2:30-3:00 | 실습: 의사결정 나무와 EVPI (신제품 출시) |

## Notebook 구성 (22 cells)

### 이론 부분
- **Cell 1**: 제목 및 학습 목표
- **Cell 2**: 라이브러리 임포트 (numpy, pandas, plotly)
- **Cell 3**: Part 1 - 의사결정의 본질과 인지 편향 (Herbert Simon, 6가지 편향, 집단사고)
- **Cell 4-5**: 이론 과제 12-1 및 제출란 (확증편향, Premortem, 조직 사례)
- **Cell 6**: Part 2 - 가중 점수법과 AHP 개념
- **Cell 7**: 코드 - 가중 점수법 분석 (국제진출 4개 대안 평가)
- **Cell 8**: 코드 - AHP 분석 (쌍대 비교, 일관성 검증)
- **Cell 9-10**: 이론 과제 12-2 및 제출란 (일관성 비율, 결과 비교, 실무 주의점)
- **Cell 11**: 휴식 안내
- **Cell 12**: Part 3 - 불확실성하 의사결정 (EMV, Decision Tree, EVPI, EVII)
- **Cell 13**: 코드 - 의사결정 나무와 EMV (3가지 시장상태, 4개 대안)
- **Cell 14**: 코드 - EVPI와 EVII 분석 (베이즈 정리, 정보 효율성)
- **Cell 15**: 코드 - 민감도 분석 (가중치 변화에 따른 순위 추적)
- **Cell 16-17**: 이론 과제 12-3 및 제출란 (EVPI vs EVII, 시장 조사 가치, 정보 효율성 낮을 때 전략)
- **Cell 18**: Part 4 실습 안내
- **Cell 19**: 실습 1 - 가중 점수법 (Smart City 4개 프로젝트)
- **Cell 20**: Part 5 실습 안내
- **Cell 21**: 실습 2 - 의사결정 나무 (신제품 3가지 시장상태)
- **Cell 22**: 강의 마무리 및 핵심 요약

## 핵심 내용 요약

### Part 1: 의사결정의 본질과 인지 편향

**Herbert Simon의 "제한된 합리성"**
- 완전한 정보와 무한한 인지능력을 가진 의사결정자 가정 불가능
- 우리는 최적(Optimize)하기보다는 만족(Satisfice)한다

**6가지 주요 인지 편향**

| 편향 | 의미 | 대응책 |
|------|------|--------|
| 확증편향 | 자신의 믿음을 지지하는 정보만 수집 | Red Team, Devil's Advocate |
| 앵커링 편향 | 처음 제시된 수치에 과도하게 의존 | 독립적 추정, 다중 기준점 |
| 과신편향 | 자신의 능력을 과대평가 | 사전 분석, 외부 벤치마킹 |
| 손실회피 | 이득보다 손실을 2배 이상 두려워함 | Expected Value 분석 |
| 가용성편향 | 최근/기억하기 쉬운 사건에 과대 가중 | 데이터 기반 통계 |
| 군집편향 | 합리성과 무관하게 다수를 따라감 | 독립적 분석 |

**집단사고(Groupthink) 방지법**
1. Premortem (Gary Klein): "프로젝트가 실패했다고 가정하고 원인 찾기"
2. Red Team: 의도적으로 다른 관점에서 비판
3. Delphi Method: 익명의 다회차 의견 수렴
4. 외부 전문가 초청

### Part 2: 가중 점수법과 AHP

**가중 점수법 (Weighted Scoring Method)**

$$W_i = \sum_{j=1}^{n} w_j \times s_{ij}$$

**4단계 프로세스**:
1. 기준 정의
2. 가중치 부여 (상대적 중요도)
3. 대안별 점수 매기기 (1-10점)
4. 가중 점수 계산

장점: 계산 간단, 직관적
단점: 가중치 결정의 자의성

**AHP (Analytic Hierarchy Process)**

**Saaty 9점 척도**:
- 1: 동등함
- 3: 약간 중요
- 5: 명확히 중요
- 7: 매우 중요
- 9: 절대적으로 중요

**Step 1-3**:
1. 쌍대 비교 행렬 구성
2. 우선순위 벡터 계산 (Column Normalization)
3. 일관성 검증: CR < 0.10이면 합리적

### Part 3: 불확실성하 의사결정과 정보의 가치

**기댓값 의사결정 (EMV)**

$$EMV(a) = \sum_{s=1}^{S} P(s) \times V(a, s)$$

**의사결정 나무 (Decision Tree)**
- 결정 노드 (□): 의사결정자의 선택
- 확률 노드 (○): 자연의 불확실한 상태
- 분석 방법: Rollback (역순 계산)

**정보의 가치**

1. **EVPI (Expected Value of Perfect Information)**
   $$EVPI = E[V|PI] - \max(EMV)$$
   정보 수집에 투자할 수 있는 최대 금액

2. **EVII (Expected Value of Imperfect Information)**
   $$EVII = E[V|\text{Signal}] - \max(EMV)$$
   실제 정보(예: 시장 조사)의 기댓값

3. **정보 효율성**
   $$\text{Efficiency} = \frac{EVII}{EVPI}$$
   - 0.5~0.8: 높은 가치의 정보 (투자 권장)
   - < 0.3: 낮은 가치의 정보 (투자 미권장)

## 사용된 데이터

### 가중 점수법 (국제진출 전략)
```
대안: 
  A: Domestic Franchise (국내 프랜차이즈)
  B: Foreign Direct Investment (해외 직접투자)
  C: Joint Venture (합작투자)
  D: Licensing (기술 라이선싱)

기준 (가중치):
  마케팅성(0.30), 수익성(0.25), 위험(0.20), 실행성(0.15), 시너지(0.10)

점수: A[7,6,8,9,7], B[9,8,4,5,6], C[8,7,6,7,8], D[5,5,9,8,4]
```

### 의사결정 나무 (신시장 진출)
```
대안:
  직접투자, 합작투자, 라이선싱, 중단

시장상태 (확률):
  호황(30%), 정상(50%), 불황(20%)

보상 (억원):
  직접투자:   [300, 100, -150]
  합작투자:   [150,  80,  -30]
  라이선싱:   [ 50,  40,   20]
  중단:       [  0,   0,    0]
```

## 실습 과제

### Part 4: 스마트시티 프로젝트 우선순위
- 4개 대안: Transportation, Energy, Safety, Environment
- 5개 기준: Citizen Satisfaction(0.30), Cost Efficiency(0.25), Tech Maturity(0.20), Scalability(0.15), Urgency(0.10)
- 과제: 가중 점수법으로 최우선 프로젝트 선정

### Part 5: 신제품 출시 의사결정
- 4개 대안: Full Launch, Limited Launch, License Out, Cancel
- 3개 시장상태: Success(25%), Moderate(50%), Failure(25%)
- 과제: EMV와 EVPI 계산으로 의사결정 분석

## 핵심 메시지

> **"다기준 의사결정은 직관을 제거하는 것이 아니라, 직관을 구조화하고 검증하는 도구다."**

조직 문화가 "데이터로 말한다"가 되려면:
- 의사결정 기준이 명시되어야 함
- 가중치가 정당화되어야 함
- 민감도 분석으로 강건성이 입증되어야 함

## 다음 장 연결

**Ch 13: 실물옵션 (Real Options)**
- 의사결정 나무 → 이항 트리로 확장
- EVPI → 연기 옵션의 가치 계산
- 정적 의사결정 → 동적 의사결정 (순차 선택)

---

**노트북 사용 방법**:
```bash
# VSCode에서 열기
code ch12_lecture.ipynb

# Jupyter Notebook 서버에서 실행
jupyter notebook ch12_lecture.ipynb
```

**필요한 라이브러리**:
```bash
pip install numpy pandas plotly scipy
```

**평가 기준**:
- 이론 과제 (30점): 개념 이해도, 조사의 깊이
- 실습 과제 (30점): 코드 정확성, 결과 해석
- 참여도 (20점): 수업 중 토론 질문
- 이해도 (20점): 선택형 이해도 퀴즈
