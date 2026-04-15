# Part 4, 5 바이브코딩 버전

## Part 4. 실습: AI 기반 뉴스 트렌드 분석

---

### 실습 7-1-A: AI로 배터리 산업 뉴스 분석하기

#### 📌 시나리오

당신은 배터리 소재 기업 C사의 전략기획팀에 소속되어 있습니다. CEO로부터 다음 과제를 받았습니다:

> "경쟁사 CATL이 나트륨이온 배터리를 대량 생산한다는 뉴스를 봤어. 이게 우리한테 기회일까, 위협일까? **최근 3개월 배터리 산업 뉴스를 분석해서 주요 트렌드를 정리해줘.**"

**목표:**
1. NewsAPI로 배터리 산업 뉴스 수집 (키워드: "battery", "solid-state", "sodium-ion" 등)
2. 키워드별 트렌드 시각화 (시계열 차트)
3. LLM(GPT/Claude)으로 주요 이슈 요약 및 전략적 함의 도출

---

#### 🤖 AI에게 이렇게 질문하세요

```
AI야, 나를 도와줘.

나는 배터리 산업의 뉴스 트렌드를 분석하고 싶어.
NewsAPI를 사용해서 다음을 해줘:

1. NewsAPI로 "battery technology" 키워드로 최근 3개월 뉴스 수집하는 Python 코드 작성
2. 수집한 뉴스에서 주요 키워드(Solid-state, Sodium-ion, LFP, Recycling) 언급 빈도를 집계
3. 주별 언급 빈도를 Plotly로 시각화 (선 그래프)
4. OpenAI GPT API로 전체 뉴스를 요약해서 "주요 트렌드 3가지"를 추출

각 단계마다 코드를 주고, 실행 결과를 해석하는 방법도 알려줘.
```

---

#### 📦 데이터 로드

아래 코드를 먼저 실행하세요. (시뮬레이션 데이터를 로드합니다)

```python
# 실습용 시뮬레이션 데이터 (실제로는 NewsAPI로 수집)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 시뮬레이션: 최근 3개월 뉴스 데이터
end_date = datetime.now()
start_date = end_date - timedelta(days=90)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

keywords = ['Solid-state', 'Sodium-ion', 'LFP', 'Recycling', 'Supply Chain']
news_sim = []

for date in dates:
    for kw in keywords:
        # 각 키워드별 일별 언급 빈도 (랜덤)
        base_count = {'Solid-state': 5, 'Sodium-ion': 3, 'LFP': 7, 
                     'Recycling': 4, 'Supply Chain': 8}[kw]
        count = int(np.random.poisson(base_count))
        news_sim.append({'date': date, 'keyword': kw, 'mentions': count})

news_df = pd.DataFrame(news_sim)
news_df['week'] = news_df['date'].dt.isocalendar().week

print("✅ 시뮬레이션 뉴스 데이터 로드 완료!")
print(f"   총 {len(news_df)} 건의 키워드 언급 데이터")
print(f"   기간: {start_date.date()} ~ {end_date.date()}")
print(f"   키워드: {', '.join(keywords)}")
```

---

#### ✏️ 1단계: 뉴스 수집 코드 작성

**AI에게 요청:**
> "NewsAPI를 사용해서 'battery technology' 키워드로 최근 3개월 뉴스를 수집하는 Python 코드를 작성해줘. API 키는 'YOUR_API_KEY'로 설정하고, 결과를 DataFrame으로 변환해줘."

```python
# ========== 여기서부터 AI가 작성한 코드를 붙여넣으세요 ==========

# TODO: AI가 작성한 NewsAPI 코드를 여기에 붙여넣고 실행하세요

# 힌트: from newsapi import News ApiClient
#       api = NewsApiClient(api_key='YOUR_API_KEY')
#       articles = api.get_everything(q='battery technology', ...)

# ========== 여기까지 ==========
```

**⚠️ 주의:** 실제 실습에서는 NewsAPI 키가 필요합니다. 수업에서는 위의 시뮬레이션 데이터를 사용합니다.

---

#### ✏️ 2단계: 키워드 트렌드 시각화

**AI에게 요청:**
> "news_df 데이터프레임에서 주별로 키워드 언급 빈도를 집계하고, Plotly로 선 그래프를 그려줘. 각 키워드는 다른 색상으로, 범례도 추가해줘."

```python
# ========== 여기서부터 AI가 작성한 코드를 붙여넣으세요 ==========

# TODO: AI가 작성한 시각화 코드를 여기에 붙여넣고 실행하세요

# 힌트: weekly = news_df.groupby(['week', 'keyword'])['mentions'].sum()
#       fig = px.line(weekly, ...)

# ========== 여기까지 ==========
```

---

#### ✏️ 3단계: LLM으로 요약

**AI에게 요청:**
> "OpenAI GPT API를 사용해서 위 뉴스 트렌드 데이터를 요약해줘. 입력으로 각 키워드의 성장률을 계산하고, GPT에게 '주요 트렌드 3가지와 전략적 함의'를 요청하는 프롬프트를 작성해줘."

```python
# ========== 여기서부터 AI가 작성한 코드를 붙여넣으세요 ==========

# TODO: AI가 작성한 LLM 요약 코드를 여기에 붙여넣고 실행하세요

# 힌트: import openai
#       response = openai.ChatCompletion.create(
#           model="gpt-4",
#           messages=[{"role": "user", "content": prompt}]
#       )

# ========== 여기까지 ==========
```

---

#### 📚 참고 (정답 힌트)

<details>
<summary>💡 1단계 힌트 보기</summary>

```python
from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta

api = NewsApiClient(api_key='YOUR_API_KEY')

end_date = datetime.now()
start_date = end_date - timedelta(days=90)

articles = api.get_everything(
    q='battery technology',
    from_param=start_date.strftime('%Y-%m-%d'),
    to=end_date.strftime('%Y-%m-%d'),
    language='en',
    sort_by='publishedAt',
    page_size=100
)

news_list = []
for article in articles['articles']:
    news_list.append({
        'title': article['title'],
        'source': article['source']['name'],
        'date': article['publishedAt'],
        'description': article['description']
    })

news_df = pd.DataFrame(news_list)
print(f"수집한 뉴스: {len(news_df)}건")
```

</details>

<details>
<summary>💡 2단계 힌트 보기</summary>

```python
import plotly.express as px

# 주별 집계
weekly = news_df.groupby(['week', 'keyword'])['mentions'].sum().reset_index()

# 시각화
fig = px.line(weekly, x='week', y='mentions', color='keyword',
              title='Battery Industry: Weekly Keyword Mentions',
              labels={'mentions': 'Weekly Mentions', 'week': 'Week'})
fig.update_layout(height=450)
fig.show()

print("💡 해석:")
print("   - 급격히 증가하는 키워드 = 약신호 가능성")
print("   - 안정적인 키워드 = 성숙 기술")
```

</details>

<details>
<summary>💡 3단계 힌트 보기</summary>

```python
import openai

# 성장률 계산
growth = []
for kw in keywords:
    kw_data = news_df[news_df['keyword'] == kw]
    first_month = kw_data[kw_data['date'] < start_date + timedelta(days=30)]['mentions'].sum()
    last_month = kw_data[kw_data['date'] > end_date - timedelta(days=30)]['mentions'].sum()
    growth_rate = (last_month - first_month) / first_month * 100 if first_month > 0 else 0
    growth.append(f"{kw}: {growth_rate:.0f}%")

prompt = f"""
배터리 산업 뉴스 트렌드 데이터:
{"\\n".join(growth)}

위 데이터를 분석하여:
1. 주요 트렌드 3가지
2. 각 트렌드의 전략적 함의
3. C사(배터리 소재 기업)에 대한 권장 사항

간결하게 요약하라.
"""

openai.api_key = "YOUR_API_KEY"
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

</details>

---

**도전과제:**
1. 감성 분석 추가: VADER나 Transformers 라이브러리로 각 뉴스의 감성 점수를 계산하세요
2. 자동화: 매일 실행되는 cron job을 만들어 최신 뉴스를 자동 수집하도록 설정하세요
3. 대시보드: Streamlit으로 실시간 트렌드 대시보드를 만들어보세요

---

### 실습 7-1-B: AI로 약신호 탐지 시스템 만들기

#### 📌 시나리오

CEO가 추가 과제를 줍니다:

> "트렌드 분석은 좋았어. 그런데 **'약신호(Weak Signal)'를 자동으로 탐지하는 시스템**을 만들 수 있을까? 초기에는 별로 눈에 안 띄다가 갑자기 급등하는 키워드를 알림으로 받고 싶어."

**목표:**
1. 이동평균 대비 이상 탐지 알고리즘 구현
2. 약신호로 판별된 키워드 분류
3. 약신호 대시보드 시각화 (성장률 × 언급량 매트릭스)

---

#### 🤖 AI에게 이렇게 질문하세요

```
AI야, 약신호 탐지 시스템을 만들고 싶어.

조건:
1. 시계열 데이터에서 최근 값이 이동평균 대비 2배 이상 증가하면 "이상"으로 판별
2. 또는 최근 1개월 성장률이 100% 이상이면 "약신호"로 분류
3. 결과를 산점도(scatter plot)로 표시: x축=성장률, y축=이동평균 비율, 버블 크기=최근 언급량

Python 코드를 단계별로 작성해줘.
```

---

#### 📦 데이터 로드

```python
# 실습 7-1-A에서 생성한 news_df 및 weekly 데이터 사용
# 만약 없다면 위의 시뮬레이션 코드를 다시 실행하세요

print("✅ 기존 뉴스 데이터 사용")
print(f"   주별 데이터: {len(weekly)} 레코드")
```

---

#### ✏️ 1단계: 이상 탐지 함수 작성

**AI에게 요청:**
> "weekly 데이터프레임에서 각 키워드별로 이동평균(window=4주)을 계산하고, 최근 값이 이동평균 대비 몇 배인지 계산하는 함수를 작성해줘."

```python
# ========== 여기서부터 AI가 작성한 코드를 붙여넣으세요 ==========

# TODO: detect_anomaly(weekly_df, keyword, window=4) 함수 작성

# ========== 여기까지 ==========
```

---

#### ✏️ 2단계: 약신호 분류

**AI에게 요청:**
> "모든 키워드에 대해 (1) 이동평균 비율, (2) 성장률(초기 vs 최근)을 계산해서 DataFrame으로 정리해줘. 그리고 'is_weak_signal' 컬럼을 추가해서 조건을 만족하면 True로 표시해줘."

```python
# ========== 여기서부터 AI가 작성한 코드를 붙여넣으세요 ==========

# TODO: 약신호 분류 코드

# ========== 여기까지 ==========
```

---

#### ✏️ 3단계: 대시보드 시각화

**AI에게 요청:**
> "위에서 만든 DataFrame을 사용해서 Plotly 산점도를 그려줘. x축=성장률, y축=이동평균 비율, 버블 크기=최근 언급량, 색상=is_weak_signal (True=빨강, False=파랑). 기준선도 추가해줘 (x=100%, y=2.0)."

```python
# ========== 여기서부터 AI가 작성한 코드를 붙여넣으세요 ==========

# TODO: 산점도 시각화 코드

# ========== 여기까지 ==========
```

---

#### 📚 참고 (정답값)

**예상 결과:**
- Solid-state: 성장률 80%, 비율 1.5 → 정상
- Sodium-ion: 성장률 250%, 비율 3.2 → 🚨 약신호!
- Recycling: 성장률 120%, 비율 1.8 → ⚠️ 주의

**도전과제:**
1. 알림 시스템: 약신호 탐지 시 이메일/Slack으로 알림 발송
2. 역사적 검증: 과거 약신호가 강신호로 전환된 사례 분석 (예: ChatGPT 관련 키워드)

---

## Part 5. 실습: AI 기반 경쟁 인텔리전스

---

### 실습 7-1-C: AI로 경쟁사 특허 분석하기

#### 📌 시나리오

C사 CEO가 말합니다:

> "경쟁사 CATL과 삼성SDI가 어떤 기술에 투자하고 있는지 알고 싶어. **두 회사의 최근 3년 특허 출원 동향을 비교 분석**해줘. 어떤 기술 분야에 집중하고 있는지, 우리랑 격차는 어느 정도인지 파악해야 해."

**목표:**
1. USPTO PatentsView API로 경쟁사 특허 데이터 수집
2. 기술 분야별 포트폴리오 시각화 (레이더 차트)
3. LLM으로 전략 추론 ("경쟁사는 X 기술에 집중 중")

---

#### 🤖 AI에게 이렇게 질문하세요

```
AI야, 특허 분석을 도와줘.

목표:
1. USPTO PatentsView API로 "CATL"과 "Samsung SDI"의 최근 3년 특허 검색
2. CPC 코드로 기술 분야 분류 (H01M=배터리, B60L=전기차, G06F=AI 등)
3. 기업별 기술 포트폴리오를 레이더 차트로 비교
4. OpenAI GPT로 전략 차이 분석

단계별 코드를 작성해줘.
```

---

#### 📦 데이터 로드

```python
# 실습용: 시뮬레이션 특허 데이터
companies = ["CATL", "Samsung SDI"]
tech_areas = ["Solid-state (H01M)", "Li-ion Cathode (H01M)", 
              "BMS (H02J)", "Recycling (C01G)", "AI/Software (G06F)"]

patent_sim = []
for company in companies:
    for tech in tech_areas:
        if company == "CATL":
            # CATL은 Cathode, BMS에 강점
            base = {'Solid-state (H01M)': 10, 'Li-ion Cathode (H01M)': 25,
                   'BMS (H02J)': 20, 'Recycling (C01G)': 8, 'AI/Software (G06F)': 5}[tech]
        else:
            # Samsung SDI는 Solid-state, AI에 강점
            base = {'Solid-state (H01M)': 22, 'Li-ion Cathode (H01M)': 12,
                   'BMS (H02J)': 10, 'Recycling (C01G)': 6, 'AI/Software (G06F)': 15}[tech]
        
        patent_sim.append({'company': company, 'tech_area': tech, 'patents': base})

patent_df = pd.DataFrame(patent_sim)
print("✅ 시뮬레이션 특허 데이터 로드 완료!")
print(patent_df)
```

---

#### ✏️ 1단계: 특허 데이터 수집

**AI에게 요청:**
> "USPTO PatentsView API로 'CATL'의 특허를 검색하는 코드를 작성해줘. 검색 조건은 2023-2026년, assignee_organization='CATL'이고, 결과에서 patent_number, patent_title, cpc_subgroup_id를 가져와줘."

```python
# ========== 여기서부터 AI가 작성한 코드를 붙여넣으세요 ==========

# TODO: API 코드

# ========== 여기까지 ==========
```

---

#### ✏️ 2단계: 기술 포트폴리오 비교

**AI에게 요청:**
> "patent_df를 사용해서 CATL과 Samsung SDI의 기술 포트폴리오를 레이더 차트(Scatterpolar)로 비교해줘. 각 기업은 다른 색상으로 표시하고, 범례도 추가해줘."

```python
# ========== 여기서부터 AI가 작성한 코드를 붙여넣으세요 ==========

#TODO: 레이더 차트 코드

# ========== 여기까지 ==========
```

---

#### ✏️ 3단계: 전략 추론

**AI에게 요청:**
> "위 특허 데이터를 OpenAI GPT에게 보내서 '두 기업의 기술 전략 차이와 C사(배터리 소재 기업)에 대한 시사점'을 요약해달라고 요청하는 프롬프트를 작성해줘."

```python
# ========== 여기서부터 AI가 작성한 코드를 붙여넣으세요 ==========

# TODO: LLM 분석 코드

# ========== 여기까지 ==========
```

---

#### 📚 참고 (정답 힌트)

**예상 분석 결과:**
- CATL: Li-ion Cathode (양극재) 특허 집중 → 기존 기술 최적화 전략
- Samsung SDI: Solid-state (전고체) 특허 집중 → 차세대 기술 선점 전략
- C사 시사점: 전고체용 신소재 개발에 집중하면 Samsung SDI와 협업 기회

---

### 실습 7-1-D: AI로 데이터 기반 SWOT 작성하기

#### 📌 시나리オ

CEO의 최종 요청:

> "지금까지 분석한 내용을 바탕으로 **C사의 SWOT 분석서**를 작성해줘. 각 항목마다 **데이터 근거**를 명시하고, 기회/위협은 **정량화**(영향도, 확률)해줘. 그리고 AI에게 전략 우선순위를 추천받아."

**목표:**
1. 데이터 기반 SWOT 매트릭스 작성 (표 형식)
2. 기회/위협 정량화 (영향도 × 확률 매트릭스)
3. LLM으로 전략 우선순위 도출

---

#### 🤖 AI에게 이렇게 질문하세요

```
AI야, 데이터 기반 SWOT 분석을 도와줘.

입력:
- C사 배경: 정밀화학 소재 기업, 배터리 진출 검토 중, R&D 역량 보유
- 앞서 분석한 트렌드 데이터 (Sodium-ion 급부상, Solid-state 경쟁)
- 경쟁사 특허 데이터 (CATL, Samsung SDI 포지셔닝)

출력:
1. SWOT 매트릭스 (Plotly Table)
2. 기회/위협 정량화 산점도 (x=확률, y=영향도, 버블=우선순위)
3. GPT로 "상위 3개 전략 과제" 도출

코드를 작성해줘.
```

---

#### 📦 데이터 준비

```python
# SWOT 초안 데이터 (나중에 AI로 보강)
swot_draft = {
    'Strengths': [
        "소재 합성 기술 (특허 32건)",
        "R&D 인력 (박사급 45명)",
        "정밀화학 제조 역량"
    ],
    'Weaknesses': [
        "배터리 산업 경험 없음",
        "브랜드 인지도 없음",
        "규모 열위 (매출 CATL 대비 1/50)"
    ],
    'Opportunities': [
        ("Sodium-ion 시장 부상", "성장률 250%", 0.7, 0.8),  # (항목, 데이터, 확률, 영향도)
        ("전고체 소재 수요 증가", "특허 80% 증가", 0.6, 0.9),
        ("재활용 규제 강화", "EU 법안 통과", 0.8, 0.6)
    ],
    'Threats': [
        ("대기업 시장 독점"기존 player 시장점유율 90%", 0.9, 0.7),
        ("기술 격차", "특허 수 10배 차이", 0.7, 0.8),
        ("가격 경쟁 심화", "중국 기업 저가 공세", 0.8, 0.6)
    ]
}

print("✅ SWOT 초안 데이터 준비 완료")
```

---

#### ✏️ 1단계: SWOT 매트릭스 시각화

**AI에게 요청:**
> "swot_draft 딕셔너리를 사용해서 Plotly Table로 SWOT 매트릭스를 만들어줘. 2x2 레이아웃으로 Strengths, Weaknesses는 위쪽, Opportunities, Threats는 아래쪽에 표시해줘."

```python
# ========== 여기서부터 AI가 작성한 코드를 붙여넣으세요 ==========

# TODO: SWOT 테이블 시각화

# ========== 여기까지 ==========
```

---

#### ✏️ 2단계: 기회/위협 정량화

**AI에게 요청:**
> "Opportunities와 Threats 항목을 산점도로 표시해줘. x축=확률, y축=영향도, 버블 크기는 확률×영향도(우선순위 대용). 색상은 기회=초록, 위협=빨강으로 구분해줘."

```python
# ========== 여기서부터 AI가 작성한 코드를 붙여넣으세요 ==========

# TODO: 정량화 매트릭스 시각화

# ========== 여기까지 ==========
```

---

#### ✏️ 3단계: 전략 우선순위 도출

**AI에게 요청:**
> "위 SWOT 데이터를 OpenAI GPT에게 제공하고, '상위 3개 전략 과제와 실행 방안'을 추천받는 프롬프트를 작성해줘."

```python
# ========== 여기서부터 AI가 작성한 코드를 붙여넣으세요 ==========

# TODO: LLM 전략 추천 코드

# ========== 여기까지 ==========
```

---

#### 📚 참고 (정답 예시)

**GPT 추천 전략 (예상):**
1. **Sodium-ion 소재 집중 개발** (기회 × 강점)
   - 실행: R&D 인력 20명 배치, 6개월 내 프로토타입 개발
   
2. **전고체 배터리 소재 파트너십** (기회 × 약점 보완)
   - 실행: Samsung SDI와 공동 개발 제안, 소재 공급 계약

3. **재활용 소재 사업 진출** (기회 × 위협 회피)
   - 실행: 신규 시장, 대기업과 직접 경쟁 회피

---

**도전과제:**
1. 실시간 SWOT 업데이트: 매주 뉴스/특허 데이터로 SWOT을 자동 갱신하는 파이프라인 구축
2. 시나리오 플래닝: 각 전략의 best/worst case를 시뮬레이션하여 민감도 분석
3. 대시보드: Streamlit으로 "경영진용 전략 대시보드" 제작 (SWOT, 트렌드, 특허가 한 화면에)

---
