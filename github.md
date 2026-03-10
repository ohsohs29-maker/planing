### 주의사항: 모든 컴퓨터 조작에서 한글 사용금지하고 반드시 영어문자를 사용해야 한다. 폴더명, 파일명 등등. 한글 사랑과 별개의 문제로서 피할 수 없는 문제이다. 

### 검색 - powershell - 작업표시줄에 고정 


# Git & GitHub 빠른 시작 가이드

## 1단계: Git 설치 (3분)

**Windows**: https://git-scm.com/download/win → 다운로드 후 설치 (기본값 OK)


설치 확인:

```bash
git --version
```

## 2단계: GitHub 가입 (2분)

1. https://github.com 접속
2. "Sign up" → 학교 이메일로 가입
3. 이메일 인증

### 3단계: Python 설치

1. https://www.python.org/downloads/ 에서 **Python 3.12** 이상 다운로드
2. 설치 프로그램 실행
3. **⚠️ 첫 화면에서 반드시 "Add python.exe to PATH" 체크** ← 가장 중요!
4. **"Install Now"** 클릭

설치 확인 (Windows: `Win+R` → `cmd` 입력 → 확인):

```bash
python --version
```

### 4단계: VSCode 설치

1. https://code.visualstudio.com/ 에서 다운로드 후 설치
2. 설치 시 **"Add to PATH" 옵션 체크** 권장
3. 설치 후 실행하여 다음 확장(Extensions)을 설치:
   - **Python** (Microsoft) — Python 개발 지원
   - **Jupyter** (Microsoft) — 노트북(.ipynb) 실행 지원

> 좌측 사이드바 확장 아이콘(□) 클릭 → 검색창에 "Python", "Jupyter" 입력 → Install


`Python 3.x.x`가 출력되면 성공입니다. 만약 `'python' is not recognized...` 오류가 나면 PATH 등록이 안 된 것이므로 Python을 제거 후 3번을 확인하며 재설치하세요.

## 5단계: 로컬 연결 (1분)

>  VSCode 통합 터미널 권장.

```bash
git config --global user.name "홍길동"
git config --global user.email "학교이메일@ac.kr"
```


### 6단계: 저장소 클론

명령 프롬프트 열기 (Windows: `Win+R` → `cmd` 입력 → 확인):

```bash
cd C:\
git clone https://github.com/LeeSeogMin/planing.git
```

`C:\planing` 폴더가 생성되면 성공입니다.

### 7단계: VSCode에서 프로젝트 열기

> **반드시 4단계(가상환경 설정)가 "설정 완료!" 메시지로 끝난 뒤 진행하세요.**

1. VSCode 실행
2. **파일 → 폴더 열기** (`Ctrl+K, Ctrl+O`) → `C:\planing` 선택
3. 왼쪽 탐색기에서 `ch01` → `ch01.ipynb` 클릭

### 6단계: 커널 선택 및 실행

1. 노트북 우측 상단 **"커널 선택"** 클릭 후, 확장선택 후, 파이썬 환경 선택
2. **"Python 환경..."** → **"Python 3"** 선택
3. 첫 번째 코드 셀부터 순차 실행 (▶)

> 커널 목록에 "AI 기획 강의 (Python 3)"가 안 보이면 4단계(`python setup_env.py`)가 정상 완료되었는지 확인하세요.

### 4단계: 환경 자동 설정
vscode를 열고 터미널-새터미널 열고 실행

```powershell
python setup_env.py
```

이 스크립트가 자동으로 처리하는 항목:
- `.venv` 가상환경 생성
- 필요한 패키지 일괄 설치
- Jupyter 커널 등록

"설정 완료!" 메시지가 나오면 성공입니다.



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
만일 github copilot 실패햇을때 아래를 사용합니다. 

## Gemini CLI 설치 및 실행

### 1단계: Node.js 설치 확인
- Gemini CLI는 npm 방식 설치를 권장하므로 Node.js 필수
- https://nodejs.org 에서 LTS 버전 다운로드 후 설치
- npm은 Node.js와 함께 자동 설치됨

### 2단계: vscode 에서 새터미널 열기
- PowerShell 실행

### 3단계: Gemini CLI 설치
```bash
npm install -g @google/gemini-cli 

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned 
```

### 4단계: 실행
```bash
gemini   

중간에 브라우저에서 구글로 로그인하는 과정을 거치니 브라우저를 잘 살핀다. 

구글로그인 후에 성공했다는 메시지와 함께 r 을 입력하라는 메시지가 나오니 따라서 하면 잠시 후에 로그인된다. 

터미널에서 마우스 우클릭 후 패널위치를 오른쪽으로 한다. 
```

### 5단계: 인증 설정
처음 실행 시 다음 중 선택:
- **Google 계정 로그인** (권장)
- **Gemini API Key**
- **Vertex AI**




# GitHub Copilot 대학생 무료 사용 방법

GitHub는 **Copilot Free** (제한된 무료 플랜)와 **Copilot Pro** (고급 기능 풀버전)을 구분해서 운영하고 있으며,

**검증된 학생**은 **Copilot Pro**를 **학생 신분 유지 기간 동안 완전 무료**로 사용할 수 있습니다.

### 현재(2026년) 상황 요약

| 구분               | 대상자                     | 가격     | 주요 제한사항                                   | 모델/기능 수준          |
| ------------------ | -------------------------- | -------- | ----------------------------------------------- | ----------------------- |
| Copilot Free       | 누구나                     | 무료     | 월 2,000 코드 완성, 50 채팅 등 매우 제한적      | 기본 모델               |
| Copilot Pro        | 일반인                     | 월 $10   | 300 premium requests + 추가 과금 가능           | 최신 모델 풀 액세스     |
| Copilot Pro (학생) | GitHub Education 검증 학생 | **무료** | 월 300 premium requests (초과 시 다음달 초기화) | Pro 풀 기능 + 최신 모델 |

→ **대학생이 진짜 원하는 건 Copilot Pro 무료**이며, 이를 위해서는 **GitHub Student Developer Pack** 승인이 필수입니다.

### 2026년 최신 정확한 등록 절차 (단계별)

1. **GitHub 계정 준비**
   - 이미 계정이 있다면 로그인
   - 없다면 [https://github.com](https://github.com/) 에서 새로 생성 (학교 이메일 추천)
   - 기존 계정이 있으면 설정(https://github.com/settings/emails)에서 학교 이메일(.ac.kr)을 추가
   - **학교 이메일을 primary email(기본 이메일)로 설정** (드롭다운에서 선택 후 Save → 인증 인식을 도움. 나중에 개인 이메일로 되돌릴 수 있음)
   - 이메일 추가 후 verification link(인증 링크)를 클릭해 verified 상태로 만들기
2. **GitHub Student Developer Pack 신청 페이지 이동**

   https://education.github.com/pack

   또는 [https://education.github.com](https://education.github.com/) → "Get your pack" 클릭

3. **학생 신분 증명** (가장 중요한 단계)

   대부분의 한국 대학생이 성공하는 순서 (우선순위 높은 순) :

   | 순위 | 증빙 방법                       | 성공률    | 소요시간    | 비고                                        |
   | ---- | ------------------------------- | --------- | ----------- | ------------------------------------------- |
   | 1    | 학교 공식 이메일 (.ac.kr)       | 매우 높음 | 즉시~수시간 | 대부분 자동 승인                            |
   | 2    | 학생증 사진 (재학증명서) 업로드 | 높음      | 1~5일       | 선명하게 촬영, 이름·학번·유효기간 보여야 함 |
   | 3    | 재학증명서 pdf 업로드           | 높음      | 1~7일       | 최근 3개월 이내 발급본                      |
   | 4    | 등록금 영수증 + 신분증          | 중간      | 3~10일      | 최후의 수단                                 |

   → **한국 4년제 대학 재학생이라면 대부분 학교 이메일만으로 1~24시간 내 자동 승인**됩니다.

4. **승인 확인**
   - https://education.github.com/pack 에서 "Your pack" 상태 확인
   - 승인 메일 도착 (보통 "You're all set!" 제목)
5. **Copilot Pro 무료 활성화** (승인 후 바로 가능)

   두 가지 방법 중 편한 것 선택:

   방법 A (가장 확실)
   - https://github.com/settings/copilot 이동
   - "Code, planning, and automation" → Copilot 클릭
   - 학생 혜택으로 무료 가입 버튼 나타남 → 클릭

   방법 B
   - https://github.com/features/copilot 로 이동
   - 학생으로 인식되면 "무료로 시작" 또는 "Claim free access" 버튼 등장

   방법 C (학생/교사 전용 무료 signup 페이지)
   - https://github.com/github-copilot/free_signup 으로 직접 이동

   > **주의**: 신용카드 입력이 요구되면 진행하지 마십시오. 학생 혜택은 완전 무료이며 결제 정보가 필요하지 않습니다.

6. **VS Code 등 에디터에서 사용 시작**
   - GitHub 계정으로 Copilot 확장 로그인
   - 학생 혜택이 정상 적용되어 풀 Pro 기능 사용 가능

### 주의사항 (2026년 기준 자주 발생하는 문제)

- 승인 후에도 바로 안 보일 때 → 72시간까지 기다린 뒤 재로그인 시도 (Incognito/시크릿 모드 사용 추천). 혜택 동기화에 72시간~최대 2주가 소요될 수 있음
- "무료 버튼이 안 보임" → 캐시 지우기 / 다른 브라우저 시도 / primary email을 학교 이메일로 재설정 후 대기 / https://github.com/settings/copilot 직접 들어가기
- 인증 실패 시 → GitHub Support(https://support.github.com/contact/education)에 티켓 제출 (카테고리: "Student having trouble redeeming offers")
- 월 300 premium requests 제한은 학생도 동일 (과거에는 무제한이었으나 2025년 중반부터 변경됨)
- 졸업하면 자동으로 Pro 유료 전환됨 → 재학생 기간에 최대한 활용 권장
- 공식 문서 참조: https://docs.github.com/en/education

위 방법은 2026년 2월 6일 기준 GitHub 공식 문서 및 실제 학생 사례들을 종합한 **현재 가장 정확한 절차**입니다.

학교 이메일이 있다면 거의 100% 성공한다고 봐도 무방합니다.

# GitHub Copilot 사용 설명서

이 문서는 VS Code에서 GitHub Copilot을 활용하는 방법을 안내합니다. 아래 캡처 이미지를 참고하여 주요 기능과 메뉴를 설명합니다.

---

## 1. Copilot Chat 세션 시작

- **새 챗 세션(New Chat Session)**: Ctrl+N 단축키로 새로운 Copilot Chat 세션을 시작할 수 있습니다.
  **Continue In**: 챗 세션을 실행할 환경을 선택할 수 있습니다.
  - Local(로컬): 현재 PC에서 직접 Copilot 챗을 실행합니다. 빠른 응답과 파일 접근이 가능합니다.
  - Background(@cli): 명령줄 환경에서 Copilot을 실행합니다. 백그라운드 작업이나 자동화에 적합합니다.
  - Cloud(@cloud): 클라우드 서버에서 Copilot 챗을 실행합니다. 대용량 작업, 서버 리소스 활용, 원격 협업에 유리합니다.

---

## 2. 에이전트 및 역할 선택

- **Agent 메뉴**: Ctrl+Shift+I로 Copilot의 역할을 선택할 수 있습니다.
  - coder: 코드 작성
  - graphic: 그래픽/다이어그램 생성
  - planner: 집필 계획
  - researcher: 자료 조사
  - reviewer: 품질 검토
  - writer: 원고 작성
- **Plan/Ask**: 계획 수립 또는 질문 모드로 전환 가능
- **커스텀 에이전트 구성**: 필요에 따라 직접 에이전트 역할을 추가/설정할 수 있습니다.

---

## 3. 도구(툴) 설정

- **Configure Tools**: Copilot이 사용할 수 있는 도구를 선택/해제할 수 있습니다.
  - 예시: edit(파일 편집), execute(코드 실행), search(검색), todo(할 일 관리), web(웹 정보 수집) 등
  - MCP 서버 기반의 확장 도구도 활성화 가능
- **도구 선택 후 OK 버튼으로 적용**

---

## 4. MCP 서버 및 확장 기능

- **MCP 서버 알림**: 새로운 MCP 서버(예: mcp-server-time, GitHub 등)가 활성화되면 알림이 표시됩니다.
- **자동 시작 옵션**: "Automatically start MCP servers" 체크박스로 서버 자동 실행 설정 가능

---

## 5. Copilot 모델 선택

- **모델 선택**: 하단 메뉴에서 GPT-4.1 등 Copilot이 사용할 모델을 선택할 수 있습니다.

---

## 6. 기타 기능

- **챗 세션 관리**: 여러 챗 세션을 동시에 운영 가능
- **도구/에이전트/모델 조합**: 작업 목적에 따라 자유롭게 조합하여 활용

---
