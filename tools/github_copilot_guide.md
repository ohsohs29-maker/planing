# GitHub Copilot 학생 가이드

## 수업 활용 안내

본 과목에서는 **GitHub Copilot**을 활용하여 Python 실습을 진행합니다.
대학생은 **GitHub Student Developer Pack**을 통해 **Copilot Pro를 무료**로 사용할 수 있습니다.

---

## 1. GitHub Copilot 개요

### Copilot 플랜 비교 (2026년 기준)

| 구분 | 대상 | 가격 | 제한 | 기능 수준 |
|------|------|------|------|----------|
| **Copilot Free** | 누구나 | 무료 | 월 2,000 코드 완성, 50 채팅 | 기본 모델 |
| **Copilot Pro** | 일반인 | 월 $10 | 월 300 premium requests | 최신 모델 풀 액세스 |
| **Copilot Pro (학생)** | GitHub Education 인증 학생 | **무료** | 월 300 premium requests | Pro 풀 기능 |

> **핵심**: 대학생이라면 **Copilot Pro 무료** 사용 가능!

---

## 2. GitHub Student Developer Pack 등록 절차

### Step 1: GitHub 계정 준비

1. [https://github.com](https://github.com) 접속
2. 계정이 없으면 새로 생성 (**학교 이메일 권장**)
3. 이미 계정이 있으면 로그인

### Step 2: Student Developer Pack 신청

1. [https://education.github.com/pack](https://education.github.com/pack) 접속
2. **"Get your pack"** 클릭

### Step 3: 학생 신분 증명 (가장 중요!)

| 순위 | 증빙 방법 | 성공률 | 소요시간 |
|------|----------|--------|----------|
| 1 | **학교 이메일 (.ac.kr)** | 매우 높음 | 즉시~수시간 |
| 2 | 학생증 사진 업로드 | 높음 | 1~5일 |
| 3 | 재학증명서 PDF 업로드 | 높음 | 1~7일 |
| 4 | 등록금 영수증 + 신분증 | 중간 | 3~10일 |

> **팁**: 한국 4년제 대학 재학생은 **학교 이메일만으로 1~24시간 내 자동 승인**됩니다.

### Step 4: 승인 확인

- [https://education.github.com/pack](https://education.github.com/pack) 에서 "Your pack" 상태 확인
- 승인 메일 도착 (제목: "You're all set!")

### Step 5: Copilot Pro 무료 활성화

**방법 A (권장)**:
1. [https://github.com/settings/copilot](https://github.com/settings/copilot) 이동
2. "Code, planning, and automation" → Copilot 클릭
3. 학생 혜택으로 무료 가입 버튼 클릭

**방법 B**:
1. [https://github.com/features/copilot](https://github.com/features/copilot) 이동
2. "무료로 시작" 또는 "Claim free access" 버튼 클릭

### Step 6: VS Code에서 사용 시작

1. VS Code 실행
2. Extensions에서 "GitHub Copilot" 설치
3. GitHub 계정으로 로그인
4. Copilot Pro 기능 사용 시작!

---

## 3. VS Code에서 Copilot 사용하기

### 기본 설정

1. **VS Code 설치**: [https://code.visualstudio.com](https://code.visualstudio.com)
2. **Extensions 설치**:
   - GitHub Copilot
   - GitHub Copilot Chat
3. **로그인**: Command Palette (Ctrl+Shift+P) → "Copilot: Sign In"

### 핵심 기능

| 기능 | 단축키 | 설명 |
|------|--------|------|
| 코드 자동완성 | (자동) | 코드 작성 중 자동 제안 |
| Copilot Chat | Ctrl+Shift+I | AI와 대화하며 코드 작성 |
| Agent 모드 | Chat에서 선택 | 외부 도구 연동 가능 |

### Agent 모드 활성화

1. Copilot Chat 열기 (Ctrl+Shift+I)
2. 상단에서 **Agent** 선택 (Ask/Edit/Agent 중)
3. 설정에서 활성화:
   ```json
   "github.copilot.chat.agent.enabled": true
   ```

---

## 4. MCP (Model Context Protocol)

### MCP란?

**MCP**는 Copilot의 기능을 외부 도구/API/데이터 소스와 연동하여 확장하는 프로토콜입니다.

| 항목 | 설명 |
|------|------|
| **목적** | 외부 도구 (GitHub, DB, Figma 등) 직접 호출 |
| **사용법** | @github, @postgres 등으로 호출 |
| **제한** | Tool 호출 시 premium requests 차감 |

### 데이터사이언스 추천 MCP 서버

| 순위 | MCP 서버 | 주요 기능 | 설치 난이도 |
|------|---------|----------|------------|
| 1 | **Pandas MCP** | DataFrame 로드, 전처리, 통계, 시각화 | ★★☆☆☆ |
| 2 | **Jupyter MCP** | 노트북 실시간 제어, 셀 실행, 플롯 표시 | ★★☆☆☆ |
| 3 | **SQL/Postgres MCP** | 자연어 → SQL 변환, 쿼리 실행 | ★★☆☆☆ |
| 4 | **Data Exploration MCP** | CSV 탐색, 요약 통계, 아웃라이어 탐지 | ★☆☆☆☆ |
| 5 | **Matplotlib MCP** | 차트 자동 생성 | ★★☆☆☆ |

### MCP 서버 추가 방법

**방법 A: 명령 팔레트**
1. Command Palette (Ctrl+Shift+P)
2. "MCP: Add Server" 선택
3. Registry에서 원하는 서버 선택

**방법 B: 설정 파일**

`.vscode/mcp.json` 파일 생성:

```json
{
  "mcp": {
    "servers": {
      "github": {
        "command": "docker",
        "args": ["run", "-i", "--rm", "-e", "GITHUB_TOKEN", "ghcr.io/github/github-mcp-server:latest"],
        "env": { "GITHUB_TOKEN": "${input:github_token}" }
      }
    }
  }
}
```

---

## 5. Agent Skills

### Skills란?

**Skills**는 Copilot에게 특정 작업에 대한 전문 지침을 주입하는 기능입니다.

| 항목 | Agent Skills | MCP |
|------|-------------|-----|
| **목적** | 작업 규칙/스타일 주입 | 외부 도구 호출 |
| **형태** | `.github/skills/` 폴더 내 SKILL.md | MCP 서버 실행 |
| **난이도** | Markdown 작성만으로 가능 | 서버 설정 필요 |
| **quota** | 거의 소모 안 함 | 소모됨 |

### 데이터사이언스 추천 Skills

| Skill 이름 | 발동 키워드 | 주요 지침 |
|-----------|------------|----------|
| **pandas-eda-standard** | "eda", "explore", "summary" | df.info(), describe(), 결측치 처리, 분포 시각화 |
| **sql-query-best-practice** | "sql", "query", "select" | CTE 사용, 인덱스 고려, 명확한 alias |
| **data-viz-publication-ready** | "plot", "visualize", "chart" | seaborn 스타일, 한글 폰트, 논문 퀄리티 |
| **jupyter-notebook-structure** | "notebook", "jupyter" | 표준 섹션 구조 자동 적용 |

### Skills 만들기

1. 저장소 루트에 폴더 생성:
   ```
   .github/skills/pandas-eda-standard/
   ```

2. `SKILL.md` 파일 작성:

```markdown
---
name: Pandas EDA Standard
description: 표준 EDA 절차를 자동으로 수행합니다.
when: "eda", "explore", "summary", "describe" 키워드 포함 시
priority: high
---

## 지침

1. df.info()로 데이터 타입 확인
2. df.describe()로 기술 통계 출력
3. 결측치 비율 계산 및 시각화
4. 수치형 변수 분포 히스토그램
5. 상관계수 히트맵 생성

## 출력 형식

- 각 단계별 Markdown 설명 포함
- 시각화는 영어 레이블 사용 (폰트 호환성)
```

3. VS Code 재시작 또는 Reload Window (Ctrl+R)

4. Copilot Chat에서 "이 데이터 EDA 해줘"라고 요청하면 자동 적용!

---

## 6. 권장 설정 조합

### 본 수업 권장 조합

**MCP 서버** (2개):
- Pandas MCP
- Jupyter MCP

**Skills** (3개):
- pandas-eda-standard
- data-viz-publication-ready
- jupyter-notebook-structure

### quota 관리 팁

| 팁 | 설명 |
|----|------|
| MCP는 2개 이하로 | 3개 이상 시 quota 소모 급증 |
| Skills는 마음껏 | quota 거의 소모 안 함 |
| 복잡한 작업은 직접 코딩 | MCP 호출보다 효율적 |

---

## 7. 문제 해결

| 문제 | 해결 방법 |
|------|----------|
| 승인 후 바로 안 보임 | 5~30분 기다리거나 로그아웃 후 재로그인 |
| 무료 버튼이 안 보임 | 캐시 지우기 / 다른 브라우저 시도 |
| MCP 서버 안 보임 | VS Code Reload / Insiders 사용 |
| Token 오류 (401/403) | PAT 권한 확인 (repo, issues) |
| Skills가 안 먹힘 | VS Code 1.108 미만 / Agent 모드 꺼짐 확인 |

---

## 8. 주의사항

1. **월 300 premium requests 제한**
   - 학생도 동일하게 적용
   - 초과 시 다음 달 초기화

2. **졸업 시 자동 유료 전환**
   - 재학생 기간에 최대한 활용 권장

3. **보안 주의**
   - MCP는 임의 코드 실행 가능
   - 신뢰할 수 있는 데이터만 연결

---

## 9. 참고 링크

- GitHub Education: [https://education.github.com](https://education.github.com)
- Copilot 설정: [https://github.com/settings/copilot](https://github.com/settings/copilot)
- MCP 서버 Registry: [https://github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)
- Copilot 공식 문서: [https://docs.github.com/ko/copilot](https://docs.github.com/ko/copilot)

---
