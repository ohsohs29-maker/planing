"""
한신대학교 계열제 효과 분석: 데이터 전처리
================================================================================
- 목적: SDID 분석을 위한 패널 데이터 구축
- 처치군: 한신대학교 (2023년 계열제 도입)
- 대조군: 20개 유사 대학 (Donor pool)
- 기간: 2016-2025년 (10년)
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. Donor Pool 정의
# ============================================================================

# 처치군
TREATED_UNIT = "한신대학교"
TREATMENT_YEAR = 2023

# 대조군 (20개 대학)
DONOR_POOL = [
    # 경기도 (10개)
    "대진대학교",
    "협성대학교",
    "강남대학교",
    "성결대학교",
    "평택대학교",
    "용인대학교",
    "안양대학교",
    "수원대학교",
    "한세대학교",
    "경기대학교",  # 2025년 계열제 도입 → 2016-2024 사용
    # 충청권 (7개)
    "남서울대학교",
    "중부대학교",
    "배재대학교",
    "대전대학교",
    "목원대학교",
    "서원대학교",
    "한서대학교",
    # 강원권 (2개)
    "한림대학교",
    "상지대학교",
    # 추가 (1개)
    "세명대학교",
]

# 분석 기간
YEARS = list(range(2016, 2026))  # 2016-2025

# ============================================================================
# 2. 데이터 로드 및 필터링
# ============================================================================

def load_and_filter_data():
    """원본 데이터에서 분석 대상 추출"""

    # 원본 데이터 로드
    df = pd.read_excel('../data.xlsx')

    # 분석 대상 대학 목록
    all_units = [TREATED_UNIT] + DONOR_POOL

    # 필터링 조건
    mask = (
        (df['학교명'].isin(all_units)) &
        (df['공시연도'].isin(YEARS)) &
        (df['본분교명'] == '본교')
    )

    df_filtered = df[mask].copy()

    # 필요한 컬럼만 선택
    columns = [
        '공시연도', '학교명', '지역명',
        '입학정원', '정원내모집인원', '정원내지원자', '정원내입학자',
        '재학생수(학부)', '전임교원수(학부+대학원)', '경쟁률'
    ]
    df_filtered = df_filtered[columns].copy()

    # 컬럼명 정리
    df_filtered.columns = [
        '연도', '대학명', '지역',
        '입학정원', '모집인원', '지원자수', '입학자수',
        '재학생수', '전임교원수', '경쟁률'
    ]

    return df_filtered

# ============================================================================
# 3. 패널 데이터 구축
# ============================================================================

def create_panel_data(df):
    """Long format 패널 데이터 생성"""

    # 정렬
    df = df.sort_values(['대학명', '연도']).reset_index(drop=True)

    # 처치군 더미
    df['treated'] = (df['대학명'] == TREATED_UNIT).astype(int)

    # 처치 후 더미
    df['post'] = (df['연도'] >= TREATMENT_YEAR).astype(int)

    # DID 상호작용 항
    df['treat_post'] = df['treated'] * df['post']

    # 대학 ID (고정효과용)
    df['unit_id'] = pd.Categorical(df['대학명']).codes

    # 상대연도 (Event Study용)
    df['relative_year'] = df['연도'] - TREATMENT_YEAR

    return df

# ============================================================================
# 4. Wide format 변환 (SCM/SDID용)
# ============================================================================

def create_wide_format(df):
    """경쟁률 Wide format 생성 (대학 × 연도)"""

    pivot = df.pivot(index='대학명', columns='연도', values='경쟁률')

    # 처치군을 첫 번째 행으로
    treated_row = pivot.loc[[TREATED_UNIT]]
    control_rows = pivot.drop(TREATED_UNIT)

    pivot_ordered = pd.concat([treated_row, control_rows])

    return pivot_ordered

# ============================================================================
# 5. 데이터 품질 검사
# ============================================================================

def check_data_quality(df, pivot):
    """데이터 품질 검사"""

    report = []
    report.append("=" * 70)
    report.append("데이터 품질 검사 보고서")
    report.append("=" * 70)

    # 1. 대학별 데이터 수
    report.append("\n[1] 대학별 관측치 수")
    counts = df.groupby('대학명').size()
    for univ, count in counts.items():
        status = "✓" if count == 10 else "✗"
        report.append(f"  {status} {univ}: {count}개")

    # 2. 결측치 확인
    report.append("\n[2] 결측치 현황")
    missing = pivot.isnull().sum(axis=1)
    for univ, miss in missing.items():
        if miss > 0:
            report.append(f"  ✗ {univ}: {miss}개 연도 결측")
    if missing.sum() == 0:
        report.append("  ✓ 결측치 없음")

    # 3. 처치군(한신대) 데이터 확인
    report.append("\n[3] 한신대학교 경쟁률 추이")
    hanshin = pivot.loc[TREATED_UNIT]
    for year, rate in hanshin.items():
        marker = "◀ 계열제 도입" if year == 2023 else ""
        report.append(f"  {year}: {rate:.1f}배 {marker}")

    # 4. 대조군 요약 통계
    report.append("\n[4] 대조군 경쟁률 요약 (처치 전: 2016-2022)")
    control_pivot = pivot.drop(TREATED_UNIT)
    pre_cols = [y for y in pivot.columns if y < TREATMENT_YEAR]
    pre_mean = control_pivot[pre_cols].mean(axis=1)

    report.append(f"  대조군 평균: {pre_mean.mean():.2f}배")
    report.append(f"  대조군 표준편차: {pre_mean.std():.2f}")
    report.append(f"  한신대 평균: {hanshin[pre_cols].mean():.2f}배")

    # 5. 동질성 체크
    report.append("\n[5] 대조군 동질성 (처치 전 평균 경쟁률)")
    for univ, rate in pre_mean.sort_values().items():
        diff = rate - hanshin[pre_cols].mean()
        report.append(f"  {univ}: {rate:.2f}배 (차이: {diff:+.2f})")

    report.append("\n" + "=" * 70)

    return "\n".join(report)

# ============================================================================
# 6. 메인 실행
# ============================================================================

def main():
    print("=" * 70)
    print("SDID 분석 데이터 전처리")
    print("=" * 70)

    # 1. 데이터 로드 및 필터링
    print("\n[1] 데이터 로드 중...")
    df = load_and_filter_data()
    print(f"  → {len(df)}개 관측치 추출 완료")
    print(f"  → {df['대학명'].nunique()}개 대학")

    # 2. 패널 데이터 구축
    print("\n[2] 패널 데이터 구축 중...")
    df_panel = create_panel_data(df)
    print(f"  → Long format: {df_panel.shape}")

    # 3. Wide format 변환
    print("\n[3] Wide format 변환 중...")
    df_wide = create_wide_format(df)
    print(f"  → Wide format: {df_wide.shape}")

    # 4. 데이터 품질 검사
    print("\n[4] 데이터 품질 검사 중...")
    quality_report = check_data_quality(df_panel, df_wide)
    print(quality_report)

    # 5. 데이터 저장
    print("\n[5] 데이터 저장 중...")

    # Long format (분석용)
    df_panel.to_csv('panel_data_long.csv', index=False, encoding='utf-8-sig')
    print("  → panel_data_long.csv 저장 완료")

    # Wide format (SCM/SDID용)
    df_wide.to_csv('panel_data_wide.csv', encoding='utf-8-sig')
    print("  → panel_data_wide.csv 저장 완료")

    # 품질 보고서
    with open('data_quality_report.txt', 'w', encoding='utf-8') as f:
        f.write(quality_report)
    print("  → data_quality_report.txt 저장 완료")

    print("\n" + "=" * 70)
    print("데이터 전처리 완료")
    print("=" * 70)

    return df_panel, df_wide

if __name__ == "__main__":
    df_panel, df_wide = main()
