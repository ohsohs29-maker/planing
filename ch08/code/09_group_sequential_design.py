"""
Group Sequential Design (GSD) 시뮬레이션

베이지언 적응형 실험설계의 핵심 요소인 그룹 순차 설계를 시뮬레이션합니다.
O'Brien-Fleming 경계를 사용하여 조기 종료 시점을 탐색합니다.

출력:
- gsd_simulation_results.csv: 시뮬레이션 결과
- gsd_simulation_plot.png: 시각화
- gsd_analysis_report.txt: 분석 보고서
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_original_data():
    """원본 SDID 데이터 로드"""
    df = pd.read_csv('panel_data_long.csv', encoding='utf-8-sig')

    # 열 이름 영문화
    df = df.rename(columns={
        '연도': 'year',
        '대학명': 'university',
        '경쟁률': 'competition_rate'
    })

    # 처치군(한신대)과 대조군 분리
    treated = df[df['university'] == '한신대학교']
    controls = df[df['university'] != '한신대학교']

    return treated, controls, df

def simulate_data_accumulation(treated, controls, n_looks=5, total_periods=None):
    """
    데이터 누적 과정 시뮬레이션

    Parameters:
    -----------
    treated : DataFrame
        처치군 데이터
    controls : DataFrame
        대조군 데이터
    n_looks : int
        중간 분석 횟수
    total_periods : int
        총 관찰 기간 (None이면 전체 데이터 사용)

    Returns:
    --------
    list : 각 look에서의 데이터셋
    """
    if total_periods is None:
        total_periods = len(treated)

    # Look 시점 계산 (균등 간격)
    look_times = np.linspace(1, total_periods, n_looks, dtype=int)

    datasets = []
    for t in look_times:
        # t 시점까지의 누적 데이터
        treated_subset = treated.iloc[:t]
        controls_subset = controls[controls['year'] <= treated.iloc[t-1]['year']]

        datasets.append({
            'look': len(datasets) + 1,
            'time': t,
            'year': treated.iloc[t-1]['year'],
            'treated': treated_subset,
            'controls': controls_subset,
            'n_treated': len(treated_subset),
            'n_controls': len(controls_subset)
        })

    return datasets

def calculate_cumulative_effect(datasets):
    """
    누적 데이터에서 처치 효과 추정 (단순 차분법)

    실제 SDID는 복잡하므로, 여기서는 단순화된 추정 사용:
    - 처치 후 평균 변화량을 효과로 근사
    """
    results = []

    for data in datasets:
        treated_df = data['treated']

        # 처치 전후 구분 (2019년 이후가 처치)
        pre_treatment = treated_df[treated_df['year'] < 2019]
        post_treatment = treated_df[treated_df['year'] >= 2019]

        if len(post_treatment) == 0:
            # 아직 처치 시점 전
            effect = 0
            se = np.nan
        else:
            # 처치 전후 평균 차이
            pre_mean = pre_treatment['competition_rate'].mean()
            post_mean = post_treatment['competition_rate'].mean()
            effect = post_mean - pre_mean

            # 표준오차 계산 (단순 추정)
            se = np.sqrt(
                pre_treatment['competition_rate'].var() / len(pre_treatment) +
                post_treatment['competition_rate'].var() / len(post_treatment)
            )

        results.append({
            'look': data['look'],
            'time': data['time'],
            'year': data['year'],
            'n_obs': data['n_treated'],
            'effect': effect,
            'se': se,
            'z_score': effect / se if se > 0 else np.nan
        })

    return pd.DataFrame(results)

def obrien_fleming_bounds(n_looks, alpha=0.05, two_sided=True):
    """
    O'Brien-Fleming 경계 계산

    Parameters:
    -----------
    n_looks : int
        중간 분석 횟수
    alpha : float
        전체 유의수준
    two_sided : bool
        양측 검정 여부

    Returns:
    --------
    dict : upper/lower bounds (z-scores)
    """
    # O'Brien-Fleming 경계는 시간의 제곱근에 반비례
    # Z_k = c / sqrt(k/K), where c는 전체 alpha를 보정하는 상수

    # 근사적 O'Brien-Fleming 상수 (Lan-DeMets 근사)
    if two_sided:
        alpha_per_side = alpha / 2
    else:
        alpha_per_side = alpha

    # 최종 시점 경계 (표준 정규분포)
    z_final = stats.norm.ppf(1 - alpha_per_side)

    # 각 look에서의 정보 비율
    information_fractions = np.arange(1, n_looks + 1) / n_looks

    # O'Brien-Fleming 경계
    bounds_upper = z_final / np.sqrt(information_fractions)

    if two_sided:
        bounds_lower = -bounds_upper
    else:
        bounds_lower = np.full(n_looks, -np.inf)

    return {
        'information_fraction': information_fractions,
        'upper': bounds_upper,
        'lower': bounds_lower,
        'alpha': alpha,
        'two_sided': two_sided
    }

def evaluate_stopping_rule(cumulative_results, bounds):
    """
    조기 종료 규칙 평가

    Parameters:
    -----------
    cumulative_results : DataFrame
        누적 분석 결과
    bounds : dict
        O'Brien-Fleming 경계

    Returns:
    --------
    dict : 조기 종료 결정
    """
    stop_decision = None
    stop_look = None

    for idx, row in cumulative_results.iterrows():
        look = int(row['look'])
        z_score = row['z_score']

        if np.isnan(z_score):
            continue

        # 경계 확인
        upper_bound = bounds['upper'][look - 1]
        lower_bound = bounds['lower'][look - 1]

        if z_score >= upper_bound:
            stop_decision = 'reject_H0_positive'
            stop_look = look
            break
        elif z_score <= lower_bound:
            stop_decision = 'reject_H0_negative'
            stop_look = look
            break

    if stop_decision is None:
        stop_decision = 'continue_to_end'
        stop_look = len(cumulative_results)

    return {
        'decision': stop_decision,
        'stop_look': stop_look,
        'total_looks': len(cumulative_results)
    }

def plot_gsd_results(cumulative_results, bounds, stopping_decision, output_file):
    """
    GSD 결과 시각화
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 상단: 효과 크기 추이와 신뢰구간
    ax1 = axes[0]

    looks = cumulative_results['look']
    effects = cumulative_results['effect']
    ses = cumulative_results['se']

    # 95% 신뢰구간
    ci_upper = effects + 1.96 * ses
    ci_lower = effects - 1.96 * ses

    ax1.plot(looks, effects, 'o-', color='steelblue', linewidth=2, markersize=8, label='추정 효과')
    ax1.fill_between(looks, ci_lower, ci_upper, alpha=0.2, color='steelblue', label='95% CI')
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.5, label='τ=0')

    # 조기 종료 시점 표시
    if stopping_decision['decision'] != 'continue_to_end':
        stop_look = stopping_decision['stop_look']
        stop_effect = cumulative_results.iloc[stop_look - 1]['effect']
        ax1.scatter([stop_look], [stop_effect], color='red', s=200, marker='X',
                   label=f"조기 종료 (Look {stop_look})", zorder=5)

    ax1.set_xlabel('중간 분석 시점 (Look)', fontsize=12)
    ax1.set_ylabel('누적 처치 효과 추정치', fontsize=12)
    ax1.set_title('Group Sequential Design: 누적 효과 추이', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 하단: Z-score와 O'Brien-Fleming 경계
    ax2 = axes[1]

    z_scores = cumulative_results['z_score']

    ax2.plot(looks, z_scores, 'o-', color='darkgreen', linewidth=2, markersize=8, label='Z-score')
    ax2.plot(looks, bounds['upper'], 'r--', linewidth=2, label='O\'Brien-Fleming 상한')
    ax2.plot(looks, bounds['lower'], 'r--', linewidth=2, label='O\'Brien-Fleming 하한')
    ax2.axhline(0, color='gray', linestyle='-', linewidth=1)

    # 조기 종료 시점 표시
    if stopping_decision['decision'] != 'continue_to_end':
        stop_look = stopping_decision['stop_look']
        stop_z = cumulative_results.iloc[stop_look - 1]['z_score']
        ax2.scatter([stop_look], [stop_z], color='red', s=200, marker='X',
                   label=f"조기 종료 (Look {stop_look})", zorder=5)

    ax2.set_xlabel('중간 분석 시점 (Look)', fontsize=12)
    ax2.set_ylabel('Z-score', fontsize=12)
    ax2.set_title('O\'Brien-Fleming 경계와 검정 통계량', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ GSD 시각화 저장: {output_file}")
    plt.close()

def main():
    print("=" * 70)
    print("Group Sequential Design (GSD) 시뮬레이션")
    print("=" * 70)

    # 1. 원본 데이터 로드
    print("\n[1] 원본 데이터 로드")
    treated, controls, df = load_original_data()
    print(f"  - 처치군(한신대): {len(treated)} 관찰치")
    print(f"  - 대조군: {len(controls)} 관찰치")
    print(f"  - 기간: {treated['year'].min()} ~ {treated['year'].max()}")

    # 2. GSD 설정
    n_looks = 5
    alpha = 0.05
    print(f"\n[2] GSD 설정")
    print(f"  - 중간 분석 횟수: {n_looks}")
    print(f"  - 전체 유의수준 (α): {alpha}")
    print(f"  - 경계 함수: O'Brien-Fleming")

    # 3. O'Brien-Fleming 경계 계산
    print(f"\n[3] O'Brien-Fleming 경계 계산")
    bounds = obrien_fleming_bounds(n_looks, alpha=alpha, two_sided=True)

    print(f"\n  Look  |  정보비율  |  상한(Z)  |  하한(Z)")
    print(f"  " + "-" * 50)
    for i in range(n_looks):
        print(f"    {i+1}   |    {bounds['information_fraction'][i]:.2f}    |  {bounds['upper'][i]:6.3f}  |  {bounds['lower'][i]:6.3f}")

    # 4. 데이터 누적 시뮬레이션
    print(f"\n[4] 데이터 누적 과정 시뮬레이션")
    datasets = simulate_data_accumulation(treated, controls, n_looks=n_looks)

    for data in datasets:
        print(f"  Look {data['look']}: {data['year']}년까지 ({data['n_treated']} 관찰치)")

    # 5. 누적 효과 추정
    print(f"\n[5] 누적 효과 추정")
    cumulative_results = calculate_cumulative_effect(datasets)

    print(f"\n  Look  |  연도  |  효과  |  SE  |  Z-score")
    print(f"  " + "-" * 60)
    for _, row in cumulative_results.iterrows():
        if not np.isnan(row['z_score']):
            print(f"    {int(row['look'])}   | {int(row['year'])} | {row['effect']:6.3f} | {row['se']:.3f} | {row['z_score']:6.3f}")
        else:
            print(f"    {int(row['look'])}   | {int(row['year'])} | {row['effect']:6.3f} | {row['se']:.3f} |   N/A")

    # 6. 조기 종료 규칙 평가
    print(f"\n[6] 조기 종료 규칙 평가")
    stopping_decision = evaluate_stopping_rule(cumulative_results, bounds)

    print(f"\n  결정: {stopping_decision['decision']}")
    print(f"  종료 시점: Look {stopping_decision['stop_look']} / {stopping_decision['total_looks']}")

    if stopping_decision['decision'] == 'reject_H0_negative':
        print(f"  ✓ 계열제가 경쟁률에 부정적 영향을 미쳤다는 강한 증거")
        print(f"  ✓ Look {stopping_decision['stop_look']}에서 조기 종료 가능")

        # 실제 종료 시점의 데이터 정보
        stop_data = cumulative_results.iloc[stopping_decision['stop_look'] - 1]
        print(f"  - 종료 시점: {int(stop_data['year'])}년")
        print(f"  - 누적 효과: {stop_data['effect']:.3f}")
        print(f"  - Z-score: {stop_data['z_score']:.3f}")
    elif stopping_decision['decision'] == 'reject_H0_positive':
        print(f"  ✓ 계열제가 경쟁률에 긍정적 영향을 미쳤다는 강한 증거")
        print(f"  ✓ Look {stopping_decision['stop_look']}에서 조기 종료 가능")
    else:
        print(f"  ⚠️ 조기 종료 조건 미충족, 최종 시점까지 데이터 수집 필요")

    # 7. 결과 저장
    print(f"\n[7] 결과 저장")

    # CSV 저장
    csv_file = 'gsd_simulation_results.csv'
    cumulative_results.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"  ✓ CSV 저장: {csv_file}")

    # JSON 저장
    json_file = 'gsd_simulation_summary.json'
    summary = {
        'settings': {
            'n_looks': n_looks,
            'alpha': alpha,
            'boundary_function': 'O\'Brien-Fleming',
            'two_sided': bounds['two_sided']
        },
        'bounds': {
            f'look_{i+1}': {
                'information_fraction': float(bounds['information_fraction'][i]),
                'upper_bound': float(bounds['upper'][i]),
                'lower_bound': float(bounds['lower'][i])
            }
            for i in range(n_looks)
        },
        'cumulative_results': cumulative_results.to_dict(orient='records'),
        'stopping_decision': stopping_decision
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON 저장: {json_file}")

    # 텍스트 보고서 저장
    report_file = 'gsd_analysis_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Group Sequential Design (GSD) 분석 보고서\n")
        f.write("=" * 70 + "\n\n")

        f.write("[1] 설계 개요\n\n")
        f.write(f"  중간 분석 횟수: {n_looks}\n")
        f.write(f"  전체 유의수준 (α): {alpha}\n")
        f.write(f"  경계 함수: O'Brien-Fleming\n")
        f.write(f"  검정 유형: {'양측 검정' if bounds['two_sided'] else '단측 검정'}\n\n")

        f.write("[2] 조기 종료 결과\n\n")
        f.write(f"  결정: {stopping_decision['decision']}\n")
        f.write(f"  종료 시점: Look {stopping_decision['stop_look']} / {stopping_decision['total_looks']}\n\n")

        if stopping_decision['decision'] == 'reject_H0_negative':
            f.write("  ✓ 계열제가 경쟁률에 부정적 영향을 미쳤다는 강한 증거\n")
            f.write(f"  ✓ Look {stopping_decision['stop_look']}에서 조기 종료 가능\n\n")

            stop_data = cumulative_results.iloc[stopping_decision['stop_look'] - 1]
            f.write(f"  조기 종료 시점 정보:\n")
            f.write(f"    - 연도: {int(stop_data['year'])}\n")
            f.write(f"    - 누적 효과: {stop_data['effect']:.3f}\n")
            f.write(f"    - 표준오차: {stop_data['se']:.3f}\n")
            f.write(f"    - Z-score: {stop_data['z_score']:.3f}\n")
            f.write(f"    - 경계 (하한): {bounds['lower'][stopping_decision['stop_look'] - 1]:.3f}\n\n")

        f.write("[3] 실무적 함의\n\n")
        if stopping_decision['stop_look'] < stopping_decision['total_looks']:
            saved_looks = stopping_decision['total_looks'] - stopping_decision['stop_look']
            f.write(f"  • 조기 종료로 {saved_looks}회 중간 분석 절약\n")
            f.write(f"  • 리소스 효율성: {saved_looks}/{stopping_decision['total_looks']} = {saved_looks/stopping_decision['total_looks']*100:.1f}% 절감\n")
            f.write(f"  • 빠른 의사결정: {int(stop_data['year'])}년에 정책 수정 가능\n\n")
        else:
            f.write(f"  • 조기 종료 조건 미충족\n")
            f.write(f"  • 최종 시점까지 데이터 수집 필요\n")
            f.write(f"  • 효과가 약하거나 불확실할 가능성\n\n")

        f.write("[4] 베이지언 관점과의 비교\n\n")
        f.write("  • GSD: 빈도주의적 접근, Type I Error 제어\n")
        f.write("  • 베이지언: 사후확률 기반, 직관적 해석\n")
        f.write("  • 공통점: 데이터 누적에 따른 적응적 의사결정\n")
        f.write("  • 차이점: GSD는 경계 함수로 조기 종료 결정\n\n")

    print(f"  ✓ 텍스트 보고서 저장: {report_file}")

    # 8. 시각화
    print(f"\n[8] 시각화 생성")
    plot_file = 'gsd_simulation_plot.png'
    plot_gsd_results(cumulative_results, bounds, stopping_decision, plot_file)

    # 9. 최종 요약
    print("\n" + "=" * 70)
    print("해석 및 결론")
    print("=" * 70)

    print(f"\n[GSD의 장점]")
    print(f"  • 조기 종료 가능성: 리소스 절감 및 빠른 의사결정")
    print(f"  • Type I Error 제어: O'Brien-Fleming 경계로 전체 α 유지")
    print(f"  • 윤리적 이점: 명확한 효과 발견 시 조기 종료")

    print(f"\n[한신대 계열제 실험 적용]")
    if stopping_decision['stop_look'] < stopping_decision['total_looks']:
        stop_year = int(cumulative_results.iloc[stopping_decision['stop_look'] - 1]['year'])
        print(f"  ✓ {stop_year}년에 이미 부정적 효과 감지 가능했음")
        print(f"  ✓ GSD 적용 시 {stopping_decision['total_looks'] - stopping_decision['stop_look']}년 절감")
        print(f"  → 정책 수정을 더 빨리 결정할 수 있었음")
    else:
        print(f"  • 효과가 명확하지 않아 전체 기간 관찰 필요")
        print(f"  • 베이지언 접근으로 보완적 분석 권장")

    print(f"\n[다음 단계]")
    print(f"  • Thompson Sampling 시뮬레이션")
    print(f"  • 베이지언 적응형 randomization")
    print(f"  • 민감도 분석 확장")

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()
