#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3.2: 데이터 시점 민감도 분석 (Data Cutoff Sensitivity Analysis)
====================================================================

목적:
- 2019, 2020, 2021 각 시점에서의 베이지언 분석 수행
- 처치 효과가 시간에 따라 어떻게 변화하는지 평가
- 조기 데이터만으로도 일관된 결론을 얻을 수 있었는지 검증

분석 시나리오:
1. Cutoff 2019: 계열제 도입 직후 1년 데이터
2. Cutoff 2020: 2년 누적 데이터
3. Cutoff 2021: 전체 데이터 (baseline)

주요 질문:
- 2019년 데이터만으로도 부정적 효과 감지 가능했는가?
- 효과 크기가 시간에 따라 변화하는가?
- 베이즈 팩터가 시간에 따라 증가하는가?

출력:
- data_cutoff_sensitivity_summary.json
- data_cutoff_sensitivity_report.txt
- data_cutoff_sensitivity_plot.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_panel_data():
    """패널 데이터 로드"""
    df = pd.read_csv('panel_data_long.csv', encoding='utf-8-sig')
    df = df.rename(columns={
        '연도': 'year',
        '대학명': 'university',
        '경쟁률': 'competition_rate'
    })
    return df

def compute_simple_did(df, cutoff_year):
    """
    간단한 Difference-in-Differences 추정치 계산

    Parameters:
    -----------
    df : DataFrame
        패널 데이터
    cutoff_year : int
        분석 종료 연도 (inclusive)

    Returns:
    --------
    did_estimate : float
        DID 효과 추정치
    """
    # 데이터 필터링
    df_filtered = df[df['year'] <= cutoff_year].copy()

    # 한신대 vs 대조군
    treated = df_filtered[df_filtered['university'] == '한신대학교']
    controls = df_filtered[df_filtered['university'] != '한신대학교']

    # 처치 전후 구분 (2018 이전 vs 2018 이후)
    treated_pre = treated[treated['year'] < 2018]['competition_rate'].mean()
    treated_post = treated[treated['year'] >= 2018]['competition_rate'].mean()

    # 대조군은 대학별 평균의 평균 (가중치 동일)
    control_pre = controls[controls['year'] < 2018].groupby('university')['competition_rate'].mean().mean()
    control_post = controls[controls['year'] >= 2018].groupby('university')['competition_rate'].mean().mean()

    # NaN 체크
    if pd.isna(treated_pre) or pd.isna(treated_post) or pd.isna(control_pre) or pd.isna(control_post):
        return np.nan

    # DID 추정치
    did_estimate = (treated_post - treated_pre) - (control_post - control_pre)

    return did_estimate

def bootstrap_did_effect(df, cutoff_year, n_bootstrap=500, random_seed=42):
    """
    부트스트랩을 통한 DID 효과 추정치 분포 생성

    Parameters:
    -----------
    df : DataFrame
        패널 데이터
    cutoff_year : int
        분석 종료 연도
    n_bootstrap : int
        부트스트랩 반복 횟수
    random_seed : int
        난수 시드

    Returns:
    --------
    bootstrap_estimates : array
        부트스트랩 효과 추정치 분포
    """
    np.random.seed(random_seed)

    df_filtered = df[df['year'] <= cutoff_year].copy()

    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        # 대학 단위 부트스트랩 (시간 불변 특성 보존)
        universities = df_filtered['university'].unique()
        sampled_universities = np.random.choice(universities, size=len(universities), replace=True)

        # 샘플링된 대학 데이터 결합
        df_boot = pd.concat([df_filtered[df_filtered['university'] == univ] for univ in sampled_universities])

        # DID 추정
        did_boot = compute_simple_did(df_boot, cutoff_year)

        # NaN이 아닌 경우만 추가
        if not pd.isna(did_boot):
            bootstrap_estimates.append(did_boot)

    # 최소 10개 이상의 유효한 부트스트랩 샘플 필요
    if len(bootstrap_estimates) < 10:
        raise ValueError(f"Too few valid bootstrap samples: {len(bootstrap_estimates)}/{n_bootstrap}")

    return np.array(bootstrap_estimates)

def savage_dickey_bf(tau_samples, prior_mean=0, prior_sd=1.0, tau_null=0):
    """
    Savage-Dickey density ratio를 사용한 베이즈 팩터 계산

    BF_10 = prior(tau=0) / posterior(tau=0)
    """
    # 사전분포에서 tau=0의 밀도
    prior_density = stats.norm.pdf(tau_null, loc=prior_mean, scale=prior_sd)

    # 사후분포에서 tau=0의 밀도 (KDE 사용)
    kde = stats.gaussian_kde(tau_samples)
    posterior_density = kde.evaluate([tau_null])[0]

    # BF_10 = prior_density / posterior_density
    bf_10 = prior_density / posterior_density

    return bf_10

def credible_interval_hpd(samples, credibility=0.95):
    """HPD 신용구간 계산"""
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    interval_size = int(np.floor(credibility * n))
    n_intervals = n - interval_size

    interval_widths = sorted_samples[interval_size:] - sorted_samples[:n_intervals]
    min_idx = np.argmin(interval_widths)

    hpd_lower = sorted_samples[min_idx]
    hpd_upper = sorted_samples[min_idx + interval_size]

    return hpd_lower, hpd_upper

def interpret_bayes_factor(bf_10):
    """Kass & Raftery (1995) 해석 척도"""
    if bf_10 > 100:
        return "결정적 증거"
    elif bf_10 > 30:
        return "매우 강한 증거"
    elif bf_10 > 10:
        return "강한 증거"
    elif bf_10 > 3:
        return "실질적 증거"
    elif bf_10 > 1:
        return "약한 증거"
    else:
        return "H0를 지지하는 증거"

def analyze_cutoff_year(df, cutoff_year, prior_mean=0, prior_sd=1.0):
    """
    특정 시점까지의 데이터로 베이지언 분석 수행

    Parameters:
    -----------
    df : DataFrame
        전체 패널 데이터
    cutoff_year : int
        분석 종료 연도
    prior_mean : float
        사전분포 평균
    prior_sd : float
        사전분포 표준편차

    Returns:
    --------
    results : dict
        분석 결과
    """
    print(f"  [{cutoff_year}] 데이터 부트스트랩 중...")
    tau_samples = bootstrap_did_effect(df, cutoff_year)

    print(f"  [{cutoff_year}] 베이즈 팩터 계산 중...")
    bf_10 = savage_dickey_bf(tau_samples, prior_mean, prior_sd)

    print(f"  [{cutoff_year}] 신용구간 계산 중...")
    hpd_lower, hpd_upper = credible_interval_hpd(tau_samples)

    # 사후확률
    prob_negative = np.mean(tau_samples < 0)
    prob_positive = np.mean(tau_samples > 0)

    # 사후 요약통계
    posterior_mean = np.mean(tau_samples)
    posterior_sd = np.std(tau_samples)
    posterior_median = np.median(tau_samples)

    # 증거 강도
    evidence_strength = interpret_bayes_factor(bf_10)

    results = {
        'cutoff_year': cutoff_year,
        'n_years_post_treatment': cutoff_year - 2017,  # 2018년 처치 시작
        'tau_samples': tau_samples,
        'bayes_factor_10': bf_10,
        'log_bf_10': np.log(bf_10) if bf_10 > 0 else -np.inf,
        'evidence_strength': evidence_strength,
        'prob_tau_negative': prob_negative,
        'prob_tau_positive': prob_positive,
        'hpd_95_lower': hpd_lower,
        'hpd_95_upper': hpd_upper,
        'posterior_mean': posterior_mean,
        'posterior_sd': posterior_sd,
        'posterior_median': posterior_median,
        'includes_zero': (hpd_lower < 0 < hpd_upper)
    }

    return results

def main():
    print("=" * 70)
    print("Phase 3.2: 데이터 시점 민감도 분석")
    print("=" * 70)
    print()

    # 1. 패널 데이터 로드
    print("[1] 패널 데이터 로드 중...")
    df = load_panel_data()
    print(f"  총 관측치: {len(df)}")
    print(f"  대학 수: {df['university'].nunique()}")
    print(f"  연도 범위: {df['year'].min()}-{df['year'].max()}")
    print()

    # 2. 시점 시나리오 정의
    print("[2] 분석 시점 정의...")
    cutoff_years = [2019, 2020, 2021]
    print(f"  분석 시점: {cutoff_years}")
    print()

    # 3. 각 시점별 분석
    print("[3] 각 시점별 베이지언 분석 수행...")
    all_results = []

    for cutoff_year in cutoff_years:
        print(f"\n--- Cutoff Year: {cutoff_year} ---")
        results = analyze_cutoff_year(df, cutoff_year)
        all_results.append(results)

        print(f"  평균 효과: {results['posterior_mean']:.3f}")
        print(f"  BF₁₀: {results['bayes_factor_10']:.2e}")
        print(f"  증거 강도: {results['evidence_strength']}")
        print(f"  P(τ<0|data): {results['prob_tau_negative']:.4f}")

    print()

    # 4. 결과 요약
    print("[4] 결과 요약 생성...")
    print()
    print("=" * 80)
    print("데이터 시점별 베이지언 분석 결과")
    print("=" * 80)
    print()

    for res in all_results:
        print(f"시점: {res['cutoff_year']} (처치 후 {res['n_years_post_treatment']}년)")
        print(f"  BF₁₀ = {res['bayes_factor_10']:.2e}")
        print(f"  log(BF₁₀) = {res['log_bf_10']:.2f}")
        print(f"  증거 강도: {res['evidence_strength']}")
        print(f"  P(τ<0|data) = {res['prob_tau_negative']:.4f}")
        print(f"  95% HPD CI: [{res['hpd_95_lower']:.3f}, {res['hpd_95_upper']:.3f}]")
        print(f"  사후 평균: {res['posterior_mean']:.3f} (SD={res['posterior_sd']:.3f})")
        print(f"  0 포함 여부: {res['includes_zero']}")
        print()

    # 5. JSON 저장
    print("[5] 결과 JSON 저장...")
    json_file = 'data_cutoff_sensitivity_summary.json'

    json_results = {
        'analysis_info': {
            'phase': 'Phase 3.2',
            'title': '데이터 시점 민감도 분석',
            'cutoff_years': cutoff_years,
            'treatment_start_year': 2018
        },
        'results_by_year': []
    }

    for res in all_results:
        year_dict = {
            'cutoff_year': int(res['cutoff_year']),
            'n_years_post_treatment': int(res['n_years_post_treatment']),
            'bayes_factor': {
                'bf_10': float(res['bayes_factor_10']),
                'log_bf_10': float(res['log_bf_10']),
                'evidence_strength': res['evidence_strength']
            },
            'posterior_probabilities': {
                'prob_tau_negative': float(res['prob_tau_negative']),
                'prob_tau_positive': float(res['prob_tau_positive'])
            },
            'credible_interval_95': {
                'hpd_lower': float(res['hpd_95_lower']),
                'hpd_upper': float(res['hpd_95_upper']),
                'includes_zero': bool(res['includes_zero'])
            },
            'posterior_summary': {
                'mean': float(res['posterior_mean']),
                'sd': float(res['posterior_sd']),
                'median': float(res['posterior_median'])
            }
        }
        json_results['results_by_year'].append(year_dict)

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)

    print(f"  저장 완료: {json_file}")
    print()

    # 6. 텍스트 보고서
    print("[6] 텍스트 보고서 생성...")
    report_file = 'data_cutoff_sensitivity_report.txt'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("데이터 시점 민감도 분석 보고서\n")
        f.write("=" * 70 + "\n\n")

        f.write("[분석 개요]\n\n")
        f.write(f"  분석 단계: Phase 3.2\n")
        f.write(f"  분석 시점: {cutoff_years}\n")
        f.write(f"  처치 시작 연도: 2018\n")
        f.write(f"  부트스트랩 샘플: 500\n\n")

        f.write("=" * 70 + "\n")
        f.write("[시점별 결과]\n")
        f.write("=" * 70 + "\n\n")

        for res in all_results:
            f.write(f"시점: {res['cutoff_year']} (처치 후 {res['n_years_post_treatment']}년)\n")
            f.write(f"{'─' * 70}\n")
            f.write(f"  BF₁₀ = {res['bayes_factor_10']:.2e}\n")
            f.write(f"  log(BF₁₀) = {res['log_bf_10']:.2f}\n")
            f.write(f"  증거 강도: {res['evidence_strength']}\n")
            f.write(f"  P(τ<0|data) = {res['prob_tau_negative']:.4f}\n")
            f.write(f"  P(τ>0|data) = {res['prob_tau_positive']:.4f}\n")
            f.write(f"  95% HPD CI: [{res['hpd_95_lower']:.3f}, {res['hpd_95_upper']:.3f}]\n")
            f.write(f"  0 포함 여부: {'Yes' if res['includes_zero'] else 'No'}\n")
            f.write(f"  사후 평균: {res['posterior_mean']:.3f}\n")
            f.write(f"  사후 표준편차: {res['posterior_sd']:.3f}\n")
            f.write(f"  사후 중앙값: {res['posterior_median']:.3f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("[주요 발견사항]\n")
        f.write("=" * 70 + "\n\n")

        # 효과 크기 변화
        f.write("1. 효과 크기의 시간적 변화\n")
        effects = [res['posterior_mean'] for res in all_results]
        if len(effects) > 1:
            trend = "증가" if effects[-1] > effects[0] else "감소" if effects[-1] < effects[0] else "유지"
            f.write(f"   → 2019: {effects[0]:.3f} → 2020: {effects[1]:.3f} → 2021: {effects[2]:.3f}\n")
            f.write(f"   → 부정적 효과가 시간에 따라 {trend}\n\n")

        # 베이즈 팩터 변화
        f.write("2. 증거 강도의 시간적 변화\n")
        bfs = [res['bayes_factor_10'] for res in all_results]
        f.write(f"   → 2019: BF={bfs[0]:.2e} ({all_results[0]['evidence_strength']})\n")
        f.write(f"   → 2020: BF={bfs[1]:.2e} ({all_results[1]['evidence_strength']})\n")
        f.write(f"   → 2021: BF={bfs[2]:.2e} ({all_results[2]['evidence_strength']})\n")

        if all([res['bayes_factor_10'] > 100 for res in all_results]):
            f.write("   → 모든 시점에서 결정적 증거\n\n")
        else:
            first_decisive = next((res['cutoff_year'] for res in all_results if res['bayes_factor_10'] > 100), None)
            if first_decisive:
                f.write(f"   → {first_decisive}년부터 결정적 증거 시작\n\n")
            else:
                f.write("   → 결정적 증거 미도달\n\n")

        # 조기 감지 가능성
        f.write("3. 조기 감지 가능성 (2019년 데이터)\n")
        res_2019 = all_results[0]
        if res_2019['bayes_factor_10'] > 10 and res_2019['prob_tau_negative'] > 0.95:
            f.write("   → 2019년 데이터만으로도 부정적 효과 감지 가능\n")
            f.write(f"   → BF₁₀ = {res_2019['bayes_factor_10']:.2e} (강한 증거 이상)\n")
            f.write(f"   → P(τ<0|data) = {res_2019['prob_tau_negative']:.4f} (95% 이상)\n\n")
        else:
            f.write("   → 2019년 데이터만으로는 강한 결론 어려움\n\n")

        f.write("=" * 70 + "\n")
        f.write("[결론]\n")
        f.write("=" * 70 + "\n\n")

        if all([res['bayes_factor_10'] > 100 for res in all_results]):
            f.write("  모든 시점에서 계열제가 경쟁률에 부정적 영향을 미쳤다는\n")
            f.write("  결정적 증거가 관찰되었습니다.\n\n")

            if res_2019['bayes_factor_10'] > 10:
                f.write("  2019년 데이터만으로도 조기에 정책 수정을 권고할 수 있었습니다.\n")
                f.write("  베이지언 적응형 설계를 사용했다면 더 빠른 의사결정이 가능했습니다.\n\n")
        else:
            f.write("  초기 시점에서는 증거가 약하지만,\n")
            f.write("  시간이 지남에 따라 증거가 강화되는 패턴을 보입니다.\n\n")

    print(f"  저장 완료: {report_file}")
    print()

    # 7. 시각화
    print("[7] 시각화 생성...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (1) 베이즈 팩터 변화
    ax1 = axes[0, 0]
    years = [res['cutoff_year'] for res in all_results]
    log_bfs = [res['log_bf_10'] for res in all_results]

    ax1.plot(years, log_bfs, 'o-', linewidth=3, markersize=12, color='steelblue', label='log(BF₁₀)')
    ax1.axhline(np.log(100), color='red', linestyle='--', linewidth=2, label='결정적 증거 기준 (BF=100)')
    ax1.axhline(np.log(10), color='orange', linestyle='--', linewidth=2, label='강한 증거 기준 (BF=10)')
    ax1.set_xlabel('데이터 종료 연도', fontsize=12)
    ax1.set_ylabel('log(BF₁₀)', fontsize=12)
    ax1.set_title('(A) 베이즈 팩터의 시간적 변화', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(years)

    # (2) 사후확률 P(τ<0|data) 변화
    ax2 = axes[0, 1]
    probs = [res['prob_tau_negative'] for res in all_results]

    ax2.plot(years, probs, 's-', linewidth=3, markersize=12, color='darkgreen', label='P(τ<0|data)')
    ax2.axhline(0.95, color='red', linestyle='--', linewidth=2, label='95% 기준')
    ax2.set_xlabel('데이터 종료 연도', fontsize=12)
    ax2.set_ylabel('P(τ<0|data)', fontsize=12)
    ax2.set_ylim([0, 1.05])
    ax2.set_title('(B) 부정적 효과 사후확률 변화', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(years)

    # (3) 95% HPD 신용구간 변화
    ax3 = axes[1, 0]

    for i, res in enumerate(all_results):
        color = 'green' if not res['includes_zero'] else 'orange'
        ax3.plot([res['hpd_95_lower'], res['hpd_95_upper']], [i, i],
                 'o-', linewidth=3, markersize=10, color=color, alpha=0.7,
                 label=f"{res['cutoff_year']}")
        ax3.plot(res['posterior_mean'], i, 'D', markersize=8, color='darkblue')

    ax3.set_yticks(range(len(all_results)))
    ax3.set_yticklabels([f"{res['cutoff_year']}" for res in all_results], fontsize=11)
    ax3.set_xlabel('효과 추정치 (τ)', fontsize=12)
    ax3.set_ylabel('데이터 종료 연도', fontsize=12)
    ax3.set_title('(C) 시점별 95% HPD 신용구간', fontsize=14, fontweight='bold')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='귀무가설 (τ=0)')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(axis='x', alpha=0.3)

    # (4) 사후분포 비교
    ax4 = axes[1, 1]

    colors = ['blue', 'green', 'red']
    for i, res in enumerate(all_results):
        ax4.hist(res['tau_samples'], bins=50, alpha=0.5, color=colors[i],
                 label=f"{res['cutoff_year']}", density=True)

    ax4.axvline(0, color='black', linestyle='--', linewidth=2, label='τ=0')
    ax4.set_xlabel('효과 추정치 (τ)', fontsize=12)
    ax4.set_ylabel('밀도', fontsize=12)
    ax4.set_title('(D) 시점별 사후분포 비교', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plot_file = 'data_cutoff_sensitivity_plot.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  저장 완료: {plot_file}")
    print()

    # 8. 최종 요약
    print("=" * 70)
    print("분석 완료!")
    print("=" * 70)
    print()
    print("생성된 파일:")
    print(f"  1. {json_file} - JSON 요약")
    print(f"  2. {report_file} - 텍스트 보고서")
    print(f"  3. {plot_file} - 시각화")
    print()

    print("주요 결론:")
    if all([res['bayes_factor_10'] > 100 for res in all_results]):
        print("  ✓ 모든 시점에서 결정적 증거")
        if res_2019['bayes_factor_10'] > 10:
            print("  ✓ 2019년 데이터만으로도 조기 감지 가능")
        print("  → 베이지언 적응형 설계로 더 빠른 의사결정 가능했음")
    else:
        print("  ⚠ 초기 시점에서 증거 약함")
        print("  → 시간에 따라 증거 강화")
    print()

if __name__ == '__main__':
    main()
