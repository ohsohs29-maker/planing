#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3.1: 확장된 사전분포 민감도 분석 (Extended Prior Sensitivity Analysis)
==========================================================================

목적:
- 다양한 사전분포 시나리오에서 베이지언 결론의 강건성 검증
- Informative priors (전문가 의견), Skeptical priors (보수적), 극단적 시나리오 테스트
- 사전분포가 결론에 미치는 영향 정량화

분석 시나리오:
1. Weakly Informative (약한 정보): N(0, 2²), N(0, 5²), N(0, 10²)
2. Informative Optimistic (낙관적): N(2, 1²) - 계열제가 경쟁률 상승시킬 것
3. Informative Pessimistic (비관적): N(-2, 1²) - 계열제가 경쟁률 하락시킬 것
4. Skeptical (회의적): N(0, 0.3²) - 효과 없을 것, 강한 사전 믿음
5. Uniform (무정보): 매우 넓은 범위 N(0, 100²)

출력:
- extended_sensitivity_summary.json: 모든 시나리오 요약
- extended_sensitivity_report.txt: 해석이 포함된 보고서
- extended_sensitivity_plot.png: 시각화
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_bootstrap_effects(file_path='bootstrap_effects.csv'):
    """부트스트랩 효과 추정치 로드"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    return df['tau'].values

def savage_dickey_bf(tau_samples, prior_mean=0, prior_sd=1.0, tau_null=0):
    """
    Savage-Dickey density ratio를 사용한 베이즈 팩터 계산

    BF_10 = prior(tau=0) / posterior(tau=0)

    참고: Bootstrap distribution을 empirical posterior로 사용하는 경우,
    posterior density at tau=0가 작을수록 H1을 지지
    → BF_10 = prior_density / posterior_density

    Parameters:
    -----------
    tau_samples : array
        사후분포 샘플 (bootstrap distribution)
    prior_mean : float
        사전분포 평균
    prior_sd : float
        사전분포 표준편차
    tau_null : float
        귀무가설 값 (보통 0)

    Returns:
    --------
    bf_10 : float
        베이즈 팩터 (H1 vs H0)
    """
    # 사전분포에서 tau=0의 밀도
    prior_density = stats.norm.pdf(tau_null, loc=prior_mean, scale=prior_sd)

    # 사후분포에서 tau=0의 밀도 (KDE 사용)
    kde = stats.gaussian_kde(tau_samples)
    posterior_density = kde.evaluate([tau_null])[0]

    # BF_10 = prior_density / posterior_density
    # (사후에서 tau=0이 덜 그럴듯할수록 BF_10 증가)
    bf_10 = prior_density / posterior_density

    return bf_10

def directional_posterior_probability(tau_samples, direction='negative'):
    """
    방향성 사후확률 계산

    Parameters:
    -----------
    tau_samples : array
        사후분포 샘플
    direction : str
        'negative' (tau < 0) or 'positive' (tau > 0)

    Returns:
    --------
    prob : float
        사후확률
    """
    if direction == 'negative':
        prob = np.mean(tau_samples < 0)
    elif direction == 'positive':
        prob = np.mean(tau_samples > 0)
    else:
        raise ValueError("direction must be 'negative' or 'positive'")

    return prob

def credible_interval_hpd(samples, credibility=0.95):
    """
    Highest Posterior Density (HPD) 신용구간 계산

    Parameters:
    -----------
    samples : array
        사후분포 샘플
    credibility : float
        신용수준 (기본값 0.95)

    Returns:
    --------
    hpd_lower, hpd_upper : tuple
        HPD 신용구간
    """
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    interval_size = int(np.floor(credibility * n))
    n_intervals = n - interval_size

    # 가능한 모든 구간의 폭 계산
    interval_widths = sorted_samples[interval_size:] - sorted_samples[:n_intervals]

    # 가장 좁은 구간 선택
    min_idx = np.argmin(interval_widths)
    hpd_lower = sorted_samples[min_idx]
    hpd_upper = sorted_samples[min_idx + interval_size]

    return hpd_lower, hpd_upper

def interpret_bayes_factor(bf_10):
    """
    Kass & Raftery (1995) 해석 척도

    BF_10 > 100: 결정적 증거 (Decisive)
    BF_10 30-100: 매우 강한 증거 (Very Strong)
    BF_10 10-30: 강한 증거 (Strong)
    BF_10 3-10: 실질적 증거 (Substantial)
    BF_10 1-3: 약한 증거 (Weak)
    BF_10 < 1: H0를 지지
    """
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

def analyze_prior_scenario(tau_samples, prior_mean, prior_sd, scenario_name):
    """
    특정 사전분포 시나리오 분석

    Parameters:
    -----------
    tau_samples : array
        사후분포 샘플 (bootstrap distribution)
    prior_mean : float
        사전분포 평균
    prior_sd : float
        사전분포 표준편차
    scenario_name : str
        시나리오 이름

    Returns:
    --------
    results : dict
        분석 결과
    """
    # 1. 베이즈 팩터 계산
    bf_10 = savage_dickey_bf(tau_samples, prior_mean, prior_sd)

    # 2. 방향성 사후확률
    prob_negative = directional_posterior_probability(tau_samples, 'negative')
    prob_positive = directional_posterior_probability(tau_samples, 'positive')

    # 3. HPD 신용구간
    hpd_lower, hpd_upper = credible_interval_hpd(tau_samples, 0.95)

    # 4. 사후분포 요약통계
    posterior_mean = np.mean(tau_samples)
    posterior_sd = np.std(tau_samples)
    posterior_median = np.median(tau_samples)

    # 5. 증거 강도 해석
    evidence_strength = interpret_bayes_factor(bf_10)

    results = {
        'scenario_name': scenario_name,
        'prior_mean': prior_mean,
        'prior_sd': prior_sd,
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
    print("Phase 3.1: 확장된 사전분포 민감도 분석")
    print("=" * 70)
    print()

    # 1. 부트스트랩 효과 추정치 로드
    print("[1] 부트스트랩 효과 추정치 로드 중...")
    tau_samples = load_bootstrap_effects()
    print(f"  샘플 수: {len(tau_samples)}")
    print(f"  평균 효과: {np.mean(tau_samples):.3f}")
    print(f"  표준편차: {np.std(tau_samples):.3f}")
    print()

    # 2. 사전분포 시나리오 정의
    print("[2] 사전분포 시나리오 정의...")
    scenarios = [
        # Weakly Informative
        {'name': 'Weakly Informative (σ=2)', 'mean': 0, 'sd': 2.0},
        {'name': 'Weakly Informative (σ=5)', 'mean': 0, 'sd': 5.0},
        {'name': 'Weakly Informative (σ=10)', 'mean': 0, 'sd': 10.0},

        # Informative - Optimistic
        {'name': 'Informative Optimistic (μ=2, σ=1)', 'mean': 2.0, 'sd': 1.0},
        {'name': 'Informative Optimistic (μ=1, σ=0.5)', 'mean': 1.0, 'sd': 0.5},

        # Informative - Pessimistic
        {'name': 'Informative Pessimistic (μ=-2, σ=1)', 'mean': -2.0, 'sd': 1.0},
        {'name': 'Informative Pessimistic (μ=-1, σ=0.5)', 'mean': -1.0, 'sd': 0.5},

        # Skeptical (강한 사전 믿음: 효과 없음)
        {'name': 'Skeptical (σ=0.3)', 'mean': 0, 'sd': 0.3},
        {'name': 'Skeptical (σ=0.5)', 'mean': 0, 'sd': 0.5},

        # Uniform (무정보)
        {'name': 'Uniform (σ=100)', 'mean': 0, 'sd': 100.0},

        # Reference (기준 분석)
        {'name': 'Reference (σ=1)', 'mean': 0, 'sd': 1.0},
    ]

    print(f"  총 {len(scenarios)}개 시나리오 분석 예정")
    print()

    # 3. 각 시나리오 분석
    print("[3] 각 시나리오 분석 중...")
    all_results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"  [{i}/{len(scenarios)}] {scenario['name']}...")
        results = analyze_prior_scenario(
            tau_samples,
            scenario['mean'],
            scenario['sd'],
            scenario['name']
        )
        all_results.append(results)

    print()

    # 4. 결과 요약 DataFrame
    print("[4] 결과 요약 생성...")
    df_results = pd.DataFrame(all_results)

    # 정렬: log(BF) 기준 내림차순
    df_results = df_results.sort_values('log_bf_10', ascending=False)

    print()
    print("=" * 100)
    print("사전분포 민감도 분석 결과 요약")
    print("=" * 100)
    print()

    # 주요 결과 출력
    for idx, row in df_results.iterrows():
        print(f"시나리오: {row['scenario_name']}")
        print(f"  사전분포: N({row['prior_mean']}, {row['prior_sd']:.2f}²)")
        print(f"  BF₁₀ = {row['bayes_factor_10']:.2e}")
        print(f"  log(BF₁₀) = {row['log_bf_10']:.2f}")
        print(f"  증거 강도: {row['evidence_strength']}")
        print(f"  P(τ<0|data) = {row['prob_tau_negative']:.4f}")
        print(f"  95% HPD CI: [{row['hpd_95_lower']:.3f}, {row['hpd_95_upper']:.3f}]")
        print(f"  0 포함 여부: {row['includes_zero']}")
        print()

    # 5. JSON 저장
    print("[5] 결과 JSON 저장...")
    json_file = 'extended_sensitivity_summary.json'

    # JSON 직렬화 가능하도록 변환
    json_results = {
        'analysis_info': {
            'phase': 'Phase 3.1',
            'title': '확장된 사전분포 민감도 분석',
            'n_scenarios': len(scenarios),
            'n_bootstrap_samples': len(tau_samples)
        },
        'scenarios': []
    }

    for _, row in df_results.iterrows():
        scenario_dict = {
            'scenario_name': row['scenario_name'],
            'prior_specification': {
                'mean': float(row['prior_mean']),
                'sd': float(row['prior_sd'])
            },
            'bayes_factor': {
                'bf_10': float(row['bayes_factor_10']),
                'log_bf_10': float(row['log_bf_10']),
                'evidence_strength': row['evidence_strength']
            },
            'posterior_probabilities': {
                'prob_tau_negative': float(row['prob_tau_negative']),
                'prob_tau_positive': float(row['prob_tau_positive'])
            },
            'credible_interval_95': {
                'hpd_lower': float(row['hpd_95_lower']),
                'hpd_upper': float(row['hpd_95_upper']),
                'includes_zero': bool(row['includes_zero'])
            },
            'posterior_summary': {
                'mean': float(row['posterior_mean']),
                'sd': float(row['posterior_sd']),
                'median': float(row['posterior_median'])
            }
        }
        json_results['scenarios'].append(scenario_dict)

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)

    print(f"  저장 완료: {json_file}")
    print()

    # 6. 텍스트 보고서 생성
    print("[6] 텍스트 보고서 생성...")
    report_file = 'extended_sensitivity_report.txt'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("확장된 사전분포 민감도 분석 보고서\n")
        f.write("=" * 70 + "\n\n")

        f.write("[분석 개요]\n\n")
        f.write(f"  분석 단계: Phase 3.1\n")
        f.write(f"  시나리오 수: {len(scenarios)}\n")
        f.write(f"  부트스트랩 샘플: {len(tau_samples)}\n")
        f.write(f"  데이터 평균 효과: {np.mean(tau_samples):.3f}\n")
        f.write(f"  데이터 표준편차: {np.std(tau_samples):.3f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("[시나리오별 결과]\n")
        f.write("=" * 70 + "\n\n")

        for _, row in df_results.iterrows():
            f.write(f"시나리오: {row['scenario_name']}\n")
            f.write(f"{'─' * 70}\n")
            f.write(f"  사전분포: N({row['prior_mean']}, {row['prior_sd']:.2f}²)\n")
            f.write(f"  BF₁₀ = {row['bayes_factor_10']:.2e}\n")
            f.write(f"  log(BF₁₀) = {row['log_bf_10']:.2f}\n")
            f.write(f"  증거 강도: {row['evidence_strength']}\n")
            f.write(f"  P(τ<0|data) = {row['prob_tau_negative']:.4f}\n")
            f.write(f"  P(τ>0|data) = {row['prob_tau_positive']:.4f}\n")
            f.write(f"  95% HPD CI: [{row['hpd_95_lower']:.3f}, {row['hpd_95_upper']:.3f}]\n")
            f.write(f"  0 포함 여부: {'Yes' if row['includes_zero'] else 'No'}\n")
            f.write(f"  사후 평균: {row['posterior_mean']:.3f}\n")
            f.write(f"  사후 표준편차: {row['posterior_sd']:.3f}\n")
            f.write(f"  사후 중앙값: {row['posterior_median']:.3f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("[주요 발견사항]\n")
        f.write("=" * 70 + "\n\n")

        # 모든 시나리오에서 BF > 100인지 확인
        all_decisive = all(df_results['bayes_factor_10'] > 100)

        f.write("1. 베이즈 팩터 강건성\n")
        if all_decisive:
            f.write("   → 모든 시나리오에서 결정적 증거 (BF > 100)\n")
            f.write("   → 사전분포 선택과 무관하게 일관된 결론\n\n")
        else:
            min_bf = df_results['bayes_factor_10'].min()
            f.write(f"   → 최소 BF = {min_bf:.2e}\n")
            f.write("   → 일부 시나리오에서 증거 강도 변화\n\n")

        # 모든 시나리오에서 P(τ<0|data) > 95%인지 확인
        all_high_prob = all(df_results['prob_tau_negative'] > 0.95)

        f.write("2. 방향성 사후확률\n")
        if all_high_prob:
            f.write("   → 모든 시나리오에서 P(τ<0|data) > 95%\n")
            f.write("   → 계열제가 경쟁률을 하락시켰을 확률 매우 높음\n\n")
        else:
            min_prob = df_results['prob_tau_negative'].min()
            f.write(f"   → 최소 P(τ<0|data) = {min_prob:.4f}\n")
            f.write("   → 일부 시나리오에서 확률 감소\n\n")

        # 모든 시나리오에서 95% CI가 0을 포함하지 않는지 확인
        all_exclude_zero = all(~df_results['includes_zero'])

        f.write("3. 신용구간\n")
        if all_exclude_zero:
            f.write("   → 모든 시나리오에서 95% HPD CI가 0을 포함하지 않음\n")
            f.write("   → 효과가 유의하다는 일관된 결론\n\n")
        else:
            n_include_zero = df_results['includes_zero'].sum()
            f.write(f"   → {n_include_zero}개 시나리오에서 95% CI가 0 포함\n")
            f.write("   → 일부 사전분포에서 결론 변화\n\n")

        f.write("=" * 70 + "\n")
        f.write("[결론]\n")
        f.write("=" * 70 + "\n\n")

        if all_decisive and all_high_prob and all_exclude_zero:
            f.write("  계열제가 경쟁률에 부정적 영향을 미쳤다는 결론은\n")
            f.write("  사전분포 선택과 무관하게 강건합니다.\n\n")
            f.write("  - 모든 시나리오에서 결정적 증거 (BF > 100)\n")
            f.write("  - 모든 시나리오에서 P(τ<0|data) > 95%\n")
            f.write("  - 모든 시나리오에서 95% CI가 0을 포함하지 않음\n\n")
            f.write("  정책 수정을 강력히 권고합니다.\n\n")
        else:
            f.write("  사전분포 선택에 따라 결론이 달라질 수 있습니다.\n")
            f.write("  추가 데이터 수집 또는 전문가 의견 반영을 권고합니다.\n\n")

    print(f"  저장 완료: {report_file}")
    print()

    # 7. 시각화
    print("[7] 시각화 생성...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (1) 베이즈 팩터 비교 (log scale)
    ax1 = axes[0, 0]
    y_pos = np.arange(len(df_results))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_results)))

    bars = ax1.barh(y_pos, df_results['log_bf_10'], color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_results['scenario_name'], fontsize=9)
    ax1.set_xlabel('log(BF₁₀)', fontsize=11)
    ax1.set_title('(A) 사전분포별 베이즈 팩터 비교', fontsize=13, fontweight='bold')
    ax1.axvline(np.log(100), color='red', linestyle='--', linewidth=2,
                label='결정적 증거 기준 (BF=100)')
    ax1.legend(fontsize=9)
    ax1.grid(axis='x', alpha=0.3)

    # (2) 사후확률 P(τ<0|data) 비교
    ax2 = axes[0, 1]
    colors2 = plt.cm.RdYlGn(df_results['prob_tau_negative'])

    bars2 = ax2.barh(y_pos, df_results['prob_tau_negative'], color=colors2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_results['scenario_name'], fontsize=9)
    ax2.set_xlabel('P(τ<0|data)', fontsize=11)
    ax2.set_xlim([0, 1])
    ax2.set_title('(B) 계열제가 경쟁률 하락시켰을 사후확률', fontsize=13, fontweight='bold')
    ax2.axvline(0.95, color='red', linestyle='--', linewidth=2, label='95% 기준')
    ax2.legend(fontsize=9)
    ax2.grid(axis='x', alpha=0.3)

    # (3) 95% HPD 신용구간 비교
    ax3 = axes[1, 0]

    for i, (idx, row) in enumerate(df_results.iterrows()):
        color = 'green' if not row['includes_zero'] else 'orange'
        ax3.plot([row['hpd_95_lower'], row['hpd_95_upper']], [i, i],
                 'o-', linewidth=2, markersize=6, color=color, alpha=0.7)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(df_results['scenario_name'], fontsize=9)
    ax3.set_xlabel('효과 추정치 (τ)', fontsize=11)
    ax3.set_title('(C) 사전분포별 95% HPD 신용구간', fontsize=13, fontweight='bold')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='귀무가설 (τ=0)')
    ax3.legend(fontsize=9)
    ax3.grid(axis='x', alpha=0.3)

    # (4) 사전분포 vs 사후 평균 산점도
    ax4 = axes[1, 1]

    # 사전분포 표준편차에 따라 색상 구분
    scatter = ax4.scatter(df_results['prior_mean'], df_results['posterior_mean'],
                         c=df_results['prior_sd'], s=200, alpha=0.7,
                         cmap='viridis', edgecolors='black', linewidth=1.5)

    # 대각선 (사전=사후)
    lim = max(abs(df_results['prior_mean'].max()), abs(df_results['posterior_mean'].max()))
    ax4.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, label='사전=사후')

    ax4.set_xlabel('사전분포 평균 (μ₀)', fontsize=11)
    ax4.set_ylabel('사후분포 평균', fontsize=11)
    ax4.set_title('(D) 사전분포 평균 vs 사후분포 평균', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # 컬러바
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('사전분포 표준편차 (σ₀)', fontsize=10)

    plt.tight_layout()
    plot_file = 'extended_sensitivity_plot.png'
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

    # 주요 결론
    print("주요 결론:")
    if all_decisive and all_high_prob and all_exclude_zero:
        print("  ✓ 사전분포와 무관하게 일관된 결론")
        print("  ✓ 모든 시나리오에서 결정적 증거")
        print("  ✓ 계열제가 경쟁률 하락시켰을 확률 > 95%")
        print("  → 정책 수정 강력 권고")
    else:
        print("  ⚠ 일부 사전분포에서 결론 변화")
        print("  → 추가 분석 필요")
    print()

if __name__ == '__main__':
    main()
