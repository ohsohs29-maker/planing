"""
베이즈 팩터(Bayes Factor) 계산

귀무가설 H0: τ=0 vs 대립가설 H1: τ≠0 (또는 τ<0)에 대한
베이즈 팩터를 계산하여 증거의 강도를 평가합니다.

Savage-Dickey density ratio 방법 사용:
BF_10 = posterior_density(τ=0) / prior_density(τ=0)

출력:
- bayes_factor_report.txt: 베이즈 팩터 분석 보고서
- bayes_factor_summary.json: JSON 요약
- bayes_factor_visualization.png: 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
import json

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_bootstrap_data(filepath):
    """Bootstrap 효과 데이터 로드"""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df['tau'].values

def savage_dickey_bf(tau_samples, prior_mean=0, prior_sd=1.0, tau_null=0):
    """
    Savage-Dickey density ratio를 사용한 베이즈 팩터 계산

    BF_10 = p(τ=tau_null | data) / p(τ=tau_null)
          = posterior_density(tau_null) / prior_density(tau_null)

    Parameters:
    -----------
    tau_samples : array-like
        사후 샘플 (bootstrap 또는 MCMC)
    prior_mean : float
        사전분포 평균
    prior_sd : float
        사전분포 표준편차
    tau_null : float
        귀무가설 값 (일반적으로 0)

    Returns:
    --------
    dict : 베이즈 팩터 및 관련 정보
    """

    # 사전분포 밀도 at tau_null
    prior_density = stats.norm.pdf(tau_null, loc=prior_mean, scale=prior_sd)

    # 사후분포 밀도 추정 (KDE)
    kde = stats.gaussian_kde(tau_samples)
    posterior_density = kde.evaluate([tau_null])[0]

    # 베이즈 팩터 (H1: τ≠0 vs H0: τ=0)
    # BF_10 = P(data|H1) / P(data|H0)
    # Savage-Dickey: BF_01 = posterior(0) / prior(0)
    # 따라서 BF_10 = 1 / BF_01
    bf_01 = posterior_density / prior_density
    bf_10 = 1 / bf_01

    # 로그 베이즈 팩터
    log_bf_10 = np.log(bf_10)

    # 증거 강도 분류 (Kass & Raftery, 1995)
    if bf_10 < 1:
        evidence_strength = "H0 지지"
        interpretation = "데이터가 귀무가설을 지지"
    elif 1 <= bf_10 < 3:
        evidence_strength = "약한 증거"
        interpretation = "H1에 대한 약한 증거"
    elif 3 <= bf_10 < 10:
        evidence_strength = "실질적 증거"
        interpretation = "H1에 대한 실질적 증거"
    elif 10 <= bf_10 < 30:
        evidence_strength = "강한 증거"
        interpretation = "H1에 대한 강한 증거"
    elif 30 <= bf_10 < 100:
        evidence_strength = "매우 강한 증거"
        interpretation = "H1에 대한 매우 강한 증거"
    else:
        evidence_strength = "결정적 증거"
        interpretation = "H1에 대한 결정적 증거"

    return {
        'bf_10': bf_10,
        'bf_01': bf_01,
        'log_bf_10': log_bf_10,
        'prior_density_at_null': prior_density,
        'posterior_density_at_null': posterior_density,
        'evidence_strength': evidence_strength,
        'interpretation': interpretation,
        'prior_mean': prior_mean,
        'prior_sd': prior_sd,
        'tau_null': tau_null
    }

def calculate_bf_directional(tau_samples, prior_mean=0, prior_sd=1.0):
    """
    방향성 베이즈 팩터 계산

    H0: τ = 0
    H1: τ < 0 (계열제가 경쟁률 하락 유발)

    BF_10 = P(τ<0 | data) / P(τ<0)
    """

    # 사전확률: P(τ<0)
    prior_prob_negative = stats.norm.cdf(0, loc=prior_mean, scale=prior_sd)

    # 사후확률: P(τ<0 | data)
    posterior_prob_negative = np.mean(tau_samples < 0)

    # 베이즈 팩터
    bf_10_directional = (posterior_prob_negative / (1 - posterior_prob_negative)) / \
                        (prior_prob_negative / (1 - prior_prob_negative))

    return {
        'bf_10_directional': bf_10_directional,
        'log_bf_10_directional': np.log(bf_10_directional),
        'prior_prob_negative': prior_prob_negative,
        'posterior_prob_negative': posterior_prob_negative
    }

def plot_bayes_factor(tau_samples, bf_results, output_file):
    """
    베이즈 팩터 시각화
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 상단 좌: 사전분포 vs 사후분포
    ax1 = axes[0, 0]

    # 사후분포 (KDE)
    kde = stats.gaussian_kde(tau_samples)
    x_range = np.linspace(tau_samples.min(), tau_samples.max(), 1000)
    posterior_pdf = kde.evaluate(x_range)

    # 사전분포
    prior_mean = bf_results['point']['prior_mean']
    prior_sd = bf_results['point']['prior_sd']
    prior_pdf = stats.norm.pdf(x_range, loc=prior_mean, scale=prior_sd)

    ax1.plot(x_range, posterior_pdf, 'b-', linewidth=2, label='사후분포 (data 반영)')
    ax1.plot(x_range, prior_pdf, 'r--', linewidth=2, label=f'사전분포 N({prior_mean}, {prior_sd}²)')

    # τ=0 지점 표시
    tau_null = bf_results['point']['tau_null']
    ax1.axvline(tau_null, color='green', linestyle=':', linewidth=2, label='τ=0 (귀무가설)')

    # 밀도 비율 시각화
    prior_density_null = bf_results['point']['prior_density_at_null']
    posterior_density_null = bf_results['point']['posterior_density_at_null']

    ax1.scatter([tau_null], [prior_density_null], color='red', s=200, marker='o',
                label=f'사전밀도(0) = {prior_density_null:.4f}', zorder=5)
    ax1.scatter([tau_null], [posterior_density_null], color='blue', s=200, marker='s',
                label=f'사후밀도(0) = {posterior_density_null:.4f}', zorder=5)

    ax1.set_xlabel('처치 효과 (τ)', fontsize=11)
    ax1.set_ylabel('밀도', fontsize=11)
    ax1.set_title('Savage-Dickey Density Ratio', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 상단 우: 베이즈 팩터 해석
    ax2 = axes[0, 1]
    ax2.axis('off')

    bf_10 = bf_results['point']['bf_10']
    log_bf_10 = bf_results['point']['log_bf_10']
    evidence = bf_results['point']['evidence_strength']

    text_content = f"""
    베이즈 팩터 (BF₁₀) 결과

    BF₁₀ = {bf_10:.2f}
    log(BF₁₀) = {log_bf_10:.2f}

    증거 강도: {evidence}

    해석:
    {bf_results['point']['interpretation']}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Kass & Raftery (1995) 기준:
    BF < 1:     H0 지지
    1-3:        약한 증거
    3-10:       실질적 증거
    10-30:      강한 증거
    30-100:     매우 강한 증거
    > 100:      결정적 증거
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━

    방향성 BF (H1: τ<0):
    BF₁₀ = {bf_results['directional']['bf_10_directional']:.2f}

    P(τ<0 | data) = {bf_results['directional']['posterior_prob_negative']:.3f}
    P(τ<0) = {bf_results['directional']['prior_prob_negative']:.3f}
    """

    ax2.text(0.1, 0.5, text_content, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 하단 좌: 사전분포 민감도 분석
    ax3 = axes[1, 0]

    prior_sds = [0.3, 0.5, 1.0, 2.0, 5.0]
    bf_values = []
    evidence_labels = []

    for sd in prior_sds:
        bf_result = savage_dickey_bf(tau_samples, prior_mean=0, prior_sd=sd, tau_null=0)
        bf_values.append(bf_result['bf_10'])
        evidence_labels.append(bf_result['evidence_strength'])

    colors = ['red' if bf < 1 else 'orange' if bf < 10 else 'green' if bf < 100 else 'darkgreen' for bf in bf_values]

    ax3.barh(range(len(prior_sds)), bf_values, color=colors, edgecolor='black')
    ax3.set_yticks(range(len(prior_sds)))
    ax3.set_yticklabels([f'σ={sd}' for sd in prior_sds], fontsize=10)
    ax3.set_xlabel('BF₁₀', fontsize=11)
    ax3.set_title('사전분포 민감도 분석', fontsize=13, fontweight='bold')
    ax3.axvline(1, color='red', linestyle='--', linewidth=1.5, label='BF=1 (중립)')
    ax3.axvline(10, color='orange', linestyle='--', linewidth=1.5, label='BF=10 (강한 증거)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='x')

    # 로그 스케일 추가 표시
    for i, (bf, label) in enumerate(zip(bf_values, evidence_labels)):
        ax3.text(bf + max(bf_values)*0.05, i, f'{bf:.1f}\n({label})',
                 va='center', fontsize=8)

    # 하단 우: 베이즈 팩터 로그 스케일
    ax4 = axes[1, 1]

    prior_sds_extended = np.linspace(0.1, 5.0, 50)
    bf_values_extended = []

    for sd in prior_sds_extended:
        bf_result = savage_dickey_bf(tau_samples, prior_mean=0, prior_sd=sd, tau_null=0)
        bf_values_extended.append(bf_result['log_bf_10'])

    ax4.plot(prior_sds_extended, bf_values_extended, 'b-', linewidth=2)
    ax4.axhline(0, color='red', linestyle='--', linewidth=1.5, label='log(BF)=0 (BF=1, 중립)')
    ax4.axhline(np.log(10), color='orange', linestyle='--', linewidth=1.5, label='log(BF)=2.3 (BF=10, 강한 증거)')
    ax4.axhline(np.log(100), color='green', linestyle='--', linewidth=1.5, label='log(BF)=4.6 (BF=100, 결정적)')
    ax4.set_xlabel('사전분포 표준편차 (σ)', fontsize=11)
    ax4.set_ylabel('log(BF₁₀)', fontsize=11)
    ax4.set_title('사전분포 σ에 따른 log(BF) 변화', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 베이즈 팩터 시각화 저장: {output_file}")
    plt.close()

def main():
    print("=" * 70)
    print("베이즈 팩터 (Bayes Factor) 계산")
    print("=" * 70)

    # 1. Bootstrap 데이터 로드
    bootstrap_file = 'bootstrap_effects.csv'
    print(f"\n[1] Bootstrap 데이터 로드: {bootstrap_file}")
    tau_samples = load_bootstrap_data(bootstrap_file)
    print(f"  - 샘플 수: {len(tau_samples)}")
    print(f"  - 평균: {np.mean(tau_samples):.4f}")
    print(f"  - P(τ<0): {np.mean(tau_samples < 0):.4f}")

    # 2. Savage-Dickey 베이즈 팩터 계산 (Point null)
    print("\n[2] Savage-Dickey Bayes Factor 계산 (τ=0)")

    # 기본 사전분포: N(0, 1)
    bf_result_point = savage_dickey_bf(tau_samples, prior_mean=0, prior_sd=1.0, tau_null=0)

    print(f"\n  사전분포: N(0, 1²)")
    print(f"  귀무가설: τ = 0")
    print(f"\n  결과:")
    print(f"    BF₁₀ (H1 vs H0): {bf_result_point['bf_10']:.2f}")
    print(f"    log(BF₁₀): {bf_result_point['log_bf_10']:.2f}")
    print(f"    증거 강도: {bf_result_point['evidence_strength']}")
    print(f"    해석: {bf_result_point['interpretation']}")

    # 3. 방향성 베이즈 팩터 (Directional test)
    print("\n[3] 방향성 베이즈 팩터 계산 (H1: τ<0)")

    bf_result_directional = calculate_bf_directional(tau_samples, prior_mean=0, prior_sd=1.0)

    print(f"\n  사전확률 P(τ<0): {bf_result_directional['prior_prob_negative']:.4f}")
    print(f"  사후확률 P(τ<0|data): {bf_result_directional['posterior_prob_negative']:.4f}")
    print(f"\n  BF₁₀ (방향성): {bf_result_directional['bf_10_directional']:.2f}")
    print(f"  log(BF₁₀): {bf_result_directional['log_bf_10_directional']:.2f}")

    # 4. 사전분포 민감도 분석
    print("\n[4] 사전분포 민감도 분석")

    prior_sds = [0.3, 0.5, 1.0, 2.0, 5.0]
    sensitivity_results = []

    print("\n  Prior σ  |  BF₁₀  |  log(BF)  |  증거 강도")
    print("  " + "-" * 55)

    for sd in prior_sds:
        bf_result = savage_dickey_bf(tau_samples, prior_mean=0, prior_sd=sd, tau_null=0)
        sensitivity_results.append(bf_result)

        print(f"    {sd:4.1f}   | {bf_result['bf_10']:6.2f} | {bf_result['log_bf_10']:7.2f}  | {bf_result['evidence_strength']}")

    # 5. 결과 저장
    print("\n[5] 결과 저장")

    # JSON 저장
    bf_summary = {
        'point': bf_result_point,
        'directional': bf_result_directional,
        'sensitivity': [
            {
                'prior_sd': sd,
                'bf_10': result['bf_10'],
                'log_bf_10': result['log_bf_10'],
                'evidence_strength': result['evidence_strength']
            }
            for sd, result in zip(prior_sds, sensitivity_results)
        ]
    }

    json_file = 'bayes_factor_summary.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(bf_summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON 저장: {json_file}")

    # 텍스트 보고서 저장
    report_file = 'bayes_factor_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("베이즈 팩터 분석 보고서\n")
        f.write("=" * 70 + "\n\n")

        f.write("[1] 점 귀무가설 검정 (H0: τ=0 vs H1: τ≠0)\n\n")
        f.write(f"  사전분포: N(0, 1²)\n")
        f.write(f"  BF₁₀ = {bf_result_point['bf_10']:.2f}\n")
        f.write(f"  log(BF₁₀) = {bf_result_point['log_bf_10']:.2f}\n")
        f.write(f"  증거 강도: {bf_result_point['evidence_strength']}\n")
        f.write(f"  해석: {bf_result_point['interpretation']}\n\n")

        f.write("[2] 방향성 검정 (H0: τ=0 vs H1: τ<0)\n\n")
        f.write(f"  사전확률 P(τ<0) = {bf_result_directional['prior_prob_negative']:.4f}\n")
        f.write(f"  사후확률 P(τ<0|data) = {bf_result_directional['posterior_prob_negative']:.4f}\n")
        f.write(f"  BF₁₀ (방향성) = {bf_result_directional['bf_10_directional']:.2f}\n\n")

        f.write("[3] 사전분포 민감도 분석\n\n")
        f.write("  Prior σ  |  BF₁₀  |  증거 강도\n")
        f.write("  " + "-" * 40 + "\n")
        for sd, result in zip(prior_sds, sensitivity_results):
            f.write(f"    {sd:4.1f}   | {result['bf_10']:6.2f} | {result['evidence_strength']}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("결론\n")
        f.write("=" * 70 + "\n\n")

        if bf_result_point['bf_10'] > 100:
            conclusion = "계열제가 경쟁률에 영향을 미쳤다는 결정적 증거가 있습니다."
        elif bf_result_point['bf_10'] > 30:
            conclusion = "계열제가 경쟁률에 영향을 미쳤다는 매우 강한 증거가 있습니다."
        elif bf_result_point['bf_10'] > 10:
            conclusion = "계열제가 경쟁률에 영향을 미쳤다는 강한 증거가 있습니다."
        elif bf_result_point['bf_10'] > 3:
            conclusion = "계열제가 경쟁률에 영향을 미쳤다는 실질적 증거가 있습니다."
        else:
            conclusion = "증거가 불충분하거나 약합니다."

        f.write(f"  {conclusion}\n\n")

        if bf_result_directional['posterior_prob_negative'] > 0.95:
            f.write(f"  사후확률 분석:\n")
            f.write(f"  - P(τ<0|data) = {bf_result_directional['posterior_prob_negative']:.3f}\n")
            f.write(f"  - 계열제가 경쟁률을 하락시켰을 확률이 95% 이상입니다.\n")
            f.write(f"  - 정책 수정을 강력히 권고합니다.\n\n")

    print(f"  ✓ 텍스트 보고서 저장: {report_file}")

    # 6. 시각화
    print("\n[6] 시각화 생성")
    plot_file = 'bayes_factor_visualization.png'
    plot_bayes_factor(tau_samples, bf_summary, plot_file)

    # 7. 최종 요약
    print("\n" + "=" * 70)
    print("최종 요약")
    print("=" * 70)

    print(f"\n[베이즈 팩터 해석]")
    print(f"  • BF₁₀ = {bf_result_point['bf_10']:.2f}")
    print(f"  • {bf_result_point['interpretation']}")

    if bf_result_point['bf_10'] > 10:
        print(f"\n  ✓ 계열제가 경쟁률에 영향을 미쳤다는 강한 증거")
        print(f"  ✓ 빈도주의 p-value와 일관된 결론")
    else:
        print(f"\n  ⚠️ 증거가 다소 약함, 사전분포 민감도 고려 필요")

    print(f"\n[방향성 분석]")
    print(f"  • P(τ<0|data) = {bf_result_directional['posterior_prob_negative']:.3f}")
    print(f"  • 계열제가 경쟁률을 하락시켰을 확률 ≈ {bf_result_directional['posterior_prob_negative']*100:.1f}%")

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()
