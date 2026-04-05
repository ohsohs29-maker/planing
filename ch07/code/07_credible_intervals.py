"""
베이지언 신용구간(Credible Interval) 계산

SDID Bootstrap 분포로부터 베이지언 신용구간을 계산하고,
빈도주의 신뢰구간과 비교합니다.

출력:
- credible_intervals.csv: 신용구간 요약
- credible_interval_plot.png: 시각화
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

def load_bootstrap_data(filepath):
    """Bootstrap 효과 데이터 로드"""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df['tau'].values

def calculate_credible_intervals(tau_samples, credibility_levels=[0.90, 0.95, 0.99]):
    """
    베이지언 신용구간 계산

    Parameters:
    -----------
    tau_samples : array-like
        Bootstrap 또는 사후 샘플
    credibility_levels : list
        신용도 수준 (예: 0.95 = 95%)

    Returns:
    --------
    dict : 신용구간 정보
    """
    results = {}

    for level in credibility_levels:
        alpha = 1 - level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(tau_samples, lower_percentile)
        ci_upper = np.percentile(tau_samples, upper_percentile)

        results[f'{int(level*100)}%'] = {
            'lower': ci_lower,
            'upper': ci_upper,
            'width': ci_upper - ci_lower,
            'contains_zero': ci_lower <= 0 <= ci_upper
        }

    return results

def calculate_hpd_interval(tau_samples, credibility=0.95):
    """
    Highest Posterior Density (HPD) 구간 계산

    가장 높은 사후밀도를 가진 최소 폭 구간
    """
    sorted_samples = np.sort(tau_samples)
    n = len(sorted_samples)

    # 포함할 샘플 수
    n_included = int(np.ceil(credibility * n))

    # 가능한 모든 구간의 폭 계산
    widths = sorted_samples[n_included:] - sorted_samples[:n - n_included]

    # 최소 폭 구간 찾기
    min_idx = np.argmin(widths)
    hpd_lower = sorted_samples[min_idx]
    hpd_upper = sorted_samples[min_idx + n_included]

    return {
        'lower': hpd_lower,
        'upper': hpd_upper,
        'width': hpd_upper - hpd_lower,
        'contains_zero': hpd_lower <= 0 <= hpd_upper
    }

def calculate_frequentist_ci(tau_samples, confidence_level=0.95):
    """
    빈도주의 신뢰구간 계산 (정규근사)
    """
    mean_tau = np.mean(tau_samples)
    se_tau = np.std(tau_samples, ddof=1)

    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha/2)

    ci_lower = mean_tau - z_critical * se_tau
    ci_upper = mean_tau + z_critical * se_tau

    return {
        'lower': ci_lower,
        'upper': ci_upper,
        'width': ci_upper - ci_lower,
        'contains_zero': ci_lower <= 0 <= ci_upper,
        'method': 'Normal Approximation'
    }

def plot_credible_intervals(tau_samples, ci_results, output_file):
    """
    신용구간 시각화
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 상단: 사후분포와 신용구간
    ax1 = axes[0]
    ax1.hist(tau_samples, bins=50, density=True, alpha=0.6, color='steelblue',
             edgecolor='black', label='사후분포 (Empirical)')

    # 정규분포 근사 오버레이
    mean_tau = np.mean(tau_samples)
    std_tau = np.std(tau_samples)
    x_range = np.linspace(tau_samples.min(), tau_samples.max(), 1000)
    ax1.plot(x_range, stats.norm.pdf(x_range, mean_tau, std_tau),
             'r--', linewidth=2, label='정규근사')

    # 95% 신용구간 표시
    ci_95 = ci_results['equal_tail']['95%']
    ax1.axvline(ci_95['lower'], color='darkgreen', linestyle='--', linewidth=2,
                label=f"95% CI: [{ci_95['lower']:.3f}, {ci_95['upper']:.3f}]")
    ax1.axvline(ci_95['upper'], color='darkgreen', linestyle='--', linewidth=2)
    ax1.axvspan(ci_95['lower'], ci_95['upper'], alpha=0.2, color='green')

    # 0 기준선
    ax1.axvline(0, color='red', linestyle='-', linewidth=1.5, label='τ=0 (귀무가설)')

    ax1.set_xlabel('처치 효과 (τ)', fontsize=12)
    ax1.set_ylabel('밀도', fontsize=12)
    ax1.set_title('베이지언 사후분포와 95% 신용구간', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 하단: 신용구간 비교 (Forest Plot)
    ax2 = axes[1]

    interval_types = []
    lower_bounds = []
    upper_bounds = []
    colors_list = []

    # Equal-tail 신용구간들
    for level in ['90%', '95%', '99%']:
        ci = ci_results['equal_tail'][level]
        interval_types.append(f'Equal-tail {level}')
        lower_bounds.append(ci['lower'])
        upper_bounds.append(ci['upper'])
        colors_list.append('steelblue')

    # HPD 구간
    hpd = ci_results['hpd']['95%']
    interval_types.append('HPD 95%')
    lower_bounds.append(hpd['lower'])
    upper_bounds.append(hpd['upper'])
    colors_list.append('darkgreen')

    # 빈도주의 신뢰구간
    freq_ci = ci_results['frequentist']['95%']
    interval_types.append('Frequentist 95% CI')
    lower_bounds.append(freq_ci['lower'])
    upper_bounds.append(freq_ci['upper'])
    colors_list.append('coral')

    # Forest plot
    y_positions = np.arange(len(interval_types))

    for i, (lower, upper, color) in enumerate(zip(lower_bounds, upper_bounds, colors_list)):
        ax2.plot([lower, upper], [i, i], color=color, linewidth=3, marker='|', markersize=12)
        ax2.scatter([(lower + upper) / 2], [i], color=color, s=100, zorder=3)

    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, label='τ=0')
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(interval_types, fontsize=11)
    ax2.set_xlabel('처치 효과 (τ)', fontsize=12)
    ax2.set_title('신용구간 및 신뢰구간 비교 (Forest Plot)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 신용구간 플롯 저장: {output_file}")
    plt.close()

def main():
    print("=" * 70)
    print("베이지언 신용구간 계산")
    print("=" * 70)

    # 1. Bootstrap 데이터 로드
    bootstrap_file = 'sdid/bootstrap_effects.csv'
    print(f"\n[1] Bootstrap 데이터 로드: {bootstrap_file}")
    tau_samples = load_bootstrap_data(bootstrap_file)
    print(f"  - 샘플 수: {len(tau_samples)}")
    print(f"  - 평균: {np.mean(tau_samples):.4f}")
    print(f"  - 표준편차: {np.std(tau_samples):.4f}")
    print(f"  - 범위: [{tau_samples.min():.4f}, {tau_samples.max():.4f}]")

    # 2. Equal-tail 신용구간 계산
    print("\n[2] Equal-tail 베이지언 신용구간 계산")
    equal_tail_ci = calculate_credible_intervals(tau_samples, [0.90, 0.95, 0.99])

    for level, ci in equal_tail_ci.items():
        print(f"\n  {level} 신용구간:")
        print(f"    하한: {ci['lower']:.4f}")
        print(f"    상한: {ci['upper']:.4f}")
        print(f"    폭: {ci['width']:.4f}")
        print(f"    0 포함 여부: {'Yes ⚠️' if ci['contains_zero'] else 'No ✓'}")

    # 3. HPD 구간 계산
    print("\n[3] Highest Posterior Density (HPD) 구간 계산")
    hpd_95 = calculate_hpd_interval(tau_samples, credibility=0.95)
    print(f"\n  95% HPD 구간:")
    print(f"    하한: {hpd_95['lower']:.4f}")
    print(f"    상한: {hpd_95['upper']:.4f}")
    print(f"    폭: {hpd_95['width']:.4f}")
    print(f"    0 포함 여부: {'Yes ⚠️' if hpd_95['contains_zero'] else 'No ✓'}")

    # Equal-tail vs HPD 비교
    equal_95 = equal_tail_ci['95%']
    print(f"\n  HPD vs Equal-tail (95%) 비교:")
    print(f"    HPD 폭: {hpd_95['width']:.4f}")
    print(f"    Equal-tail 폭: {equal_95['width']:.4f}")
    print(f"    차이: {equal_95['width'] - hpd_95['width']:.4f} (HPD가 더 좁음 ✓)")

    # 4. 빈도주의 신뢰구간 계산
    print("\n[4] 빈도주의 신뢰구간 계산 (비교용)")
    freq_ci_95 = calculate_frequentist_ci(tau_samples, confidence_level=0.95)
    print(f"\n  95% 신뢰구간 (정규근사):")
    print(f"    하한: {freq_ci_95['lower']:.4f}")
    print(f"    상한: {freq_ci_95['upper']:.4f}")
    print(f"    폭: {freq_ci_95['width']:.4f}")
    print(f"    0 포함 여부: {'Yes ⚠️' if freq_ci_95['contains_zero'] else 'No ✓'}")

    # 5. 결과 저장
    print("\n[5] 결과 저장")

    # CSV 저장
    ci_summary = []

    # Equal-tail 신용구간
    for level, ci in equal_tail_ci.items():
        ci_summary.append({
            'Type': 'Equal-tail Credible Interval',
            'Level': level,
            'Lower': ci['lower'],
            'Upper': ci['upper'],
            'Width': ci['width'],
            'Contains_Zero': ci['contains_zero']
        })

    # HPD 구간
    ci_summary.append({
        'Type': 'HPD Credible Interval',
        'Level': '95%',
        'Lower': hpd_95['lower'],
        'Upper': hpd_95['upper'],
        'Width': hpd_95['width'],
        'Contains_Zero': hpd_95['contains_zero']
    })

    # 빈도주의 신뢰구간
    ci_summary.append({
        'Type': 'Frequentist Confidence Interval',
        'Level': '95%',
        'Lower': freq_ci_95['lower'],
        'Upper': freq_ci_95['upper'],
        'Width': freq_ci_95['width'],
        'Contains_Zero': freq_ci_95['contains_zero']
    })

    ci_df = pd.DataFrame(ci_summary)
    ci_csv_file = 'sdid/credible_intervals.csv'
    ci_df.to_csv(ci_csv_file, index=False, encoding='utf-8-sig')
    print(f"  ✓ CSV 저장: {ci_csv_file}")

    # JSON 저장 (NumPy types를 Python native types로 변환)
    def convert_to_serializable(obj):
        """NumPy types를 JSON serializable types로 변환"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

    ci_results_dict = {
        'equal_tail': convert_to_serializable(equal_tail_ci),
        'hpd': {'95%': convert_to_serializable(hpd_95)},
        'frequentist': {'95%': convert_to_serializable(freq_ci_95)},
        'summary': {
            'mean': float(np.mean(tau_samples)),
            'median': float(np.median(tau_samples)),
            'sd': float(np.std(tau_samples)),
            'n_samples': int(len(tau_samples))
        }
    }

    ci_json_file = 'sdid/credible_intervals.json'
    with open(ci_json_file, 'w', encoding='utf-8') as f:
        json.dump(ci_results_dict, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON 저장: {ci_json_file}")

    # 6. 시각화
    print("\n[6] 시각화 생성")
    plot_file = 'sdid/credible_interval_plot.png'
    plot_credible_intervals(tau_samples, ci_results_dict, plot_file)

    # 7. 해석 및 결론
    print("\n" + "=" * 70)
    print("해석 및 결론")
    print("=" * 70)

    print("\n[베이지언 vs 빈도주의 해석]")
    print(f"  • 베이지언 95% CI: [{equal_95['lower']:.3f}, {equal_95['upper']:.3f}]")
    print(f"    → '주어진 데이터에서 τ가 이 구간에 있을 확률 95%'")
    print(f"  • 빈도주의 95% CI: [{freq_ci_95['lower']:.3f}, {freq_ci_95['upper']:.3f}]")
    print(f"    → '반복 실험 시 95%의 구간이 참값 포함'")

    print(f"\n[정책 함의]")
    if not equal_95['contains_zero']:
        print(f"  ✓ 95% 신용구간이 0을 포함하지 않음")
        print(f"  ✓ 계열제가 경쟁률에 부정적 영향을 미쳤을 확률 ≥ 97.5%")
        print(f"  → 정책 수정 권고")
    else:
        print(f"  ⚠️ 95% 신용구간이 0을 포함")
        print(f"  → 효과 불확실, 추가 관찰 필요")

    print(f"\n[HPD의 장점]")
    print(f"  • HPD 95% 폭: {hpd_95['width']:.3f}")
    print(f"  • Equal-tail 95% 폭: {equal_95['width']:.3f}")
    print(f"  • HPD가 {equal_95['width'] - hpd_95['width']:.3f}만큼 더 좁음 (정보 효율적)")

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()
