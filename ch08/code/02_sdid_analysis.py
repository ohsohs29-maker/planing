"""
한신대학교 계열제 효과 분석: DID, SCM, SDID 비교
================================================================================
- 방법론:
  1. 전통적 DID (Two-Way Fixed Effects)
  2. SCM (Synthetic Control Method)
  3. SDID (Synthetic Difference-in-Differences)
- 참고: Arkhangelsky et al. (2021), Abadie et al. (2010)
================================================================================
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 설정
# ============================================================================

TREATED_UNIT = "한신대학교"
TREATMENT_YEAR = 2023

# ============================================================================
# 1. 데이터 로드
# ============================================================================

def load_data():
    """전처리된 데이터 로드"""
    df_long = pd.read_csv('panel_data_long.csv')
    df_wide = pd.read_csv('panel_data_wide.csv', index_col=0)
    return df_long, df_wide

# ============================================================================
# 2. 전통적 DID (TWFE)
# ============================================================================

def traditional_did(df):
    """
    이원 고정효과 DID 추정
    Y_it = α_i + λ_t + δ·D_it + ε_it
    """
    print("\n" + "=" * 70)
    print("2. 전통적 DID (Two-Way Fixed Effects)")
    print("=" * 70)

    # TWFE 회귀분석
    formula = '경쟁률 ~ C(대학명) + C(연도) + treat_post'
    model = smf.ols(formula, data=df).fit()

    # DID 계수 추출
    did_coef = model.params['treat_post']
    did_se = model.bse['treat_post']
    did_pval = model.pvalues['treat_post']
    did_ci = model.conf_int().loc['treat_post']

    print(f"\n  처치효과 (δ): {did_coef:.3f}")
    print(f"  표준오차: {did_se:.3f}")
    print(f"  t-통계량: {did_coef/did_se:.3f}")
    print(f"  p-value: {did_pval:.4f}")
    print(f"  95% CI: [{did_ci[0]:.3f}, {did_ci[1]:.3f}]")
    print(f"  R²: {model.rsquared:.3f}")

    results = {
        'method': 'DID',
        'effect': did_coef,
        'se': did_se,
        'pvalue': did_pval,
        'ci_lower': did_ci[0],
        'ci_upper': did_ci[1]
    }

    return results, model

# ============================================================================
# 3. SCM (Synthetic Control Method)
# ============================================================================

def synthetic_control(df_wide):
    """
    합성통제법: 최적 가중치로 합성 대조군 구성
    min_w ||Y_treated_pre - Y_donors_pre @ w||²
    s.t. Σw = 1, w >= 0
    """
    print("\n" + "=" * 70)
    print("3. SCM (Synthetic Control Method)")
    print("=" * 70)

    # 데이터 준비
    years = df_wide.columns.astype(int)
    pre_years = [y for y in years if y < TREATMENT_YEAR]
    post_years = [y for y in years if y >= TREATMENT_YEAR]

    # 처치군과 대조군 분리
    Y_treated = df_wide.loc[TREATED_UNIT].values
    Y_donors = df_wide.drop(TREATED_UNIT).values

    Y_treated_pre = Y_treated[:len(pre_years)]
    Y_donors_pre = Y_donors[:, :len(pre_years)]

    n_donors = Y_donors.shape[0]
    donor_names = df_wide.drop(TREATED_UNIT).index.tolist()

    # 목적함수: 사전 기간 MSPE 최소화
    def scm_objective(w):
        synthetic = Y_donors_pre.T @ w
        return np.sum((Y_treated_pre - synthetic) ** 2)

    # 제약조건
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_donors

    # 초기값
    w_init = np.ones(n_donors) / n_donors

    # 최적화
    result = minimize(
        scm_objective, w_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    w_optimal = result.x

    # 합성 대조군 계산
    Y_synthetic = Y_donors.T @ w_optimal

    # 처치효과 계산
    treatment_effects = Y_treated - Y_synthetic
    pre_effects = treatment_effects[:len(pre_years)]
    post_effects = treatment_effects[len(pre_years):]

    avg_effect = np.mean(post_effects)

    # MSPE 계산
    mspe_pre = np.mean(pre_effects ** 2)
    rmspe_pre = np.sqrt(mspe_pre)

    # 가중치 집중도 (HHI)
    hhi = np.sum(w_optimal ** 2)

    print(f"\n  [사전 적합도]")
    print(f"  MSPE (사전): {mspe_pre:.4f}")
    print(f"  RMSPE (사전): {rmspe_pre:.4f}")
    print(f"  HHI (가중치 집중도): {hhi:.3f}")

    print(f"\n  [처치효과]")
    print(f"  평균 처치효과: {avg_effect:.3f}")
    for i, year in enumerate(post_years):
        print(f"    {year}년: {post_effects[i]:.3f}")

    print(f"\n  [주요 Donor 가중치]")
    sorted_idx = np.argsort(w_optimal)[::-1]
    for rank, idx in enumerate(sorted_idx[:5], 1):
        if w_optimal[idx] > 0.01:
            print(f"    {rank}. {donor_names[idx]}: {w_optimal[idx]:.3f}")

    results = {
        'method': 'SCM',
        'effect': avg_effect,
        'weights': dict(zip(donor_names, w_optimal)),
        'Y_synthetic': Y_synthetic,
        'mspe_pre': mspe_pre,
        'hhi': hhi,
        'post_effects': post_effects
    }

    return results

# ============================================================================
# 4. SDID (Synthetic Difference-in-Differences)
# ============================================================================

def synthetic_did(df_wide, zeta=0.01):
    """
    합성 이중차분법: 단위 가중치 + 시간 가중치 동시 최적화
    - Arkhangelsky et al. (2021)
    """
    print("\n" + "=" * 70)
    print("4. SDID (Synthetic Difference-in-Differences)")
    print("=" * 70)

    # 데이터 준비
    years = df_wide.columns.astype(int).tolist()
    T = len(years)
    T0 = len([y for y in years if y < TREATMENT_YEAR])  # 사전 기간 길이

    Y_treated = df_wide.loc[TREATED_UNIT].values
    Y_donors = df_wide.drop(TREATED_UNIT).values

    n_donors = Y_donors.shape[0]
    donor_names = df_wide.drop(TREATED_UNIT).index.tolist()

    # -----------------------------------------------------------------------
    # 4.1 단위 가중치 (ω) 최적화: 사전 기간 매칭
    # -----------------------------------------------------------------------

    Y_treated_pre = Y_treated[:T0]
    Y_donors_pre = Y_donors[:, :T0]

    def unit_weight_objective(omega):
        synthetic_pre = Y_donors_pre.T @ omega
        return np.sum((Y_treated_pre - synthetic_pre) ** 2) + zeta * np.sum(omega ** 2)

    constraints_omega = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds_omega = [(0, 1)] * n_donors
    omega_init = np.ones(n_donors) / n_donors

    result_omega = minimize(
        unit_weight_objective, omega_init,
        method='SLSQP',
        bounds=bounds_omega,
        constraints=constraints_omega
    )
    omega = result_omega.x

    # -----------------------------------------------------------------------
    # 4.2 시간 가중치 (λ) 계산: 처치 직전에 높은 가중치
    # -----------------------------------------------------------------------

    # 간단한 방법: 최근 시점에 더 높은 가중치 (지수 감쇠)
    lambda_pre = np.zeros(T)
    decay = 0.8
    for t in range(T0):
        lambda_pre[t] = decay ** (T0 - 1 - t)
    lambda_pre[:T0] = lambda_pre[:T0] / np.sum(lambda_pre[:T0])

    # 사후 기간은 균등 가중치
    lambda_post = np.zeros(T)
    n_post = T - T0
    lambda_post[T0:] = 1.0 / n_post

    # -----------------------------------------------------------------------
    # 4.3 SDID 처치효과 계산
    # -----------------------------------------------------------------------

    # 합성 대조군 (가중 평균)
    Y_synthetic = Y_donors.T @ omega

    # 사전 기간 가중 평균
    treated_pre_weighted = np.sum(lambda_pre[:T0] * Y_treated[:T0])
    synthetic_pre_weighted = np.sum(lambda_pre[:T0] * Y_synthetic[:T0])

    # 사후 기간 가중 평균
    treated_post_weighted = np.sum(lambda_post[T0:] * Y_treated[T0:])
    synthetic_post_weighted = np.sum(lambda_post[T0:] * Y_synthetic[T0:])

    # SDID 효과 = (처치군 사후-사전) - (합성대조군 사후-사전)
    tau_sdid = (treated_post_weighted - treated_pre_weighted) - \
               (synthetic_post_weighted - synthetic_pre_weighted)

    # -----------------------------------------------------------------------
    # 4.4 Bootstrap 표준오차
    # -----------------------------------------------------------------------

    n_bootstrap = 500
    bootstrap_effects = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        # 대조군 리샘플링
        idx = np.random.choice(n_donors, n_donors, replace=True)
        Y_donors_boot = Y_donors[idx]

        # 단위 가중치 재계산
        Y_donors_pre_boot = Y_donors_boot[:, :T0]

        def boot_objective(w):
            synth = Y_donors_pre_boot.T @ w
            return np.sum((Y_treated_pre - synth) ** 2) + zeta * np.sum(w ** 2)

        result_boot = minimize(
            boot_objective, omega_init,
            method='SLSQP',
            bounds=bounds_omega,
            constraints=constraints_omega,
            options={'maxiter': 100}
        )
        omega_boot = result_boot.x

        # 효과 계산
        Y_synth_boot = Y_donors_boot.T @ omega_boot
        t_pre = np.sum(lambda_pre[:T0] * Y_treated[:T0])
        s_pre = np.sum(lambda_pre[:T0] * Y_synth_boot[:T0])
        t_post = np.sum(lambda_post[T0:] * Y_treated[T0:])
        s_post = np.sum(lambda_post[T0:] * Y_synth_boot[T0:])

        tau_boot = (t_post - t_pre) - (s_post - s_pre)
        bootstrap_effects.append(tau_boot)

    se_bootstrap = np.std(bootstrap_effects)
    ci_lower = np.percentile(bootstrap_effects, 2.5)
    ci_upper = np.percentile(bootstrap_effects, 97.5)

    # p-value (양측 검정)
    pvalue = 2 * min(
        np.mean(np.array(bootstrap_effects) >= 0),
        np.mean(np.array(bootstrap_effects) <= 0)
    )

    # -----------------------------------------------------------------------
    # 4.5 결과 출력
    # -----------------------------------------------------------------------

    print(f"\n  [SDID 추정 결과]")
    print(f"  처치효과 (τ): {tau_sdid:.3f}")
    print(f"  표준오차 (Bootstrap): {se_bootstrap:.3f}")
    print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  p-value: {pvalue:.4f}")

    print(f"\n  [시간 가중치 (λ) - 사전 기간]")
    for t, year in enumerate(years[:T0]):
        print(f"    {year}: {lambda_pre[t]:.3f}")

    print(f"\n  [주요 단위 가중치 (ω)]")
    sorted_idx = np.argsort(omega)[::-1]
    for rank, idx in enumerate(sorted_idx[:5], 1):
        if omega[idx] > 0.01:
            print(f"    {rank}. {donor_names[idx]}: {omega[idx]:.3f}")

    # 사전 적합도
    mspe_pre = np.mean((Y_treated[:T0] - Y_synthetic[:T0]) ** 2)
    hhi = np.sum(omega ** 2)

    print(f"\n  [적합도]")
    print(f"  MSPE (사전): {mspe_pre:.4f}")
    print(f"  HHI (가중치 집중도): {hhi:.3f}")

    results = {
        'method': 'SDID',
        'effect': tau_sdid,
        'se': se_bootstrap,
        'pvalue': pvalue,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_effects': bootstrap_effects,
        'omega': dict(zip(donor_names, omega)),
        'lambda_pre': lambda_pre[:T0],
        'Y_synthetic': Y_synthetic,
        'mspe_pre': mspe_pre,
        'hhi': hhi
    }

    return results

# ============================================================================
# 5. 결과 비교 및 시각화
# ============================================================================

def compare_results(did_results, scm_results, sdid_results, df_wide):
    """방법론 간 결과 비교"""

    print("\n" + "=" * 70)
    print("5. 방법론 비교")
    print("=" * 70)

    # 비교 테이블
    print("\n  [추정 결과 비교]")
    print("-" * 70)
    print(f"  {'방법론':<10} {'처치효과':>10} {'표준오차':>10} {'p-value':>10} {'95% CI':>20}")
    print("-" * 70)

    # DID
    print(f"  {'DID':<10} {did_results['effect']:>10.3f} {did_results['se']:>10.3f} "
          f"{did_results['pvalue']:>10.4f} [{did_results['ci_lower']:.2f}, {did_results['ci_upper']:.2f}]")

    # SCM
    print(f"  {'SCM':<10} {scm_results['effect']:>10.3f} {'N/A':>10} "
          f"{'N/A':>10} {'N/A':>20}")

    # SDID
    print(f"  {'SDID':<10} {sdid_results['effect']:>10.3f} {sdid_results['se']:>10.3f} "
          f"{sdid_results['pvalue']:>10.4f} [{sdid_results['ci_lower']:.2f}, {sdid_results['ci_upper']:.2f}]")

    print("-" * 70)

    # 시각화
    years = df_wide.columns.astype(int).tolist()
    Y_treated = df_wide.loc[TREATED_UNIT].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 경쟁률 추이 비교
    ax1 = axes[0, 0]
    ax1.plot(years, Y_treated, 'o-', linewidth=2, markersize=8, label='한신대 (실제)', color='navy')
    ax1.plot(years, scm_results['Y_synthetic'], 's--', linewidth=2, markersize=6, label='합성 한신대 (SCM)', color='green')
    ax1.plot(years, sdid_results['Y_synthetic'], '^--', linewidth=2, markersize=6, label='합성 한신대 (SDID)', color='orange')
    ax1.axvline(x=2022.5, color='red', linestyle='--', alpha=0.7, label='계열제 도입')
    ax1.set_xlabel('연도', fontsize=11)
    ax1.set_ylabel('경쟁률 (배수)', fontsize=11)
    ax1.set_title('경쟁률 추이: 실제 vs 합성 대조군', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (2) 연도별 처치효과
    ax2 = axes[0, 1]
    effects_scm = Y_treated - scm_results['Y_synthetic']
    effects_sdid = Y_treated - sdid_results['Y_synthetic']

    x = np.arange(len(years))
    width = 0.35
    ax2.bar(x - width/2, effects_scm, width, label='SCM', color='green', alpha=0.7)
    ax2.bar(x + width/2, effects_sdid, width, label='SDID', color='orange', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=6.5, color='red', linestyle='--', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years, rotation=45)
    ax2.set_xlabel('연도', fontsize=11)
    ax2.set_ylabel('처치효과 (실제 - 합성)', fontsize=11)
    ax2.set_title('연도별 처치효과', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # (3) 단위 가중치 비교
    ax3 = axes[1, 0]
    donors = list(sdid_results['omega'].keys())
    scm_weights = [scm_results['weights'].get(d, 0) for d in donors]
    sdid_weights = [sdid_results['omega'].get(d, 0) for d in donors]

    # 상위 10개만
    combined = list(zip(donors, scm_weights, sdid_weights))
    combined.sort(key=lambda x: x[2], reverse=True)
    top10 = combined[:10]

    donors_top = [x[0].replace('대학교', '') for x in top10]
    scm_top = [x[1] for x in top10]
    sdid_top = [x[2] for x in top10]

    y_pos = np.arange(len(donors_top))
    ax3.barh(y_pos - 0.2, scm_top, 0.4, label='SCM', color='green', alpha=0.7)
    ax3.barh(y_pos + 0.2, sdid_top, 0.4, label='SDID', color='orange', alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(donors_top)
    ax3.set_xlabel('가중치', fontsize=11)
    ax3.set_title('단위 가중치 (상위 10개)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # (4) 방법론 비교 요약
    ax4 = axes[1, 1]
    methods = ['DID\n(기존)', 'DID\n(확장)', 'SCM', 'SDID']
    effects = [
        -2.775,  # 기존 DID (4개 대학)
        did_results['effect'],
        scm_results['effect'],
        sdid_results['effect']
    ]
    colors = ['gray', 'blue', 'green', 'orange']

    bars = ax4.bar(methods, effects, color=colors, alpha=0.7, edgecolor='black')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    for bar, eff in zip(bars, effects):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.3,
                f'{eff:.2f}', ha='center', va='top', fontsize=10, fontweight='bold', color='white')

    ax4.set_ylabel('처치효과', fontsize=11)
    ax4.set_title('방법론별 처치효과 비교', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    print("\n  → comparison_results.png 저장 완료")
    plt.close()

# ============================================================================
# 6. 메인 실행
# ============================================================================

def main():
    print("=" * 70)
    print("한신대학교 계열제 효과 분석: DID vs SCM vs SDID")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    df_long, df_wide = load_data()
    print(f"  → Long format: {df_long.shape}")
    print(f"  → Wide format: {df_wide.shape}")

    # 2. 전통적 DID
    did_results, did_model = traditional_did(df_long)

    # 3. SCM
    scm_results = synthetic_control(df_wide)

    # 4. SDID
    sdid_results = synthetic_did(df_wide)

    # 5. 결과 비교
    compare_results(did_results, scm_results, sdid_results, df_wide)

    # 6. 결과 저장
    print("\n" + "=" * 70)
    print("6. 결과 저장")
    print("=" * 70)

    results_summary = pd.DataFrame([
        {
            '방법론': 'DID (기존 4개 대학)',
            '처치효과': -2.775,
            '표준오차': 0.990,
            'p-value': 0.0082,
            '95% CI 하한': -4.785,
            '95% CI 상한': -0.765,
            '비고': '대조군 이질성 문제'
        },
        {
            '방법론': 'DID (확장 20개 대학)',
            '처치효과': did_results['effect'],
            '표준오차': did_results['se'],
            'p-value': did_results['pvalue'],
            '95% CI 하한': did_results['ci_lower'],
            '95% CI 상한': did_results['ci_upper'],
            '비고': 'Donor pool 확장'
        },
        {
            '방법론': 'SCM',
            '처치효과': scm_results['effect'],
            '표준오차': np.nan,
            'p-value': np.nan,
            '95% CI 하한': np.nan,
            '95% CI 상한': np.nan,
            '비고': f"MSPE={scm_results['mspe_pre']:.3f}, HHI={scm_results['hhi']:.3f}"
        },
        {
            '방법론': 'SDID',
            '처치효과': sdid_results['effect'],
            '표준오차': sdid_results['se'],
            'p-value': sdid_results['pvalue'],
            '95% CI 하한': sdid_results['ci_lower'],
            '95% CI 상한': sdid_results['ci_upper'],
            '비고': '이중 강건성'
        }
    ])

    results_summary.to_csv('analysis_results.csv', index=False, encoding='utf-8-sig')
    print("  → analysis_results.csv 저장 완료")

    # SDID 가중치 저장
    omega_df = pd.DataFrame({
        '대학명': list(sdid_results['omega'].keys()),
        'SDID_가중치': list(sdid_results['omega'].values()),
        'SCM_가중치': [scm_results['weights'].get(d, 0) for d in sdid_results['omega'].keys()]
    })
    omega_df = omega_df.sort_values('SDID_가중치', ascending=False)
    omega_df.to_csv('unit_weights.csv', index=False, encoding='utf-8-sig')
    print("  → unit_weights.csv 저장 완료")

    print("\n" + "=" * 70)
    print("분석 완료")
    print("=" * 70)

    return did_results, scm_results, sdid_results

if __name__ == "__main__":
    did_results, scm_results, sdid_results = main()
