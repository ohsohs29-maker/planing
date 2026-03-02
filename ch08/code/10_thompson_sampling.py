"""
Thompson Sampling 시뮬레이션

베이지언 Multi-Armed Bandits를 사용한 Response-Adaptive Randomization (RAR) 시뮬레이션
여러 정책 대안(계열제 variants)에 대한 적응형 할당 전략을 평가합니다.

출력:
- thompson_sampling_results.csv: 시뮬레이션 결과
- thompson_sampling_plot.png: 시각화
- thompson_sampling_report.txt: 분석 보고서
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)  # 재현성

def define_policy_arms():
    """
    정책 대안 정의

    3가지 시나리오:
    1. 계열제 (현행): 평균 경쟁률 하락
    2. 학과제 (복원): 평균 경쟁률 회복
    3. 혼합제 (하이브리드): 중간 효과
    """
    arms = {
        '계열제 (현행)': {
            'true_mean': -2.0,  # 경쟁률 변화량
            'true_sd': 0.8
        },
        '학과제 (복원)': {
            'true_mean': 1.5,  # 경쟁률 회복
            'true_sd': 1.0
        },
        '혼합제 (하이브리드)': {
            'true_mean': 0.5,  # 작은 개선
            'true_sd': 0.9
        }
    }
    return arms

def initialize_priors(arms, prior_mean=0, prior_sd=2.0):
    """
    사전분포 초기화 (약한 정보 사전분포)

    Normal-Normal conjugate:
    Prior: N(μ₀, σ₀²)
    Posterior: N(μₙ, σₙ²)
    """
    priors = {}
    for arm_name in arms.keys():
        priors[arm_name] = {
            'mu': prior_mean,  # 사전 평균
            'sigma': prior_sd,  # 사전 표준편차
            'n_obs': 0,  # 관찰 횟수
            'sum_rewards': 0,  # 보상 합계
            'sum_sq_rewards': 0  # 보상 제곱 합계
        }
    return priors

def sample_reward(arm, true_params):
    """
    주어진 정책(arm)에서 보상 샘플링

    Parameters:
    -----------
    arm : str
        정책 이름
    true_params : dict
        각 정책의 true distribution parameters

    Returns:
    --------
    float : 경쟁률 변화량 샘플
    """
    params = true_params[arm]
    reward = np.random.normal(params['true_mean'], params['true_sd'])
    return reward

def thompson_sampling_update(posteriors, arm, reward, data_sd=1.0):
    """
    Thompson Sampling: 사후분포 업데이트 (Normal-Normal conjugate)

    Known variance 가정 하에 posterior update:
    μₙ = (σ₀² * Σx + σ² * μ₀) / (n*σ₀² + σ²)
    σₙ² = (σ₀² * σ²) / (n*σ₀² + σ²)

    Parameters:
    -----------
    posteriors : dict
        현재 사후분포 파라미터
    arm : str
        선택된 정책
    reward : float
        관찰된 보상
    data_sd : float
        데이터 분산 (known)
    """
    post = posteriors[arm]

    # 관찰 추가
    post['n_obs'] += 1
    post['sum_rewards'] += reward

    # Posterior update (Normal-Normal conjugate, known variance)
    n = post['n_obs']
    prior_mu = 0  # 초기 사전 평균
    prior_var = 4.0  # 초기 사전 분산 (σ₀² = 2²)
    data_var = data_sd ** 2

    # Posterior 평균
    posterior_var = 1 / (1/prior_var + n/data_var)
    posterior_mu = posterior_var * (prior_mu/prior_var + post['sum_rewards']/data_var)

    # 업데이트
    post['mu'] = posterior_mu
    post['sigma'] = np.sqrt(posterior_var)

def select_arm_thompson(posteriors):
    """
    Thompson Sampling: 사후분포에서 샘플링하여 arm 선택

    Returns:
    --------
    str : 선택된 정책 이름
    """
    samples = {}
    for arm, post in posteriors.items():
        # 사후분포에서 샘플
        samples[arm] = np.random.normal(post['mu'], post['sigma'])

    # 최대 샘플 값을 가진 arm 선택
    selected_arm = max(samples, key=samples.get)
    return selected_arm

def run_thompson_sampling(arms, n_trials=100, prior_mean=0, prior_sd=2.0):
    """
    Thompson Sampling 시뮬레이션 실행

    Parameters:
    -----------
    arms : dict
        정책 대안과 true parameters
    n_trials : int
        총 시행 횟수
    prior_mean : float
        사전분포 평균
    prior_sd : float
        사전분포 표준편차

    Returns:
    --------
    dict : 시뮬레이션 결과
    """
    posteriors = initialize_priors(arms, prior_mean, prior_sd)

    # 결과 저장
    history = {
        'trial': [],
        'selected_arm': [],
        'reward': [],
        'cumulative_reward': [],
        'regret': []  # 최선 대비 손실
    }

    # 최선의 정책 (true mean이 가장 큰 정책)
    best_arm = max(arms.keys(), key=lambda k: arms[k]['true_mean'])
    best_mean = arms[best_arm]['true_mean']

    cumulative_reward = 0
    cumulative_regret = 0

    # 각 정책별 선택 횟수 추적
    arm_counts = {arm: [] for arm in arms.keys()}
    arm_means = {arm: [] for arm in arms.keys()}

    for t in range(1, n_trials + 1):
        # 1. Thompson Sampling으로 arm 선택
        selected_arm = select_arm_thompson(posteriors)

        # 2. 보상 관찰
        reward = sample_reward(selected_arm, arms)

        # 3. 사후분포 업데이트
        thompson_sampling_update(posteriors, selected_arm, reward, data_sd=1.0)

        # 4. Cumulative reward와 regret 계산
        cumulative_reward += reward
        instant_regret = best_mean - reward
        cumulative_regret += instant_regret

        # 5. 결과 저장
        history['trial'].append(t)
        history['selected_arm'].append(selected_arm)
        history['reward'].append(reward)
        history['cumulative_reward'].append(cumulative_reward)
        history['regret'].append(cumulative_regret)

        # 6. 각 정책별 선택 비율 저장 (시각화용)
        for arm in arms.keys():
            count = posteriors[arm]['n_obs']
            arm_counts[arm].append(count)
            arm_means[arm].append(posteriors[arm]['mu'])

    # 결과 정리
    results = {
        'history': pd.DataFrame(history),
        'final_posteriors': posteriors,
        'arm_counts': arm_counts,
        'arm_means': arm_means,
        'best_arm': best_arm,
        'total_reward': cumulative_reward,
        'total_regret': cumulative_regret
    }

    return results

def plot_thompson_sampling(results, arms, output_file):
    """
    Thompson Sampling 결과 시각화
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    history = results['history']
    n_trials = len(history)

    # 1. 누적 보상 추이
    ax1 = axes[0, 0]
    ax1.plot(history['trial'], history['cumulative_reward'], linewidth=2, color='steelblue')
    ax1.set_xlabel('시행 횟수', fontsize=11)
    ax1.set_ylabel('누적 보상 (경쟁률 변화)', fontsize=11)
    ax1.set_title('누적 보상 추이', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. 누적 Regret 추이
    ax2 = axes[0, 1]
    ax2.plot(history['trial'], history['regret'], linewidth=2, color='coral')
    ax2.set_xlabel('시행 횟수', fontsize=11)
    ax2.set_ylabel('누적 Regret', fontsize=11)
    ax2.set_title('누적 Regret (최선 대비 손실)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. 정책별 선택 비율 추이
    ax3 = axes[1, 0]

    for arm in arms.keys():
        counts = np.array(results['arm_counts'][arm])
        trials = np.arange(1, n_trials + 1)
        proportions = counts / trials
        ax3.plot(trials, proportions, linewidth=2, label=arm)

    ax3.set_xlabel('시행 횟수', fontsize=11)
    ax3.set_ylabel('선택 비율', fontsize=11)
    ax3.set_title('정책별 선택 비율 추이', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. 사후분포 평균 추이
    ax4 = axes[1, 1]

    for arm in arms.keys():
        means = results['arm_means'][arm]
        ax4.plot(range(1, n_trials + 1), means, linewidth=2, label=arm)

        # True mean 수평선
        true_mean = arms[arm]['true_mean']
        ax4.axhline(true_mean, linestyle='--', linewidth=1, alpha=0.6)

    ax4.set_xlabel('시행 횟수', fontsize=11)
    ax4.set_ylabel('사후 평균 (μₙ)', fontsize=11)
    ax4.set_title('사후분포 평균 수렴', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Thompson Sampling 시각화 저장: {output_file}")
    plt.close()

def main():
    print("=" * 70)
    print("Thompson Sampling 시뮬레이션")
    print("=" * 70)

    # 1. 정책 대안 정의
    print("\n[1] 정책 대안 정의")
    arms = define_policy_arms()

    for arm_name, params in arms.items():
        print(f"  - {arm_name}: μ = {params['true_mean']:.2f}, σ = {params['true_sd']:.2f}")

    # 최선의 정책
    best_arm = max(arms.keys(), key=lambda k: arms[k]['true_mean'])
    print(f"\n  최선의 정책 (Oracle): {best_arm} (μ = {arms[best_arm]['true_mean']:.2f})")

    # 2. 시뮬레이션 설정
    n_trials = 100
    print(f"\n[2] 시뮬레이션 설정")
    print(f"  - 총 시행 횟수: {n_trials}")
    print(f"  - 사전분포: N(0, 2²) (약한 정보)")
    print(f"  - 데이터 분산 (known): σ² = 1")

    # 3. Thompson Sampling 실행
    print(f"\n[3] Thompson Sampling 실행")
    results = run_thompson_sampling(arms, n_trials=n_trials, prior_mean=0, prior_sd=2.0)

    print(f"  ✓ 시뮬레이션 완료")

    # 4. 최종 결과 분석
    print(f"\n[4] 최종 결과 분석")

    print(f"\n  [최종 정책별 선택 횟수]")
    for arm in arms.keys():
        count = results['final_posteriors'][arm]['n_obs']
        proportion = count / n_trials
        print(f"    {arm}: {count}회 ({proportion*100:.1f}%)")

    print(f"\n  [최종 사후분포 파라미터]")
    for arm in arms.keys():
        post = results['final_posteriors'][arm]
        true_mean = arms[arm]['true_mean']
        bias = post['mu'] - true_mean
        print(f"    {arm}:")
        print(f"      사후 평균: {post['mu']:.3f} (True: {true_mean:.2f}, Bias: {bias:+.3f})")
        print(f"      사후 SD: {post['sigma']:.3f}")

    print(f"\n  [성과 지표]")
    print(f"    총 누적 보상: {results['total_reward']:.2f}")
    print(f"    총 누적 Regret: {results['total_regret']:.2f}")
    print(f"    평균 Regret: {results['total_regret']/n_trials:.3f}")

    # 최선의 정책과 비교
    oracle_reward = arms[best_arm]['true_mean'] * n_trials
    efficiency = (results['total_reward'] / oracle_reward) * 100 if oracle_reward != 0 else 0
    print(f"    Oracle 대비 효율: {efficiency:.1f}%")

    # 5. 결과 저장
    print(f"\n[5] 결과 저장")

    # CSV 저장
    csv_file = 'thompson_sampling_results.csv'
    results['history'].to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"  ✓ CSV 저장: {csv_file}")

    # JSON 저장
    json_file = 'thompson_sampling_summary.json'
    summary = {
        'settings': {
            'n_trials': n_trials,
            'arms': {
                arm: {
                    'true_mean': float(params['true_mean']),
                    'true_sd': float(params['true_sd'])
                }
                for arm, params in arms.items()
            },
            'best_arm': best_arm
        },
        'final_posteriors': {
            arm: {
                'mu': float(post['mu']),
                'sigma': float(post['sigma']),
                'n_obs': int(post['n_obs'])
            }
            for arm, post in results['final_posteriors'].items()
        },
        'performance': {
            'total_reward': float(results['total_reward']),
            'total_regret': float(results['total_regret']),
            'average_regret': float(results['total_regret'] / n_trials),
            'oracle_efficiency_pct': float(efficiency)
        }
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON 저장: {json_file}")

    # 텍스트 보고서 저장
    report_file = 'thompson_sampling_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Thompson Sampling 시뮬레이션 보고서\n")
        f.write("=" * 70 + "\n\n")

        f.write("[1] 시뮬레이션 개요\n\n")
        f.write(f"  목적: 베이지언 Multi-Armed Bandits를 통한 정책 대안 평가\n")
        f.write(f"  방법: Thompson Sampling (Response-Adaptive Randomization)\n")
        f.write(f"  시행 횟수: {n_trials}\n")
        f.write(f"  정책 대안 수: {len(arms)}\n\n")

        f.write("[2] 정책 대안\n\n")
        for arm_name, params in arms.items():
            f.write(f"  • {arm_name}\n")
            f.write(f"    - True Mean: {params['true_mean']:.2f}\n")
            f.write(f"    - True SD: {params['true_sd']:.2f}\n")
        f.write(f"\n  최선의 정책: {best_arm}\n\n")

        f.write("[3] 최종 결과\n\n")
        f.write("  정책별 선택 횟수:\n")
        for arm in arms.keys():
            count = results['final_posteriors'][arm]['n_obs']
            proportion = count / n_trials
            f.write(f"    {arm}: {count}회 ({proportion*100:.1f}%)\n")

        f.write("\n  사후분포 파라미터:\n")
        for arm in arms.keys():
            post = results['final_posteriors'][arm]
            true_mean = arms[arm]['true_mean']
            bias = post['mu'] - true_mean
            f.write(f"    {arm}:\n")
            f.write(f"      사후 평균: {post['mu']:.3f} (True: {true_mean:.2f}, Bias: {bias:+.3f})\n")
            f.write(f"      사후 SD: {post['sigma']:.3f}\n")

        f.write("\n  성과 지표:\n")
        f.write(f"    총 누적 보상: {results['total_reward']:.2f}\n")
        f.write(f"    총 누적 Regret: {results['total_regret']:.2f}\n")
        f.write(f"    평균 Regret: {results['total_regret']/n_trials:.3f}\n")
        f.write(f"    Oracle 대비 효율: {efficiency:.1f}%\n\n")

        f.write("[4] 해석 및 함의\n\n")

        # 최다 선택 정책
        most_selected = max(results['final_posteriors'].keys(),
                           key=lambda k: results['final_posteriors'][k]['n_obs'])
        most_count = results['final_posteriors'][most_selected]['n_obs']

        f.write(f"  • Thompson Sampling이 '{most_selected}'를 가장 많이 선택 ({most_count}/{n_trials})\n")

        if most_selected == best_arm:
            f.write(f"  ✓ 최선의 정책을 성공적으로 식별\n")
        else:
            f.write(f"  ⚠️ 최선의 정책({best_arm})과 다른 정책 선택\n")

        f.write(f"\n  • Exploration-Exploitation 균형:\n")
        f.write(f"    - Thompson Sampling은 사후분포 샘플링으로 자동 균형 달성\n")
        f.write(f"    - 누적 Regret: {results['total_regret']:.2f}\n")
        f.write(f"    - 이론적 최선 대비 효율: {efficiency:.1f}%\n\n")

        f.write("[5] 한신대 계열제 실험 적용\n\n")
        f.write("  만약 Thompson Sampling을 사용했다면:\n")
        f.write("  1. 계열제 실험 초기부터 여러 정책 대안 동시 시험 가능\n")
        f.write("  2. 성과가 좋은 정책에 더 많은 학생 할당 (윤리적 이점)\n")
        f.write("  3. 빠른 학습과 적응으로 최적 정책 조기 발견\n")
        f.write("  4. 전체 학생 복지 향상 (최선 정책에 대한 노출 증가)\n\n")

        f.write("  현행 방식 vs Thompson Sampling:\n")
        f.write("  • 현행: 단일 정책 전면 시행 → 실패 시 큰 손실\n")
        f.write("  • Thompson Sampling: 적응형 할당 → 손실 최소화, 학습 최대화\n\n")

        f.write("[6] 결론\n\n")
        f.write("  Thompson Sampling은 다음과 같은 장점을 제공합니다:\n")
        f.write("  1. 자동 Exploration-Exploitation 균형\n")
        f.write("  2. 윤리적 실험 설계 (좋은 정책에 더 많은 할당)\n")
        f.write("  3. 빠른 적응과 학습\n")
        f.write("  4. 베이지언 사후분포 직관적 해석\n\n")

        f.write("  한신대 계열제 실험에 적용하면:\n")
        f.write("  • 여러 정책 대안 동시 시험으로 최적 정책 조기 발견\n")
        f.write("  • 학생 복지 최대화 (좋은 정책 우선 할당)\n")
        f.write("  • 데이터 기반 적응형 의사결정\n\n")

    print(f"  ✓ 텍스트 보고서 저장: {report_file}")

    # 6. 시각화
    print(f"\n[6] 시각화 생성")
    plot_file = 'thompson_sampling_plot.png'
    plot_thompson_sampling(results, arms, plot_file)

    # 7. 최종 요약
    print("\n" + "=" * 70)
    print("해석 및 결론")
    print("=" * 70)

    print(f"\n[Thompson Sampling의 장점]")
    print(f"  • 자동 Exploration-Exploitation: 사후분포 샘플링")
    print(f"  • 윤리적 실험: 좋은 정책에 더 많은 할당")
    print(f"  • 빠른 학습: 베이지언 사후 업데이트")

    print(f"\n[시뮬레이션 결과]")
    print(f"  • 최다 선택 정책: {most_selected} ({most_count}/{n_trials})")
    print(f"  • Oracle 대비 효율: {efficiency:.1f}%")
    print(f"  • 평균 Regret: {results['total_regret']/n_trials:.3f}")

    print(f"\n[한신대 계열제 실험 적용]")
    print(f"  ✓ 여러 정책 대안 동시 시험 → 최적 정책 조기 발견")
    print(f"  ✓ 적응형 할당 → 학생 복지 최대화")
    print(f"  ✓ 데이터 기반 의사결정 → 증거 기반 정책 수정")

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()
