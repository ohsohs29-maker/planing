"""
12장 실습 코드 12-2: AHP(Analytic Hierarchy Process) 분석
Saaty의 AHP 기법을 구현하여 다기준 의사결정을 수행한다.
쌍대 비교, 일관성 검증, 종합 우선순위 계산을 포함한다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform

# 한글 폰트 설정
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("12-2. AHP(Analytic Hierarchy Process) 분석 실습")
print("=" * 60)

# ============================================================
# 1. AHP 기본 개념
# ============================================================
print("\n[1] AHP 쌍대 비교 척도")
print("-" * 40)

# Saaty의 9점 척도
saaty_scale = {
    1: '동등하게 중요',
    3: '약간 더 중요',
    5: '상당히 더 중요',
    7: '매우 더 중요',
    9: '극히 더 중요',
    2: '1과 3의 중간',
    4: '3과 5의 중간',
    6: '5와 7의 중간',
    8: '7과 9의 중간'
}

print("Saaty의 9점 척도:")
for value, meaning in sorted(saaty_scale.items()):
    print(f"  {value}: {meaning}")

# ============================================================
# 2. 평가 계층 구조 설정
# ============================================================
print("\n[2] 평가 계층 구조")
print("-" * 40)

# 평가 기준
criteria = ['시장성', '수익성', '리스크', '실행가능성', '시너지']
alternatives = ['A: 국내 프랜차이즈', 'B: 해외 직접투자', 'C: 합작투자', 'D: 라이선싱']

print("목표: 최적의 신규 사업 진출 방식 선정")
print(f"평가 기준: {', '.join(criteria)}")
print(f"대안: {', '.join([a.split(':')[0] for a in alternatives])}")

# ============================================================
# 3. 기준 간 쌍대 비교 매트릭스
# ============================================================
print("\n[3] 기준 간 쌍대 비교")
print("-" * 40)

# 기준 간 쌍대 비교 매트릭스 (전문가 판단)
# 행이 열보다 중요하면 1보다 큰 값, 덜 중요하면 1보다 작은 값
criteria_pairwise = np.array([
    #시장성  수익성  리스크  실행    시너지
    [1,     2,     3,     3,     5],      # 시장성
    [1/2,   1,     2,     2,     4],      # 수익성
    [1/3,   1/2,   1,     2,     3],      # 리스크
    [1/3,   1/2,   1/2,   1,     2],      # 실행가능성
    [1/5,   1/4,   1/3,   1/2,   1]       # 시너지
])

criteria_df = pd.DataFrame(criteria_pairwise, 
                           index=criteria, 
                           columns=criteria)
print("\n기준 간 쌍대 비교 매트릭스:")
print(criteria_df.round(2))

# ============================================================
# 4. AHP 핵심 함수
# ============================================================

def calculate_priority_vector(pairwise_matrix):
    """
    고유벡터 방법으로 우선순위 벡터 계산
    """
    # 열 합계로 정규화
    col_sums = pairwise_matrix.sum(axis=0)
    normalized = pairwise_matrix / col_sums
    
    # 행 평균 = 우선순위 벡터
    priority_vector = normalized.mean(axis=1)
    
    return priority_vector, normalized

def calculate_consistency_ratio(pairwise_matrix, priority_vector):
    """
    일관성 비율(CR) 계산
    CR = CI / RI
    CI = (λmax - n) / (n - 1)
    """
    n = len(priority_vector)
    
    # λmax 계산
    weighted_sum = pairwise_matrix @ priority_vector
    lambda_max = np.mean(weighted_sum / priority_vector)
    
    # 일관성 지수(CI)
    ci = (lambda_max - n) / (n - 1)
    
    # 무작위 지수(RI) - Saaty의 표
    ri_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_table[n]
    
    # 일관성 비율(CR)
    cr = ci / ri if ri > 0 else 0
    
    return lambda_max, ci, cr

# ============================================================
# 5. 기준 우선순위 계산
# ============================================================
print("\n[4] 기준 우선순위 계산")
print("-" * 40)

criteria_priority, criteria_normalized = calculate_priority_vector(criteria_pairwise)
lambda_max, ci, cr = calculate_consistency_ratio(criteria_pairwise, criteria_priority)

print("\n정규화된 쌍대 비교 매트릭스:")
print(pd.DataFrame(criteria_normalized, index=criteria, columns=criteria).round(3))

print("\n기준 우선순위 벡터:")
for i, (crit, priority) in enumerate(zip(criteria, criteria_priority)):
    print(f"  {crit}: {priority:.4f} ({priority*100:.1f}%)")

print(f"\n일관성 검증:")
print(f"  λmax = {lambda_max:.4f}")
print(f"  일관성 지수(CI) = {ci:.4f}")
print(f"  일관성 비율(CR) = {cr:.4f}")
print(f"  판정: {'일관성 있음 (CR < 0.10)' if cr < 0.10 else '일관성 부족 (재검토 필요)'}")

# ============================================================
# 6. 대안 간 쌍대 비교 (각 기준별)
# ============================================================
print("\n[5] 대안 간 쌍대 비교 (기준별)")
print("-" * 40)

# 각 기준에 대한 대안 간 쌍대 비교 매트릭스
alternative_pairwise = {
    '시장성': np.array([
        [1,   1/3, 1/2, 3],
        [3,   1,   2,   5],
        [2,   1/2, 1,   4],
        [1/3, 1/5, 1/4, 1]
    ]),
    '수익성': np.array([
        [1,   1/2, 1/2, 2],
        [2,   1,   2,   4],
        [2,   1/2, 1,   3],
        [1/2, 1/4, 1/3, 1]
    ]),
    '리스크': np.array([  # 리스크는 낮을수록 좋음
        [1,   4,   2,   1/2],
        [1/4, 1,   1/3, 1/5],
        [1/2, 3,   1,   1/3],
        [2,   5,   3,   1]
    ]),
    '실행가능성': np.array([
        [1,   4,   2,   2],
        [1/4, 1,   1/2, 1/2],
        [1/2, 2,   1,   1],
        [1/2, 2,   1,   1]
    ]),
    '시너지': np.array([
        [1,   2,   1/2, 3],
        [1/2, 1,   1/2, 2],
        [2,   2,   1,   4],
        [1/3, 1/2, 1/4, 1]
    ])
}

# 각 기준별 대안 우선순위 계산
alternative_priorities = {}
alt_names = ['A', 'B', 'C', 'D']

print("\n기준별 대안 우선순위:")
for criterion in criteria:
    priority, _ = calculate_priority_vector(alternative_pairwise[criterion])
    _, _, cr = calculate_consistency_ratio(alternative_pairwise[criterion], priority)
    alternative_priorities[criterion] = priority
    
    print(f"\n  [{criterion}] (CR={cr:.3f})")
    for alt, p in zip(alt_names, priority):
        print(f"    {alt}: {p:.4f}")

# ============================================================
# 7. 종합 우선순위 계산
# ============================================================
print("\n[6] 종합 우선순위 계산")
print("-" * 40)

# 대안별 가중 우선순위 매트릭스
priority_matrix = np.column_stack([alternative_priorities[c] for c in criteria])
priority_df = pd.DataFrame(priority_matrix, 
                           index=alt_names,
                           columns=criteria)

print("\n대안-기준 우선순위 매트릭스:")
print(priority_df.round(4))

# 종합 우선순위 = 대안 우선순위 × 기준 가중치
overall_priority = priority_matrix @ criteria_priority

print("\n종합 우선순위:")
for alt, priority in sorted(zip(alternatives, overall_priority), 
                            key=lambda x: x[1], reverse=True):
    print(f"  {alt}: {priority:.4f} ({priority*100:.1f}%)")

best_alternative = alternatives[np.argmax(overall_priority)]
print(f"\n최적 대안: {best_alternative}")

# ============================================================
# 8. 민감도 분석
# ============================================================
print("\n[7] 민감도 분석")
print("-" * 40)

# 기준 가중치 변동에 따른 종합 우선순위 변화
sensitivity_data = {}
weight_range = np.linspace(0.05, 0.50, 10)

for target_criterion in criteria:
    sensitivity_data[target_criterion] = []
    target_idx = criteria.index(target_criterion)
    
    for new_weight in weight_range:
        # 가중치 재조정
        adjusted_weights = criteria_priority.copy()
        old_weight = adjusted_weights[target_idx]
        adjusted_weights[target_idx] = new_weight
        
        # 나머지 가중치 비례 조정
        scale_factor = (1 - new_weight) / (1 - old_weight)
        for i in range(len(adjusted_weights)):
            if i != target_idx:
                adjusted_weights[i] *= scale_factor
        
        # 종합 우선순위 재계산
        new_overall = priority_matrix @ adjusted_weights
        sensitivity_data[target_criterion].append(new_overall)

# 가장 민감한 기준 식별
sensitivity_range = {}
for crit in criteria:
    data = np.array(sensitivity_data[crit])
    # 각 대안의 순위 변동폭 계산
    rank_changes = 0
    for i in range(len(data) - 1):
        current_rank = np.argsort(-data[i])
        next_rank = np.argsort(-data[i+1])
        if not np.array_equal(current_rank, next_rank):
            rank_changes += 1
    sensitivity_range[crit] = rank_changes

print("기준별 순위 전환 횟수:")
for crit, changes in sorted(sensitivity_range.items(), key=lambda x: x[1], reverse=True):
    print(f"  {crit}: {changes}회")

# ============================================================
# 9. 시각화
# ============================================================
print("\n[8] 결과 시각화")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 9-1. 기준 우선순위
ax1 = axes[0, 0]
bars = ax1.barh(criteria, criteria_priority, color='#3498db')
ax1.set_xlabel('우선순위')
ax1.set_title('기준 우선순위 (AHP)')
for i, v in enumerate(criteria_priority):
    ax1.text(v + 0.01, i, f'{v:.3f}', va='center')

# 9-2. 종합 우선순위
ax2 = axes[0, 1]
colors = ['#2ecc71' if alt == best_alternative else '#95a5a6' for alt in alternatives]
bars = ax2.barh(range(len(alternatives)), overall_priority, color=colors)
ax2.set_yticks(range(len(alternatives)))
ax2.set_yticklabels(alternatives)
ax2.set_xlabel('종합 우선순위')
ax2.set_title('대안별 종합 우선순위')
for i, v in enumerate(overall_priority):
    ax2.text(v + 0.01, i, f'{v:.3f}', va='center')

# 9-3. 기준별 대안 우선순위 히트맵
ax3 = axes[1, 0]
im = ax3.imshow(priority_df.values, cmap='YlGnBu', aspect='auto')
ax3.set_xticks(range(len(criteria)))
ax3.set_xticklabels(criteria, rotation=45, ha='right')
ax3.set_yticks(range(len(alt_names)))
ax3.set_yticklabels(alt_names)
ax3.set_title('기준별 대안 우선순위')
plt.colorbar(im, ax=ax3)

# 값 표시
for i in range(len(alt_names)):
    for j in range(len(criteria)):
        ax3.text(j, i, f'{priority_df.iloc[i, j]:.2f}', 
                ha='center', va='center', color='black', fontsize=9)

# 9-4. 민감도 분석 (시장성 기준)
ax4 = axes[1, 1]
sensitivity_df = pd.DataFrame(
    np.array(sensitivity_data['시장성']),
    columns=alt_names
)
for alt in alt_names:
    ax4.plot(weight_range, sensitivity_df[alt], marker='o', label=alt, markersize=4)
ax4.set_xlabel('시장성 가중치')
ax4.set_ylabel('종합 우선순위')
ax4.set_title('민감도 분석 (시장성 기준)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/callii/Documents/strategy/practice/chapter12/code/12-2-ahp-analysis.png', 
            dpi=150, bbox_inches='tight')
plt.close()
print("그래프 저장 완료: 12-2-ahp-analysis.png")

# ============================================================
# 10. 결과 요약
# ============================================================
print("\n" + "=" * 60)
print("AHP 분석 결과 요약")
print("=" * 60)

print(f"""
1. 기준 우선순위 (합계 = 1.00)
""")
for crit, priority in sorted(zip(criteria, criteria_priority), 
                             key=lambda x: x[1], reverse=True):
    print(f"   {crit}: {priority:.4f} ({priority*100:.1f}%)")

print(f"""
2. 종합 순위
""")
for rank, (alt, priority) in enumerate(
    sorted(zip(alternatives, overall_priority), key=lambda x: x[1], reverse=True), 1):
    print(f"   {rank}위: {alt} ({priority:.4f})")

print(f"""
3. 일관성 검증
   - 기준 간 비교 CR = {cr:.4f} ({'통과' if cr < 0.10 else '미통과'})
   - 허용 기준: CR < 0.10

4. 민감도 분석
   - 가장 민감한 기준: {max(sensitivity_range, key=sensitivity_range.get)}
   - 강건성: {'높음' if max(sensitivity_range.values()) <= 2 else '보통' if max(sensitivity_range.values()) <= 4 else '낮음'}

5. 의사결정 권고
   - 최적 대안: {best_alternative}
   - AHP와 가중점수법 결과 일치 여부 확인 권장
""")
