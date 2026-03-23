"""
12장 실습 코드 12-1: 가중 점수법(Weighted Scoring Method)
다기준 의사결정의 기본 기법인 가중 점수법을 구현하고
신규 사업 진출 대안을 평가한다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정 (크로스플랫폼)
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("12-1. 가중 점수법(Weighted Scoring Method) 실습")
print("=" * 60)

# ============================================================
# 1. 평가 기준 및 대안 설정
# ============================================================
print("\n[1] 평가 기준 및 대안 설정")
print("-" * 40)

# 신규 사업 진출 대안
alternatives = ['A: 국내 프랜차이즈', 'B: 해외 직접투자', 'C: 합작투자', 'D: 라이선싱']

# 평가 기준과 가중치
criteria = {
    '시장성': {'weight': 0.30, 'description': '시장 규모 및 성장 잠재력'},
    '수익성': {'weight': 0.25, 'description': '예상 수익률 및 마진'},
    '리스크': {'weight': 0.20, 'description': '투자 위험도 (낮을수록 좋음)'},
    '실행가능성': {'weight': 0.15, 'description': '조직 역량 및 자원 적합성'},
    '시너지': {'weight': 0.10, 'description': '기존 사업과의 시너지'}
}

print("평가 기준 및 가중치:")
for name, info in criteria.items():
    print(f"  - {name}: {info['weight']:.0%} ({info['description']})")

# 가중치 합계 검증
total_weight = sum(c['weight'] for c in criteria.values())
print(f"\n가중치 합계: {total_weight:.2f} (정상)")

# ============================================================
# 2. 대안별 점수 부여 (1-10점 척도)
# ============================================================
print("\n[2] 대안별 점수 평가 (1-10점 척도)")
print("-" * 40)

# 전문가 평가 결과 (1-10점)
scores = {
    'A: 국내 프랜차이즈': {'시장성': 7, '수익성': 6, '리스크': 8, '실행가능성': 9, '시너지': 7},
    'B: 해외 직접투자': {'시장성': 9, '수익성': 8, '리스크': 4, '실행가능성': 5, '시너지': 6},
    'C: 합작투자': {'시장성': 8, '수익성': 7, '리스크': 6, '실행가능성': 7, '시너지': 8},
    'D: 라이선싱': {'시장성': 5, '수익성': 5, '리스크': 9, '실행가능성': 8, '시너지': 4}
}

# 점수 매트릭스 생성
score_df = pd.DataFrame(scores).T
print("\n원점수 매트릭스:")
print(score_df)

# ============================================================
# 3. 가중 점수 계산
# ============================================================
print("\n[3] 가중 점수 계산")
print("-" * 40)

# 가중치 배열
weights = np.array([c['weight'] for c in criteria.values()])

# 가중 점수 계산
weighted_scores = score_df.copy()
for col in weighted_scores.columns:
    weighted_scores[col] = score_df[col] * criteria[col]['weight']

print("\n가중 점수 매트릭스:")
print(weighted_scores.round(2))

# 종합 점수 계산
total_scores = weighted_scores.sum(axis=1)
print("\n종합 가중 점수:")
for alt, score in total_scores.sort_values(ascending=False).items():
    print(f"  {alt}: {score:.2f}점")

# 최적 대안
best_alternative = total_scores.idxmax()
print(f"\n최적 대안: {best_alternative} ({total_scores.max():.2f}점)")

# ============================================================
# 4. 민감도 분석
# ============================================================
print("\n[4] 민감도 분석")
print("-" * 40)

def calculate_weighted_score(weights_dict, scores_dict):
    """가중 점수 계산 함수"""
    total = 0
    for criterion, weight in weights_dict.items():
        total += scores_dict[criterion] * weight
    return total

# 각 기준의 가중치를 ±20% 변동시키며 순위 변화 분석
sensitivity_results = []

base_weights = {name: info['weight'] for name, info in criteria.items()}

for criterion in criteria.keys():
    for delta in [-0.10, -0.05, 0, 0.05, 0.10]:
        # 가중치 조정
        adjusted_weights = base_weights.copy()
        adjusted_weights[criterion] += delta
        
        # 다른 기준 가중치 비례 조정
        others = [c for c in criteria.keys() if c != criterion]
        adjustment_factor = (1 - adjusted_weights[criterion]) / sum(base_weights[c] for c in others)
        for other in others:
            adjusted_weights[other] = base_weights[other] * adjustment_factor
        
        # 각 대안의 점수 재계산
        alt_scores = {}
        for alt, alt_scores_dict in scores.items():
            alt_scores[alt] = calculate_weighted_score(adjusted_weights, alt_scores_dict)
        
        # 1위 대안 결정
        best = max(alt_scores, key=alt_scores.get)
        
        sensitivity_results.append({
            '기준': criterion,
            '가중치변동': f'{delta:+.0%}',
            '1위': best.split(':')[0],
            '점수': alt_scores[best]
        })

sens_df = pd.DataFrame(sensitivity_results)
print("\n가중치 변동에 따른 1위 대안 변화:")
pivot_sens = sens_df.pivot(index='기준', columns='가중치변동', values='1위')
print(pivot_sens)

# 순위 전환점 분석
print("\n순위 전환점 분석:")
for criterion in criteria.keys():
    subset = sens_df[sens_df['기준'] == criterion]
    unique_winners = subset['1위'].unique()
    if len(unique_winners) > 1:
        print(f"  - {criterion}: 가중치 변동 시 1위가 변경됨 ({', '.join(unique_winners)})")
    else:
        print(f"  - {criterion}: 가중치 변동에도 1위 유지 ({unique_winners[0]})")

# ============================================================
# 5. 시각화
# ============================================================
print("\n[5] 결과 시각화")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 5-1. 대안별 종합 점수
ax1 = axes[0]
colors = ['#2ecc71' if alt == best_alternative else '#3498db' for alt in total_scores.index]
bars = ax1.barh(range(len(total_scores)), total_scores.values, color=colors)
ax1.set_yticks(range(len(total_scores)))
ax1.set_yticklabels(total_scores.index)
ax1.set_xlabel('종합 가중 점수')
ax1.set_title('대안별 종합 점수 비교')
ax1.set_xlim(0, 10)

# 점수 레이블 추가
for i, (alt, score) in enumerate(total_scores.items()):
    ax1.text(score + 0.1, i, f'{score:.2f}', va='center', fontsize=10)

# 5-2. 기준별 기여도 스택 차트
ax2 = axes[1]
bottom = np.zeros(len(alternatives))
criterion_names = list(criteria.keys())
colors_criteria = plt.cm.Set2(np.linspace(0, 1, len(criterion_names)))

for i, criterion in enumerate(criterion_names):
    values = weighted_scores[criterion].values
    ax2.barh(range(len(alternatives)), values, left=bottom, 
             label=criterion, color=colors_criteria[i])
    bottom += values

ax2.set_yticks(range(len(alternatives)))
ax2.set_yticklabels(alternatives)
ax2.set_xlabel('가중 점수')
ax2.set_title('기준별 점수 기여도')
ax2.legend(loc='lower right', fontsize=9)
ax2.set_xlim(0, 10)

plt.tight_layout()
plt.savefig('/Users/callii/Documents/strategy/practice/chapter12/code/12-1-weighted-scoring.png', 
            dpi=150, bbox_inches='tight')
plt.close()
print("그래프 저장 완료: 12-1-weighted-scoring.png")

# ============================================================
# 6. 결과 요약
# ============================================================
print("\n" + "=" * 60)
print("분석 결과 요약")
print("=" * 60)

print(f"""
1. 평가 개요
   - 평가 대안: {len(alternatives)}개
   - 평가 기준: {len(criteria)}개
   - 평가 척도: 1-10점

2. 종합 순위
""")

for rank, (alt, score) in enumerate(total_scores.sort_values(ascending=False).items(), 1):
    print(f"   {rank}위: {alt} ({score:.2f}점)")

print(f"""
3. 의사결정 권고
   - 최적 대안: {best_alternative}
   - 2위 대안과의 점수 차이: {total_scores.max() - total_scores.sort_values(ascending=False).iloc[1]:.2f}점
   
4. 민감도 분석 결과
   - 가중치 ±10% 변동 범위 내에서 최적 대안이 변경되는지 확인
   - 강건한 의사결정을 위해 핵심 기준의 가중치 재검토 권장
""")
