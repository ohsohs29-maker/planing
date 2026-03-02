"""
12장 실습 코드 12-3: 의사결정 나무와 정보의 가치
불확실성하의 의사결정을 위한 의사결정 나무 분석과
정보의 가치(EVPI, EVII) 계산을 수행한다.
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
print("12-3. 의사결정 나무와 정보의 가치 분석")
print("=" * 60)

# ============================================================
# 1. 문제 정의
# ============================================================
print("\n[1] 의사결정 문제 정의")
print("-" * 40)

print("""
문제: 신규 해외 시장 진출 결정
- 대안 1: 직접 투자 (대규모)
- 대안 2: 합작 투자 (중규모)  
- 대안 3: 라이선싱 (소규모)
- 대안 4: 진출 포기

시장 상황:
- 호황: 30% 확률
- 보통: 50% 확률
- 불황: 20% 확률
""")

# ============================================================
# 2. 보상 매트릭스 정의
# ============================================================
print("\n[2] 보상 매트릭스 (단위: 억 원)")
print("-" * 40)

# 대안 정의
alternatives = ['직접투자', '합작투자', '라이선싱', '포기']

# 시장 상황별 확률
market_states = ['호황', '보통', '불황']
probabilities = np.array([0.30, 0.50, 0.20])

# 보상 매트릭스 (각 대안 × 시장 상황)
payoff_matrix = np.array([
    [300, 100, -150],   # 직접투자: 고위험-고수익
    [150, 80, -30],     # 합작투자: 중위험-중수익
    [50, 40, 20],       # 라이선싱: 저위험-저수익
    [0, 0, 0]           # 포기: 확실한 0
])

payoff_df = pd.DataFrame(payoff_matrix, 
                         index=alternatives,
                         columns=market_states)
print(payoff_df)

# ============================================================
# 3. 기대 화폐 가치(EMV) 계산
# ============================================================
print("\n[3] 기대 화폐 가치(EMV) 계산")
print("-" * 40)

emv = payoff_matrix @ probabilities

print("\n대안별 기대 화폐 가치:")
for alt, value in zip(alternatives, emv):
    print(f"  {alt}: {value:.1f}억 원")

best_alternative_idx = np.argmax(emv)
best_alternative = alternatives[best_alternative_idx]
best_emv = emv[best_alternative_idx]

print(f"\nEMV 기준 최적 대안: {best_alternative} ({best_emv:.1f}억 원)")

# ============================================================
# 4. 완전 정보의 기대가치(EVPI)
# ============================================================
print("\n[4] 완전 정보의 기대가치(EVPI)")
print("-" * 40)

# 각 상황에서의 최적 대안 보상
best_payoff_per_state = np.max(payoff_matrix, axis=0)
print("\n각 시장 상황에서의 최적 보상:")
for state, prob, payoff in zip(market_states, probabilities, best_payoff_per_state):
    best_alt = alternatives[np.argmax(payoff_matrix[:, market_states.index(state)])]
    print(f"  {state} (확률 {prob:.0%}): {payoff:.0f}억 원 ({best_alt})")

# 완전 정보하의 기대가치
ev_with_perfect_info = np.sum(best_payoff_per_state * probabilities)
print(f"\n완전 정보하의 기대가치(EV|PI): {ev_with_perfect_info:.1f}억 원")

# EVPI 계산
evpi = ev_with_perfect_info - best_emv
print(f"완전 정보의 기대가치(EVPI): {evpi:.1f}억 원")
print(f"\n해석: 시장 상황을 완벽히 예측할 수 있는 정보에")
print(f"      최대 {evpi:.1f}억 원까지 지불할 가치가 있음")

# ============================================================
# 5. 불완전 정보의 기대가치(EVII)
# ============================================================
print("\n[5] 불완전 정보의 기대가치(EVII)")
print("-" * 40)

print("""
시장 조사 결과:
- 긍정적 신호: P(긍정|호황)=0.8, P(긍정|보통)=0.4, P(긍정|불황)=0.1
- 부정적 신호: P(부정|호황)=0.2, P(부정|보통)=0.6, P(부정|불황)=0.9
""")

# 조건부 확률 (likelihood)
# P(Signal | State)
likelihood = {
    '긍정': np.array([0.8, 0.4, 0.1]),  # P(긍정|호황,보통,불황)
    '부정': np.array([0.2, 0.6, 0.9])   # P(부정|호황,보통,불황)
}

# 신호 확률 계산: P(Signal) = Σ P(Signal|State) × P(State)
signal_probs = {}
for signal, likelihoods in likelihood.items():
    signal_probs[signal] = np.sum(likelihoods * probabilities)
    print(f"P({signal}) = {signal_probs[signal]:.3f}")

# 베이지안 사후 확률: P(State|Signal) = P(Signal|State) × P(State) / P(Signal)
posterior = {}
for signal, likelihoods in likelihood.items():
    posterior[signal] = (likelihoods * probabilities) / signal_probs[signal]

print("\n사후 확률 P(상태|신호):")
posterior_df = pd.DataFrame(posterior, index=market_states)
print(posterior_df.round(3))

# 각 신호에 대한 최적 의사결정과 기대값
print("\n신호별 최적 의사결정:")
ev_given_signals = {}

for signal in ['긍정', '부정']:
    # 각 대안의 기대값 (신호 조건부)
    emv_given_signal = payoff_matrix @ posterior[signal]
    best_idx = np.argmax(emv_given_signal)
    best_alt = alternatives[best_idx]
    best_value = emv_given_signal[best_idx]
    
    ev_given_signals[signal] = best_value
    print(f"\n  {signal} 신호 시:")
    for alt, val in zip(alternatives, emv_given_signal):
        marker = "← 최적" if alt == best_alt else ""
        print(f"    {alt}: {val:.1f}억 원 {marker}")

# 정보 사용 시 기대가치
ev_with_imperfect_info = sum(ev_given_signals[s] * signal_probs[s] 
                              for s in ['긍정', '부정'])
print(f"\n불완전 정보하의 기대가치: {ev_with_imperfect_info:.1f}억 원")

# EVII 계산
evii = ev_with_imperfect_info - best_emv
print(f"불완전 정보의 기대가치(EVII): {evii:.1f}억 원")

# 정보 효율성
info_efficiency = (evii / evpi) * 100
print(f"정보 효율성: {info_efficiency:.1f}% (EVII/EVPI)")

print(f"\n해석: 시장 조사에 최대 {evii:.1f}억 원까지 지불할 가치가 있음")
print(f"      (완전 정보 가치의 {info_efficiency:.1f}%)")

# ============================================================
# 6. 의사결정 나무 시각화
# ============================================================
print("\n[6] 의사결정 나무 시각화")
print("-" * 40)

fig, ax = plt.subplots(figsize=(16, 10))

# 노드 위치 정의
decision_node = (0.1, 0.5)
alt_nodes = [(0.3, 0.85), (0.3, 0.6), (0.3, 0.35), (0.3, 0.1)]
state_nodes = [
    [(0.55, 0.95), (0.55, 0.85), (0.55, 0.75)],  # 직접투자
    [(0.55, 0.70), (0.55, 0.60), (0.55, 0.50)],  # 합작투자
    [(0.55, 0.45), (0.55, 0.35), (0.55, 0.25)],  # 라이선싱
    [(0.55, 0.1, None)]  # 포기 (단일 결과)
]

# 의사결정 노드 (사각형)
decision_square = plt.Rectangle((decision_node[0]-0.03, decision_node[1]-0.03), 
                                  0.06, 0.06, fill=True, color='#3498db')
ax.add_patch(decision_square)
ax.text(decision_node[0], decision_node[1], '?', ha='center', va='center', 
        fontsize=14, color='white', fontweight='bold')

# 대안 노드 (원) 및 가지
for i, (alt, node) in enumerate(zip(alternatives, alt_nodes)):
    # 가지 (의사결정 → 대안)
    ax.annotate('', xy=node, xytext=decision_node,
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    # 대안 노드 (원 = 불확실성 노드)
    if i < 3:  # 직접투자, 합작투자, 라이선싱
        circle = plt.Circle(node, 0.025, fill=True, color='#e74c3c')
        ax.add_patch(circle)
        
        # 대안 레이블
        ax.text(node[0]-0.05, node[1], f'{alt}\nEMV={emv[i]:.0f}', 
                ha='right', va='center', fontsize=9)
        
        # 상태 노드 및 보상
        for j, (state, prob) in enumerate(zip(market_states, probabilities)):
            state_node = state_nodes[i][j]
            # 가지
            ax.plot([node[0]+0.025, state_node[0]], [node[1], state_node[1]], 
                   'gray', lw=1)
            # 보상
            payoff = payoff_matrix[i, j]
            color = '#27ae60' if payoff >= 0 else '#c0392b'
            ax.text(state_node[0]+0.02, state_node[1], 
                   f'{state}({prob:.0%}): {payoff:+.0f}억', 
                   ha='left', va='center', fontsize=8, color=color)
    else:  # 포기
        ax.text(node[0]+0.02, node[1], f'{alt}: 0억', 
                ha='left', va='center', fontsize=9)

# 최적 대안 표시
best_node = alt_nodes[best_alternative_idx]
ax.plot([decision_node[0], best_node[0]], 
        [decision_node[1], best_node[1]], 
        color='#27ae60', lw=3, zorder=0)
ax.text(0.2, best_node[1]+0.05, '★ 최적', color='#27ae60', fontsize=10, fontweight='bold')

# 범례
ax.text(0.75, 0.95, '의사결정 나무 범례', fontsize=11, fontweight='bold')
ax.text(0.75, 0.90, '□ 의사결정 노드', fontsize=9)
ax.text(0.75, 0.85, '○ 불확실성 노드', fontsize=9)
ax.text(0.75, 0.80, f'EMV 최적: {best_alternative}', fontsize=9, color='#27ae60')

# 정보 가치 요약
ax.text(0.75, 0.70, '정보의 가치', fontsize=11, fontweight='bold')
ax.text(0.75, 0.65, f'EMV (정보 없음): {best_emv:.1f}억', fontsize=9)
ax.text(0.75, 0.60, f'EV|PI (완전정보): {ev_with_perfect_info:.1f}억', fontsize=9)
ax.text(0.75, 0.55, f'EVPI: {evpi:.1f}억', fontsize=9, color='#2980b9')
ax.text(0.75, 0.50, f'EVII: {evii:.1f}억', fontsize=9, color='#8e44ad')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('의사결정 나무: 해외 시장 진출 전략', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/callii/Documents/strategy/practice/chapter12/code/12-3-decision-tree.png', 
            dpi=150, bbox_inches='tight')
plt.close()
print("그래프 저장 완료: 12-3-decision-tree.png")

# ============================================================
# 7. 위험 선호도 분석
# ============================================================
print("\n[7] 위험 선호도별 분석")
print("-" * 40)

# 확실성 등가 계산 (지수 효용 함수)
def exponential_utility(x, risk_tolerance):
    """지수 효용 함수"""
    if risk_tolerance == float('inf'):  # 위험 중립
        return x
    return 1 - np.exp(-x / risk_tolerance)

# 다양한 위험 허용도에서의 확실성 등가
risk_tolerances = [50, 100, 200, 500, float('inf')]
risk_labels = ['매우 회피적', '회피적', '중립적', '선호적', '완전 중립']

print("\n위험 허용도별 대안 선호:")
for rt, label in zip(risk_tolerances, risk_labels):
    ce_values = []
    for i, alt in enumerate(alternatives):
        payoffs = payoff_matrix[i]
        utilities = [exponential_utility(p, rt) for p in payoffs]
        expected_utility = np.sum(np.array(utilities) * probabilities)
        # 역변환으로 확실성 등가 계산
        if rt == float('inf'):
            ce = expected_utility
        else:
            ce = -rt * np.log(1 - expected_utility) if expected_utility < 1 else rt * 10
        ce_values.append(ce)
    
    best_idx = np.argmax(ce_values)
    print(f"\n  {label} (RT={rt}):")
    print(f"    최적 대안: {alternatives[best_idx]}")

# ============================================================
# 8. 결과 요약
# ============================================================
print("\n" + "=" * 60)
print("의사결정 분석 결과 요약")
print("=" * 60)

print(f"""
1. 기대 화폐 가치(EMV) 분석
   - 직접투자: {emv[0]:.1f}억 원
   - 합작투자: {emv[1]:.1f}억 원  
   - 라이선싱: {emv[2]:.1f}억 원
   - 포기: {emv[3]:.1f}억 원
   
   EMV 최적 대안: {best_alternative} ({best_emv:.1f}억 원)

2. 정보의 가치
   - 완전 정보의 기대가치(EVPI): {evpi:.1f}억 원
   - 불완전 정보의 기대가치(EVII): {evii:.1f}억 원
   - 정보 효율성: {info_efficiency:.1f}%

3. 의사결정 권고
   - 시장 조사 비용이 {evii:.1f}억 원 미만이면 조사 실시 권장
   - 조사 없이 결정 시: {best_alternative} 선택
   - 긍정적 신호 시: 직접투자 고려
   - 부정적 신호 시: 라이선싱 또는 포기 고려

4. 리스크 관리
   - 위험 회피적 의사결정자: 라이선싱 선호
   - 위험 중립적 의사결정자: {best_alternative} 선호
   - 손실 최소화 필요 시: 포기 검토
""")
