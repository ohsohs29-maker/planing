import numpy as np
np.random.seed(42)

# Step 1
print("Step 1. 변수 6개 OK")

# Step 2
edges = [
    ('Change Success','Trust','+'),('Trust','Participation','+'),
    ('Participation','Change Success','+'),('Change Speed','Resistance','+'),
    ('Resistance','Change Success','-'),('Change Speed','Fatigue','+'),
    ('Fatigue','Participation','-'),('Change Success','Change Speed','+'),
]
print("Step 2. CLD 8개 엣지, R1/B1/B2 OK")

# Step 3
params = {
    'months':24,'init_success':0.15,'init_trust':0.78,'init_participation':0.80,
    'init_resistance':0.10,'init_fatigue':0.10,'trust_gain':0.15,'trust_decay':0.08,
    'resistance_rate':0.12,'fatigue_rate':0.10,'fatigue_recovery':0.03,'speed_multiplier':1.5,
}
print("Step 3. 파라미터 12개 OK")

# Step 4
m = 24
s = np.zeros(m); tr = np.zeros(m); pa = np.zeros(m)
re = np.zeros(m); fa = np.zeros(m); sp = np.zeros(m)
s[0], tr[0], pa[0] = 0.15, 0.78, 0.80
re[0], fa[0], sp[0] = 0.10, 0.10, 0.3

for i in range(1, m):
    sp[i] = min(sp[i-1] * (1.5 if i == 4 else 1.0), 1.0)
    re[i] = np.clip(re[i-1] + 0.12*sp[i-1]*(1-re[i-1]) - 0.02*tr[i-1], 0, 1)
    fa[i] = np.clip(fa[i-1] + 0.10*sp[i-1] - 0.03, 0, 1)
    pa[i] = np.clip(pa[i-1] + 0.1*(tr[i-1]-0.5) - 0.15*fa[i-1], 0, 1)
    s[i] = np.clip(s[i-1] + 0.08*pa[i-1] - 0.1*re[i-1], 0, 1)
    if s[i] > s[i-1]:
        tr[i] = np.clip(tr[i-1] + 0.15*(s[i]-0.3), 0, 1)
    else:
        tr[i] = np.clip(tr[i-1] - 0.08*(1-s[i]), 0, 1)

print(f"Step 4. 시뮬레이션 결과:")
print(f"  3M:  success={s[3]:.2f}, trust={tr[3]:.2f}")
print(f"  6M:  success={s[6]:.2f}, resistance={re[6]:.2f}")
print(f"  9M:  success={s[9]:.2f}, fatigue={fa[9]:.2f}")
print(f"  24M: success={s[-1]:.2f}, trust={tr[-1]:.2f}")

# Step 5
def run_sim(p):
    m = p['months']
    s = np.zeros(m); t = np.zeros(m); pa = np.zeros(m)
    r = np.zeros(m); f = np.zeros(m); sp = np.zeros(m)
    s[0], t[0], pa[0] = p['init_success'], p['init_trust'], p['init_participation']
    r[0], f[0], sp[0] = p['init_resistance'], p['init_fatigue'], 0.3
    for i in range(1, m):
        sp[i] = min(sp[i-1] * (p['speed_multiplier'] if i == 4 else 1.0), 1.0)
        r[i] = np.clip(r[i-1] + p['resistance_rate']*sp[i-1]*(1-r[i-1]) - 0.02*t[i-1], 0, 1)
        f[i] = np.clip(f[i-1] + p['fatigue_rate']*sp[i-1] - p['fatigue_recovery'], 0, 1)
        pa[i] = np.clip(pa[i-1] + 0.1*(t[i-1]-0.5) - 0.15*f[i-1], 0, 1)
        s[i] = np.clip(s[i-1] + 0.08*pa[i-1] - 0.1*r[i-1], 0, 1)
        if s[i] > s[i-1]:
            t[i] = np.clip(t[i-1] + p['trust_gain']*(s[i]-0.3), 0, 1)
        else:
            t[i] = np.clip(t[i-1] - p['trust_decay']*(1-s[i]), 0, 1)
    return s

print("Step 5. 민감도 분석:")
for pn in ['resistance_rate', 'fatigue_rate', 'trust_gain']:
    res = {}
    for label, mult in [('Low', 0.5), ('Base', 1.0), ('High', 1.5)]:
        p = params.copy()
        p[pn] = params[pn] * mult
        res[label] = run_sim(p)[-1]
    spread = res['High'] - res['Low']
    print(f"  {pn:20s}: L={res['Low']:.2f} B={res['Base']:.2f} H={res['High']:.2f} (spread={spread:.2f})")

# Step 6
cn = {}
for a, b, _ in edges:
    cn[a] = cn.get(a, 0) + 1
    cn[b] = cn.get(b, 0) + 1
print("Step 6. 레버리지 포인트:")
for n, c in sorted(cn.items(), key=lambda x: -x[1])[:3]:
    print(f"  {n}: {c}개 연결")

print("\nALL 6 STEPS OK")
