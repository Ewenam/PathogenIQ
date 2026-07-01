"""Calibrate the CLR squash_k against a realistic endemic baseline + a labeled
outbreak/endemic test set. Reports sensitivity/specificity at the alert line
theta=0.6 for relative vs CLR(k), so we can pick a defensible default."""
import numpy as np, pandas as pd
from pathogeniq.scoring.baseline_risk import (
    AbundanceBaseline, score_sample_relative, SQUASH_K)

rng = np.random.default_rng(7)
THETA = 0.6

def clean(ps=(.35,.05), st=(.05,.02), ac=(.01,.006), ec=(.002,.001)):
    v = {"Pseudomonas":max(0,rng.normal(*ps)), "Streptococcus":max(0,rng.normal(*st)),
         "Acinetobacter":max(0,rng.normal(*ac)), "Escherichia":max(0,rng.normal(*ec))}
    rest=max(.01,1-sum(v.values())); v.update({"Bg1":rest*.6,"Bg2":rest*.4})
    s=pd.Series(v); return s/s.sum()

# realistic clean baseline (endemic flora present at normal levels)
base = pd.concat([clean() for _ in range(30)], axis=1).clip(lower=0)

# labeled test set: (name, series, should_alert)
tests = []
# --- should alert (true outbreaks / spikes) ---
for _ in range(8): tests.append(("vibrio_outbreak", pd.Series({"Vibrio":rng.uniform(.2,.5),"Pseudomonas":.3,"Bg1":.3}), True))
for _ in range(6): tests.append(("yersinia_bsl3", pd.Series({"Yersinia":rng.uniform(.02,.1),"Pseudomonas":.35,"Bg1":.6}), True))
for _ in range(8): tests.append(("salmonella_borderline", pd.Series({"Salmonella":rng.uniform(.08,.16),"Pseudomonas":.33,"Bg1":.5}), True))
for _ in range(8): tests.append(("acineto_spike", clean(ac=(.30,.03)), True))
# --- should NOT alert (endemic / near-threshold) ---
for _ in range(10): tests.append(("endemic_normal", clean(), False))
for _ in range(8): tests.append(("endemic_flora_confounder", clean(ps=(.18,.03),st=(.06,.02),ac=(.03,.01)), False))
for _ in range(6): tests.append(("pseudomonas_bloom", clean(ps=(.55,.05)), False))  # endemic just high, still baseline genus

def evaluate(space, k):
    bl = AbundanceBaseline.fit(base, space=space)
    tp=fp=tn=fn=0
    for _, s, should in tests:
        r = score_sample_relative("x", s/s.sum(), bl, squash_k=k)
        alert = r.score >= THETA
        if should and alert: tp+=1
        elif should and not alert: fn+=1
        elif not should and alert: fp+=1
        else: tn+=1
    sens = tp/(tp+fn) if tp+fn else None
    spec = tn/(tn+fp) if tn+fp else None
    youden = (sens or 0)+(spec or 0)-1
    return sens, spec, youden, (tp,fp,tn,fn)

print(f"{'config':22s} {'sens':>5s} {'spec':>5s} {'youden':>7s}  TP/FP/TN/FN")
sens,spec,y,c = evaluate("relative", SQUASH_K["relative"])
print(f"{'relative k=3.0':22s} {sens:5.2f} {spec:5.2f} {y:7.2f}  {c}")
best=None
for k in [0.75,1.0,1.25,1.5,2.0,2.5,3.0]:
    sens,spec,y,c = evaluate("clr", k)
    tag = ""
    if best is None or y>best[0]: best=(y,k); tag=""
    print(f"{'clr k='+str(k):22s} {sens:5.2f} {spec:5.2f} {y:7.2f}  {c}")
print(f"\nbest CLR k by Youden: {best[1]} (J={best[0]:.2f})")
