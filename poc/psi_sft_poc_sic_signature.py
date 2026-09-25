import numpy as np
from scipy import special
from psi_sft_poc import dfp_ddm, sic, make_psi
rng = np.random.default_rng(3)
psi = make_psi((-55.,50.), 100, .02); print(f"AGRT β 網格上限 = {psi.beta.max():.2f}(DDM 等效 β=32 在網格外 → β 被釘在上限附近)")
# SIC 移植驗證:大 salience 差(Houpt 的 a=3, v=2, 漂移 H=8·? 用 drift 3.0 vs 1.0),250/cell,10 reps
a,ter,sdv = 3.,.1,.2; H,L = 3.0, 1.0
print("大 salience 差 (drift 3.0 vs 1.0, a=3) 250/cell ×10:顯著比例")
for arch, rule in (("PAR","OR"),("PAR","AND"),("SER","OR"),("SER","AND"),("COA",None)):
    res=[]
    for rep in range(10):
        cells={}
        for name,(c1,c2) in dict(HH=(H,H),HL=(H,L),LH=(L,H),LL=(L,L)).items():
            rt,cr = dfp_ddm(250,c1,c2,a,ter,sdv,arch,rule,rng); cells[name]=rt[cr==1]
        s=sic(**cells)
        dom_ok = all(p<.05 for k,p in s['Dominance'].items() if '>' in k) and not any(p<.05 for k,p in s['Dominance'].items() if '<' in k)
        res.append((s['Dplus'][1]<.05, s['Dminus'][1]<.05, s['MIC'][1]<.05, np.sign(s['MIC'][0]), dom_ok))
    r=np.array(res,float).mean(axis=0)
    print(f"  {arch}-{rule or '':3}  D+ {r[0]:.1f}  D- {r[1]:.1f}  MIC {r[2]:.1f} (sign {r[3]:+.1f})  dominance {r[4]:.1f}")
