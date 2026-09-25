import numpy as np, time
from scipy import stats, optimize
from psi_sft_poc import make_psi, salience_levels, simdiffT, dfp_ddm, sic

rng = np.random.default_rng(7)
x_range = (-55.0, 50.0); lapse = .02; ml = 1-np.sqrt(1-lapse)
# A. Psi 移植本身的回復:受試者就是 Psi 假設的累積常態 (alpha=6, beta=15)
print("A. 累積常態受試者 alpha=6 beta=15, lapse .02, 20 次重複")
for n in (72, 144, 300):
    est = []
    for rep in range(20):
        psi = make_psi(x_range, 100, lapse)
        for _ in range(n):
            p1 = .5*ml + (1-ml)*stats.norm.cdf(psi.nextIntensity, 6, 15)
            psi.update(int(rng.uniform() < p1))
        est.append(psi.estimateLambda())
    est = np.array(est)
    print(f"  n={n:3d}  alpha {est[:,0].mean():5.2f} ± {est[:,0].std():4.2f}   beta {est[:,1].mean():5.2f} ± {est[:,1].std():4.2f}")

# B. DDM 受試者:真值改用 simdiffT 自己的 p = 1/(1+exp(-a·drift))(平均掉 sdv)
thres50=6.; a,v,ter,sdv = 1.45,1.6,.1,.25
drift = lambda x: (x-thres50)/(x_range[1]-thres50)*v
xx = np.linspace(*x_range, 400)
pc = np.array([np.mean(1/(1+np.exp(-a*rng.normal(drift(x), sdv, 4000)))) for x in xx])
pmf = lambda x, al, be: .5*ml + (1-ml)*stats.norm.cdf(x, al, be)
true_ab = optimize.minimize(lambda p: np.sqrt(np.mean((pmf(xx,*p)-pc)**2)), [6, 20]).x
print(f"\nB. DDM 受試者的等效 alpha/beta(用 simdiffT 的 p 公式)= {true_ab.round(2)};"
      f"  x=50 時 P(correct)={pc[-1]:.3f}  → .99 在刺激範圍內不可達")
est=[]
for rep in range(10):
    psi = make_psi(x_range, 100, lapse)
    for _ in range(144):
        _, r = simdiffT(1, a, drift(psi.nextIntensity), sdv, ter, rng=rng); psi.update(int(r[0]))
    est.append(psi.estimateLambda())
est=np.array(est); print(f"  Psi 144 trials ×10: alpha {est[:,0].mean():5.2f} ± {est[:,0].std():4.2f}  beta {est[:,1].mean():5.2f} ± {est[:,1].std():4.2f}")
al, be = est.mean(axis=0)
for pH, pL in ((.95,.80),(.90,.70)):
    H, L = salience_levels(al, be, ml, [pH, pL])
    _, xH = simdiffT(3000, a, drift(H), sdv, ter, rng=rng); _, xL = simdiffT(3000, a, drift(L), sdv, ter, rng=rng)
    print(f"  目標 {pH}/{pL}: H={H:6.2f} (acc {xH.mean():.3f})  L={L:6.2f} (acc {xL.mean():.3f})")
    for ncell in (250, 1000):
        hits = {}
        for arch, rule in (("PAR","OR"),("PAR","AND"),("SER","OR"),("SER","AND"),("COA",None)):
            res = []
            for rep in range(5):
                cells = {}
                for name,(c1,c2) in dict(HH=(H,H),HL=(H,L),LH=(L,H),LL=(L,L)).items():
                    rt, cr = dfp_ddm(ncell, drift(c1), drift(c2), a, ter, sdv, arch, rule, rng); cells[name]=rt[cr==1]
                s = sic(**cells); res.append((s['Dplus'][1]<.05, s['Dminus'][1]<.05, s['MIC'][1]<.05))
            res=np.array(res).mean(axis=0); hits[f"{arch}-{rule or ''}"] = f"D+ {res[0]:.1f} D- {res[1]:.1f} MIC {res[2]:.1f}"
        print(f"    {ncell:4d}/cell  " + "  |  ".join(f"{k}: {v}" for k,v in hits.items()))
