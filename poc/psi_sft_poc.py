"""
概念驗證:GRTv3_ada 的 Psi(AGRT.py agrtPsiObject,去掉 PsychoPy)接 adaptiveSFT 的
DDM 模擬(diffIRT::simdiffT 移植)與 sft::sic 移植,全程 numpy/scipy。
對應 R:psiSimulation_functions.R Est.Trial.Psi.Color / psi_color_ddm,
adaptiveSFT_functions.R dfp_ddm,sft/R/sic.R sic / sic.test / siDominance / mic.test。
"""
import time, sys
import numpy as np
from scipy import stats, special, optimize

# ─────────────────────────────────────────────────────────────────────
# 1. Psi(逐字抄 AGRT.py:75-184 agrtPsiObject;不需要 psychopy)
# ─────────────────────────────────────────────────────────────────────
class PsiObject:
    def __init__(self, x, alpha, beta, xPrecision, aPrecision, bPrecision, delta=0, prior=None):
        self.x = np.linspace(x[0], x[1], xPrecision, True)
        self.alpha = np.linspace(alpha[0], alpha[1], aPrecision)
        self.beta = np.linspace(beta[0], beta[1], bPrecision)
        self.r = np.arange(2); self.delta = delta
        self._r = self.r.reshape((2,1,1,1))
        self._alpha = self.alpha.reshape((1,-1,1,1))
        self._beta = self.beta.reshape((1,1,-1,1))
        self._x = self.x.reshape((1,1,1,-1))
        self._probLambda = np.full((1,len(self.alpha),len(self.beta),1), 1/(len(self.alpha)*len(self.beta)))
        # AGRT.py:133(含 lapse)
        self._probResponseGivenLambdaX = np.array([0,1]).reshape(2,1,1,1) + np.array([1,-1]).reshape(2,1,1,1) * \
            ((self.delta/2) + (1-self.delta) * stats.norm.cdf(self._alpha, loc=self._x, scale=self._beta))
        self.update(None)
    def update(self, response=None):                         # AGRT.py:140-158
        if response is not None:
            self._probLambda = self._probLambdaGivenXResponse[response,:,:,self.nextIntensityIndex].reshape((1,len(self.alpha),len(self.beta),1))
        self._probResponseGivenX = np.sum(self._probResponseGivenLambdaX*self._probLambda, axis=(1,2)).reshape((2,1,1,-1))
        self._probLambdaGivenXResponse = self._probLambda*self._probResponseGivenLambdaX/self._probResponseGivenX
        p = self._probLambdaGivenXResponse
        self._entropyXResponse = -np.sum(p*np.log10(p, out=np.zeros_like(p), where=p>0), axis=(1,2)).reshape((2,1,1,-1))
        self._expectedEntropyX = np.sum(self._entropyXResponse*self._probResponseGivenX, axis=0).reshape((1,1,1,-1))
        self.nextIntensityIndex = np.argmin(self._expectedEntropyX, axis=3)[0][0][0]
        self.nextIntensity = self.x[self.nextIntensityIndex]
    def estimateLambda(self):                                # AGRT.py:160-161
        pl = self._probLambda.squeeze()
        return (float(np.sum(self.alpha*pl.sum(axis=1))), float(np.sum(self.beta*pl.sum(axis=0))))

def make_psi(dim_range, steps=100, lapse=0.0):               # AGRT.py:287-311 的 β 範圍公式
    ml = 1 - np.sqrt(1-lapse)
    bmax = (np.average(dim_range)-dim_range[0]) / (np.sqrt(2)*special.erfinv((2*(.99-ml/2)-ml)/(1-ml)-1))
    return PsiObject(dim_range, dim_range, [bmax/steps, bmax], steps, steps, steps, delta=ml)

def salience_levels(alpha, beta, delta, p_list):
    """SFT 版的 estimateThreshold:同一側、多個目標正確率。
    psiSimulation_functions.R:171  inv.pm.function(y,a,b,d) = qnorm((y-.5d)/(1-d), a, b)"""
    return [alpha + beta*stats.norm.ppf((p-.5*delta)/(1-delta)) for p in p_list]

# ─────────────────────────────────────────────────────────────────────
# 2. DDM 受試者(diffIRT::simdiffT 逐行移植;a=界線, mv=平均漂移, sv=漂移 SD, ter)
# ─────────────────────────────────────────────────────────────────────
def simdiffT(N, a, mv, sv, ter, vp=1.0, max_iter=19999, eps=1e-15, rng=None):
    rng = rng or np.random.default_rng()
    rt = np.empty(N); p = np.empty(N)
    for jj in range(N):
        drift = rng.normal(mv, sv)
        p[jj] = np.exp(a*drift)/(1+np.exp(a*drift))
        lmb = drift**2/(2*vp**2) + np.pi**2*vp**2/(2*a**2)
        FF = np.pi**2*vp**4/(np.pi**2*vp**4 + drift**2*a**2)
        rej = 0
        while True:
            v = rng.uniform(); u = rng.uniform()
            sh1, sh2, sh3, i = 1.0, 0.0, 0.0, 0
            while abs(sh1-sh2) > eps or abs(sh2-sh3) > eps:
                sh1, sh2 = sh2, sh3; i += 1
                sh3 = sh2 + (2*i+1)*(-1)**i*(1-u)**(FF*(2*i+1)**2)
            ev = 1 + (1-u)**(-FF)*sh3
            if v <= ev:
                rt[jj] = abs(np.log(1-u))/lmb + ter; break
            rej += 1
            if rej == max_iter: raise RuntimeError("Rejection algorithm failed")
    x = (p > rng.uniform(size=N)).astype(int)
    return rt, x

def dfp_ddm(N, d1, d2, a, ter, sdv, arch, rule, rng, pmix=.5):   # adaptiveSFT_functions.R:115-150
    if arch == "COA":
        return simdiffT(N, a, d1+d2, sdv, ter, rng=rng)
    rt1, x1 = simdiffT(N, a, d1, sdv, ter, rng=rng); rt2, x2 = simdiffT(N, a, d2, sdv, ter, rng=rng)
    if arch == "PAR":
        if rule == "OR":
            rt = np.minimum(rt1, rt2); cr = np.where(rt1 < rt2, x1, x2)
        else:
            rt = np.maximum(rt1, rt2); cr = x1 & x2
    else:
        if rule == "OR":
            s = rng.uniform(size=N) < pmix; rt = np.where(s, rt1, rt2); cr = np.where(s, x1, x2)
        else:
            rt = rt1 + rt2; cr = x1 & x2
    return rt, cr.astype(int)

# ─────────────────────────────────────────────────────────────────────
# 3. SIC(sft/R/sic.R 移植)
# ─────────────────────────────────────────────────────────────────────
def ecdf(s): s = np.sort(s); return lambda t: np.searchsorted(s, t, side='right')/len(s)
def sic(HH, HL, LH, LL):
    RTall = np.unique(np.concatenate([HH, HL, LH, LL]))
    F = {k: ecdf(v) for k, v in dict(HH=HH, HL=HL, LH=LH, LL=LL).items()}
    N = 1/(1/len(HH)+1/len(HL)+1/len(LH)+1/len(LL))
    sicall = F['LH'](RTall)+F['HL'](RTall)-F['HH'](RTall)-F['LL'](RTall)   # sic.R:145
    Dp = max(0, sicall.max()); Dm = abs(min(0, sicall.min()))               # sic.R:170-183
    out = dict(SIC=(RTall, sicall), Dplus=(Dp, np.exp(-2*N*Dp**2)), Dminus=(Dm, np.exp(-2*N*Dm**2)))
    # siDominance:sic.R:189-216(ks.test 單尾、漸近)
    dom = {}
    for name, (A, B) in dict(hh_hl=(HH,HL), hh_lh=(HH,LH), hl_ll=(HL,LL), lh_ll=(LH,LL)).items():
        g = stats.ks_2samp(A, B, alternative='greater', method='asymp')
        l = stats.ks_2samp(A, B, alternative='less', method='asymp')
        dom['S.'+name.replace('_', ' > S.')] = g.pvalue; dom['S.'+name.replace('_', ' < S.')] = l.pvalue
    out['Dominance'] = dom
    # mic.test(ART):sic.R:219-248
    allrt = np.concatenate([HH, HL, LH, LL]); n1,n2,n3,n4 = map(len, (HH,HL,LH,LL))
    h1 = np.r_[np.ones(n1+n2), np.zeros(n3+n4)]; h2 = np.r_[np.ones(n1), np.zeros(n2), np.ones(n3), np.zeros(n4)]
    mA0 = allrt[h1==0].mean(); mA1 = allrt[h1==1].mean(); mB0 = allrt[h2==0].mean(); mB1 = allrt[h2==1].mean()
    adj = np.round(allrt - (1-h1)*mA0 - h1*mA1 - (1-h2)*mB0 - h2*mB1, 15)
    ranks = stats.rankdata(adj, method='average')
    Xf = np.c_[np.ones_like(h1), h1, h2, h1*h2]; Xr = Xf[:, :3]
    rss = lambda X: np.sum((ranks - X @ np.linalg.lstsq(X, ranks, rcond=None)[0])**2)
    Fstat = (rss(Xr)-rss(Xf))/(rss(Xf)/(len(ranks)-4))
    MIC = (LL.mean()-LH.mean()) - (HL.mean()-HH.mean())
    out['MIC'] = (MIC, stats.f.sf(Fstat, 1, len(ranks)-4))
    return out

# ─────────────────────────────────────────────────────────────────────
# 4. 跑起來:psiSimulation_functions.R Est.Trial.Psi.Color 的設定
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = np.random.default_rng(seed)
    x_range = (-55.0, 50.0); thres50 = 6.0                     # psiSimulation_functions.R:15,93
    a, v, ter, sdv = 1.45, 1.6, .1, .25                        # :96-99
    lapse = 0.02
    drift = lambda x: (x-thres50)/(x_range[1]-thres50)*v       # :104

    # 「真值」:R 腳本的做法 —— 把 DDM 的 P(correct) 曲線用 pm.function 擬合(26MAR2019.R:130-136)
    xx = np.linspace(*x_range, 1000); pc = 1/(1+np.exp(-2*a*drift(xx)))
    ml = 1-np.sqrt(1-lapse)
    pmf = lambda x, al, be: .5*ml + (1-ml)*stats.norm.cdf(x, al, be)
    true_ab = optimize.minimize(lambda p: np.sqrt(np.mean((pmf(xx, *p)-pc)**2)), [6, 15]).x

    for n_trials in (72, 144, 300):
        t0 = time.time(); psi = make_psi(x_range, 100, lapse)
        for _ in range(n_trials):
            _, r = simdiffT(1, a, drift(psi.nextIntensity), sdv, ter, rng=rng)
            psi.update(int(r[0]))
        al, be = psi.estimateLambda(); dt = time.time()-t0
        H, L = salience_levels(al, be, ml, [.99, .90])
        # 驗證:把 H / L 丟回 DDM 跑 2000 試看實際正確率
        _, xH = simdiffT(2000, a, drift(H), sdv, ter, rng=rng); _, xL = simdiffT(2000, a, drift(L), sdv, ter, rng=rng)
        print(f"Psi {n_trials:3d} trials  {dt:5.1f}s   alpha={al:6.2f} (true {true_ab[0]:5.2f})  beta={be:5.2f} (true {true_ab[1]:5.2f})"
              f"   H={H:6.2f} -> acc {xH.mean():.3f} (目標 .99)   L={L:6.2f} -> acc {xL.mean():.3f} (目標 .90)")

    # DFP + SIC:兩個通道都用同一組 H/L(色與方位在 R 裡各校一次,這裡示範用一次)
    print()
    for arch, rule in (("PAR","OR"), ("PAR","AND"), ("SER","OR"), ("SER","AND"), ("COA",None)):
        t0 = time.time()
        cells = {}
        for name, (c1, c2) in dict(HH=(H,H), HL=(H,L), LH=(L,H), LL=(L,L)).items():
            rt, cr = dfp_ddm(250, drift(c1), drift(c2), a, ter, sdv, arch, rule, rng)
            cells[name] = rt[cr == 1]
        s = sic(**cells)
        print(f"{arch}-{rule or '':3}  {time.time()-t0:4.1f}s  D+={s['Dplus'][0]:.3f} p={s['Dplus'][1]:.3f}   "
              f"D-={s['Dminus'][0]:.3f} p={s['Dminus'][1]:.3f}   MIC={s['MIC'][0]:+.3f} p={s['MIC'][1]:.3f}   "
              f"dominance ok={all(p<.05 for k,p in s['Dominance'].items() if '>' in k) and not any(p<.05 for k,p in s['Dominance'].items() if '<' in k)}")
