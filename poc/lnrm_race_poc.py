"""adaptiveSFT_functions.R:61-111 dlognormalrace / plognormalrace / lnrm_adjusted_integral 的 scipy 版。
sigmasq 依 R 的簽名收「變異數」再開根號(issue.md S3);m 是 0-based 的勝者索引。"""
import numpy as np
from scipy.stats import lognorm
from scipy.integrate import quad

def dlognormalrace(x, m, psi, mu, sigmasq, log=False):          # R :61-77
    sigma = np.sqrt(np.asarray(sigmasq, float)); x = np.asarray(x, float); t = x - psi
    with np.errstate(divide='ignore', invalid='ignore'):
        g = lognorm.logpdf(t, s=sigma[m], scale=np.exp(mu[m]))
        G = sum(lognorm.logsf(t, s=sigma[i], scale=np.exp(mu[i])) for i in range(len(mu)) if i != m)
        out = g + G
    out = np.where(t > 0, out, -np.inf)
    return out if log else np.exp(out)

def plognormalrace(x, m, psi, mu, sigmasq):                    # R :81-111(integrate 0..x;不用 R 的 stepsize 補救)
    return np.array([quad(dlognormalrace, psi, xi, args=(m, psi, mu, sigmasq), limit=200)[0] if xi > psi else 0.0
                     for xi in np.atleast_1d(x)])

if __name__ == "__main__":
    mu = np.array([1.5-0.8, 1.5+0.8]); sigmasq = np.array([.36, .36]); psi = .12
    tot = sum(quad(dlognormalrace, psi, np.inf, args=(m, psi, mu, sigmasq))[0] for m in (0, 1))
    p_win0 = quad(dlognormalrace, psi, np.inf, args=(0, psi, mu, sigmasq))[0]
    from scipy.stats import norm
    print(f"兩個勝者的密度總積分 = {tot:.6f}(應為 1)")
    print(f"P(累積器 0 先到) 積分 = {p_win0:.4f}   封閉解 Φ(2d/(σ√2)) = {norm.cdf(1.6/(0.6*np.sqrt(2))):.4f}")
    # 蒙地卡羅對照 CDF
    rng = np.random.default_rng(1); N = 200000
    t0 = psi + rng.lognormal(mu[0], .6, N); t1 = psi + rng.lognormal(mu[1], .6, N)
    for x in (0.5, 1.0, 2.0, 4.0):
        mc = np.mean((t0 < t1) & (t0 <= x)); print(f"  x={x}: plognormalrace={plognormalrace(x,0,psi,mu,sigmasq)[0]:.4f}  MC={mc:.4f}")
