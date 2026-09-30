"""
Psi 適應法（Kontsevich & Tyler 1999）—— psiSimulation_functions.R 的 Est.Trial.Psi.Color（:5-168）
與 Est.Trial.Psi.Orientation（:214-373）的逐行移植。兩個 R 函式除了網格常數以外逐字相同，
這裡合成一個 Psi 類別，網格由呼叫端給（預設值就是 R 的兩組常數，見 GRIDS）。
不引用、不依賴 Visual_AudioWM/AGRT.py。

模型（R :3）：

    pm.function(x, a, b, d) = ½·d + (1 − d)·Φ((x − a)/b)          P(r = 1 | x; α, β, δ)

R 的流程（行號為 Est.Trial.Psi.Color 的）：
    :24-27    x / a / b 網格用 seq(range, step)，r = (0, 1)
    :31-40    pR.LX[r, a, b, x] = (1 − r) + (2r − 1)·pm.function(x, a, b, d)
    :45-49    pL 均勻先驗（prior 是 NA）
    :52-57    pR.X[r, x]   = Σ_{a,b} pR.LX · pL
    :60-69    pL.XR[r,a,b,x] = pL · pR.LX / pR.X
    :72-77    entropy.XR[r, x] = −Σ_{a,b} pL.XR · log10(pL.XR)
    :80-83    expected.entropy.X[x] = Σ_r entropy.XR · pR.X
    :86-87    next.intensity = argmin
    :112-114  作答後 pL ← pL.XR[r, , , x_index]，然後重算 :117-150
    :154-163  α̂ = Σ a·pL，β̂ = Σ b·pL（後驗平均）
    :171      inv.pm.function(y, a, b, d) = qnorm((y − ½d)/(1 − d), a, b)
    :183-184  H = inv.pm.function(.99, …)，L = inv.pm.function(.90, …)

與 R 的兩個數值細節：(1) R 的 −Σ p·log10(p) 在 p = 0 時得 NaN（which.min 會忽略那個 x），這裡把
0·log(0) 當 0，pL.XR 只在 Φ underflow 時才會是 0；(2) 期望熵同分時取最前面的 x（見 _recompute 的註解）。
tests/test_parity_demo.py 用 R 逐行跑出的 oracle 驗證：顏色與方位兩組網格、各 300 試，每一試選的 x 相同。
"""
import numpy as np
from scipy import stats

__all__ = ["GRIDS", "Psi", "make_psi", "salience_levels", "pm_function", "inv_pm_function"]

# psiSimulation_functions.R:15-18（顏色）、:222-225（方位）：(下限, 上限, 步長)
GRIDS = {
    "colour": dict(x=(-55.0, 50.0, 1.0), a=(-25.0, 45.0, 1.0), b=(1.0, 50.0, 1.0), d=0.01),
    "orientation": dict(x=(45.0, 90.0, 0.5), a=(45.5, 75.0, 0.5), b=(1.0, 10.0, 0.5), d=0.01),
}


def _seq(lo, hi, step):
    """R 的 seq(lo, hi, step)：含兩端（浮點誤差時以 hi 為準）。"""
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    return lo + step * np.arange(n)


def pm_function(x, a, b, d):
    """psiSimulation_functions.R:3。"""
    return 0.5 * d + (1 - d) * stats.norm.cdf(x, a, b)


def inv_pm_function(y, a, b, d):
    """psiSimulation_functions.R:171。"""
    return stats.norm.ppf((np.asarray(y, dtype=float) - 0.5 * d) / (1 - d), a, b)


class Psi:
    """Est.Trial.Psi.* 的內部狀態。陣列順序同 R：[r, a, b, x]。"""

    def __init__(self, x, a, b, d=0.01, prior=None):
        self.x = _seq(*x)
        self.a = _seq(*a)
        self.b = _seq(*b)
        self.d = float(d)
        self.r = np.array([0, 1])
        # :31-40  P(R | L, X)
        pm = pm_function(self.x[None, None, :], self.a[:, None, None], self.b[None, :, None], self.d)  # [a, b, x]
        self.pR_LX = np.stack([1.0 - pm, pm])                                                     # [r, a, b, x]
        # :45-49  P(L)
        if prior is None:
            self.pL = np.full((len(self.a), len(self.b)), 1.0 / (len(self.a) * len(self.b)))
        else:
            self.pL = np.asarray(prior, dtype=float).reshape(len(self.a), len(self.b))
        self.next_index = None
        self.next_intensity = None
        self._recompute()

    def _recompute(self):
        """R :52-87 / :117-150：從目前的 pL 算到下一題的 x。"""
        pR_X = np.einsum("rabx,ab->rx", self.pR_LX, self.pL)                       # :52-57
        self.pL_XR = self.pL[None, :, :, None] * self.pR_LX / pR_X[:, None, None, :]  # :60-69
        p = self.pL_XR
        ent = -np.sum(p * np.log10(p, out=np.zeros_like(p), where=p > 0), axis=(1, 2))   # :72-77
        expected = np.sum(ent * pR_X, axis=0)                                       # :80-83
        # :86 which.min 取第一個最小值。網格對稱時（方位：α 網格中心 60.25 落在 x 網格的 60 與 60.5 之間）
        # 兩個 x 的期望熵在數學上相等，只差浮點加總順序；R 的巢狀迴圈和這裡的 einsum 順序不同，
        # 硬比 argmin 會各選各的。所以在 1e−12 內視為同分，取最前面的一個（＝R 對真正同分的行為）。
        self.next_index = int(np.argmax(expected <= expected.min() + 1e-12))
        self.next_intensity = float(self.x[self.next_index])                        # :87

    def update(self, response):
        """R :112-114：作答（0/1）後更新 pL，重算下一題。"""
        self.pL = self.pL_XR[int(response), :, :, self.next_index]
        self._recompute()

    def estimate(self):
        """R :154-163：後驗平均 (α̂, β̂)。"""
        return float(np.sum(self.a * self.pL.sum(axis=1))), float(np.sum(self.b * self.pL.sum(axis=0)))

    # 給舊呼叫端用的別名
    nextIntensity = property(lambda self: self.next_intensity)
    estimateLambda = estimate


def make_psi(dim=None, x=None, a=None, b=None, d=None, prior=None):
    """
    用 R 的網格常數建一個 Psi：make_psi("colour") / make_psi("orientation")，
    或自己給 x / a / b = (下限, 上限, 步長)、d。
    """
    g = dict(GRIDS[dim]) if dim is not None else {}
    if x is not None:
        g["x"] = x
    if a is not None:
        g["a"] = a
    if b is not None:
        g["b"] = b
    if d is not None:
        g["d"] = d
    missing = {"x", "a", "b"} - set(g)
    if missing:
        raise ValueError(f"缺網格 {sorted(missing)}：給 dim 或自己給 x / a / b")
    return Psi(g["x"], g["a"], g["b"], g.get("d", 0.01), prior)


def salience_levels(alpha, beta, delta, p_list, x_range=None):
    """
    psiSimulation_functions.R:183-184：同一側多個目標正確率（原碼 .99 / .90）→ 刺激值。
    回傳 (levels, warnings)；給 x_range 時，落在範圍外的目標會進 warnings
    （原作者自己的 .99 就在範圍外：psi Simulation_25JUNE2018.R:174 = 101.6 > 50）。
    """
    levels = [float(inv_pm_function(p, alpha, beta, delta)) for p in p_list]
    warnings = []
    if x_range is not None:
        for p, x in zip(p_list, levels):
            if not (x_range[0] <= x <= x_range[1]):
                warnings.append(f"目標 {p}: x = {x:.3f} 落在刺激範圍 [{x_range[0]}, {x_range[1]}] 之外")
    return levels, warnings
