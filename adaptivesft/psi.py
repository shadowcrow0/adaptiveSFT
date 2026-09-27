"""
Psi 適應法（Kontsevich & Tyler 1999）—— psiSimulation_functions.R 的 Est.Trial.Psi.* 那條路。

PsiObject 逐字取自 Visual_AudioWM/AGRT.py:75-184 的 agrtPsiObject（Glavan 2022，GPL），
去掉那個檔案模組層的 PsychoPy import（AGRT.py:64-67），所以離線模擬與 pytest 都能跑。
模型（AGRT.py:133 ＝ psiSimulation_functions.R:3 的 pm.function）：

    P(r = 1 | x; α, β, δ) = δ/2 + (1 − δ)·Φ((x − α)/β)

與 AGRT 的兩個差別（SFT 用）：
  1. lapse 直接用 δ，不做 AGRT.py:288 的邊際換算 1 − √(1 − λ)（那是 GRT 兩維聯合正確率用的）。
  2. 反解用 salience_levels()：同一側、多個目標正確率（psiSimulation_functions.R:171 inv.pm.function），
     不是 AGRT.estimateThreshold 的「一個正確率、α 兩側」。

β 網格上限照 AGRT.py:295-298 由刺激範圍推出（假設範圍端點 ≈ 99% 正確）；受試者比這個假設鈍時 β 會
被釘在上限（plan_grtv3ada_psi_python.md §5.2），所以 make_psi 提供 beta_max 覆寫。
"""
import numpy as np
from scipy import special, stats

__all__ = ["PsiObject", "make_psi", "salience_levels", "pm_function", "inv_pm_function"]


def pm_function(x, a, b, d):
    """psiSimulation_functions.R:3。"""
    return 0.5 * d + (1 - d) * stats.norm.cdf(x, a, b)


def inv_pm_function(y, a, b, d):
    """psiSimulation_functions.R:171：qnorm((y − .5d)/(1 − d), a, b)。"""
    return stats.norm.ppf((np.asarray(y, dtype=float) - 0.5 * d) / (1 - d), a, b)


class PsiObject:
    """AGRT.py:75-184 agrtPsiObject，陣列順序 [r, α, β, x]。"""

    def __init__(self, x, alpha, beta, xPrecision, aPrecision, bPrecision, delta=0.0, stepType="lin", prior=None):
        if stepType == "lin":
            self.x = np.linspace(x[0], x[1], xPrecision, True)
        elif stepType == "log":
            self.x = np.logspace(np.log10(x[0]), np.log10(x[1]), xPrecision, True)
        else:
            raise RuntimeError("Invalid step type. Unable to initialize PsiObject.")
        self.alpha = np.linspace(alpha[0], alpha[1], aPrecision)
        self.beta = np.linspace(beta[0], beta[1], bPrecision)
        self.r = np.arange(2)
        self.delta = delta
        self._r = self.r.reshape((2, 1, 1, 1))
        self._alpha = self.alpha.reshape((1, -1, 1, 1))
        self._beta = self.beta.reshape((1, 1, -1, 1))
        self._x = self.x.reshape((1, 1, 1, -1))
        shape = (1, len(self.alpha), len(self.beta), 1)
        if prior is None or np.shape(prior) != shape:
            self._probLambda = np.full(shape, 1.0 / (len(self.alpha) * len(self.beta)))
        else:
            self._probLambda = np.asarray(prior, dtype=float).reshape(shape)
        # AGRT.py:133（含 lapse）
        self._probResponseGivenLambdaX = (
            np.array([0, 1]).reshape(2, 1, 1, 1)
            + np.array([1, -1]).reshape(2, 1, 1, 1)
            * ((self.delta / 2) + (1 - self.delta) * stats.norm.cdf(self._alpha, loc=self._x, scale=self._beta))
        )
        self.nextIntensityIndex = None
        self.nextIntensity = None
        self.update(None)

    def update(self, response=None):
        """AGRT.py:140-158。response 為 0/1；None 只做初始化。"""
        if response is not None:
            self._probLambda = self._probLambdaGivenXResponse[response, :, :, self.nextIntensityIndex].reshape(
                (1, len(self.alpha), len(self.beta), 1))
        self._probResponseGivenX = np.sum(self._probResponseGivenLambdaX * self._probLambda, axis=(1, 2)).reshape(
            (2, 1, 1, -1))
        self._probLambdaGivenXResponse = self._probLambda * self._probResponseGivenLambdaX / self._probResponseGivenX
        p = self._probLambdaGivenXResponse
        self._entropyXResponse = -np.sum(p * np.log10(p, out=np.zeros_like(p), where=p > 0), axis=(1, 2)).reshape(
            (2, 1, 1, -1))
        self._expectedEntropyX = np.sum(self._entropyXResponse * self._probResponseGivenX, axis=0).reshape((1, 1, 1, -1))
        self.nextIntensityIndex = int(np.argmin(self._expectedEntropyX, axis=3)[0][0][0])
        self.nextIntensity = float(self.x[self.nextIntensityIndex])

    def estimateLambda(self):
        """AGRT.py:160-161：後驗平均 (α, β)。"""
        pl = self._probLambda.squeeze()
        return float(np.sum(self.alpha * pl.sum(axis=1))), float(np.sum(self.beta * pl.sum(axis=0)))

    def savePosterior(self):
        return self._probLambda


def make_psi(dim_range, steps=100, lapse=0.0, beta_max=None, prior=None):
    """
    AGRT.py:287-311 的建構方式（單維、δ = lapse）。
    beta_max 預設照 AGRT.py:295：(mean(range) − min(range)) / (√2·erfinv((2·(.99 − δ/2) − δ)/(1 − δ) − 1))。
    """
    d = float(lapse)
    if beta_max is None:
        beta_max = (np.average(dim_range) - dim_range[0]) / (
            np.sqrt(2) * special.erfinv((2 * (0.99 - d / 2) - d) / (1 - d) - 1))
    beta_range = [beta_max / steps, beta_max]
    return PsiObject(dim_range, dim_range, beta_range, steps, steps, steps, delta=d, stepType="lin", prior=prior)


def salience_levels(alpha, beta, delta, p_list, x_range=None):
    """
    SFT 版反解：同一側、多個目標正確率（psiSimulation_functions.R:183-184 用 .99 / .90）。
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
