"""
lnrm2.stan 的 PyMC 版本（目前可以轉換的部分）。

每一段都對應 lnrm2.stan 的某幾行，用中文註解說明「這段在做什麼」。
轉換時撞到的問題全部列在下面的 TODO 區塊，細節見 lnrm2_pymc_gaps.md。

用法：
    python lnrm2_pymc.py            # 用模擬資料自我測試（看能不能把真值找回來）

    from lnrm2_pymc import fit_lnrm2
    idata = fit_lnrm2(intensity, rt, correct)

驗證環境：PyMC 5.28.5 / PyTensor 2.38.3 / Python 3.11
"""

# =====================================================================
# TODO：Stan -> PyMC 轉換時撞到的問題
# =====================================================================
#
# [x] TODO-1  lognormal_lccdf 在 PyMC 沒有對應函式。
#             直覺寫法 pm.math.log1mexp(pm.logcdf(...)) 在 varZ 偏小時會回傳 -inf
#             （CDF 在 float64 飽和成 1.0 -> logcdf = -0.0 -> log(0)），
#             NUTS 初始化就會踩到，模型還沒開始跑就 SamplingError。
#             狀態：已繞過。改走常態 CDF：ln S(y) = ln Φ(−(ln y − m)/s)，見 lognormal_lccdf()。
#             實測與 scipy.stats.lognorm.logsf 在 s=0.01（值 -31149）仍完全吻合。
#
# [ ] TODO-2  Stan 的 target += 作用在 rt[tr] - psi 上（資料減參數）。
#             PyMC 的 pm.CustomDist(observed=rt - psi) 直接報錯：
#             "Variables that depend on other nodes cannot be used for observed data"。
#             狀態：改用 pm.Potential（最接近 target +=）。
#             未解的代價：Potential 不是隨機變數，做不了 posterior predictive check。
#             若之後需要 PPC，要改成 CustomDist 並把 rt - psi 搬進 logp 函式內部。
#
# [x] TODO-3  Stan 的 psi 只有宣告 real<lower=0,upper=minRT>，沒有 ~ 敘述，
#             宣告本身就是先驗。PyMC 沒有「宣告即先驗」，必須明寫 pm.Uniform。
#             狀態：已處理，數學上等價（有限區間上的 flat = Uniform）。
#             注意：原始碼註解寫 "improper flat prior on positive reals"，
#             跟宣告的邊界不一致，這是 Stan 檔本身的問題，見 lnrm2_stan_explained.md §11.3。
#
# [x] TODO-4  Stan model block 裡逐試次的 if (correct[tr]) / else。
#             PyMC 是靜態計算圖，不能對 tensor 做 Python if。
#             狀態：已處理，改用 pt.switch 向量化。
#
# [ ] TODO-5  Stan 的 transformed parameters（z[2,N]）每個抽樣都會自動存下來；
#             PyMC 預設不存，要存得包 pm.Deterministic。
#             狀態：目前不存（等同 R 端用 pars= 排除 z）。需要的話把下面 SAVE_Z 改 True。
#
# [x] TODO-6  PyTensor 預設 float32，尾端 log-survival（-1250 ~ -31149）精度不夠。
#             狀態：已處理，檔案開頭強制 floatX = "float64"（必須在 import pymc 之前）。
#
# =====================================================================

import numpy as np
import pytensor

pytensor.config.floatX = "float64"          # TODO-6：一定要在 import pymc 之前

import pytensor.tensor as pt                # noqa: E402
import pymc as pm                           # noqa: E402

SAVE_Z = False                              # TODO-5：要不要把 z 存進輸出


# ---------------------------------------------------------------------
# TODO-1 的替代品：Stan 的 lognormal_lccdf(y | m, s)
# ---------------------------------------------------------------------
def lognormal_lccdf(y, m, s):
    """
    回傳 ln S(y; m, s) = ln P(Y > y)，Y 是對數常態。

    不要用 pm.math.log1mexp(pm.logcdf(...))：s 小的時候會給 -inf。
    改用等價式子  S(y) = Φ( −(ln y − m) / s )，
    PyTensor 的常態 logcdf 內部走 erfcx，在極端尾端仍然準確。
    """
    return pm.logcdf(pm.Normal.dist(0.0, 1.0), -(pt.log(y) - m) / s)


# ---------------------------------------------------------------------
# 主體：對應 lnrm2.stan 的每個 block
# ---------------------------------------------------------------------
def build_model(intensity, rt, correct, minRT=None):
    """
    建立與 lnrm2.stan 等價的 PyMC 模型。

    intensity : (N,) 每一題的刺激強度            <- lnrm2.stan:3   intensity[N]
    rt        : (N,) 每一題按按鈕花幾秒           <- lnrm2.stan:6   rt[N]
    correct   : (N,) 每一題答對沒（1/0）          <- lnrm2.stan:4   correct[N]
    minRT     : psi 的上界，預設 rt.min()        <- lnrm2.stan:5   minRT
                （R 端 dataframe2stan() 也是用 min(rt)）
    """
    intensity = np.asarray(intensity, dtype=float)
    rt = np.asarray(rt, dtype=float)
    correct = np.asarray(correct).astype(int)
    if minRT is None:
        minRT = float(rt.min())

    with pm.Model() as model:

        # ---- data block（lnrm2.stan:1-7）------------------------------
        # 三欄資料包成 pm.Data，之後可以用 pm.set_data 換資料重跑。
        intensity_d = pm.Data("intensity", intensity)
        rt_d = pm.Data("rt", rt)
        correct_d = pm.Data("correct", correct)

        # ---- transformed data（lnrm2.stan:8-11）------------------------
        #   square_intensity = square(intensity);
        # 只跟資料有關，算一次就好。
        sq_intensity_d = pm.Data("square_intensity", intensity ** 2)

        # ---- parameters + 先驗（lnrm2.stan:13-17, 30-33）----------------
        #   mu     ~ normal(0,1);        兩個小人共同的基礎腳程
        #   alpha  ~ normal(0,2);        題目變明顯時賽跑變多不公平（一次項）
        #   alpha2 ~ normal(0,1);        效果會不會飽和（二次項）
        mu = pm.Normal("mu", 0.0, 1.0)
        alpha = pm.Normal("alpha", 0.0, 2.0)
        alpha2 = pm.Normal("alpha2", 0.0, 1.0)

        #   real<lower=0> varZ;  varZ ~ inv_gamma(1,.1);
        # 名字叫 varZ，但它在 Stan 裡是放在「標準差」的位置。
        # 這裡照樣當標準差用，等一下傳 sigma=varZ，不要開根號。
        # PyMC 的 InverseGamma(alpha, beta) 跟 Stan 的 inv_gamma(shape, scale) 參數順序一致。
        varZ = pm.InverseGamma("varZ", alpha=1.0, beta=0.1)

        #   real<lower=0,upper=minRT> psi;   （沒有 ~ 敘述）
        # TODO-3：Stan 靠宣告邊界當先驗；PyMC 要明寫。有限區間上的 flat 就是 Uniform。
        psi = pm.Uniform("psi", lower=0.0, upper=minRT)

        # ---- transformed parameters（lnrm2.stan:19-28）------------------
        #   z[1,tr] = mu - alpha*intensity[tr] - alpha2*square_intensity[tr];
        #   z[2,tr] = mu + alpha*intensity[tr] + alpha2*square_intensity[tr];
        #
        #        z1 = mu - d     "答對" 小人（減，所以變快）
        #              ^
        #              |  d = alpha*intensity + alpha2*intensity^2   （賽跑有多不公平）
        #              v
        #        z2 = mu + d     "答錯" 小人（加，所以變慢）
        d = alpha * intensity_d + alpha2 * sq_intensity_d
        z1 = mu - d
        z2 = mu + d
        if SAVE_Z:                          # TODO-5
            z1 = pm.Deterministic("z1", z1)
            z2 = pm.Deterministic("z2", z2)

        # ---- model：似然（lnrm2.stan:36-45）-----------------------------
        #   if (correct[tr]) { 贏家用 z[1], 輸家用 z[2] } else { 對調 }
        # TODO-4：不能用 Python if，改 pt.switch 一次選好全部 N 題的贏家/輸家。
        is_correct = pt.eq(correct_d, 1)
        z_win = pt.switch(is_correct, z1, z2)      # 贏的那個小人
        z_lose = pt.switch(is_correct, z2, z1)     # 輸的那個小人

        #   rt[tr] - psi   扣掉鳴槍到起跑的固定延遲，剩下真正在跑的秒數
        # clip 防止採樣途中 rt - psi <= 0 讓 log 爆掉（psi 有上界 minRT，理論上不會，但保險）。
        shifted_rt = pt.clip(rt_d - psi, 1e-12, np.inf)

        #   target += lognormal_lpdf (rt[tr] - psi | z_win,  varZ);   贏家剛好這時到終點
        #   target += lognormal_lccdf(rt[tr] - psi | z_lose, varZ);   輸家這時還沒到
        logp_win = pm.logp(pm.LogNormal.dist(mu=z_win, sigma=varZ), shifted_rt)
        logccdf_lose = lognormal_lccdf(shifted_rt, z_lose, varZ)   # TODO-1

        # TODO-2：Stan 的 target += 對應 pm.Potential。
        # 把 N 題的分數全部加起來，一次加進對數後驗。
        pm.Potential("race_loglik", pt.sum(logp_win + logccdf_lose))

    return model, minRT


def fit_lnrm2(intensity, rt, correct, minRT=None, draws=1000, tune=1000,
              chains=4, target_accept=0.9, random_seed=None, **kwargs):
    """建模 + 抽樣，回傳 arviz InferenceData。"""
    model, minRT = build_model(intensity, rt, correct, minRT)
    with model:
        return pm.sample(
            draws=draws, tune=tune, chains=chains, target_accept=target_accept,
            random_seed=random_seed,
            # 明確給 psi 初始值，別讓預設 jitter 把它丟到貼近 minRT 的邊界
            initvals={"psi": 0.3 * minRT},
            **kwargs,
        )


# ---------------------------------------------------------------------
# 自我測試用：從模型本身生資料
# ---------------------------------------------------------------------
def simulate(n=500, mu=1.5, alpha=0.8, alpha2=-0.15, sigma=0.6, psi=0.12, seed=7):
    """
    照 lnrm2.stan 的世界觀生一批資料：
    兩個小人各抽一個完成時間，誰小誰贏；rt = 贏家的時間，correct = 是否答對小人贏。
    """
    rng = np.random.default_rng(seed)
    intensity = rng.uniform(0.0, 3.0, n)
    d = alpha * intensity + alpha2 * intensity ** 2
    t_correct = psi + np.exp(rng.normal(mu - d, sigma))     # 答對小人
    t_error = psi + np.exp(rng.normal(mu + d, sigma))       # 答錯小人
    rt = np.minimum(t_correct, t_error)                     # 先到終點的那個
    correct = (t_correct < t_error).astype(int)
    return intensity, rt, correct


if __name__ == "__main__":
    import arviz as az

    truth = dict(mu=1.5, alpha=0.8, alpha2=-0.15, sigma=0.6, psi=0.12)
    intensity, rt, correct = simulate(n=500, **truth)
    print(f"N={len(rt)}  minRT={rt.min():.4f}  正確率={correct.mean():.3f}")

    idata = fit_lnrm2(intensity, rt, correct, random_seed=1, progressbar=False)

    summary = az.summary(idata, var_names=["mu", "alpha", "alpha2", "varZ", "psi"])
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]].to_string())
    print("\n真值: mu=1.5  alpha=0.8  alpha2=-0.15  varZ(=sigma)=0.6  psi=0.12")
    print("divergences:", int(idata.sample_stats.diverging.sum()))
    print("\n注意：mu 與 psi 在後驗上高度相關，N=500 時兩者會互相補償，")
    print("      這是模型本身的性質（lnrm2_derivations.md），不是轉換錯誤。")
