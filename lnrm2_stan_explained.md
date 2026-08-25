# `lnrm2.stan` 流程拆解

> 對象：`adaptiveSFT/lnrm2.stan`
> 模型類別：**Log-Normal Race Model (LNRM)**，帶非決策時間位移 `psi`，
> 以二次多項式把刺激強度（intensity / salience）映射到累積器的漂移速度。

---

## 1. 一句話總結

這個 Stan 檔在做的事情是：**用「兩個對數常態累積器賽跑」的模型，同時擬合反應時間 (RT)
與正確率 (accuracy)，把「刺激強度 → 作業難度」的關係估成一條二次曲線**，
好讓外層的 adaptive 程序能反推「要用多強的刺激才能達到指定的難度水準」。

它不是一個獨立的分析腳本，而是 `adaptiveSFT_functions.R` 裡
`find_salience_polynomial(polynomial_order = 2)` 所呼叫的擬合引擎。

---

## 2. 背景：什麼是 Log-Normal Race Model

賽跑模型 (race model) 的假設是：每個可能的反應各自對應一個獨立的證據累積器，
誰先到達門檻，誰就決定了受試者的反應。觀察到的資料只有兩個數字
——**哪一個贏了**（correct / incorrect）以及**贏在什麼時候**（RT）。

LNRM 把「第 *i* 個累積器的完成時間」寫成

$$T_i = \psi + \exp(X_i), \qquad X_i \sim \mathcal{N}(z_i,\ \sigma^2)$$

也就是說 $T_i - \psi$ 服從對數常態分佈。其中

* $\psi$ = **非決策時間 (non-decision time)**，編碼與運動反應的固定成本，
  所有累積器共用；
* $z_i$ = 第 *i* 個累積器的**對數尺度速度參數**。
  在對數常態下 $z_i$ **越小代表越快**（注意方向與 drift rate 相反）。

本模型只有兩個累積器：

| 索引 | 意義 | 對應的反應 |
|---|---|---|
| `z[1,]` | 正確反應的累積器 | 贏 ⇒ `correct == 1` |
| `z[2,]` | 錯誤反應的累積器 | 贏 ⇒ `correct == 0` |

---

## 3. 逐段拆解

### 3.1 `data` — 輸入什麼

```stan
int<lower=1> N;                       // 試次總數
real intensity[N];                    // 每一試次的刺激強度 / 顯著度
int<lower=0,upper=1> correct[N];      // 該試次是否答對
real<lower=0> minRT;                  // 全體 RT 的最小值
real<lower=0> rt[N];                  // 每一試次的反應時間
```

這份資料是由 R 端的 `dataframe2stan()` 打包的
（`adaptiveSFT_functions.R:168`）：

```r
standat <- with(dat, list(N = dim(dat)[1], intensity = intensity,
                          correct = correct, minRT = min(rt), rt = rt))
```

`minRT` 之所以要當成資料傳進來，是因為它被用來當作 `psi` 的上界
——非決策時間不可能比觀察到最快的那個反應還長，否則 `rt - psi` 會變成負數，
對數常態密度沒有定義。

### 3.2 `transformed data` — 預先算好平方項

```stan
real square_intensity[N];
square_intensity = square(intensity);
```

純粹是效率考量。`intensity²` 只跟資料有關、與參數無關，
所以放在 `transformed data` 只會在編譯後算一次，不會每個 leapfrog step 重算。

### 3.3 `parameters` — 要估什麼

| 參數 | 範圍 | 角色 |
|---|---|---|
| `mu` | 實數 | 兩個累積器速度的**共同基準**（整體反應快慢） |
| `alpha` | 實數 | intensity 的**一次項**係數 |
| `alpha2` | 實數 | intensity 的**二次項**係數（曲率） |
| `varZ` | > 0 | 對數常態的尺度參數（見 §6 的命名陷阱） |
| `psi` | (0, minRT) | 非決策時間 |

### 3.4 `transformed parameters` — 難度如何進入模型

```stan
for (tr in 1:N) {
   z[1,tr] = mu - alpha * intensity[tr] - alpha2 * square_intensity[tr];
   z[2,tr] = mu + alpha * intensity[tr] + alpha2 * square_intensity[tr];
}
```

這是整個模型的核心設計。定義**難度函數**

$$d(x) \;=\; \alpha x + \alpha_2 x^2$$

則兩個累積器是圍繞共同基準 `mu` **對稱地**被推開的：

$$z_1 = \mu - d(x), \qquad z_2 = \mu + d(x)$$

意義是：

* $d(x) > 0$ ⇒ 正確累積器變快（$z_1$ 變小）、錯誤累積器變慢 ⇒ 正確率上升、RT 下降。
* $d(x) = 0$ ⇒ 兩者完全相同 ⇒ 正確率 50%（純猜測）。
* 刺激強度的效果不必是線性的：`alpha2` 讓它可以是一條**拋物線**，
  能捕捉常見的「一開始效果很大、後來飽和」的心理物理曲線形狀。

這個對稱參數化的好處是 `mu` 只管**整體速度**、`d(x)` 只管**難度**，
兩者的角色分得比較乾淨。

### 3.5 `model` — 先驗與似然

**先驗：**

```stan
varZ  ~ inv_gamma(1, .1);   // 尺度參數，弱訊息、限制為正
mu    ~ normal(0, 1);
alpha ~ normal(0, 2);       // 一次項給比較寬的先驗
alpha2~ normal(0, 1);       // 二次項收得比較緊 → 偏好接近線性的解
```

`psi` 沒有明寫先驗。原始碼註解寫的是 *"improper flat prior on positive reals"*，
但實際上因為宣告時已經有 `<lower=0, upper=minRT>`，
Stan 會自動套用 logit 轉換並加上對應的 Jacobian，
**實際得到的是 (0, minRT) 上的均勻分佈**——是一個 proper prior，不是 improper 的。
註解與實作在這裡有落差。

**似然：race 的關鍵**

```stan
if (correct[tr]) {
   target += lognormal_lpdf (rt[tr] - psi | z[1,tr], varZ);   // 贏家的密度
   target += lognormal_lccdf(rt[tr] - psi | z[2,tr], varZ);   // 輸家還沒完成
} else {
   target += lognormal_lpdf (rt[tr] - psi | z[2,tr], varZ);
   target += lognormal_lccdf(rt[tr] - psi | z[1,tr], varZ);
}
```

這兩行是整個模型的靈魂。推導如下：

我們觀察到的是 $(T, m)$ ——完成時間 $T=t$ 以及贏家身分 $m$。
「$m$ 在 $t$ 時刻獲勝」這個事件等於：

1. 累積器 $m$ **恰好**在 $t$ 完成 → 密度 $f_m(t)$
2. **且**所有其他累積器在 $t$ 時**都還沒**完成 → 存活函數 $\prod_{i \ne m} S_i(t)$

因為累積器彼此獨立，聯合密度就是相乘：

$$\mathcal{L}(t, m) \;=\; f_m(t)\;\prod_{i \neq m} S_i(t)
\;=\; f_m(t)\;\prod_{i \neq m}\bigl[1 - F_i(t)\bigr]$$

取對數之後乘法變加法，就是上面那兩個 `target +=`。
Stan 的 `lognormal_lccdf` 就是 $\log S(t) = \log[1-F(t)]$
（lccdf = **l**og **c**omplementary **c**umulative **d**istribution **f**unction，
也就是 log survival function），而且它是**數值穩定**地直接算出來的，
不是先算 CDF 再做 `log(1 - ...)`。這點在移植到別的框架時特別重要（見 §7）。

同樣的邏輯在 R 端有一份對照實作，可以互相驗證
（`adaptiveSFT_functions.R:61` 的 `dlognormalrace`）：

```r
g <- dlnorm(x - psi, mu[m], sigma[m], log = TRUE)              # 贏家密度
G <- G + plnorm(x - psi, mu[i], sigma[i],
                lower = FALSE, log = TRUE)                     # 輸家存活
rval <- exp(g + G)
```

`lower = FALSE` 就是取上尾，等價於 `lccdf`。

---

## 4. 由參數推回正確率

因為兩個累積器共用同一個 $\sigma$，正確率有封閉解。
反應正確 $\iff T_1 < T_2 \iff X_1 < X_2$，而

$$X_1 - X_2 \sim \mathcal{N}\bigl(z_1 - z_2,\ 2\sigma^2\bigr),
\qquad z_1 - z_2 = -2d(x)$$

所以

$$P(\text{correct} \mid x) \;=\; P(X_1 - X_2 < 0)
\;=\; \Phi\!\left(\frac{2\,d(x)}{\sigma\sqrt{2}}\right)
\;=\; \Phi\!\left(\frac{\sqrt{2}\,d(x)}{\sigma}\right)$$

注意正確率只取決於 **$d(x)/\sigma$ 這個比值**，與 `mu` 和 `psi` 無關。
`mu` 與 `psi` 只影響 RT 的位置，不影響正確率。這是一個很有用的檢查：

* 想調**正確率** → 動 `alpha` / `alpha2` / `varZ`
* 想調**整體 RT** → 動 `mu` / `psi`

也因此 `mu` 與 `psi` 之間存在明顯的**相關性**（兩者都在拉 RT 的位置），
在後驗上通常會看到一條斜的脊 (ridge)，是這個模型採樣時的主要困難來源之一。

### 4.1 數值驗證

上面兩個結論（似然是一個合法的聯合密度、以及正確率的封閉解）都經過蒙地卡羅與
數值積分交叉驗證。取 `mu=1.5, alpha=0.8, alpha2=-0.15, sigma=0.6, psi=0.12, x=1.0`：

```
m=0 (correct)    ∫ f_1(t)S_2(t) dt = 0.937247
m=1 (incorrect)  ∫ f_2(t)S_1(t) dt = 0.062753
                 總和              = 1.000000     ← 確認是合法密度
P(correct) 由積分求得 = 0.937247
P(correct) 由 Φ(√2 d/σ) = 0.937247               ← 兩者一致
```

正確率公式另外也用 40 萬次模擬賽跑檢查過，在 x = 0.5 ~ 3.0 的範圍內
模擬值與理論值的差距都在 ±0.0006 以內（蒙地卡羅誤差範圍內）。

§5 提到的二次式反解也驗證過是精確的：把 `h_targ.dist` 代回
$\alpha_2 x^2 + \alpha x$ 確實恰好等於 `h_targ / 2`。

---

## 5. 它在 adaptiveSFT 整體流程中的位置

`lnrm2.stan` 是 adaptive 校準迴圈裡的「量尺」。完整流程：

```
  受試者做 method-of-constant-stimuli 作業
              │
              ▼
  data.frame(intensity, rt, correct)
              │  dataframe2stan()                   [R:168]
              ▼
  standat = {N, intensity, correct, minRT, rt}
              │  stan(file = "lnrm2.stan", ...)     [R:216]
              ▼
  後驗樣本 {mu, alpha, alpha2, varZ, psi}
              │  解 d(x) = target / 2
              ▼
  high / low salience 兩個刺激強度值
              │
              ▼
  拿去跑 Double Factorial Paradigm (DFP) / SFT 主實驗
```

反推的那一步在 `find_salience_polynomial()`
（`adaptiveSFT_functions.R:230` 起）。給定目標難度 `h_targ`，
解這個二次方程式

$$\alpha_2 x^2 + \alpha x - \tfrac{1}{2}\,\texttt{h\_targ} = 0$$

程式碼裡對應的是：

```r
h_targ.dist <- (-alpha/alpha2 - sqrt((alpha/alpha2)^2 + 2/alpha2 * h_targ)) / 2
```

這是在**每一個後驗樣本上**都解一次，最後取 `mean(..., na.rm = TRUE)`
——所以得到的是「後驗平均的 salience」，而不是「用後驗平均參數解出的 salience」。
兩者不一樣，前者比較保守，因為它有把參數的不確定性帶進去。

`if (post.diff$alpha2 < 0)` 這個條件只挑向下開口的拋物線，
也就是「效果會飽和」的那種形狀；`alpha2 > 0` 時整段不會執行，
`h_targ.dist` 會維持未定義狀態。這是實務上要留意的分支。

SFT 為什麼需要這一步：Double Factorial Paradigm 要求每個通道都有
**high / low salience 兩個水準**，而且這兩個水準在受試者之間必須是「等難度」的，
否則後續的 SIC / MIC 分析會被個別差異污染。這個 Stan 模型就是用來
**針對每個受試者個別校準**出他自己的 high / low 強度。

---

## 6. 命名陷阱：`varZ` 其實不是變異數

這是讀這份程式碼最容易踩到的地方。

Stan 的 `lognormal(mu, sigma)` 第二個參數依定義是**標準差**（在對數尺度上），
不是變異數。所以

```stan
lognormal_lpdf(rt[tr] - psi | z[1,tr], varZ)
                                        ^^^^ 這裡被當成 sigma 使用
```

儘管變數叫 `varZ`（variance of Z），它在似然裡扮演的是 **σ** 的角色。

會取這個名字，推測是因為先驗選了 `inv_gamma(1, .1)`
——inverse-gamma 是常態**變異數**的共軛先驗，命名沿用了那個直覺。

而 R 端的對照函式則是另一套慣例，它收 `sigmasq` 然後**開根號**：

```r
dlognormalrace <- function(x, m, psi, mu, sigmasq) {
  sigma <- sqrt(sigmasq)          # <- 這裡有開根號
```

**所以 Stan 的 `varZ` 與 R 的 `sigmasq` 不是同一個尺度。**
若要把 Stan 的後驗樣本餵給 R 的 `dlognormalrace` / `plognormalrace`
（`simulateLNRM_ogival.R:214` 附近就是這樣做的），必須先確認要不要平方。
Python 端的 `adaptive_sft2.py:23` 也沿用了 R 的 `sigmasq` 慣例。
移植到任何新框架時，**一律以 Stan 的用法為準：它是 sigma。**

---

## 7. 其他值得注意的地方

**`psi` 的硬邊界。** 當 `psi → minRT` 時，最快的那個試次會有 `rt - psi → 0`，
而對數常態密度在 0 處趨近於 0，log 密度趨近 $-\infty$。
所以似然本身就會把 `psi` 從上界推開——這是好事（自帶保護），
但也造成後驗在邊界附近極度不對稱，是 divergence 的常見來源。
`adaptive_sft2.py:160` 用 `psi: .1 * dat['minRT']` 當初始值，
就是在避開這個邊界。

**`z` 被存成 `transformed parameters`。** 這代表每一個後驗抽樣都會存下
`2 × N` 個數字。N 一大，輸出檔案會膨脹得很誇張。
R 端用 `pars = c("mu","alpha","alpha2","varZ","psi")` 過濾掉了
（`adaptiveSFT_functions.R:216`），但如果直接跑這個模型而沒指定 `pars`，
記憶體會是個問題。要更乾淨的話，可以把 `z` 移進 `model` block 當區域變數。

**可辨識性。** 承 §4，正確率只認得 $d(x)/\sigma$。
如果資料裡 intensity 的變化範圍不夠大、或正確率都貼在天花板/地板，
`alpha`、`alpha2`、`varZ` 三者會嚴重共線。二次項的先驗
`alpha2 ~ normal(0,1)` 收得比一次項緊，多少有正則化的作用，
但根本的解法是確保校準階段的 intensity 有涵蓋到中間的正確率區段。

**只有 `lnrm2.stan` 在版本庫裡。** R 與 Python 程式碼還引用了
`lnrm0.stan`、`lnrm1.stan`（一次項版本）、`lnrm2a.stan`（ogival / logistic 版本），
但這些檔案目前不在 repo 中。`lnrm2a.stan` 應該是把 $d(x)$ 換成
`L * inv_logit(slope * (x - midpoint))` 的變體，
從 `getPr_ogival()`（`adaptiveSFT_functions.R:169`）的內容可以反推出來。

**迴圈可以向量化。** `model` block 裡的 `for` 迴圈在 Stan 中是可以接受的，
但改成向量化形式（用 `segment` 或先切成 correct / incorrect 兩組）
通常能明顯加速，因為可以少建很多 autodiff 節點。

---

## 8. 對照速查表

| 概念 | 數學 | Stan | R (`adaptiveSFT_functions.R`) |
|---|---|---|---|
| 贏家密度 | $\log f_m(t-\psi)$ | `lognormal_lpdf` | `dlnorm(..., log=TRUE)` |
| 輸家存活 | $\log S_i(t-\psi)$ | `lognormal_lccdf` | `plnorm(..., lower=FALSE, log=TRUE)` |
| 尺度參數 | $\sigma$ | `varZ`（實為 SD） | `sqrt(sigmasq)` |
| 難度函數 | $d(x)=\alpha x + \alpha_2 x^2$ | `alpha*i + alpha2*i^2` | 同 |
| 正確率 | $\Phi(\sqrt{2}\,d/\sigma)$ | — | — |

---

## 9. 移植到 PyMC

> 本節所有結論都在本機環境實測驗證過：
> **PyMC 5.28.5 / PyTensor 2.38.3 / Python 3.11**。
> 標示「已驗證」的都有實際跑過的數值輸出佐證。

### 9.1 結論先講

**Python 生態圈目前沒有現成、解析形式的 Log-Normal Race Model 套件。**
最忠實的做法就是用 `pm.Potential` 自己把似然寫出來——這反而比套用現有套件
更貼近原 Stan 模型，也更輕量。第 9.5 節有完整可跑的移植碼。

### 9.2 API 對照表

| Stan | PyMC v5 |
|---|---|
| `target += ...` | `pm.Potential("name", expr)` |
| `lognormal_lpdf(y \| m, s)` | `pm.logp(pm.LogNormal.dist(mu=m, sigma=s), y)` |
| `lognormal_lccdf(y \| m, s)` | **沒有直接對應——見 §9.3，這是最大的坑** |
| `inv_gamma(1, .1)` | `pm.InverseGamma("varZ", alpha=1, beta=0.1)`（參數順序一致） |
| `real<lower=0,upper=minRT> psi` | `pm.Uniform("psi", lower=0, upper=minRT)` |
| `for` 迴圈 + `if/else` | `pt.switch(cond, a, b)` 向量化 |

注意 v3 → v5 的兩個變化：`pm.DensityDist` 已被 `pm.CustomDist` 取代；
`rv.logp(x)` 這種呼叫方式已移除，必須用模組層級的 `pm.logp(rv, x)`。

### 9.3 ⚠️ 最大的坑：`lccdf` 沒有安全的直接對應

PyMC 沒有 `logsf` / `logccdf` API。網路上（以及多數 porting guide）給的標準答案是

```python
lccdf = pm.math.log1mexp(pm.logcdf(rv, value))   # ❌ 會在關鍵區域壞掉
```

先說符號慣例（實測釐清，因為 PyTensor 與 PyMC 兩邊的 docstring 互相矛盾）：
在目前版本中 `pt.log1mexp(x)` 與 `pm.math.log1mexp(x)` **算的都是 `log(1-exp(x))`，
要求 `x ≤ 0`**（正數輸入回傳 `NaN`）。因為 `logcdf ≤ 0`，所以**不需要手動加負號**。
舊版曾經預設吃正數（`log(1-exp(-x))`），`negative_input` 參數就是那個歷史遺跡，
現已棄用，新程式碼不要傳它。

**但符號對了，這個寫法還是會壞。** 實測：

```
sigma     真值 lccdf        log1mexp 路線
0.6       -11.046460       -11.046459887   ✓
0.2       -81.307775       -81.307778005   ✓
0.05    -1250.565570             -inf      ✗
0.033   -2865.061096             -inf      ✗
0.01   -31149.836613             -inf      ✗
```

原因：當 σ 偏小時，輸家的 CDF 在 float64 下**飽和到恰好 1.0**，
`logcdf` 回傳 `-0.0`，於是 `log1mexp(-0.0) = log(0) = -inf`。
真值其實是 -1250、-2865 這種**有限**的大負數。
Stan 的 `lognormal_lccdf` 是直接算尾端機率的，沒有這個問題。

**後果是致命的**：NUTS 的 `jitter+adapt_diag` 初始化會隨機探到小 σ 的區域，
模型會在還沒開始採樣前就直接拋出

```
SamplingError: Initial evaluation of model at starting point failed!
{'varZ': -1.92, 'psi': -1.56, 'race': -inf}
```

**正確做法**——繞過 `logcdf`，改用常態存活函數。因為

$$S_{\text{lognormal}}(x) = P(X > x) = P\!\left(Z > \tfrac{\log x - \mu}{\sigma}\right)
= \Phi\!\left(-\tfrac{\log x - \mu}{\sigma}\right)$$

所以：

```python
def lognormal_lccdf(x, mu, sigma):
    """等價於 Stan 的 lognormal_lccdf，且在極端尾端數值穩定。"""
    return pm.logcdf(pm.Normal.dist(0.0, 1.0), -(pt.log(x) - mu) / sigma)
```

PyTensor 的常態 `logcdf` 內部用 `erfcx` 處理尾端，所以是穩定的。
已驗證：上表所有 σ（含 σ=0.01，lccdf = −31149）都與 `scipy.stats.lognorm.logsf`
**完全吻合**。

### 9.4 次要的坑：`CustomDist` 的 `observed` 不能依賴參數

如果你選 `pm.CustomDist` 而非 `pm.Potential`，這樣寫會直接報錯（已驗證）：

```python
pm.CustomDist("rt_obs", ..., observed=rt - psi)
# TypeError: Variables that depend on other nodes cannot be used for
#            observed data. The data variable was: Sub.0
```

`observed=` 只能是常數資料。`rt - psi` 的位移必須搬進 `logp` 函式**內部**，
把 `psi` 當成 `logp(value, ..., psi)` 的一個參數。
`pm.Potential` 沒有這個限制——這也是本題推薦用 `Potential` 的原因之一。
（代價是 `Potential` 不是隨機變數，做不了 posterior predictive check。）

### 9.5 完整移植碼（已驗證可跑）

完整可執行版本見同目錄的 **`lnrm2_pymc.py`**。核心如下：

```python
import numpy as np, pytensor
pytensor.config.floatX = "float64"          # 建議：避免 float32 精度問題
import pytensor.tensor as pt, pymc as pm

def lognormal_lccdf(x, mu, sigma):          # Stan lognormal_lccdf 的穩定替代
    return pm.logcdf(pm.Normal.dist(0.0, 1.0), -(pt.log(x) - mu) / sigma)

with pm.Model() as model:
    mu     = pm.Normal("mu", 0, 1)
    alpha  = pm.Normal("alpha", 0, 2)
    alpha2 = pm.Normal("alpha2", 0, 1)
    varZ   = pm.InverseGamma("varZ", alpha=1, beta=0.1)   # ← 當 sigma 用，見 §6
    psi    = pm.Uniform("psi", lower=0, upper=minRT)

    d      = alpha * intensity + alpha2 * intensity**2     # transformed parameters
    z1, z2 = mu - d, mu + d

    ok     = pt.constant(correct == 1)                     # 取代 Stan 的 if/else
    zw, zl = pt.switch(ok, z1, z2), pt.switch(ok, z2, z1)

    u = pt.clip(pt.constant(rt) - psi, 1e-12, np.inf)      # rt - psi，含數值防護

    lp    = pm.logp(pm.LogNormal.dist(mu=zw, sigma=varZ), u)   # 贏家密度
    lccdf = lognormal_lccdf(u, zl, varZ)                       # 輸家存活
    pm.Potential("race", pt.sum(lp + lccdf))                   # target +=

    idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                      initvals={"psi": 0.3 * minRT})
```

**驗證結果。** 逐點比對：在三組不同參數下，這個 PyMC 圖算出的每一筆試次
log-likelihood 與 `scipy` 參考實作（即 Stan 所計算的量）的最大絕對誤差為
`4e-8 ~ 2e-6`，相對誤差穩定在 ~3e-8——純粹是浮點捨入，結構完全正確。

用 N=500 的模擬資料（真值 `mu=1.5, alpha=0.8, alpha2=-0.15, sigma=0.6, psi=0.12`）
跑 4 chains × 1000 draws，15 秒完成：

```
         mean     sd  hdi_3%  hdi_97%  ess_bulk  r_hat
mu      1.387  0.039   1.314    1.462    2312.0   1.00
alpha   0.683  0.068   0.560    0.812    1599.0   1.00
alpha2 -0.115  0.025  -0.163   -0.068    1727.0   1.01
varZ    0.591  0.033   0.533    0.650    1455.0   1.00
psi     0.156  0.073   0.003    0.266    1354.0   1.00
divergences: 0
```

`varZ`（0.591 vs 0.6）與 `alpha2` 回收得很好，**但 `psi` 系統性高估（0.156 vs 0.12）
而 `mu` 系統性低估（1.387 vs 1.5）**——這正是 §4 預測的 `mu`/`psi` 後驗脊：
兩者都在拉 RT 的位置，N=500 還不足以把它們分開。這不是移植錯誤，
原始的 Stan 模型會有一模一樣的行為。實務上要嘛加大 N，要嘛給 `psi` 更強的先驗。

### 9.6 現有套件生態（2026-08 查證）

| 套件 | 狀態 | 能不能直接用 |
|---|---|---|
| **HSSM** | 活躍，v0.4.0（2026-07-15），需 Python ≥3.12。建立在 PyMC 5 + Bambi 上，定位是 HDDM 的接班人 | **不行。** 有 LBA / racing-diffusion / Poisson race，但**沒有 lognormal race**。而且那些模型用的是 LAN（神經網路近似似然），不是本模型的解析 lpdf × lccdf |
| **HDDM** | **已停擺。** 最後版本 1.0.1（2023-12-03），依賴早已淘汰的 PyMC 2 | 不建議新專案使用 |
| **ssm-simulators** | 活躍，v0.13.2（2026-07-16） | 是 HSSM 的模擬器後端，用來訓練 LAN，不提供解析似然 |
| `pm.LogNormal` / `pm.Wald` | PyMC 內建 | 只是單一分布，race 邏輯仍要自己組 |

> HSSM 的版本與日期我直接查了 PyPI 確認；
> 「內部沒有 LNR 設定檔、LBA 走 LAN 近似」這部分來自對其原始碼樹的檢視，
> 未在本機安裝驗證（HSSM 需要 Python ≥3.12，與本次驗證環境不符）。

### 9.7 版本與效能注意事項

**PyMC 6 需要 Python ≥ 3.12。** 這點很容易誤判：在 Python 3.11 下
`pip install pymc` 只會裝到 5.28.5，看起來像是「最新版」，
但 PyPI 上其實已有 6.3.1（2026-08-16）。`pip index versions pymc` 也只會列出 5.x。
若要用 PyMC 6，先確認 Python 版本。

PyMC 6 把 **nutpie**（Rust 實作的 NUTS）設為預設 sampler，不改模型碼就能受益。
在 PyMC 5 下可以自己指定：

```python
pm.sample(nuts_sampler="nutpie")    # 或 "numpyro"（需額外安裝）
```

其他建議：

* **全程向量化**。用 `pt.switch` 取代 Python 迴圈——逐筆建圖會產生巨大的計算圖，
  compile 時間就先炸掉了。
* **明確給 `initvals`**。`psi` 的邊界問題（§7）在 PyMC 一樣存在，
  `initvals={"psi": 0.3*minRT}` 比依賴預設 jitter 可靠得多。
* **設 `floatX="float64"`**。預設 float32 在這個模型的尾端計算上精度不夠。
* 把資料包成 `pm.Data` 方便之後用 `pm.set_data` 換資料重跑，
  這在 adaptive 迴圈裡每次加新試次都要重新擬合的情境特別有用。

---

## 10. 概念補充：race model、coactive、以及為什麼要同時擬合

### 10.1 Race model 是什麼

Race model（賽跑模型）屬於**序列取樣模型 (sequential sampling models)** 這個大家族。
它的假設是：每一個可能的反應各自配備一個獨立的證據累積器，
證據隨時間累積，**誰先撞到自己的門檻，誰就決定了受試者的反應**。

關鍵在於：實驗只觀察得到 $(T, m)$ ——完成時間與贏家身分——
而 $T$ 是所有潛在完成時間的**最小值**：

$$T = \min_i T_i, \qquad m = \arg\min_i T_i$$

這是一個**次序統計量 (order statistic)**。§3.5 那個 $f_m(t)\prod_{i\neq m}S_i(t)$
的似然形式並不是隨便湊的，它就是「最小值 + 誰是最小值」這個聯合事件的密度。
`lccdf`（log survival）出現在這裡，是因為「其他人都還沒完成」＝「其他人的完成時間都大於 $t$」。

**與 DDM（漂移擴散模型）的對比**，這是最容易混淆的地方：

| | DDM | Race model |
|---|---|---|
| 累積器數量 | **一個** | 每個反應一個 |
| 門檻 | 兩個（上下界） | 每個累積器各自一個 |
| 證據性質 | **相對**證據（A 減 B） | **絕對**證據（各自獨立累積） |
| 反應由什麼決定 | 撞到哪一個邊界 | 誰先撞到門檻 |

DDM 只能處理二選一；race model 天然可以擴展到 $n$ 個選項，加一個累積器就好。
這個家族的成員包括 LBA（Linear Ballistic Accumulator）、
racing diffusion、Poisson counter，以及本檔採用的 **LNR（log-normal race）**。
LNR 的優點是似然有**解析解**（就是那兩行 `lpdf` + `lccdf`），
不需要數值積分，也不需要像 LBA 那樣用神經網路近似（見 §9.6）。

### 10.2 跟 coactive 的關係：注意這裡有「兩層」race

這是讀這個 repo 最容易搞混的地方。**專案裡有兩個不同層次的競爭，不要混為一談：**

**層次一 —— 反應層次的 race（`lnrm2.stan` 所在的層次）。**
「正確」與「錯誤」兩個累積器在賽跑。這描述的是**單一個決策**如何產生 (RT, 正確與否)。
它的用途是**校準**，不涉及任何架構判斷。

**層次二 —— 通道層次的架構（SFT / DFP 所在的層次）。**
Double Factorial Paradigm 同時呈現**兩個通道**（例如兩個目標特徵），
問的是：這兩個通道如何組合？答案的候選就是
parallel（平行）、serial（序列）、**coactive（共同激發）**。

**Coactive 恰恰是「通道之間沒有 race」的那一種架構。** 對比如下：

* **Parallel race**：每個通道有自己的累積器、自己的門檻。通道各自跑完，
  再用 `min`（OR／自我終止）或 `max`（AND／窮盡）把**完成時間**組合起來。
  證據**沒有**匯流，匯流的是完成時間。
* **Coactive**：兩個通道的**證據**在任何門檻被跨越之前就先匯流進**同一個**累積器。
  全程只有一個累積器、一個門檻。通道之間根本沒有賽跑可言。

這個區別在 repo 裡有字面上的示範，`dfp_ddm()`（`adaptiveSFT_functions.R:114`）：

```r
if (architecture == "COA") {
    channel12 <- simdiffT(N, a, drift.1 + drift.2, sdv, ter)   # 一個過程，drift 相加
    rt <- channel12$rt
} else {
    channel1 <- simdiffT(N, a, drift.1, sdv, ter)              # 兩個獨立過程
    channel2 <- simdiffT(N, a, drift.2, sdv, ter)
    if (architecture == "PAR") {
      if (stopping.rule == "OR") {
        rt <- pmin(channel1$rt, channel2$rt)                   # 賽跑 = 取 min
```

**COA 分支是 `drift.1 + drift.2` 餵進單一個 `simdiffT`（證據層次匯流）；
PAR 分支是兩次 `simdiffT` 再 `pmin`（完成時間層次競爭）。**
一行程式碼就把「coactive vs. race」的理論差異講完了。

**怎麼分辨？** 靠 SFT 的兩個診斷量：

| 架構 | MIC | SIC 形狀 |
|---|---|---|
| Parallel OR（自我終止） | > 0 | 全正 |
| Parallel AND（窮盡） | < 0 | 全負 |
| Serial OR | = 0 | 恆為 0 |
| Serial AND | = 0 | 先負後正，正負面積相等 |
| **Coactive** | **> 0** | **先一個小的負向下凹，再轉成大的正向** |

注意 **coactive 與 parallel-OR 的 MIC 同號**，光看 MIC 分不出來，
必須靠 SIC 早期那個小小的負向下凹才能認出 coactive。
這個下凹很淺，**對雜訊極度敏感**——這件事直接決定了下一節的重要性。

**那 `lnrm2.stan` 在這裡扮演什麼角色？**
它**不判斷架構**，它是讓架構判斷得以成立的前置作業。
SIC/MIC 的整套推導建立在 **selective influence（選擇性影響）** 這個前提上：
改變通道 1 的顯著度，必須**只**影響通道 1 的處理時間分佈。而且 high/low 兩個水準
必須真的在每個受試者身上都造成可測量、方向一致的差異。

如果 salience 選得不好——兩個水準都在天花板（太簡單、都是滿分），
或 low 掉到地板（純猜測，累積器模型根本不適用）——SIC 就只剩雜訊，
那個用來辨認 coactive 的淺下凹會被完全淹沒。更糟的是，如果用**固定**的物理強度值，
不同受試者的敏感度不同，同一個數值對每個人的難度不一樣，
群體分析等於把不同架構的人混在一起平均。

**所以 `lnrm2.stan` 存在的理由，就是逐人把 high/low 校準到「等難度」。**

### 10.3 為什麼要同時擬合 RT 與正確率

以下四點各自獨立，強度由弱到強。

**（a）正確率單獨無法辨識模型。** 由 §4，$P(\text{correct}) = \Phi(\sqrt{2}\,d/\sigma)$
——正確率**只**認得 $d/\sigma$ 這個比值，對 `mu`、`psi` 完全沒有約束力。
實測四組參數（模擬 20 萬試次）：

```
   mu      d  sigma   psi | P(correct)  mean RT   sd RT
  1.5    0.6    0.6  0.12 |     0.9216    2.908   1.636
  1.5    0.2    0.2  0.12 |     0.9216    3.816   0.696
  0.3    0.6    0.6  0.12 |     0.9216    0.960   0.493
  1.5    0.6    0.6  0.35 |     0.9216    3.138   1.636
```

**正確率完全相同（0.9216），平均 RT 卻從 0.96 到 3.82、標準差從 0.49 到 1.64。**
只看正確率，這四組參數是完全無法區分的。

實務後果很直接：如果只用正確率擬合，`alpha`、`alpha2`、`varZ` 的**絕對尺度**
根本不可辨識（只有比值可辨識）。實測直接這樣跑，NUTS 會產生 **543 個 divergences**，
必須改成用比值參數化才會收斂。

**（b）錯誤反應的 RT 帶著被丟掉的資訊。** 這個模型對「錯誤反應比正確反應慢還是快」
做了明確的預測。實測（40 萬試次）：

```
  d=0.2:  P(corr)=0.682   correct RT=3.469   error RT=3.926   → 錯誤較慢
  d=0.6:  P(corr)=0.921   correct RT=2.801   error RT=4.226   → 錯誤較慢
  d=1.0:  P(corr)=0.991   correct RT=2.061   error RT=4.387   → 錯誤較慢
```

模型預測**錯誤系統性地更慢，而且差距隨難度下降（d 上升）而擴大**。
這是一個可以拿去對照真實資料的檢驗點；只擬合正確率就等於自願放棄它。

**（c）校準精度確實會提升，但幅度是溫和的。**
用同一批模擬資料，比較「同時擬合」與「只用正確率」對校準目標
$x^*$（達到 85% 正確率所需的刺激強度）的後驗不確定性
（兩者都已確認 0 divergences）：

```
    N   方法                    x* 均值   後驗SD   HDI寬
  200   A 同時擬合 RT+ACC        0.7185   0.1026  0.3719
  200   B 只用正確率              0.6983   0.1253  0.4622   → SD 大 1.22 倍
  500   A 同時擬合 RT+ACC        0.7448   0.0667  0.2467
  500   B 只用正確率              0.7429   0.0882  0.3264   → SD 大 1.32 倍
```

同時擬合讓後驗 SD 縮小約 1.2–1.3 倍，換算成試次數大約等於
**多做 1.5–1.7 倍的試次**。在校準階段試次數就是最稀缺的資源，這個效益是真的，
但要誠實說：**它並不是壓倒性的**。真正決定性的理由是下面這一點。

**（d）最強的理由：下游的 SFT 分析吃的就是 RT 分佈。**
SIC 是定義在**存活函數**上的量——它整個是 RT 分佈的泛函，
而且如 §10.2 所述，辨認 coactive 靠的是 SIC 早期那個很淺的負向下凹。

如果校準模型根本不模 RT，你就**無從保證**選出來的 high/low 會在 RT 分佈上
產生 SIC 所需要的分離。你可能挑到一組「正確率漂亮地分開、RT 幾乎重疊」的水準，
那麼正確率上看起來校準得很好，SIC 卻什麼都測不到。

換句話說：**校準模型應該和後續要分析的那個過程屬於同一個模型族。**
用 race model 校準、再用 SFT 分析 RT 分佈，兩者的假設是一致的；
用純正確率的 psychometric function 校準、再去分析 RT，中間有一段是斷開的。
這才是 `lnrm2.stan` 要把 `lpdf` 和 `lccdf` 兩項都寫進似然的根本理由。
