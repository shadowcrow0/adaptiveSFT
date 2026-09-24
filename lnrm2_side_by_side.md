# `lnrm2.stan` 逐項對照 + 不相容項目

三段：

```
   Part 1  逐項對照     lnrm2.stan 每一個 block  →  數學式 / Stan code / 中文 / English
   Part 2  不相容項目   哪個語言（哪個版本）做了什麼，所以該怎麼做（全部是提案，未套用）
   Part 3  總表         A–G 一行一列
```

**命名約定（全文適用）。** Stan 變數 `varZ` 被放在 `lognormal_lpdf` / `lognormal_lccdf`
第二個分布參數的位置（`lnrm2.stan:38-39`）。Stan 的 `lognormal(m, s)` 第二個參數定義為
**標準差**。因此本文公式一律把 `varZ` 寫成 s，不開根號、不平方。

**Naming convention (applies throughout).** The Stan variable `varZ` sits in the second
distribution-parameter slot of `lognormal_lpdf` / `lognormal_lccdf` (`lnrm2.stan:38-39`).
Stan's `lognormal(m, s)` defines that slot as the **standard deviation**. So every formula
below writes `varZ` as s and never takes its square root or squares it.

其他符號 / other symbols:

| 符號 | 程式碼 | 意義 |
|---|---|---|
| n | `tr` | 試次編號 trial index, n = 1 … N |
| xₙ | `intensity[n]` | 刺激強度 |
| cₙ | `correct[n]` | 0 / 1 |
| tₙ | `rt[n]` | 反應時間 |
| t_min | `minRT` | min of all tₙ |
| μ, α, α₂, s, ψ | `mu`, `alpha`, `alpha2`, `varZ`, `psi` | 參數 |
| yₙ | `rt[tr] - psi` | tₙ − ψ |

「程式碼字面寫了什麼」放在 **Stan code** 小節；**數學公式** 與解釋是從程式碼翻譯 / 推導出來的。

---

## Part 1 — 逐項對照

### 1. `data` block

#### 數學公式

```
      N ∈ ℤ,  N ≥ 1
      xₙ ∈ ℝ                       n = 1 … N
      cₙ ∈ {0, 1}
      t_min ≥ 0
      tₙ ≥ 0
```

#### Stan code

`lnrm2.stan:1-7`

```stan
data { 
   int<lower=1> N;
   real intensity[N];
   int<lower=0,upper=1> correct[N];
   real<lower=0> minRT;
   real<lower=0> rt[N];
}
```

#### 中文解釋

五個輸入：試次數 `N`、強度 `intensity`、對錯 `correct`、最小反應時間 `minRT`、反應時間 `rt`。
`<lower=...>` 在 `data` block 裡只是輸入檢查，資料不合範圍時 Stan 直接報錯。
這五項由 `adaptiveSFT_functions.R:168-172` 的 `dataframe2stan()` 打包，`minRT = min(rt)`。

#### English explanation

Five inputs: the trial count `N`, stimulus `intensity`, the 0/1 `correct` flag, the
smallest reaction time `minRT`, and the reaction times `rt`. Bounds in the `data` block are
input checks only; out-of-range data make Stan error out. The list is assembled by
`dataframe2stan()` at `adaptiveSFT_functions.R:168-172`, with `minRT = min(rt)`.

---

### 2. `transformed data` block

#### 數學公式

```
      xₙ²          n = 1 … N
```

#### Stan code

`lnrm2.stan:8-11`

```stan
transformed data {
   real square_intensity[N];
   square_intensity = square(intensity);
}
```

#### 中文解釋

宣告長度 N 的 `square_intensity`，內容是 `intensity` 逐元素平方。
這個 block 在資料載入後只執行一次，不隨抽樣重算。

#### English explanation

Declares `square_intensity` of length N and fills it with the element-wise square of
`intensity`. This block runs once after the data are loaded, not on every draw.

---

### 3. `parameters` block

#### 數學公式

```
      α  ∈ ℝ
      α₂ ∈ ℝ
      μ  ∈ ℝ
      s  ∈ (0, ∞)                  ← 程式碼裡叫 varZ
      ψ  ∈ (0, t_min)
```

#### Stan code

`lnrm2.stan:12-18`

```stan
parameters{
   real alpha;
   real alpha2;
   real mu;
   real<lower=0> varZ;
   real<lower=0,upper=minRT> psi;
}
```

#### 中文解釋

五個待估參數。`alpha`、`alpha2`、`mu` 無限制；`varZ` 下界 0；
`psi` 的上界是 `data` block 傳進來的 `minRT`，保證 `rt[tr] - psi > 0`。
有界參數會讓 Stan 在內部做變數變換並自動加 Jacobian（見第 8 項）。

#### English explanation

Five free parameters. `alpha`, `alpha2`, `mu` are unbounded; `varZ` has lower bound 0;
`psi` is capped at `minRT` from the `data` block so that `rt[tr] - psi > 0` always holds.
Bounded parameters are transformed internally by Stan, which adds a Jacobian term
automatically (see item 8).

---

### 4. `transformed parameters` block

#### 數學公式

```
      dₙ   =  α·xₙ  +  α₂·xₙ²

      z₁ₙ  =  μ − dₙ
      z₂ₙ  =  μ + dₙ
```

```
            z₁ₙ = μ − dₙ
                  ^
                  |  dₙ
      ────────────+────────────  μ
                  |  dₙ
                  v
            z₂ₙ = μ + dₙ
```

#### Stan code

`lnrm2.stan:19-28`

```stan
transformed parameters {
   real z[2,N];

   for (tr in 1:N) { 
      z[1,tr] = mu - alpha * intensity[tr] - alpha2 * square_intensity[tr];
      z[2,tr] = mu + alpha * intensity[tr] + alpha2 * square_intensity[tr];
   }


}
```

#### 中文解釋

對每個試次算兩個數 `z[1,tr]`、`z[2,tr]`，以 `mu` 為中心被 dₙ 對稱推開。
dₙ 是 `intensity` 的二次式（`alpha` 線性項、`alpha2` 二次項）。
`z` 宣告在這個 block 代表每個抽樣都會存 2×N 個值；
呼叫端 `adaptiveSFT_functions.R:216-217` 用 `pars=` 把它排除在輸出外。

#### English explanation

For each trial two numbers `z[1,tr]`, `z[2,tr]` are computed, pushed symmetrically away
from `mu` by dₙ. dₙ is a quadratic in `intensity` (`alpha` linear, `alpha2` quadratic).
Because `z` lives in `transformed parameters`, 2×N values would be stored per draw; the
caller at `adaptiveSFT_functions.R:216-217` excludes it via `pars=`.

---

### 5. Priors

#### 數學公式

常態密度 / normal density:

```
                            1                (    (v − m)²  )
      p_N(v; m, σ)  =  ───────────  ·  exp   ( − ─────────  )
                         σ·√(2π)              (     2 σ²     )
```

逆伽瑪密度，代入 a = 1、b = 0.1（Γ(1) = 1）/ inverse-gamma at a = 1, b = 0.1:

```
                          b^a
      p_IG(v; a, b)  =  ────────  ·  v^(−a−1)  ·  exp(−b / v)
                         Γ(a)

      p(s)  =  0.1 · s^(−2) · exp(−0.1 / s)            s > 0
```

四個先驗 / the four priors:

```
      p(s)   =  0.1 · s^(−2) · exp(−0.1 / s)
      p(μ)   =  p_N(μ;  0, 1)
      p(α)   =  p_N(α;  0, 2)
      p(α₂)  =  p_N(α₂; 0, 1)

      p(ψ)   ∝  1        0 < ψ < t_min      (沒有 ~ 敘述，由第 17 行的界限決定)
```

#### Stan code

`lnrm2.stan:30-33`

```stan
   varZ ~ inv_gamma(1,.1);
   mu ~ normal(0,1);
   alpha ~ normal(0,2);
   alpha2 ~ normal(0,1);
```

#### 中文解釋

四個 `~` 敘述給 `varZ`、`mu`、`alpha`、`alpha2`。`psi` 沒有 `~`，
只有 `lnrm2.stan:17` 的 `<lower=0,upper=minRT>`；`lnrm2.stan:35` 的註解說它是
「正實數上的 improper flat」，但界限實際產生的是 (0, t_min) 上的均勻分布。
`inv_gamma` 放在 `varZ` 上，而 `varZ` 在似然裡是標準差（見開頭命名約定）。

#### English explanation

Four `~` statements cover `varZ`, `mu`, `alpha`, `alpha2`. `psi` has none; its prior is
whatever the bounds at `lnrm2.stan:17` imply. The comment at `lnrm2.stan:35` calls it
"improper flat on positive reals", but the bounds actually give a uniform on (0, t_min).
The `inv_gamma` prior sits on `varZ`, which the likelihood uses as a standard deviation
(see the naming convention at the top).

---

### 6. Likelihood, correct branch

#### 數學公式

對數常態密度與存活函數 / lognormal pdf and survival function:

```
                         1                    (     (ln y − m)²  )
      f(y; m, s)  =  ───────────────  ·  exp  ( − ─────────────  )      y > 0
                      y · s · √(2π)            (        2 s²      )

                                         (     ln y − m  )
      S(y; m, s)  =  1 − F(y; m, s)  = Φ ( − ──────────  )
                                         (        s       )

      Φ = 標準常態 CDF / standard normal CDF
```

cₙ = 1 時，第 n 試次貢獻 / per-trial term when cₙ = 1, with yₙ = tₙ − ψ:

```
      Lₙ  =  ln f(yₙ; z₁ₙ, s)  +  ln S(yₙ; z₂ₙ, s)

      exp(Lₙ)  =  f(yₙ; z₁ₙ, s)  ×  S(yₙ; z₂ₙ, s)
```

一般形式（wₙ = 贏家、lₙ = 輸家；此分支 wₙ = z₁ₙ、lₙ = z₂ₙ）:

```
      exp(Lₙ)  =  f(yₙ; wₙ, s)  ×  S(yₙ; lₙ, s)
```

#### Stan code

`lnrm2.stan:37-39`

```stan
      if ( correct[tr] ) {
         target += lognormal_lpdf(rt[tr] - psi | z[1,tr], varZ);
         target += lognormal_lccdf(rt[tr] - psi | z[2,tr], varZ);
```

#### 中文解釋

答對的試次：`rt[tr] - psi` 在 `z[1,tr]` 那條對數常態上取密度，
在 `z[2,tr]` 那條上取互補 CDF（存活函數），兩個對數相加進 `target`。
密度 × 存活函數的組合是 log-normal race 的標準寫法（「z[1] 先完成、z[2] 還沒完成」），
但檔案裡沒有任何取 min 的運算，這個讀法是推導，不是字面。

#### English explanation

On a correct trial, `rt[tr] - psi` is evaluated as a density under the lognormal with
location `z[1,tr]` and as a complementary CDF (survival) under the one with `z[2,tr]`; the
two logs are added to `target`. Density times survival is the standard log-normal race
form ("z[1] finished, z[2] had not yet"), but the file contains no min operation, so that
reading is a derivation, not the literal code.

---

### 7. Likelihood, error branch

#### 數學公式

cₙ = 0：z₁ₙ 與 z₂ₙ 對調 / cₙ = 0: swap z₁ₙ and z₂ₙ:

```
      Lₙ  =  ln f(yₙ; z₂ₙ, s)  +  ln S(yₙ; z₁ₙ, s)
```

兩分支合併 / both branches in one line:

```
              ⎧ z₁ₙ   若 cₙ = 1                  ⎧ z₂ₙ   若 cₙ = 1
      wₙ  =   ⎨                          lₙ  =   ⎨
              ⎩ z₂ₙ   若 cₙ = 0                  ⎩ z₁ₙ   若 cₙ = 0

      Lₙ  =  ln f(yₙ; wₙ, s)  +  ln S(yₙ; lₙ, s)
```

```
   cₙ = 1                cₙ = 0
   ──────                ──────
   f( · ; z₁ₙ )          f( · ; z₂ₙ )      ← lognormal_lpdf
        ×                     ×
   S( · ; z₂ₙ )          S( · ; z₁ₙ )      ← lognormal_lccdf
```

#### Stan code

`lnrm2.stan:41-44`

```stan
      else { 
         target += lognormal_lpdf(rt[tr] - psi | z[2,tr], varZ);
         target += lognormal_lccdf(rt[tr] - psi | z[1,tr], varZ);
      }
```

#### 中文解釋

答錯的試次結構完全相同，只把 `z[1,tr]` 與 `z[2,tr]` 互換：
密度用 `z[2,tr]`，存活函數用 `z[1,tr]`。三個引數（`rt[tr] - psi`、`z[?,tr]`、`varZ`）
在兩個分支中位置一致。

#### English explanation

The error branch has the identical structure with `z[1,tr]` and `z[2,tr]` swapped: the
density uses `z[2,tr]`, the survival function `z[1,tr]`. The three arguments
(`rt[tr] - psi`, `z[?,tr]`, `varZ`) occupy the same slots in both branches.

---

### 8. Assembled target (whole `model` block)

#### 數學公式

```
      ln p(μ, α, α₂, s, ψ | data)

          =   ln p(μ) + ln p(α) + ln p(α₂) + ln p(s)

                   N
              +   ∑   [ ln f(tₙ − ψ; wₙ, s)  +  ln S(tₙ − ψ; lₙ, s) ]
                  n=1

              +   J                            (+ 與參數無關的常數)

      其中  dₙ = α·xₙ + α₂·xₙ²,   z₁ₙ = μ − dₙ,   z₂ₙ = μ + dₙ
            wₙ = z₁ₙ 若 cₙ = 1 否則 z₂ₙ ;   lₙ = z₂ₙ 若 cₙ = 1 否則 z₁ₙ
```

J：Stan 對 `varZ` 的 `<lower=0>` 與 `psi` 的 `<lower=0,upper=minRT>` 自動加的
Jacobian，**不在程式碼裡**。

```
   priors (:30-33)  ──┐
                      ├──►  target  ◄── J (Stan 自動加，看不到)
   ∑ₙ Lₙ (:36-45)  ──┘
```

#### Stan code

`lnrm2.stan:29-48`

```stan
model {
   varZ ~ inv_gamma(1,.1);
   mu ~ normal(0,1);
   alpha ~ normal(0,2);
   alpha2 ~ normal(0,1);

   // psi has improper flat prior on positive reals
   for ( tr in 1:N) { 
      if ( correct[tr] ) {
         target += lognormal_lpdf(rt[tr] - psi | z[1,tr], varZ);
         target += lognormal_lccdf(rt[tr] - psi | z[2,tr], varZ);
      } 
      else { 
         target += lognormal_lpdf(rt[tr] - psi | z[2,tr], varZ);
         target += lognormal_lccdf(rt[tr] - psi | z[1,tr], varZ);
      }
   }
   
   
}
```

#### 中文解釋

`~` 敘述與 `target +=` 都累加到同一個對數後驗密度。整體 = 四個先驗 + N 個試次的
密度 + 存活函數 + Jacobian J。J 是 Stan 語言對有界參數的行為，程式碼裡看不到；
換到 PyMC 時它同樣自動加（`lnrm2_pymc_gaps.md` 末表）。

#### English explanation

Both `~` statements and `target +=` accumulate into one log posterior density. The total is
the four priors plus, per trial, log density plus log survival, plus the Jacobian J. J is
Stan's built-in behaviour for bounded parameters and is invisible in the code; PyMC adds it
automatically too (last table of `lnrm2_pymc_gaps.md`).

---

### 9. Downstream R solve that consumes the fit

#### 數學公式

反解：給定目標 targ，找 x 使 2·dₙ = targ / solve for the intensity x reaching a target:

```
      α₂·x² + α·x  =  targ / 2

      x² + (α/α₂)·x − targ/(2α₂)  =  0

             − α/α₂  −  √( (α/α₂)²  +  2·targ/α₂ )
      x  =  ─────────────────────────────────────────        (程式碼取「−」根)
                                2
```

判別式為負的條件 / discriminant negative when:

```
      (α/α₂)² + 2·targ/α₂ < 0   ⇔   targ  >  −α²/(2α₂)  =  2·d_max ,

                    α²
      d_max  =  − ──────         (α₂ < 0 時拋物線 dₙ(x) 的頂點高度)
                   4α₂
```

此時 `sqrt()` 回 NaN，該抽樣在 `:279` 被 `na.rm=TRUE` 丟掉。

#### Stan code

（R，不是 Stan）`adaptiveSFT_functions.R:228-233`

```r
  if (post.diff$alpha2 <0) {
  l_targ.dist <- with(post.diff,  
        (-alpha/alpha2 - sqrt( (alpha/alpha2)^2 + 2 / alpha2 * l_targ)) / 2)
  h_targ.dist <- with(post.diff,  
        (-alpha/alpha2 - sqrt( (alpha/alpha2)^2 + 2 / alpha2 * h_targ)) / 2)
  }
```

`adaptiveSFT_functions.R:279`

```r
  return(list(high=mean(h_targ.dist, na.rm=TRUE), 
```

#### 中文解釋

`extract()`（`:219`）拿到 `mu, alpha, alpha2, psi, varZ` 的後驗抽樣向量後，
對每個抽樣解二次式，得到達成 `l_targ` / `h_targ` 所需的強度，再取平均回傳 `high` / `low`。
整段包在 `if (post.diff$alpha2 < 0)` 裡，`alpha2` 是向量——這是 Part 2 A 的問題。

#### English explanation

After `extract()` (`:219`) returns posterior draw vectors for `mu, alpha, alpha2, psi,
varZ`, the quadratic is solved per draw for the intensity that reaches `l_targ` / `h_targ`,
and the means are returned as `high` / `low`. The whole block is wrapped in
`if (post.diff$alpha2 < 0)` where `alpha2` is a vector — that is problem A in Part 2.

---

## Part 2 — 不相容項目：哪個語言做了什麼，所以該怎麼做

**以下全部是提案（PROPOSAL），尚未套用到任何檔案。**
**Everything below is a PROPOSAL; nothing has been applied.**

### A. R `if()` 收到長度 > 1 的條件

#### 問題 / Problem

`find_salience_polynomial()` 在 R 4.3.3 上一呼叫就報錯 `the condition has length > 1`，
整個多項式反解路徑不能用。

Calling `find_salience_polynomial()` on R 4.3.3 fails immediately with
`the condition has length > 1`; the whole polynomial-solve path is unusable.

#### 哪個語言做了什麼 / Which language changed what

R 4.2.0：`if()` 的條件長度 > 1 時，從**警告**（只用第一個元素）改為**錯誤**。
`post.diff$alpha2` 是後驗抽樣向量，長度等於抽樣數，所以每次都觸發。

R 4.2.0 turned a length > 1 condition in `if()` from a **warning** (first element used)
into an **error**. `post.diff$alpha2` is a vector of posterior draws, so it triggers every
time.

#### 本 repo 的位置 / Where in this repo

`adaptiveSFT_functions.R:228`

```r
  if (post.diff$alpha2 <0) {
```

#### 所以該怎麼做 / What to do

三個候選語意，**結果會不同，由 owner 決定**：

Three candidate semantics; **they give different estimates, the owner must choose**:

```r
# PROPOSAL (a): 全部抽樣都 < 0 才解
if (all(post.diff$alpha2 < 0)) { ... }

# PROPOSAL (b): 逐抽樣，不符的給 NA（之後被 :279 的 na.rm 丟掉）
h_targ.dist <- with(post.diff, ifelse(alpha2 < 0,
      (-alpha/alpha2 - sqrt((alpha/alpha2)^2 + 2/alpha2 * h_targ)) / 2, NA))

# PROPOSAL (c): 用後驗平均判斷
if (mean(post.diff$alpha2) < 0) { ... }
```

(a) 最保守，只要有一個抽樣 ≥ 0 就整段不執行；(b) 保留最多資訊但改變 `high`/`low` 的定義；
(c) 最接近原作者「只用第一個元素」時代的行為。

(a) is the most conservative: one draw ≥ 0 skips the block entirely; (b) keeps the most
draws but changes what `high`/`low` mean; (c) is closest to the pre-4.2 "first element"
behaviour in spirit.

---

### B. Stan 舊陣列宣告語法

#### 問題 / Problem

`real intensity[N];` 這種寫法在現行 Stan 編不過。**本機沒裝 rstan，未實際編譯**，
依據是 Stan 語言版本紀錄。

`real intensity[N];` no longer compiles on current Stan. **Not compiled locally (no rstan
installed here)**; the basis is the Stan language changelog.

#### 哪個語言做了什麼 / Which language changed what

Stan 2.26 標記棄用、Stan 2.33 移除舊的 `type name[dims]` 陣列語法；
現行語法是 `array[dims] type name`。

Stan deprecated the old `type name[dims]` array syntax in 2.26 and removed it in 2.33;
the current syntax is `array[dims] type name`.

#### 本 repo 的位置 / Where in this repo

`lnrm2.stan:3`
```stan
   real intensity[N];
```
`lnrm2.stan:4`
```stan
   int<lower=0,upper=1> correct[N];
```
`lnrm2.stan:6`
```stan
   real<lower=0> rt[N];
```
`lnrm2.stan:9`
```stan
   real square_intensity[N];
```
`lnrm2.stan:20`
```stan
   real z[2,N];
```

#### 所以該怎麼做 / What to do

五行改寫，其餘不動。找回的 `lnrm0/1/2a.stan`（項目 G）比照辦理。

Rewrite these five lines, nothing else. Apply the same to `lnrm0/1/2a.stan` once found (item G).

```stan
// PROPOSAL
   array[N] real intensity;                 // :3
   array[N] int<lower=0,upper=1> correct;   // :4
   array[N] real<lower=0> rt;               // :6
   array[N] real square_intensity;          // :9
   array[2, N] real z;                      // :20
```

驗收：`rstan::stan_model("lnrm2.stan")` 或 `cmdstanr::cmdstan_model()` 編譯通過。

Acceptance: `rstan::stan_model("lnrm2.stan")` or `cmdstanr::cmdstan_model()` compiles.

---

### C. `na.rm=TRUE` 靜默丟掉無解的抽樣

#### 問題 / Problem

這不是版本變化，是 R 語意。當 targ 超過可達最大值 2·d_max（d_max = −α²/(4α₂)，
見 Part 1 第 9 項），`sqrt()` 收到負數回 NaN；`mean(..., na.rm=TRUE)` 把這些抽樣丟掉，
回傳的 `high`/`low` 沒有任何提示說丟了多少。

Not a version change but R semantics. When targ exceeds the reachable maximum 2·d_max
(d_max = −α²/(4α₂), Part 1 item 9), `sqrt()` of a negative discriminant gives NaN;
`mean(..., na.rm=TRUE)` drops those draws and the returned `high`/`low` carry no sign of
how many were dropped.

#### 哪個語言做了什麼 / Which language changed what

R 的設計規則：`sqrt(負數)` 回 `NaN` 並給警告；`mean(x, na.rm=TRUE)` 對 `NA` 與 `NaN`
一視同仁地剔除後平均，不回報剔除數量。

R's design rule: `sqrt(negative)` returns `NaN` with a warning; `mean(x, na.rm=TRUE)`
removes both `NA` and `NaN` before averaging and does not report how many were removed.

#### 本 repo 的位置 / Where in this repo

`adaptiveSFT_functions.R:279-280`

```r
  return(list(high=mean(h_targ.dist, na.rm=TRUE), 
               low=mean(l_targ.dist, na.rm=TRUE), fit=fitModel))
```

#### 所以該怎麼做 / What to do

回傳值多兩個欄位：被丟掉的比例。是否進一步改成警告或報錯由 owner 決定。

Add two fields with the dropped fraction. Whether to escalate to a warning or error is the
owner's call.

```r
# PROPOSAL
  return(list(high = mean(h_targ.dist, na.rm=TRUE),
              low  = mean(l_targ.dist, na.rm=TRUE),
              high_nan_frac = mean(is.na(h_targ.dist)),
              low_nan_frac  = mean(is.na(l_targ.dist)),
              fit = fitModel))
```

---

### D. `varZ` 尺度：Stan 是標準差，R helper 收變異數

#### 問題 / Problem

命名問題，不是語言版本變化。Stan 把 `varZ` 當標準差用；R 的 `dlognormalrace()`
收 `sigmasq` 再開根號。把 Stan 的 `varZ` 抽樣原封不動傳進去會差一個平方。

A naming mismatch, not a language change. Stan uses `varZ` as the sd; R's
`dlognormalrace()` takes `sigmasq` and square-roots it. Feeding Stan's `varZ` draws in
unchanged is off by a square.

#### 哪個語言做了什麼 / Which language changed what

Stan 的設計規則：`lognormal_lpdf(y | mu, sigma)` 與 `lognormal_lccdf` 第二個參數是
**標準差**。R 的設計規則：`dlnorm(x, meanlog, sdlog)` 第三個參數同樣是標準差，
所以 helper 才需要 `sqrt(sigmasq)` 把變異數轉回去。兩邊各自一致，錯在中間的命名。

Stan's rule: the second argument of `lognormal_lpdf(y | mu, sigma)` and `lognormal_lccdf`
is the **standard deviation**. R's rule: `dlnorm(x, meanlog, sdlog)` also takes the sd,
which is why the helper does `sqrt(sigmasq)`. Each side is self-consistent; the mismatch is
in the naming between them.

#### 本 repo 的位置 / Where in this repo

`lnrm2.stan:38`

```stan
         target += lognormal_lpdf(rt[tr] - psi | z[1,tr], varZ);
```

`adaptiveSFT_functions.R:61-62`

```r
dlognormalrace <- function(x, m, psi, mu, sigmasq) { 
  sigma <- sqrt(sigmasq)
```

#### 所以該怎麼做 / What to do

先決定一個尺度，再把另一邊改名或轉換，並在兩處都寫註解。兩個方向：

Pick one scale, rename or convert the other side, and comment both places. Two directions:

```r
# PROPOSAL (i): 以 Stan 現況（標準差）為準，R helper 改收 sigma
dlognormalrace <- function(x, m, psi, mu, sigma) {   # sigma = Stan varZ, sd scale
  g <- dlnorm(x-psi, mu[m], sigma[m], log=TRUE)
  ...

# PROPOSAL (ii): 以 R 命名（變異數）為準，Stan 改成 sqrt(varZ)
#   target += lognormal_lpdf(rt[tr] - psi | z[1,tr], sqrt(varZ));
#   （inv_gamma 先驗此時才真的落在變異數上）
```

(ii) 會改變模型與後驗，需重跑；(i) 不動模型。`simulateLNRM_ogival.R:214` 附近就有這種
跨語言傳遞，改動時要一起檢查。

(ii) changes the model and posterior and requires refitting; (i) leaves the model alone.
The cross-language hand-off near `simulateLNRM_ogival.R:214` must be checked with whichever
change is made.

---

### E. Python driver `adaptive_sft2.py`

#### 問題 / Problem

三件事：`^` 不是次方；`allchannels` 未定義；PyStan 2 API 已不存在。
這條路目前沒人用（`plan_r_modernization.md` P6），嚴重度低。

Three things: `^` is not a power operator; `allchannels` is undefined; the PyStan 2 API no
longer exists. This path is currently unused (`plan_r_modernization.md` P6), so severity is
low.

#### 哪個語言做了什麼 / Which language changed what

- `^`：Python 從來就是**位元 XOR**，次方是 `**`。純 bug，不是版本變化。
- `allchannels`：Python 在執行到 `:27` 時會拋 `NameError`。純 bug。
- `pystan`：PyStan 3（2021）整個移除了 `pystan.StanModel` 這套 API，改成
  `stan.build(program_code, data=...)`；目前維護中的現代路線是 `cmdstanpy`。

- `^`: in Python it has always been **bitwise XOR**; exponentiation is `**`. Plain bug.
- `allchannels`: Python raises `NameError` when `:27` runs. Plain bug.
- `pystan`: PyStan 3 (2021) removed the `pystan.StanModel` API entirely in favour of
  `stan.build(program_code, data=...)`; the maintained modern route is `cmdstanpy`.

#### 本 repo 的位置 / Where in this repo

`adaptive_sft2.py:2`
```python
import pystan
```
`adaptive_sft2.py:12`
```python
        + posterior_samples['intensity']^2 * posterior_samples['alpha2'])
```
`adaptive_sft2.py:27`
```python
    for i in range(allchannels) :
```
`adaptive_sft2.py:167`
```python
            sm = pystan.StanModel(file="lnrm2a.stan")
```

#### 所以該怎麼做 / What to do

```python
# PROPOSAL
# :12
        + posterior_samples['intensity']**2 * posterior_samples['alpha2'])

# :27  （比照 R 版 adaptiveSFT_functions.R:66 的 allchannels <- 1:length(mu)）
    for i in range(len(mu)) :

# :2 / :167  改用 cmdstanpy
from cmdstanpy import CmdStanModel
sm = CmdStanModel(stan_file="lnrm2a.stan")
fit = sm.sample(data=standat)
```

`:167` 引用的 `lnrm2a.stan` 本身不在 repo（項目 G），所以即使改完 API 也跑不起來。

`lnrm2a.stan` referenced at `:167` is missing from the repo (item G), so the script still
cannot run after the API fix.

---

### F. 轉成 PyMC 時的兩個設計規則

摘自 `lnrm2_pymc_gaps.md` §1–§2（PyMC 5.28.5 / PyTensor 2.38.3 / Python 3.11 實測），不重新推導。

Summarised from `lnrm2_pymc_gaps.md` §1–§2 (verified on PyMC 5.28.5 / PyTensor 2.38.3 /
Python 3.11); not re-derived here.

#### 問題 / Problem

(i) PyMC 沒有 `lognormal_lccdf`。(ii) `pm.CustomDist(observed=...)` 不接受依賴參數的
`observed`。這兩條各自對應 `lnrm2.stan:39,43` 與 `:38-43`，直譯會在抽樣前就失敗。

(i) PyMC has no `lognormal_lccdf`. (ii) `pm.CustomDist(observed=...)` rejects an
`observed` that depends on a parameter. They map to `lnrm2.stan:39,43` and `:38-43`
respectively; a literal port fails before sampling.

#### 哪個語言做了什麼 / Which language changed what

(i) PyMC/PyTensor 設計規則：只有 `pm.logp` 與 `pm.logcdf`，沒有 `logsf`。
常見的組合 `pm.math.log1mexp(pm.logcdf(...))` 在 s 小時 CDF 在 float64 飽和成 1.0，
`log1mexp(−0.0)` 回 −inf。實測 y = 20、m = 0.5、s = 0.05：真值 −1250.57，組合路線 −inf。
NUTS 的 `jitter+adapt_diag` 初始化會隨機落到小 s，模型在抽樣開始前就死。

```
   小 s  ──►  F(y) 在 float64 進位成 1.0
         ──►  logcdf 回 −0.0
         ──►  log1mexp(−0.0) = ln(0) = −inf      ← Stan 直接算尾端，不會這樣
```

(ii) PyMC 設計規則：`observed=` 必須是常數。`observed=rt - psi` 拋
`TypeError: Variables that depend on other nodes cannot be used for observed data`。
Stan 的 `target +=` 沒有這個限制。

(i) PyMC/PyTensor rule: only `pm.logp` and `pm.logcdf` exist, no `logsf`. The usual
composition `pm.math.log1mexp(pm.logcdf(...))` returns −inf once the CDF saturates to 1.0
in float64 at small s. Verified at y = 20, m = 0.5, s = 0.05: true value −1250.57, composed
route −inf. NUTS `jitter+adapt_diag` initialisation lands on small s at random, so the
model fails before sampling.

(ii) PyMC rule: `observed=` must be constant. `observed=rt - psi` raises
`TypeError: Variables that depend on other nodes cannot be used for observed data`.
Stan's `target +=` has no such restriction.

#### 本 repo 的位置 / Where in this repo

`lnrm2.stan:38-39`（`:42-43` 為鏡像）

```stan
         target += lognormal_lpdf(rt[tr] - psi | z[1,tr], varZ);
         target += lognormal_lccdf(rt[tr] - psi | z[2,tr], varZ);
```

#### 所以該怎麼做 / What to do

```python
# PROPOSAL (i): 走標準常態 CDF，不走 log1mexp
def lognormal_lccdf(y, m, s):
    # ln S(y; m, s) = ln Φ( −(ln y − m) / s )
    return pm.logcdf(pm.Normal.dist(0.0, 1.0), -(pt.log(y) - m) / s)

# PROPOSAL (ii-A): pm.Potential，最接近 target +=
u = pt.clip(pt.constant(rt) - psi, 1e-12, np.inf)
pm.Potential("race", pt.sum(logp_winner + lognormal_lccdf(u, z_lose, varZ)))

# PROPOSAL (ii-B): CustomDist，把位移搬進 logp
def race_logp(value, z_win, z_lose, s, psi):
    u = pt.clip(value - psi, 1e-12, np.inf)
    ...
pm.CustomDist("rt_obs", z_win, z_lose, varZ, psi, logp=race_logp, observed=rt)
```

(i) 實測與 `scipy.stats.lognorm.logsf` 在 s = 0.01 仍一致。(ii-A) 失去 posterior
predictive；(ii-B) 保留。另需 `pytensor.config.floatX = "float64"`（`lnrm2_pymc_gaps.md` §6）。

(i) matches `scipy.stats.lognorm.logsf` down to s = 0.01. (ii-A) loses posterior
predictive sampling; (ii-B) keeps it. Also force `pytensor.config.floatX = "float64"`
(`lnrm2_pymc_gaps.md` §6).

---

### G. 被引用但不在版本庫的 Stan 檔

#### 問題 / Problem

三個 `.stan` 檔被 R / Python 引用，`ls` 確認不在 repo。`lnrm2a.stan` 是 ogival 主路徑
與 Python driver 用的模型，缺了整條路徑就斷。

Three `.stan` files are referenced from R / Python but `ls` confirms they are absent.
`lnrm2a.stan` is the model used by the ogival main path and the Python driver, so that
whole path is broken without it.

#### 哪個語言做了什麼 / Which language changed what

不是語言問題，是版本庫內容缺失。

Not a language issue; the repository is missing content.

#### 本 repo 的位置 / Where in this repo

| 檔名 | 引用位置 | 引用行 |
|---|---|---|
| `lnrm0.stan` | `simulateLNRM_ogival.R:157` | `fitDiff0 <- stan(file="lnrm0.stan", data=stanData0)` |
| `lnrm1.stan` | `adaptiveSFT_functions.R:210` | `fitModel <- stan(file="lnrm1.stan", data=standatDiff,` |
| `lnrm2a.stan` | `adaptiveSFT_functions.R:185` | `fitModel <- stan(file="lnrm2a.stan", data=standatDiff,` |
| `lnrm2a.stan` | `simulateLNRM_ogival.R:62` | `fitDiff <- stan(file="lnrm2a.stan", data=standatDiff,` |
| `lnrm2a.stan` | `adaptive_sft2.py:167` | `sm = pystan.StanModel(file="lnrm2a.stan")` |

#### 所以該怎麼做 / What to do

PROPOSAL：問原作者 / 翻上游 `jhoupt/adaptiveSFT` 歷史；找不到就從 `getPr_ogival()`
反推重寫 `lnrm2a.stan`，或放棄 ogival 路徑。owner 決定（`plan_r_modernization.md` §6）。
找回後同樣要做項目 B 的語法更新。

PROPOSAL: ask the original author / search upstream `jhoupt/adaptiveSFT` history; if not
found, rewrite `lnrm2a.stan` from `getPr_ogival()` or drop the ogival path. Owner's decision
(`plan_r_modernization.md` §6). Recovered files also need the item B syntax update.

---

## Part 3 — 總表

| # | 項目 | 語言 | 版本/規則 | 位置 | 處理方式 |
|---|---|---|---|---|---|
| A | `if()` 條件長度 > 1 變成 error | R | R 4.2.0：warning → error | `adaptiveSFT_functions.R:228` | 改 `all()` / 逐抽樣 `ifelse` / `mean()`，owner 選語意 |
| B | 舊陣列語法 `real x[N]` | Stan | 2.26 棄用、2.33 移除 | `lnrm2.stan:3,4,6,9,20` | 改 `array[N] real x;`、`array[2, N] real z;`（本機未編譯） |
| C | `na.rm=TRUE` 靜默丟 NaN 抽樣 | R | 語意規則，非版本變化 | `adaptiveSFT_functions.R:279` | 回傳 `*_nan_frac`，是否升級為警告由 owner 決定 |
| D | `varZ` 是 sd，R helper 收 `sigmasq` | Stan ↔ R | 命名不一致，非版本變化 | `lnrm2.stan:38` vs `adaptiveSFT_functions.R:61-62` | 選定一個尺度、改名並加註解 |
| E | `^` 是 XOR；`allchannels` 未定義；PyStan 2 API | Python | `^` 語言規則；PyStan 3 移除 `StanModel` | `adaptive_sft2.py:12, 27, 2, 167` | `**`、`len(mu)`、改 `cmdstanpy` |
| F | 無 `lognormal_lccdf`；`observed` 不能依賴參數 | PyMC | PyMC 5.28 / PyTensor 2.38 設計規則 | `lnrm2.stan:38-43` 的對應 | `pm.logcdf(Normal(0,1), −(ln y − m)/s)`；`pm.Potential` 或位移進 `logp` |
| G | 三個 `.stan` 檔不在 repo | — | 檔案缺失 | `lnrm0/1/2a.stan` 的五個引用行 | 找回或重寫，owner 決定 |
