# `lnrm2.stan` 逐行說明

這份文件只描述**程式碼實際寫了什麼**，以及**哪一段程式碼呼叫它**。
不包含推導、模擬或建議。

---

## 1. 這個檔案在專案裡的位置

`lnrm2.stan` 不是獨立腳本。呼叫它的是 `adaptiveSFT_functions.R` 的
`find_salience_polynomial()`，在 `polynomial_order == 2` 的分支
（`adaptiveSFT_functions.R:216`）：

```r
fitModel <- stan(file = "lnrm2.stan", data = standatDiff,
                 pars = c("mu", "alpha", "alpha2", "varZ", "psi"))
```

`pars` 指定只保留這五個參數的抽樣結果。

輸入資料由同檔案的 `dataframe2stan()` 打包（`adaptiveSFT_functions.R:168`）：

```r
dataframe2stan <- function(dat) {
   standat <- with(dat, list(N = dim(dat)[1], intensity = intensity,
                             correct = correct, minRT = min(rt), rt = rt))
   return(standat)
}
```

---

## 2. `data` block

```stan
data {
   int<lower=1> N;
   real intensity[N];
   int<lower=0,upper=1> correct[N];
   real<lower=0> minRT;
   real<lower=0> rt[N];
}
```

| 變數 | 型別 | 對應到 `dataframe2stan` 的哪一項 |
|---|---|---|
| `N` | 整數，下界 1 | `dim(dat)[1]`，資料列數 |
| `intensity` | 長度 N 的實數陣列 | `dat$intensity` |
| `correct` | 長度 N 的整數陣列，限 0 或 1 | `dat$correct` |
| `minRT` | 實數，下界 0 | `min(rt)` |
| `rt` | 長度 N 的實數陣列，下界 0 | `dat$rt` |

`<lower=...>` / `<upper=...>` 在 `data` block 中的作用是輸入檢查：
傳入不符合範圍的值時 Stan 會報錯。

---

## 3. `transformed data` block

```stan
transformed data {
   real square_intensity[N];
   square_intensity = square(intensity);
}
```

宣告一個長度 N 的實數陣列 `square_intensity`，內容是 `intensity` 逐元素平方。

`transformed data` block 在資料載入後執行一次。

---

## 4. `parameters` block

```stan
parameters{
   real alpha;
   real alpha2;
   real mu;
   real<lower=0> varZ;
   real<lower=0,upper=minRT> psi;
}
```

| 參數 | 宣告的範圍 |
|---|---|
| `alpha` | 無限制 |
| `alpha2` | 無限制 |
| `mu` | 無限制 |
| `varZ` | 下界 0 |
| `psi` | 下界 0、上界 `minRT` |

`psi` 的上界用的是 `data` block 傳進來的 `minRT`。

---

## 5. `transformed parameters` block

```stan
transformed parameters {
   real z[2,N];

   for (tr in 1:N) {
      z[1,tr] = mu - alpha * intensity[tr] - alpha2 * square_intensity[tr];
      z[2,tr] = mu + alpha * intensity[tr] + alpha2 * square_intensity[tr];
   }
}
```

宣告一個 2 x N 的實數陣列 `z`，然後對每個試次 `tr` 填入兩個值：

```
z[1,tr] = mu - (alpha * intensity[tr] + alpha2 * square_intensity[tr])
z[2,tr] = mu + (alpha * intensity[tr] + alpha2 * square_intensity[tr])
```

兩列的差別只在中間那項是減還是加。

`z` 宣告在 `transformed parameters` block，代表它會跟著每一個抽樣被儲存
（每個抽樣 2 x N 個值）。上面 §1 的 `pars` 參數把它排除在輸出之外。

---

## 6. `model` block

### 6.1 先驗

```stan
varZ ~ inv_gamma(1,.1);
mu ~ normal(0,1);
alpha ~ normal(0,2);
alpha2 ~ normal(0,1);
```

四個 `~` 敘述，分別給 `varZ`、`mu`、`alpha`、`alpha2`。
`psi` 沒有 `~` 敘述，只有第 17 行宣告時的範圍限制。

原始碼在這裡有一行註解：

```stan
// psi has improper flat prior on positive reals
```

### 6.2 似然

```stan
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
```

逐試次迴圈，依 `correct[tr]` 分兩個分支。兩個分支的結構相同，
只有 `z[1,tr]` 和 `z[2,tr]` 互換：

| `correct[tr]` | `lognormal_lpdf` 用哪一列 | `lognormal_lccdf` 用哪一列 |
|---|---|---|
| 1（真） | `z[1,tr]` | `z[2,tr]` |
| 0（假） | `z[2,tr]` | `z[1,tr]` |

三個引數在兩個分支中都一樣：

- 被評估的值：`rt[tr] - psi`
- 第一個分布參數：`z[?,tr]`
- 第二個分布參數：`varZ`

依 Stan 函式手冊：

- `lognormal_lpdf(y | mu, sigma)` 回傳對數常態分布的**對數機率密度**
- `lognormal_lccdf(y | mu, sigma)` 回傳**對數互補累積分布函數**，
  即 `log(1 - CDF(y))`
- 兩者的第二個分布參數 `sigma` 定義為**標準差**

`target +=` 把值累加到 Stan 的對數後驗密度上。

---

## 7. 呼叫端的 R 程式碼

`find_salience_polynomial()` 取出後驗抽樣後（`adaptiveSFT_functions.R:219`）：

```r
post.diff <- extract(fitModel, c("mu","alpha", "alpha2", "psi", "varZ"))
```

接著解一個二次式（`adaptiveSFT_functions.R:228`）：

```r
if (post.diff$alpha2 <0) {
l_targ.dist <- with(post.diff,
      (-alpha/alpha2 - sqrt( (alpha/alpha2)^2 + 2 / alpha2 * l_targ)) / 2)
h_targ.dist <- with(post.diff,
      (-alpha/alpha2 - sqrt( (alpha/alpha2)^2 + 2 / alpha2 * h_targ)) / 2)
}
```

原始碼上方的註解寫出了它在解什麼：

```r
# If order=2
# a2 i^2 +  a1 i - 1/2 h_targ
# i^2 +  a1/a2 i - 1/(2a2)h_targ
# -a1/a2 + sqrt((a1/a2)^2+2/a2 * h_targ) / 2
```

最後回傳（`adaptiveSFT_functions.R:279`）：

```r
return(list(high=mean(h_targ.dist, na.rm=TRUE),
             low=mean(l_targ.dist, na.rm=TRUE), fit=fitModel))
```

兩點與程式碼字面有關：

- 整段解二次式的程式碼包在 `if (post.diff$alpha2 < 0)` 裡。
  `alpha2 >= 0` 時這個區塊不執行，`h_targ.dist` 與 `l_targ.dist` 不會被賦值。
- `mean()` 帶了 `na.rm = TRUE`，所以 `sqrt()` 產生 NA 的抽樣會被排除在平均之外。

---

## 8. 程式碼層面的事實記錄

**`varZ` 傳入的位置。** `varZ` 被放在 `lognormal_lpdf` 與 `lognormal_lccdf`
的第二個分布參數位置。依 Stan 函式手冊，該位置的定義是標準差。

R 端 `dlognormalrace()`（`adaptiveSFT_functions.R:61`）收的參數名為 `sigmasq`，
函式內第一行是：

```r
sigma <- sqrt(sigmasq)
```

Python 端 `adaptive_sft2.py:23` 的 `lognormalrace_pdf()` 同樣收 `sigmasq`
並在內部開根號。

**陣列宣告語法。** `real intensity[N]` 這種寫法在 Stan 2.26 標記為棄用、
2.33 移除。現行語法為 `array[N] real intensity`。

**被引用但不存在於版本庫的檔案。** 以下 Stan 檔在 R 與 Python 程式碼中被引用，
但不在此 repo 中：

| 檔名 | 引用位置 |
|---|---|
| `lnrm0.stan` | `simulateLNRM_ogival.R:157` |
| `lnrm1.stan` | `adaptiveSFT_functions.R:210` |
| `lnrm2a.stan` | `adaptiveSFT_functions.R:185`、`simulateLNRM_ogival.R:62`、`adaptive_sft2.py:167` |

---

## 9. 函式對照

| 概念 | Stan | R (`adaptiveSFT_functions.R:61`) |
|---|---|---|
| 對數機率密度 | `lognormal_lpdf` | `dlnorm(..., log=TRUE)` |
| 對數互補 CDF | `lognormal_lccdf` | `plnorm(..., lower=FALSE, log=TRUE)` |
| 第二個分布參數 | `varZ` | `sqrt(sigmasq)` |

---

## 10. 推導與數學形式：見另外兩個檔案

本文件只描述程式碼字面寫了什麼。從程式碼推導出來的結果，以及把程式碼
翻成數學式，分別放在：

| 檔案 | 內容 |
|---|---|
| `lnrm2_math.md` | 把 `lnrm2.stan` 每一行翻成數學公式 |
| `lnrm2_derivations.md` | 從程式碼推導出來的結果，每一節都先貼出涉及的程式碼 |

`lnrm2_derivations.md` 涵蓋：

```
  A. 為什麼 z[1,tr] 變小代表反應變快
  B. d_n = 0 時正確率恰好 50%
  C. 正確率的公式
  D. 二次項讓曲線可以彎
  E. 「先完成的那個決定 RT」沒有寫在程式碼裡
  F. RT 的方向不是自動的
```

## 11. 待確認的問題

以下都是從程式碼看得到、但**光看程式碼無法確定原意**的事情。
每一條都附上確切位置，可以直接拿去問熟悉這個模型的人。

### 11.1 第 38-39 行的組合是不是在表達「賽跑」

`lnrm2.stan:38-39`

```stan
target += lognormal_lpdf (rt[tr] - psi | z[1,tr], varZ);
target += lognormal_lccdf(rt[tr] - psi | z[2,tr], varZ);
```

**問題：這個「機率密度 + 互補累積分布函數」的組合，是不是在表達
「觀察到的 RT 是兩個累積器中先完成的那一個」？**

背景：全檔沒有任何取最小值的運算（見 10.5），所以這個讀法無法從程式碼直接確認。
這類模型在文獻上的名稱是 log-normal race model。

如果答案是「是」，接著問：

**當 `alpha * intensity + alpha2 * square_intensity` 上升時，
`z[1,tr]` 變小、`z[2,tr]` 變大——這兩個變化對觀察到的 RT
的影響方向是不是相反的？**

（10.6 的推導假設答案是「是」。若不是，10.6 整段要重寫。）

### 11.2 `varZ` 到底該是標準差還是變異數

三處寫法互相不一致：

| 位置 | 寫法 |
|---|---|
| `lnrm2.stan:38-39` | `varZ` 放在 `lognormal_lpdf` 第二個分布參數位置（Stan 手冊定義為標準差） |
| `lnrm2.stan:30` | `varZ ~ inv_gamma(1,.1)`（inverse-gamma 通常用作變異數的先驗） |
| `adaptiveSFT_functions.R:61-62` | 參數名為 `sigmasq`，函式內 `sigma <- sqrt(sigmasq)` |
| `adaptive_sft2.py:23` | 同樣收 `sigmasq` 並在內部開根號 |

**問題：Stan 的 `varZ` 與 R 的 `sigmasq` 是不是同一個量？
如果要把 Stan 的後驗抽樣餵給 R 的 `dlognormalrace()`，要不要先平方？**

（`simulateLNRM_ogival.R:214` 附近就有這種跨語言傳遞。）

### 11.3 `psi` 的先驗

`lnrm2.stan:35` 的註解：

```stan
// psi has improper flat prior on positive reals
```

但 `lnrm2.stan:17` 的宣告是：

```stan
real<lower=0,upper=minRT> psi;
```

**問題：註解與宣告不一致（註解說是正實數上的 improper flat，
宣告卻有上下界）。哪一個是原意？**

### 11.4 `alpha2 >= 0` 時反解不會執行

`adaptiveSFT_functions.R:228`

```r
if (post.diff$alpha2 <0) {
```

整段解二次式的程式碼包在這個條件裡。`alpha2 >= 0` 的抽樣不會進入，
`h_targ.dist` 與 `l_targ.dist` 不會被賦值。

**問題：這是刻意只處理 `alpha2 < 0` 的情況，還是遺漏？
如果後驗中有相當比例的抽樣是 `alpha2 >= 0`，應該怎麼處理？**

### 11.5 `na.rm = TRUE` 會靜默排除無解的抽樣

`adaptiveSFT_functions.R:279`

```r
return(list(high=mean(h_targ.dist, na.rm=TRUE),
             low=mean(l_targ.dist, na.rm=TRUE), fit=fitModel))
```

當 `sqrt()` 內的值為負時會產生 NA，`na.rm = TRUE` 會把這些抽樣排除在平均之外，
而且不會有任何提示。

**問題：這是刻意的嗎？回報的 salience 是否應該同時附上
「有多少比例的抽樣被排除」？**

### 11.6 被引用但不在版本庫中的檔案

| 檔名 | 引用位置 |
|---|---|
| `lnrm0.stan` | `simulateLNRM_ogival.R:157` |
| `lnrm1.stan` | `adaptiveSFT_functions.R:210` |
| `lnrm2a.stan` | `adaptiveSFT_functions.R:185`、`simulateLNRM_ogival.R:62`、`adaptive_sft2.py:167` |

**問題：這三個檔案在哪裡？`lnrm2a.stan` 尤其重要——
`find_salience_ogival()` 與 `adaptive_sft2.py` 的主要路徑都用它，
不是用 `lnrm2.stan`。**

### 11.7 陣列宣告語法

`lnrm2.stan:3,4,6,9` 用的是 `real intensity[N]` 這種寫法。
該語法在 Stan 2.26 標記棄用、2.33 移除，現行語法為 `array[N] real intensity`。

**問題：目前是用哪個版本的 Stan 跑這份程式碼？**
