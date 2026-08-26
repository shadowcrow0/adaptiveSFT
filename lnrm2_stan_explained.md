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

## 10. 補充：從程式碼推出來的幾件事

> 這一節**不是**程式碼的字面內容，是從程式碼推導出來的。
> 前面九節才是「程式寫了什麼」。

### 10.1 為什麼 `z[1,tr]` 變小代表反應變快

`z` 不是時間，它是藏在分布參數位置的一個數。所以「`z` 變小」跟
「反應變快」之間需要一座橋。

橋是這樣搭的。`lognormal_lpdf(y | z, varZ)` 描述的分布，可以寫成

```
完成時間 = psi + exp( 一個隨機數 )
                        這個隨機數的平均值就是 z
```

`exp()` 的性質是輸入越大、輸出越大。所以

```
z 小  ->  隨機數平均小  ->  exp 出來小  ->  完成時間短
```

到這裡是「平均而言變快」。但實際上成立的是更強的版本，因為

```
exp(z + 晃動)  =  exp(z) * exp(晃動)
```

`z` 跑到括號外面變成一個純粹的**倍率**，而「晃動」那部分跟 `z` 完全無關。
所以改變 `z` 不是把分布往左推，是把整條分布**等比例縮放**。

實際數字（`varZ` = 0.6，看 `exp(隨機數)` 這一項）：

| | 快的（前 10%） | 中位數 | 慢的（後 10%） |
|---|---|---|---|
| z = 1.5 | 2.08 | 4.47 | 9.66 |
| z = 1.0 | 1.26 | 2.71 | 5.84 |
| 比值 | 0.61 | 0.61 | 0.60 |

三個位置都乘上同一個倍率。不是只有平均值變小。

### 10.2 中間那一項等於 0 時，正確率恰好是 50%

當 `alpha * intensity[tr] + alpha2 * square_intensity[tr]` 等於 0 時，
`transformed parameters` 那兩行會得到

```
z[1,tr] = mu
z[2,tr] = mu
```

兩列完全相同，代表兩個分布的參數一模一樣。

兩個條件完全相同的東西比「誰先發生」，機率必然各半——否則你無法說明
憑什麼是這一個而不是那一個。

這個結論**只用到「兩者相同」**，連對數常態這個假設都不需要，
換成任何連續分布都成立。

### 10.3 正確率的公式

因為 `z` 的兩列共用同一個 `varZ`，正確率可以直接算出來，不必模擬。

把中間那一項取名叫 `d`：

```
d = alpha * intensity + alpha2 * square_intensity
```

則

```
正確率 = Phi( sqrt(2) * d / varZ )
```

`Phi` 是常態分布的累積機率函數。它唯一重要的性質是**輸入越大、輸出越大**。
所以整個公式只要看兩件事：

- `d` 在**分子**：`d` 越大，正確率越高
- `varZ` 在**分母**：`varZ` 越大（雜訊越大），正確率越低

實際數字（`varZ` = 0.6）：

| d | 正確率 |
|---|---|
| 0 | 50% |
| 0.3 | 76% |
| 1.0 | 99% |

這個公式同時說明了：**正確率只取決於 `d / varZ` 這個比值**，
跟 `mu` 和 `psi` 完全無關。`mu` 和 `psi` 只影響 RT 落在哪裡，不影響答對率。

### 10.4 二次項的作用

```
d = alpha * intensity + alpha2 * square_intensity
```

因為有 `square_intensity` 這一項，`d` 是 `intensity` 的二次式，
畫出來是拋物線。`alpha2` 等於 0 時退化成直線。

這一條來自 `transformed data` block 的 `square(intensity)`
與 `transformed parameters` block 的那兩行，沒有其他推導成分。

### 10.5 「觀察到的 RT 是兩者中先完成的那一個」——這件事沒有寫在程式碼裡

**先講清楚：`lnrm2.stan` 全檔沒有任何取最小值的運算。**
唯一出現 "min" 字樣的是第 5 行的 `minRT` 與第 17 行 `psi` 的上界，
那是另一回事。

「先完成的那個決定了 RT」這個結構，是**隱含在下面這兩行的組合裡**
（`lnrm2.stan:38-39`）：

```stan
target += lognormal_lpdf (rt[tr] - psi | z[1,tr], varZ);
target += lognormal_lccdf(rt[tr] - psi | z[2,tr], varZ);
```

一個是機率密度、一個是互補累積分布函數（也就是「還沒發生的機率」），
兩者相加（在 log 尺度上相加等於原尺度相乘）。
`else` 分支（`lnrm2.stan:42-43`）把兩列對調。

R 端 `dlognormalrace()`（`adaptiveSFT_functions.R:61-72`）是同一個結構：

```r
g <- dlnorm(x-psi, mu[m], sigma[m], log=TRUE)                    # 密度
G <- G + plnorm(x-psi, mu[i], sigma[i], lower=FALSE, log=TRUE)   # 還沒發生
rval <- exp(g + G)
```

同樣沒有取最小值的運算。

### 10.6 由此推出的一個結果：RT 的方向不是自動的

如果接受 10.5 的讀法（觀察到的 RT 是兩者中先完成的那一個），那麼當
`alpha * intensity + alpha2 * square_intensity` 這一項變大時：

- `z[1,tr]` 變小，對應「答對」的完成時間變短
- `z[2,tr]` 變大，對應「答錯」的完成時間變長

**這兩件事對觀察到的 RT 影響方向相反。** 用四個試次示範
（假設那一項變大時，答對那邊快 1、答錯那邊慢 1）：

改變前：

| 試次 | 答對 | 答錯 | 誰先 | RT |
|---|---|---|---|---|
| 1 | 3.0 | 4.0 | 對 | 3.0 |
| 2 | 5.0 | 2.0 | 錯 | 2.0 |
| 3 | 2.0 | 6.0 | 對 | 2.0 |
| 4 | 4.0 | 3.0 | 錯 | 3.0 |

改變後：

| 試次 | 答對 | 答錯 | 誰先 | RT | 變化 |
|---|---|---|---|---|---|
| 1 | 2.0 | 5.0 | 對 | 2.0 | 變快 |
| 2 | 4.0 | 3.0 | 錯 | 3.0 | **變慢** |
| 3 | 1.0 | 7.0 | 對 | 1.0 | 變快 |
| 4 | 3.0 | 4.0 | 對 | 3.0 | 不變 |

**試次 2 是關鍵**：那一次是「答錯」先完成，而它變慢了，所以該次 RT 從
2.0 變成 3.0。

所以：

- 「答對」先完成的試次，RT 變快
- 「答錯」先完成的試次，RT **變慢**

哪一邊主導，取決於兩種試次各佔多少，**必須實際計算，不能由程式碼直接看出**。
以模擬資料計算的結果（`varZ` = 0.6）：

| 那一項的值 | 答錯的比例 | 全部 RT 中位數 | 答對的 RT | 答錯的 RT |
|---|---|---|---|---|
| 0.0 | 50% | 3.353 | 3.350 | 3.357 |
| 0.3 | 24% | 3.111 | 2.946 | 3.639 |
| 1.0 | 1% | 1.770 | 1.760 | 4.076 |
| 1.5 | 0.02% | 1.119 | 1.119 | 4.144 |

答錯的 RT 確實一路上升（3.357 到 4.144），就是試次 2 那個效果。
但這種試次的比例掉到 0.02%，所以「全部 RT」仍然單調下降。

結論：**「刺激越明顯反應越快」對整體與答對的試次成立，對答錯的試次相反。**
