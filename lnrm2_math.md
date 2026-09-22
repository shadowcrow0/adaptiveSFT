# `lnrm2.stan` 的數學形式

把 `lnrm2.stan` 每一行翻成數學式。左邊是程式碼，右邊是它對應的公式。

本文所有密度公式都與 `scipy.stats` 比對過數值，確認一致。
符號一律用 ASCII，不用希臘字母。

---

## 0. 符號約定

```
  n           試次編號，n = 1 .. N
  x_n         第 n 試次的刺激強度      <- intensity[n]
  c_n         第 n 試次是否答對 (0/1)  <- correct[n]
  t_n         第 n 試次的反應時間      <- rt[n]
  t_min       所有 t_n 的最小值        <- minRT
```

參數（五個，都是要估的）：

```
  mu, alpha, alpha2, varZ, psi
```

**注意 `varZ` 這個名字。** 它在公式裡扮演的是**標準差**（不是變異數），
因為 Stan 的 `lognormal(m, s)` 第二個參數定義就是標準差，
而程式碼把 `varZ` 放在那個位置。本文一律照此處理。

---

## 1. 兩個用到的分布函數

### 對數常態機率密度 `lognormal_lpdf(y | m, s)`

```
                    1                (  (log(y) - m)^2  )
  f(y; m, s) = ------------- * exp ( - --------------- )        y > 0
                y*s*sqrt(2*pi)      (       2*s^2       )
```

Stan 的 `lognormal_lpdf` 回傳的是 `log f(y; m, s)`。

### 對數常態互補累積分布 `lognormal_lccdf(y | m, s)`

先定義標準常態的累積分布函數 `Phi`：

```
                 1      /  u
  Phi(u) = ----------- |    exp(-w^2 / 2) dw
            sqrt(2*pi)  / -inf
```

則對數常態的「大於 y 的機率」（存活函數）為：

```
                                    (   log(y) - m  )
  S(y; m, s) = 1 - F(y; m, s) = Phi ( - ----------- )
                                    (        s      )
```

Stan 的 `lognormal_lccdf` 回傳的是 `log S(y; m, s)`。

---

## 2. `transformed data` block

```stan
square_intensity = square(intensity);
```

```
  x_n^2        n = 1 .. N
```

---

## 3. `transformed parameters` block

```stan
z[1,tr] = mu - alpha * intensity[tr] - alpha2 * square_intensity[tr];
z[2,tr] = mu + alpha * intensity[tr] + alpha2 * square_intensity[tr];
```

先把重複出現的那一項取名為 `d_n`：

```
  d_n = alpha * x_n + alpha2 * x_n^2
```

則兩行變成：

```
  z_1n = mu - d_n
  z_2n = mu + d_n
```

圖示：

```
            z_1n = mu - d_n
                  ^
                  |
      ------------+------------  mu
                  |
                  v
            z_2n = mu + d_n

      兩者以 mu 為中心，被 d_n 對稱地推開
```

---

## 4. `model` block：先驗

```stan
varZ  ~ inv_gamma(1,.1);
mu    ~ normal(0,1);
alpha ~ normal(0,2);
alpha2~ normal(0,1);
```

常態密度：

```
                     1            (  (v - m)^2 )
  p_normal(v; m, s) = ----------- * exp( - --------- )
                     s*sqrt(2*pi)      (   2*s^2   )
```

逆伽瑪密度，代入 a = 1、b = 0.1（因為 Gamma(1) = 1）：

```
                        b^a                            0.1        ( -0.1 )
  p_invgamma(v; a, b) = ------ * v^(-a-1) * exp(-b/v) = ----- * exp( ---- )
                        Gamma(a)                         v^2        (  v  )
```

所以四個先驗是：

```
  p(varZ)   = 0.1 * varZ^(-2) * exp(-0.1 / varZ)        varZ > 0
  p(mu)     = p_normal(mu;     0, 1)
  p(alpha)  = p_normal(alpha;  0, 2)
  p(alpha2) = p_normal(alpha2; 0, 1)
```

`psi` 沒有 `~` 敘述，只有宣告 `real<lower=0,upper=minRT> psi`。

---

## 5. `model` block：似然

```stan
if ( correct[tr] ) {
   target += lognormal_lpdf (rt[tr] - psi | z[1,tr], varZ);
   target += lognormal_lccdf(rt[tr] - psi | z[2,tr], varZ);
} else {
   target += lognormal_lpdf (rt[tr] - psi | z[2,tr], varZ);
   target += lognormal_lccdf(rt[tr] - psi | z[1,tr], varZ);
}
```

令 `y_n = t_n - psi`（位移後的反應時間）。則第 n 試次貢獻的對數似然是：

```
  c_n = 1 (答對):   L_n = log f(y_n; z_1n, varZ) + log S(y_n; z_2n, varZ)

  c_n = 0 (答錯):   L_n = log f(y_n; z_2n, varZ) + log S(y_n; z_1n, varZ)
```

兩式只差在 `z_1n` 和 `z_2n` 對調。合併寫成一式：

```
  設  w_n = z_1n, l_n = z_2n   若 c_n = 1
      w_n = z_2n, l_n = z_1n   若 c_n = 0

  L_n = log f(y_n; w_n, varZ) + log S(y_n; l_n, varZ)
```

在原尺度上（log 相加等於原尺度相乘）：

```
  exp(L_n) = f(y_n; w_n, varZ) * S(y_n; l_n, varZ)
```

圖示：

```
   c_n = 1              c_n = 0
   --------             --------
   f( . ; z_1n )        f( . ; z_2n )      <- lognormal_lpdf
        *                    *
   S( . ; z_2n )        S( . ; z_1n )      <- lognormal_lccdf

   兩個分支的結構相同，只把 z_1n / z_2n 對調
```

---

## 6. 完整的目標函數

Stan 的 `target` 累加後，整體是（相差一個與參數無關的常數）：

```
  log p(mu, alpha, alpha2, varZ, psi | data)

      =   log p(mu)  + log p(alpha)  + log p(alpha2)  + log p(varZ)

              N
        +   sum  [ log f(t_n - psi; w_n, varZ) + log S(t_n - psi; l_n, varZ) ]
             n=1

        +   J
```

其中：

```
  d_n  = alpha * x_n + alpha2 * x_n^2
  z_1n = mu - d_n
  z_2n = mu + d_n
  w_n  = z_1n if c_n = 1 else z_2n        (贏的那一個)
  l_n  = z_2n if c_n = 1 else z_1n        (輸的那一個)
```

`J` 是 Stan 對有界參數自動加上的 Jacobian 修正項，
來自 `varZ` 的 `<lower=0>` 與 `psi` 的 `<lower=0,upper=minRT>`。
它不是程式碼裡寫出來的，是 Stan 的語言行為。

---

## 7. 一句話版本

```
  每個試次貢獻:   (贏家的機率密度) x (輸家的存活函數)

  贏家 / 輸家由 correct[n] 決定，
  兩者的分布參數只差在 d_n 的正負號。
```
