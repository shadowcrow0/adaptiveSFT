# `lnrm2.stan` 的數學形式

把 `lnrm2.stan` 每一行翻成數學式。每一節都先貼程式碼，再給對應的公式。

所有密度公式都與 `scipy.stats` 比對過數值，確認一致。

> 公式一律放在 code block 裡，用 Unicode 符號排版。
> 這樣不經過任何數學渲染器，GitHub 網頁、GitHub app、Obsidian 顯示都一致。

---

## 0. 符號約定

| 符號 | 程式碼 | 意義 |
|---|---|---|
| n | `tr` | 試次編號，n = 1 … N |
| xₙ | `intensity[n]` | 刺激強度 |
| cₙ | `correct[n]` | 是否答對，0 或 1 |
| tₙ | `rt[n]` | 反應時間 |
| t_min | `minRT` | 所有 tₙ 的最小值 |
| μ | `mu` | |
| α, α₂ | `alpha`, `alpha2` | |
| s | `varZ` | 見下方說明 |
| ψ | `psi` | |

**關於 s（程式碼裡叫 `varZ`）。** 它在公式裡扮演的是**標準差**，不是變異數。
因為 Stan 的 `lognormal(m, s)` 第二個參數定義就是標準差，而程式碼把 `varZ`
放在那個位置。本文一律寫 s，避免名稱誤導。

---

## 1. 兩個用到的分布函數

### 對數常態機率密度

Stan 的 `lognormal_lpdf(y | m, s)` 回傳 ln f(y; m, s)：

```
                         1                    (     (ln y − m)²  )
      f(y; m, s)  =  ───────────────  ·  exp  ( − ─────────────  )      y > 0
                      y · s · √(2π)            (        2 s²      )
```

### 對數常態互補累積分布（存活函數）

Φ 是標準常態的累積分布函數：

```
                    1          u
      Φ(u)  =  ─────────  ·  ∫    exp(−w² / 2) dw
                 √(2π)        −∞
```

Stan 的 `lognormal_lccdf(y | m, s)` 回傳 ln S(y; m, s)：

```
                                         (     ln y − m  )
      S(y; m, s)  =  1 − F(y; m, s)  = Φ ( − ──────────  )
                                         (        s       )
```

---

## 2. `transformed data` block

```stan
square_intensity = square(intensity);
```

```
      xₙ²          n = 1 … N
```

---

## 3. `transformed parameters` block

```stan
z[1,tr] = mu - alpha * intensity[tr] - alpha2 * square_intensity[tr];
z[2,tr] = mu + alpha * intensity[tr] + alpha2 * square_intensity[tr];
```

先把重複出現的那一項取名為 dₙ：

```
      dₙ  =  α·xₙ  +  α₂·xₙ²
```

則那兩行是：

```
      z₁ₙ  =  μ − dₙ
      z₂ₙ  =  μ + dₙ
```

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
                            1                (    (v − m)²  )
      p_N(v; m, σ)  =  ───────────  ·  exp   ( − ─────────  )
                         σ·√(2π)              (     2 σ²     )
```

逆伽瑪密度：

```
                          b^a
      p_IG(v; a, b)  =  ────────  ·  v^(−a−1)  ·  exp(−b / v)
                         Γ(a)
```

代入 a = 1、b = 0.1，因為 Γ(1) = 1：

```
      p(s)  =  0.1 · s^(−2) · exp(−0.1 / s)            s > 0
```

四個先驗合起來：

```
      p(s)   =  0.1 · s^(−2) · exp(−0.1 / s)
      p(μ)   =  p_N(μ;  0, 1)
      p(α)   =  p_N(α;  0, 2)
      p(α₂)  =  p_N(α₂; 0, 1)
```

`psi` 沒有 `~` 敘述，只有 `lnrm2.stan:17` 的宣告 `real<lower=0,upper=minRT> psi`。

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

令 yₙ = tₙ − ψ（位移後的反應時間）。第 n 試次貢獻的對數似然：

```
   cₙ = 1 (答對):   Lₙ  =  ln f(yₙ; z₁ₙ, s)  +  ln S(yₙ; z₂ₙ, s)

   cₙ = 0 (答錯):   Lₙ  =  ln f(yₙ; z₂ₙ, s)  +  ln S(yₙ; z₁ₙ, s)
```

兩式只差在 z₁ₙ 與 z₂ₙ 對調。定義贏家 wₙ 與輸家 lₙ：

```
              ⎧ z₁ₙ   若 cₙ = 1                  ⎧ z₂ₙ   若 cₙ = 1
      wₙ  =   ⎨                          lₙ  =   ⎨
              ⎩ z₂ₙ   若 cₙ = 0                  ⎩ z₁ₙ   若 cₙ = 0
```

則兩個分支合併為一式：

```
      Lₙ  =  ln f(yₙ; wₙ, s)  +  ln S(yₙ; lₙ, s)
```

在原尺度上（log 相加等於原尺度相乘）：

```
      exp(Lₙ)  =  f(yₙ; wₙ, s)  ×  S(yₙ; lₙ, s)
```

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
      ln p(μ, α, α₂, s, ψ | data)

          =   ln p(μ) + ln p(α) + ln p(α₂) + ln p(s)

                   N
              +   ∑   [ ln f(tₙ − ψ; wₙ, s)  +  ln S(tₙ − ψ; lₙ, s) ]
                  n=1

              +   J
```

其中：

```
      dₙ   =  α·xₙ + α₂·xₙ²
      z₁ₙ  =  μ − dₙ
      z₂ₙ  =  μ + dₙ
      wₙ   =  z₁ₙ 若 cₙ = 1，否則 z₂ₙ        (贏的那一個)
      lₙ   =  z₂ₙ 若 cₙ = 1，否則 z₁ₙ        (輸的那一個)
```

J 是 Stan 對有界參數自動加上的 Jacobian 修正項，來自 `varZ` 的 `<lower=0>`
與 `psi` 的 `<lower=0,upper=minRT>`。**它不是程式碼裡寫出來的**，是 Stan 的語言行為。

---

## 7. 一句話版本

```
      每個試次貢獻:

           f(yₙ; wₙ, s)     ×     S(yₙ; lₙ, s)
           ~~~~~~~~~~~~           ~~~~~~~~~~~~
           贏家的機率密度          輸家的存活函數

      贏家 / 輸家由 correct[n] 決定，
      兩者的分布參數只差在 dₙ 的正負號。
```
