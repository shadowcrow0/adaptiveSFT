# `lnrm2.stan` 的數學形式

把 `lnrm2.stan` 每一行翻成數學式。每一節都先貼程式碼，再給對應的公式。

本文所有密度公式都與 `scipy.stats` 比對過數值，確認一致。

> 數學用 LaTeX 寫，GitHub 網頁版會渲染。手機 app 可能只顯示原始碼。

---

## 0. 符號約定

| 數學符號 | 程式碼 | 意義 |
|---|---|---|
| $n$ | `tr` | 試次編號，$n = 1,\dots,N$ |
| $x_n$ | `intensity[n]` | 刺激強度 |
| $c_n$ | `correct[n]` | 是否答對，取值 $0$ 或 $1$ |
| $t_n$ | `rt[n]` | 反應時間 |
| $t_{\min}$ | `minRT` | 所有 $t_n$ 的最小值 |
| $\mu$ | `mu` | |
| $\alpha,\ \alpha_2$ | `alpha`, `alpha2` | |
| $s$ | `varZ` | 見下方說明 |
| $\psi$ | `psi` | |

**關於 $s$（程式碼裡叫 `varZ`）。** 它在公式裡扮演的是**標準差**，不是變異數。
因為 Stan 的 `lognormal(m, s)` 第二個參數定義就是標準差，而程式碼把 `varZ`
放在那個位置。本文一律用 $s$ 表示，以避免名稱誤導。

---

## 1. 兩個用到的分布函數

### 對數常態機率密度

Stan 的 `lognormal_lpdf(y | m, s)` 回傳 $\log f(y;m,s)$，其中

$$
f(y; m, s) \;=\; \frac{1}{y\,s\,\sqrt{2\pi}}\,
\exp\!\left(-\frac{(\log y - m)^2}{2 s^2}\right),
\qquad y > 0
$$

### 對數常態互補累積分布（存活函數）

令 $\Phi$ 為標準常態的累積分布函數

$$
\Phi(u) \;=\; \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{u} e^{-w^2/2}\,dw
$$

Stan 的 `lognormal_lccdf(y | m, s)` 回傳 $\log S(y;m,s)$，其中

$$
S(y; m, s) \;=\; 1 - F(y; m, s) \;=\; \Phi\!\left(-\frac{\log y - m}{s}\right)
$$

---

## 2. `transformed data` block

```stan
square_intensity = square(intensity);
```

$$
x_n^2, \qquad n = 1,\dots,N
$$

---

## 3. `transformed parameters` block

```stan
z[1,tr] = mu - alpha * intensity[tr] - alpha2 * square_intensity[tr];
z[2,tr] = mu + alpha * intensity[tr] + alpha2 * square_intensity[tr];
```

先把重複出現的那一項取名為 $d_n$：

$$
d_n \;=\; \alpha\,x_n + \alpha_2\,x_n^2
$$

則那兩行是

$$
z_{1n} \;=\; \mu - d_n, \qquad z_{2n} \;=\; \mu + d_n
$$

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

常態密度

$$
p_{\mathcal{N}}(v; m, \sigma) \;=\; \frac{1}{\sigma\sqrt{2\pi}}\,
\exp\!\left(-\frac{(v-m)^2}{2\sigma^2}\right)
$$

逆伽瑪密度

$$
p_{\mathrm{IG}}(v; a, b) \;=\; \frac{b^{a}}{\Gamma(a)}\,v^{-a-1} e^{-b/v}
$$

代入 $a = 1$、$b = 0.1$，因為 $\Gamma(1) = 1$：

$$
p(s) \;=\; 0.1\, s^{-2}\, e^{-0.1/s}, \qquad s > 0
$$

所以四個先驗是

$$
p(s) = 0.1\,s^{-2}e^{-0.1/s}, \quad
p(\mu) = p_{\mathcal{N}}(\mu; 0,1), \quad
p(\alpha) = p_{\mathcal{N}}(\alpha; 0,2), \quad
p(\alpha_2) = p_{\mathcal{N}}(\alpha_2; 0,1)
$$

`psi` 沒有 `~` 敘述，只有第 17 行的宣告 `real<lower=0,upper=minRT> psi`。

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

令 $y_n = t_n - \psi$（位移後的反應時間）。第 $n$ 試次貢獻的對數似然是

$$
\mathcal{L}_n =
\begin{cases}
\log f(y_n;\, z_{1n},\, s) \;+\; \log S(y_n;\, z_{2n},\, s), & c_n = 1\\[6pt]
\log f(y_n;\, z_{2n},\, s) \;+\; \log S(y_n;\, z_{1n},\, s), & c_n = 0
\end{cases}
$$

兩式只差在 $z_{1n}$ 與 $z_{2n}$ 對調。定義

$$
w_n = \begin{cases} z_{1n}, & c_n = 1\\ z_{2n}, & c_n = 0\end{cases}
\qquad
\ell_n = \begin{cases} z_{2n}, & c_n = 1\\ z_{1n}, & c_n = 0\end{cases}
$$

則兩個分支合併為一式

$$
\mathcal{L}_n \;=\; \log f(y_n;\, w_n,\, s) \;+\; \log S(y_n;\, \ell_n,\, s)
$$

在原尺度上（log 相加等於原尺度相乘）

$$
\exp(\mathcal{L}_n) \;=\; f(y_n;\, w_n,\, s)\,\cdot\, S(y_n;\, \ell_n,\, s)
$$

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

Stan 的 `target` 累加後，整體是（相差一個與參數無關的常數）

$$
\log p(\mu, \alpha, \alpha_2, s, \psi \mid \text{data})
\;=\;
\log p(\mu) + \log p(\alpha) + \log p(\alpha_2) + \log p(s)
$$

$$
\qquad\qquad
+\; \sum_{n=1}^{N}\Big[\,
\log f(t_n - \psi;\, w_n,\, s)
\;+\;
\log S(t_n - \psi;\, \ell_n,\, s)
\,\Big]
\;+\; J
$$

其中

$$
d_n = \alpha x_n + \alpha_2 x_n^2, \qquad
z_{1n} = \mu - d_n, \qquad
z_{2n} = \mu + d_n
$$

$$
w_n = \begin{cases} z_{1n}, & c_n = 1\\ z_{2n}, & c_n = 0\end{cases}
\qquad\text{(贏的那一個)}
\qquad
\ell_n = \begin{cases} z_{2n}, & c_n = 1\\ z_{1n}, & c_n = 0\end{cases}
\qquad\text{(輸的那一個)}
$$

$J$ 是 Stan 對有界參數自動加上的 Jacobian 修正項，來自 `varZ` 的 `<lower=0>`
與 `psi` 的 `<lower=0,upper=minRT>`。**它不是程式碼裡寫出來的**，是 Stan 的語言行為。

---

## 7. 一句話版本

每個試次貢獻

$$
\underbrace{f(y_n;\, w_n,\, s)}_{\text{贏家的機率密度}}
\;\times\;
\underbrace{S(y_n;\, \ell_n,\, s)}_{\text{輸家的存活函數}}
$$

贏家與輸家由 `correct[n]` 決定，兩者的分布參數只差在 $d_n$ 的正負號。
