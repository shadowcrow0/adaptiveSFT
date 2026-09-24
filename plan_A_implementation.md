# 方案 A 實施企劃書：最小修補的具體做法與實行流程

日期：2026-09-24
對應：`plan_r_modernization.md` 的方案 A。
**本文件是提案。下面所有「改法」都是建議寫法，尚未套用到任何檔案。**

---

## 0. 範圍與原則

```
   會動到的檔案            不會動到的
   ───────────────         ───────────────────────────
   lnrm2.stan              simulateLNRM_ogival.R
   adaptiveSFT_functions.R psiSimulation_functions.R
   （新增）tests/           psi Simulation_*.R
   （新增）找回的 .stan     adaptive_sft2.py（另案）
   README.md               lnrm2_pymc.py（不參與）
```

三條原則：

1. **每一步一個 commit**，commit 訊息寫清楚改了哪一行、為什麼。隨時可以 `git revert` 單獨一步。
2. **改語意之前先留底**：能改變估計結果的那一步（`:228`），改之前先用固定種子的模擬資料跑出舊行為的數字存檔。
3. **語法修補與語意修補分開 commit**，之後 code review 才看得出哪一個 commit 影響結果。

---

## 1. 實行流程總覽

```
   Phase 0  準備環境              Phase 1  留底
   ───────────────               ───────────────
   開分支                  ──►   固定種子生模擬資料
   記錄 R / rstan / 套件版本       用「舊 R 語意」算出反解數字存檔
   裝 rstan, sft, diffIRT              │
        │                              ▼
        │                    ┌─ 閘門 1：source() 只剩 :228 那個已知錯誤 ─┐
        ▼                    ▼                                          │
   Phase 2  Stan 語法（純機械）                                          │
   ───────────────                                                      │
   lnrm2.stan 5 行改 array[] 語法                                        │
        │                                                               │
        ▼                                                               │
   ┌─ 閘門 2：stan_model() 編譯通過 ─┐                                   │
   ▼                                                                    │
   Phase 3  修 :228（語意，需老闆拍板）                                   │
   ───────────────                                                      │
   換成向量語意；跑 Phase 1 的資料                                        │
        │                                                               │
        ▼                                                               │
   ┌─ 閘門 3：結果與留底相同（所有抽樣 alpha2<0 時）─┐                    │
   ▼                                                                    │
   Phase 4  :279 回報丟棄比例        Phase 5  varZ 尺度（只加註解＋測試）  │
        │                                                               │
        ▼                                                               │
   Phase 6  找回 lnrm0 / lnrm1 / lnrm2a.stan  ◄──── 可與 Phase 2-5 平行 ─┘
        │
        ▼
   Phase 7  回歸測試 tests/          Phase 8  全流程跑一次 → README → PR
```

---

## 2. 每個 Phase 的具體做法

### Phase 0 — 準備環境（0.5 天）

```bash
git checkout -b fix/r-modernization
Rscript -e 'cat(R.version.string, "\n"); for (p in c("rstan","StanHeaders","sft","diffIRT")) cat(p, tryCatch(as.character(packageVersion(p)), error=function(e) "未安裝"), "\n")'
Rscript -e 'install.packages(c("rstan","sft","diffIRT"))'
```

若 `diffIRT` 裝不起來（印象中曾被 CRAN 下架，未在本機確認）：

```r
install.packages("remotes")
remotes::install_version("diffIRT", version = "1.5")   # 或從 CRAN archive 指定版本
```

把上面印出來的版本貼進一個新檔 `ENVIRONMENT.md`，之後 README 會引用。

**閘門 1**：`source("adaptiveSFT_functions.R")` 無錯誤；
呼叫 `find_salience_polynomial()` 只出現 `the condition has length > 1`（已知，Phase 3 處理）。

### Phase 1 — 留底（0.5 天）

目的：在改任何語意之前，記下「舊 R（< 4.2）會算出什麼」。
舊 R 對 `if (向量)` 的行為是**只看第一個元素**，其餘照算，所以留底就是用第一個元素當條件。

新增 `tests/make_baseline.R`（提案）：

```r
# 從 lnrm2.stan 的世界觀生資料：兩個對數常態賽跑，取先到者
set.seed(20260924)
N <- 2000
truth <- list(mu=1.5, alpha=0.8, alpha2=-0.15, s=0.6, psi=0.12)
intensity <- runif(N, 0, 3)
d  <- truth$alpha*intensity + truth$alpha2*intensity^2
t1 <- truth$psi + exp(rnorm(N, truth$mu - d, truth$s))   # 答對累積器
t2 <- truth$psi + exp(rnorm(N, truth$mu + d, truth$s))   # 答錯累積器
dat <- data.frame(intensity=intensity, rt=pmin(t1,t2), correct=as.integer(t1<t2))
saveRDS(list(dat=dat, truth=truth), "tests/baseline_data.rds")
```

留底用的反解要**繞過** `:228` 的錯誤但**保留舊語意**——不改原檔，只在測試腳本裡重現舊行為：

```r
fit <- rstan::stan(file="lnrm2.stan", data=dataframe2stan(dat),
                   pars=c("mu","alpha","alpha2","varZ","psi"), seed=1)
post <- rstan::extract(fit, c("mu","alpha","alpha2","psi","varZ"))
h_targ <- 1.6; l_targ <- 0.8
if (post$alpha2[1] < 0) {                                # 舊 R 的實際行為：只看第 1 個
  src <- readLines("adaptiveSFT_functions.R")
  eval(parse(text = src[229:232]))                       # 逐字執行原式，不改內容
}
saveRDS(list(high=mean(h_targ.dist, na.rm=TRUE), low=mean(l_targ.dist, na.rm=TRUE),
             n_na=sum(is.na(h_targ.dist)), post=post), "tests/baseline_result.rds")
```

（Phase 2 之前 Stan 若編不過，這一步可先用 PyMC 引擎抽樣存 CSV 代替，`lnrm2_pymc_notes.md` 有做法；
但正式留底仍應以 rstan 為準。）

### Phase 2 — Stan 語法（0.5 天，純機械）

`lnrm2.stan` 五行，逐一對照（提案）：

| 行 | 現在 | 改成 |
|---|---|---|
| `:3` | `real intensity[N];` | `array[N] real intensity;` |
| `:4` | `int<lower=0,upper=1> correct[N];` | `array[N] int<lower=0,upper=1> correct;` |
| `:6` | `real<lower=0> rt[N];` | `array[N] real<lower=0> rt;` |
| `:9` | `real square_intensity[N];` | `array[N] real square_intensity;` |
| `:20` | `real z[2,N];` | `array[2, N] real z;` |

其他行不動。`:10` 的 `square(intensity)` 對 array 仍是向量化函式，不需改。
`array[]` 語法從 Stan 2.26 起就被接受，所以**不管目標機器的 rstan 是 2.26 還是 2.36 都能編**，
不必先確認版本再決定改不改。

**閘門 2**：

```r
m <- rstan::stan_model("lnrm2.stan")   # 無 error
```

若目標機器 rstan < 2.26（極舊），才會失敗；那時先升 rstan。

### Phase 3 — 修 `:228`（0.5 天，需老闆拍板語意）

現況（`adaptiveSFT_functions.R:228-233`）：

```r
  if (post.diff$alpha2 <0) {
  l_targ.dist <- with(post.diff,
        (-alpha/alpha2 - sqrt( (alpha/alpha2)^2 + 2 / alpha2 * l_targ)) / 2)
  h_targ.dist <- with(post.diff,
        (-alpha/alpha2 - sqrt( (alpha/alpha2)^2 + 2 / alpha2 * h_targ)) / 2)
  }
```

**哪個語言做了什麼**：R 4.2.0 起，`if()` 的條件長度 > 1 從「警告、取第一個」改為**直接 error**。
`post.diff$alpha2` 是幾千個抽樣的向量，所以整個函式在 R ≥ 4.2 一呼叫就停。

三種改法，**建議 (b)**：

```r
# (a) 全部抽樣都 < 0 才算；否則整段跳過（最接近原作者「只在向下開口時反解」的字面意圖）
if (all(post.diff$alpha2 < 0)) { ...原式不動... }

# (b) 逐抽樣：alpha2 < 0 的算，其餘給 NA，最後 na.rm 平均 —— 建議
ok <- post.diff$alpha2 < 0
l_targ.dist <- ifelse(ok, with(post.diff, (-alpha/alpha2 - sqrt((alpha/alpha2)^2 + 2/alpha2*l_targ))/2), NA_real_)
h_targ.dist <- ifelse(ok, with(post.diff, (-alpha/alpha2 - sqrt((alpha/alpha2)^2 + 2/alpha2*h_targ))/2), NA_real_)

# (c) 用後驗平均判斷
if (mean(post.diff$alpha2) < 0) { ...原式不動... }
```

為什麼建議 (b)：原式本來就是對整個向量算（`with()` 是向量化的），`:279` 也已經在對抽樣取平均，
(b) 是唯一「每個抽樣各自負責」、不會因為第一個或平均值的偶然而整段跳掉的寫法。
代價：`alpha2 ≥ 0` 的抽樣會變成 NA，被 `:279` 的 `na.rm` 吃掉——所以 Phase 4 必須回報比例。

**閘門 3**：用 Phase 1 的資料重跑。若留底的 `post$alpha2` 全部 < 0（典型情況），
(b) 的 `high`/`low` 必須與 `baseline_result.rds` **完全相同**（差 < 1e-12）——
因為此時 (b) 和舊行為算的是同一批抽樣、同一條式子。不相同就是改錯了。

### Phase 4 — `:279` 回報丟棄比例（0.5 天）

現況：

```r
  return(list(high=mean(h_targ.dist, na.rm=TRUE),
               low=mean(l_targ.dist, na.rm=TRUE), fit=fitModel))
```

提案（只**加**欄位，不改原有欄位，呼叫端不會壞）：

```r
  return(list(high=mean(h_targ.dist, na.rm=TRUE),
               low=mean(l_targ.dist, na.rm=TRUE),
               high_dropped=mean(is.na(h_targ.dist)),   # 被 na.rm 丟掉的比例
               low_dropped=mean(is.na(l_targ.dist)),
               fit=fitModel))
```

NA 有兩個來源：(1) Phase 3 的 `alpha2 ≥ 0`；(2) `targ` 超過可達上限 `2·d_max`，`sqrt` 的判別式為負。

```
      d_max  =  −α² / (4·α₂)          （α₂ < 0 時的拋物線頂點）
      h_targ 上限  =  2·d_max
```

建議在 README 寫：`*_dropped` 超過 5% 就要檢查 `h_targ` 是否訂太高。

### Phase 5 — `varZ` 尺度（0.5 天，只加註解與測試；改不改由老闆定）

事實：`lnrm2.stan:38` 把 `varZ` 放在 `lognormal_lpdf` 的**標準差**位置；
`adaptiveSFT_functions.R:61-62` 的 `dlognormalrace(x, m, psi, mu, sigmasq)` 收 `sigmasq` 後 `sqrt()`。
把 Stan 的 `varZ` 抽樣直接餵給 `dlognormalrace()` 會被多開一次根號。

提案（不改邏輯）：

1. 在 `lnrm2.stan:16` 上方加註解：`// varZ is used as the SD of the lognormal (Stan's second argument), despite the name.`
2. 在 `dlognormalrace()` 上方加註解：`// sigmasq is a VARIANCE; pass varZ^2 when using Stan draws.`
3. 在 Phase 7 加一個測試：用 `varZ^2` 餵 `dlognormalrace()` 後對 `x` 積分應 ≈ 1（是合法密度）；
   用 `varZ` 直接餵則不會是 1。這樣哪個尺度對，測試會說話。

若老闆決定改名（例如 `varZ` → `sdZ`），那是另一個 commit，且要同步改 `pars=` 清單（`:217`、`:219`）。

### Phase 6 — 找回三個 `.stan` 檔（0.5 ～ 2 天，最大變數）

已查：**這個 fork 的 git 歷史從第一個 commit（`10a86ce`）起就沒有這三個檔**，不是後來被刪的。
所以只能往外找，順序：

```
   1. 上游 jhoupt/adaptiveSFT 的所有分支與歷史      git log --all -- '*.stan'
   2. 原作者 / 實驗室共用硬碟 / 舊電腦
   3. 都找不到 → 從 R 端的用法反推重寫（只重寫 lnrm2a，lnrm0/1 若無人用可先放棄）
```

重寫 `lnrm2a.stan` 的依據：`adaptiveSFT_functions.R:186` 的 `pars=c("slope","midpoint","mu","varZ","psi")`，
以及 `:11` 的 `getPr_ogival()`：

```r
x <- with(postSamps, L * inv_logit(slope * (intensity - midpoint)))
```

也就是把 `lnrm2.stan` 的 `d_n = alpha*x + alpha2*x²` 換成 `L * inv_logit(slope*(x − midpoint))`。
**一個要先問清楚的矛盾**：`getPr_ogival()` 用到 `L`，但 `:186` 的 `pars` 清單裡沒有 `L`——
要嘛 `L` 在 stan 檔裡是固定常數，要嘛 `getPr_ogival()` 已經過時。重寫前必須確認。

### Phase 7 — 回歸測試（1 天）

新增 `tests/test_lnrm2.R`，三層：

```
   第 1 層（秒級，不需 Stan）   反解公式：給固定的 alpha/alpha2 向量，
                                 high/low 必須等於手算值；alpha2 ≥ 0 的位置必須是 NA
   第 2 層（秒級，不需 Stan）   Phase 5 的密度積分測試
   第 3 層（分鐘級，需 Stan）   Phase 1 資料 → stan() → find_salience_polynomial()
                                 → high/low 與真值誤差 < 5%，*_dropped < 5%
```

真值的 high/low 由 truth 直接代入同一條反解式算出（`alpha=0.8, alpha2=-0.15, h_targ=1.6, l_targ=0.8`
→ `high ≈ 1.333`, `low ≈ 0.559`）。

### Phase 8 — 全流程 → README → PR（1 天）

1. 從 `dataframe2stan()` 到 `find_salience_polynomial()` 用 Phase 1 資料完整跑一次，貼輸出。
2. README 加「執行環境」（貼 `ENVIRONMENT.md`）與「如何跑測試」。
3. 開 PR，描述引用本文件的 Phase 編號；每個 commit 對應一個 Phase。

---

## 3. 時程

```
   Day 1   Phase 0 ─ Phase 1 ─ Phase 2                （閘門 1、2）
   Day 2   Phase 3 ─ Phase 4 ─ Phase 5                （閘門 3；Phase 3 前先拿到老闆決定）
   Day 3   Phase 7 ─ Phase 8                          （若 Phase 6 已解決）
   Day 3-5 Phase 6                                    （視找檔結果，可與 Day 1-2 平行進行）
```

合計 **3 天（找得到檔）～ 5 天（要重寫 lnrm2a）**。假設執行者熟 R、能自己裝 rstan。

---

## 4. 驗收清單

- [ ] `ENVIRONMENT.md` 有 R、rstan、StanHeaders、sft、diffIRT 版本
- [ ] R ≥ 4.2 下 `source("adaptiveSFT_functions.R")` 無 error（閘門 1）
- [ ] `rstan::stan_model("lnrm2.stan")` 編譯通過（閘門 2）
- [ ] `find_salience_polynomial()` 可呼叫，且留底資料上結果與舊語意相同（閘門 3）
- [ ] 回傳值含 `high_dropped` / `low_dropped`
- [ ] `tests/test_lnrm2.R` 三層全過；第 3 層 high/low 誤差 < 5%
- [ ] `varZ` 尺度在兩個檔案都有註解，且有測試證明哪個尺度是對的
- [ ] 三個 `.stan` 檔：找回、重寫、或明確記錄「放棄」
- [ ] 每個 Phase 一個 commit，PR 描述對得上

---

## 5. 回滾

每個 Phase 獨立 commit，出問題就 `git revert <該 commit>`，不影響其他 Phase。
Phase 3 是唯一改變估計結果的一步，`tests/baseline_result.rds` 永久保留作為對照。

---

## 6. 需要老闆拍板（Phase 3 開工前）

| 決定 | 建議預設 | 不拍板的話 |
|---|---|---|
| `:228` 語意 (a)/(b)/(c) | **(b)** 逐抽樣 | Phase 3 卡住，Day 2 無法開始 |
| `varZ` 要不要改名 | 先不改，只加註解＋測試 | 不影響進度 |
| 三個 `.stan` 的下落 | 先問上游 | Phase 6 走最壞路線（+2 天） |
| `getPr_ogival()` 的 `L` 是常數還是過時碼 | — | 重寫 lnrm2a 前無法動 |
| `*_dropped` 門檻 | 5% | 用預設 |
