# poc/ — 「GRTv3_ada 方式」概念驗證（純 numpy / scipy，不需要 R、Stan、PsychoPy）

**這是什麼。** 這個資料夾是概念驗證（proof of concept，PoC）腳本：在正式寫套件之前，
用最短的程式證明「這條路走得通」。

**一個比喻：蓋房子前的紙板模型。** 比例對、承重對，足以說服人這個設計成立，
但不是拿來住的。正式的房子是 `adaptivesft/`。

幾個名詞：

- **GRTv3_ada 方式**：`Visual_AudioWM` 實驗裡那套「每題更新、自動調難度」的 Psi 適應程序。
  這裡驗證的是：用它取代原作者跑不起來的 R 版 Psi，接到 SFT 的流程上行不行。
- **Psi**：每答一題就重算「下一題該出哪個強度」的方法。估的是 α（閾值位置）和 β（曲線斜率）。
- **DDM / `simdiffT`**：漂移擴散模型，這裡當假受試者。`simdiffT` 是 R 套件 `diffIRT` 裡模擬它的函式。
- **`dfp_ddm`**：用 DDM 跑一場 DFP（2×2 反應時間實驗）。
- **`sft::sic`**：R 套件 `sft` 裡算 SIC（survivor interaction contrast）的函式，判斷並行 / 序列。
- **移植**：把 R 的函式用 Python 重寫，數字要對得上。
- **回復（recovery）**：先用已知參數造假資料，再估計，看估得回來嗎。
- **檢定力（power）**：真的有效果時，統計檢定抓到它的機率。

對應文件：`../plan_grtv3ada_psi_python.md`。這裡的程式是**證據**，不是正式套件。

## Psi 接 SFT：三支腳本

| 檔案 | 做什麼 | 對應原碼 |
|---|---|---|
| `psi_sft_poc.py` | Psi（逐字抄 `Visual_AudioWM/AGRT.py:75-184`，去掉 PsychoPy）＋ `simdiffT` 移植 ＋ `dfp_ddm` 移植 ＋ `sft::sic` 移植；主程式跑 72/144/300 試校準與五種架構的 SIC | `psiSimulation_functions.R`、`adaptiveSFT_functions.R:115-150`、`sft/R/sic.R` |
| `psi_sft_poc_recovery_power.py` | A：累積常態受試者的 α/β 回復；B：DDM 受試者下 β 網格上限、H/L 可達性、SIC 檢定力 | `psi Simulation_26MAR2019.R` |
| `psi_sft_poc_sic_signature.py` | 大 salience 差下五種架構的 SIC 簽名，驗證 `sic` 移植 | Townsend & Nozawa 1995 的預測 |

三支各回答一個問題。第一支：整條流程接得起來嗎？校準 72、144、300 題各試一次，再跑五種
架構看 SIC。第二支：Psi 估得準嗎？A 用「剛好符合 Psi 假設」的受試者（累積常態＝心理計量
函數是 S 形的常態累積），看 α/β 回不回得來；B 換成 DDM 受試者，看 β 會不會超出網格上限
（網格＝Psi 事先列好的候選值，超出就被釘在邊緣）、算出的 H/L 在不在實驗範圍內、SIC 抓得到
架構嗎。第三支：把顯著度差距拉大，五種架構的 SIC 曲線應該長成 Townsend & Nozawa 1995
預測的樣子，用來確認 `sic` 移植沒寫錯。

```bash
python3 -m venv venv && ./venv/bin/pip install numpy scipy
cd poc && ../venv/bin/python psi_sft_poc.py 1
../venv/bin/python psi_sft_poc_recovery_power.py
../venv/bin/python psi_sft_poc_sic_signature.py
```

第一行建一個獨立的 Python 環境並裝 numpy、scipy（兩個數值計算套件）。`psi_sft_poc.py 1`
後面的 `1` 是隨機種子，同一個種子每次跑出同樣的數字。

實測（2026-09-25，numpy 2.4.6 / scipy 1.17.1）：三支合計約 3 分鐘。

## 賽跑密度：一支腳本

| `lnrm_race_poc.py` | `dlognormalrace / plognormalrace` 的 scipy 版，對總積分、封閉解正確率、蒙地卡羅 CDF | `adaptiveSFT_functions.R:61-111` |
|---|---|---|

對應文件：`../plan_python_port.md` §2.3。

這支和 Psi 無關，是另一條路（LNRM 賽跑模型）的第一塊積木。`dlognormalrace` 是賽跑的
密度（某時間點反應的機率密度），`plognormalrace` 是它的累積版（CDF，到某時間為止已反應的
比例）。驗的三件事：密度總積分要等於 1；算出的正確率要等於有公式的封閉解；CDF 要和
大量隨機模擬（蒙地卡羅）一致。

## 注意

> 注意：`psi_sft_poc.py` 裡的 Psi 是抄 `Visual_AudioWM/AGRT.py` 的，只是當時的概念驗證。
> 正式套件 `adaptivesft/psi.py` **不用 AGRT 的方式**，是 `psiSimulation_functions.R` 的逐行移植（網格照 R）。

意思是：這裡的 Psi 和正式套件的 Psi 是同一個演算法的兩個來源。PoC 抄實驗程式 `AGRT.py`
的版本，正式套件照原作者的 R 版一行一行移植，網格常數也照 R。兩者的數字不保證相同，
正式結果一律以 `adaptivesft/psi.py` 為準。

## 所以你要做什麼

1. 平常不用動這個資料夾。它的任務（證明可行）已經完成，結論寫在 `../plan_grtv3ada_psi_python.md`。
2. 想重現那份文件裡的數字：跑上面的三行指令，約 3 分鐘。
3. 要用 Psi 做正式校準：去 `adaptivesft/psi.py` 和 `adaptivesft/experiment.py`，不要用這裡的。
4. 要看賽跑密度的最小可跑版本：`lnrm_race_poc.py`，一秒跑完。
