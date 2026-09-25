# poc/ — 「GRTv3_ada 方式」概念驗證（純 numpy / scipy，不需要 R、Stan、PsychoPy）

對應文件：`../plan_grtv3ada_psi_python.md`。這裡的程式是**證據**，不是正式套件。

| 檔案 | 做什麼 | 對應原碼 |
|---|---|---|
| `psi_sft_poc.py` | Psi（逐字抄 `Visual_AudioWM/AGRT.py:75-184`，去掉 PsychoPy）＋ `simdiffT` 移植 ＋ `dfp_ddm` 移植 ＋ `sft::sic` 移植；主程式跑 72/144/300 試校準與五種架構的 SIC | `psiSimulation_functions.R`、`adaptiveSFT_functions.R:115-150`、`sft/R/sic.R` |
| `psi_sft_poc_recovery_power.py` | A：累積常態受試者的 α/β 回復；B：DDM 受試者下 β 網格上限、H/L 可達性、SIC 檢定力 | `psi Simulation_26MAR2019.R` |
| `psi_sft_poc_sic_signature.py` | 大 salience 差下五種架構的 SIC 簽名，驗證 `sic` 移植 | Townsend & Nozawa 1995 的預測 |

```bash
python3 -m venv venv && ./venv/bin/pip install numpy scipy
cd poc && ../venv/bin/python psi_sft_poc.py 1
../venv/bin/python psi_sft_poc_recovery_power.py
../venv/bin/python psi_sft_poc_sic_signature.py
```

實測（2026-09-25，numpy 2.4.6 / scipy 1.17.1）：三支合計約 3 分鐘。
