"""
adaptivesft —— adaptiveSFT（Houpt 2018–2019，R + Stan）的 Python 版。

    race.py       賽跑 likelihood / 密度 / CDF      ← adaptiveSFT_functions.R:61-111、lnrm2.stan:37-44
    models.py     lnrm0 / lnrm1 / lnrm2 / lnrm2a    ← lnrm2.stan（其餘三支原檔遺失，重建）
    salience.py   反解 high / low                    ← adaptiveSFT_functions.R:180-281
    ddm.py        simdiffT / moc_ddm / dfp_ddm      ← diffIRT、adaptiveSFT_functions.R:115-165
    sic.py        sic / sicGroup                    ← sft/R/sic.R
    psi.py        Psi 適應法                         ← psiSimulation_functions.R、Visual_AudioWM/AGRT.py

不需要 R、Stan、PsychoPy。依賴見 requirements.txt。每個模組檔頭寫了它對應哪幾行原碼、
以及原碼哪些問題（issue.md 編號）在那裡定案。
"""
from .ddm import A_CONVENTIONS, ddm_mean_dt, ddm_p_correct, dfp_ddm, draw_participant, moc_ddm, separation, simdiffT
from .models import LINKS, d_numpy, fit_lnrm, fit_lnrm0_by_level, make_data
from .psi import PsiObject, inv_pm_function, make_psi, pm_function, salience_levels
from .race import dlognormalrace, lnrm_pointwise_loglik, lnrm_random, plognormalrace, plognormalrace_curve
from .salience import (
    ALPHA2_RULES, accuracy_to_targ, find_salience, find_salience_ogival,
    find_salience_polynomial, summarize, targ_to_accuracy,
)
from .sic import classify, mic_test, si_dominance, sic, sic_group, sic_test

__version__ = "0.1.0"
