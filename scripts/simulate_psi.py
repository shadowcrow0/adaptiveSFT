"""
psi Simulation_26MAR2019.R 的 Python 版（2018 版被它取代，不移植），兩種 a 慣例可並排跑。

    convergence      :9-148     nsamps 次 × trials 試的 Psi，DDM 受試者；α/β 收斂與平均心理計量函數
    dfp              :177-459   Psi 校準顏色與方位 → H(.99) / L(.90) → 五種架構各一場 DFP → survivor + SIC
    full-experiment  :460-1003  nParticipants 位，四種架構各一場（PAR-OR / PAR-AND / SER-OR / SER-AND）
                                → SFTresults 表（欄位同 ParallelOR_Psi_Simulation_SFTresults.csv）

「真」α/β：R :117-121 把 DDM 的 P(correct) 曲線用 pm.function 擬合，但 :117 的公式是 2·a·drift。
這裡對每種 a 慣例各算一條真曲線（ddm_p_correct(a_is_separation=…)），所以慣例選對時
Psi 的估計應該收斂到「真」值，選錯時不會 —— 這就是 decisions_for_author.md A 要看的數字。

參數照 psiSimulation_functions.R:10-18, 93-99, 179, 214-225：顏色範圍 −55–50、thres50 6；
方位範圍 45–90、thres50 63；a=1.45, v=1.6, ter=.1, sdv=.25；lapse .01；100 格網格（R 是每 1 單位一格）。
"""
import functools
import os

import numpy as np
from scipy import optimize

from _common import (ARCHS, Timer, base_parser, conventions, outdir, plot_survivor_sic, print_table,
                     sic_row, write_csv)
from adaptivesft.ddm import ddm_p_correct, dfp_ddm, simdiffT
from adaptivesft.psi import make_psi, pm_function, salience_levels

A, V, TER, SDV = 1.45, 1.6, 0.1, 0.25            # psiSimulation_functions.R:96-99
LAPSE = 0.01                                     # :12, :18
DIMS = {"colour": ((-55.0, 50.0), 6.0), "orientation": ((45.0, 90.0), 63.0)}
P_HIGH, P_LOW = 0.99, 0.90                       # :183-184
STEPS = 100


def scaled(x, dim):
    x_range, thres50 = DIMS[dim]
    return (np.asarray(x, float) - thres50) / (x_range[1] - thres50)    # :104


def true_alpha_beta(dim, sep):
    """R :113-121：把 DDM 的 P(correct) 用 pm.function 擬合成 (α, β)。"""
    x_range, _ = DIMS[dim]
    xx = np.linspace(x_range[0], x_range[1], 1000)
    pc = ddm_p_correct(A, scaled(xx, dim) * V, a_is_separation=sep)
    rmse = lambda p: np.sqrt(np.mean((pm_function(xx, p[0], p[1], LAPSE) - pc) ** 2))
    r = optimize.minimize(rmse, [DIMS[dim][1], 6.0], method="Nelder-Mead")
    return float(r.x[0]), float(r.x[1]), float(pc[-1])


def run_psi(dim, n_trials, sep, rng, beta_max=None):
    """Est.Trial.Psi.*（:5-168）：DDM 受試者回答 Psi 出的題。回傳每試後的 (α, β) 軌跡。"""
    x_range, _ = DIMS[dim]
    psi = make_psi(x_range, STEPS, LAPSE, beta_max=beta_max)
    traj = np.empty((n_trials, 2))
    for t in range(n_trials):
        _, r = simdiffT(1, A, scaled(psi.nextIntensity, dim) * V, SDV, TER, rng=rng, a_is_separation=sep)
        psi.update(int(r[0]))
        traj[t] = psi.estimateLambda()
    return traj


def sec_convergence(args, conv, sep, rng):
    nsamps, trials = (5, 60) if args.quick else (args.nsamps, args.trials)
    dim = "orientation"                                              # R 論文用方位（:9-11）
    a_true, b_true, p_top = true_alpha_beta(dim, sep)
    with Timer(f"{nsamps} × {trials} Psi"):
        trajs = np.stack([run_psi(dim, trials, sep, rng) for _ in range(nsamps)])
    rows = []
    for t in sorted(set([0, 9, 29, 59, 99, 143, trials - 1]) & set(range(trials))):
        rows.append({"convention": conv, "trial": t + 1,
                     "alpha_mean": float(trajs[:, t, 0].mean()), "alpha_q05": float(np.quantile(trajs[:, t, 0], .05)),
                     "alpha_q95": float(np.quantile(trajs[:, t, 0], .95)),
                     "beta_mean": float(trajs[:, t, 1].mean()), "beta_q05": float(np.quantile(trajs[:, t, 1], .05)),
                     "beta_q95": float(np.quantile(trajs[:, t, 1], .95)),
                     "alpha_true": a_true, "beta_true": b_true})
    print(f"  「真」α={a_true:.2f} β={b_true:.2f}（此慣例下範圍上限的 P(correct)={p_top:.3f}）")
    print_table(rows, ["trial", "alpha_mean", "alpha_q05", "alpha_q95", "beta_mean", "beta_q05", "beta_q95"], fmt="{:>11}")
    # 圖：R :63-127 的平均心理計量函數（紅→藍）+ α/β 收斂
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x_range, _ = DIMS[dim]
    xx = np.linspace(x_range[0], x_range[1], 300)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    cmap = plt.get_cmap("rainbow_r")
    for t in range(0, trials, max(1, trials // 30)):
        a_m, b_m = trajs[:, t, 0].mean(), trajs[:, t, 1].mean()
        axes[0].plot(xx, pm_function(xx, a_m, b_m, LAPSE), color=cmap(t / trials), lw=0.8)
    axes[0].plot(xx, pm_function(xx, a_true, b_true, LAPSE), color="k", lw=2, label="true (fitted DDM)")
    axes[0].plot(xx, ddm_p_correct(A, scaled(xx, dim) * V, a_is_separation=sep), ls=":", color="k", label="DDM")
    axes[0].set_title("Psychometric Function Convergence")
    axes[0].legend(fontsize=7)
    for ax, k, tv in ((axes[1], 0, a_true), (axes[2], 1, b_true)):
        ax.plot(trajs[:, :, k].mean(axis=0), lw=2)
        ax.plot(np.quantile(trajs[:, :, k], .05, axis=0), ls="--", color="k", lw=0.8)
        ax.plot(np.quantile(trajs[:, :, k], .95, axis=0), ls="--", color="k", lw=0.8)
        ax.axhline(tv, ls="--", color="red")
        ax.set_xlabel("Trial")
        ax.set_title(("Location", "Slope")[k])
    fig.suptitle(f"Psi convergence, {dim}, a = {conv}")
    fig.tight_layout()
    path = os.path.join(outdir(args, "psi_convergence"), f"Psi_convergence_{conv}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    wrote {path}")
    return rows


def calibrate(args, conv, sep, rng, trials):
    """兩個維度各跑一次 Psi，反解 H / L，回代 DDM 驗證正確率。"""
    out = {}
    rows = []
    for dim in ("colour", "orientation"):
        x_range, _ = DIMS[dim]
        traj = run_psi(dim, trials, sep, rng)
        al, be = traj[-1]
        (H, Lo), warns = salience_levels(al, be, LAPSE, [P_HIGH, P_LOW], x_range=x_range)
        a_true, b_true, _ = true_alpha_beta(dim, sep)
        H_true, L_true = salience_levels(a_true, b_true, LAPSE, [P_HIGH, P_LOW])[0]
        chk = {}
        for name, x in (("H", H), ("L", Lo)):
            _, r = simdiffT(2000, A, scaled(x, dim) * V, SDV, TER, rng=rng, a_is_separation=sep)
            chk[name] = float(r.mean())
        rows.append({"convention": conv, "dim": dim, "alpha": al, "beta": be, "alpha_true": a_true, "beta_true": b_true,
                     "H": H, "L": Lo, "H_true": H_true, "L_true": L_true, "acc_H": chk["H"], "acc_L": chk["L"],
                     "range_max": x_range[1], "warnings": "; ".join(warns)})
        out[dim] = (H, Lo)
    print_table(rows, ["dim", "alpha", "beta", "alpha_true", "beta_true", "H", "H_true", "acc_H", "L", "L_true", "acc_L", "range_max"],
                fmt="{:>11}")
    for r in rows:
        if r["warnings"]:
            print(f"    ⚠ {r['dim']}: {r['warnings']}")
    return out, rows


def sec_dfp(args, conv, sep, rng, n=100):
    """:177-459：H / L 來自 Psi，五種架構各一場，dfptrials = 100（:252）。"""
    trials = 60 if args.quick else args.trials
    hl, cal_rows = calibrate(args, conv, sep, rng, trials)
    (hc, lc), (ho, lo) = hl["colour"], hl["orientation"]
    dfp = functools.partial(dfp_ddm, a=A, ter=TER, sdv=SDV, rng=rng, a_is_separation=sep)

    def call(n_, c1, c2, arch, rule):
        return dfp(n_, scaled(c1, "colour") * V, scaled(c2, "orientation") * V, architecture=arch, stopping_rule=rule)

    rows = []
    for arch, rule in ARCHS:
        # 兩維的 H/L 不同，不能用 run_dfp_cells 的單一 high/low；手動組四格
        cells, acc = {}, {}
        try:
            for name, (c1, c2) in dict(HH=(hc, ho), HL=(hc, lo), LH=(lc, ho), LL=(lc, lo)).items():
                rt, cr = call(n, c1, c2, arch, rule)
                cells[name] = rt[cr == 1]
                acc[name] = float(cr.mean())
        except RuntimeError as e:
            print(f"    ✘ {arch}-{rule or ''}: {e}（跳過）")
            continue
        row, s = sic_row(cells, acc, f"{arch}-{rule or ''}")
        row["convention"] = conv
        rows.append(row)
        plot_survivor_sic(cells, s, f"{arch} {rule or ''}  (a = {conv})",
                          os.path.join(outdir(args, "psi_dfp"), f"psi_{arch}_{rule or 'x'}_{conv}.png"))
    print_table(rows, ["condition", "Dplus", "p_Dplus", "Dminus", "p_Dminus", "MIC", "p_MIC", "SI", "predicted"])
    return rows, cal_rows


def sec_full_experiment(args, conv, sep, rng):
    """:460-1003：nParticipants 位 × 四種架構。每位受試者各自 Psi 校準一次（R 用讀進來的 csv，這裡直接跑）。"""
    n_p, n_cell, trials = (3, 60, 60) if args.quick else (args.n_participants, args.n_trials, args.trials)
    results = []
    for sn in range(1, n_p + 1):
        hl, cal = calibrate(args, conv, sep, rng, trials)
        (hc, lc), (ho, lo) = hl["colour"], hl["orientation"]
        for arch, rule in (("PAR", "OR"), ("PAR", "AND"), ("SER", "OR"), ("SER", "AND")):
            cells, acc = {}, {}
            try:
                for name, (c1, c2) in dict(HH=(hc, ho), HL=(hc, lo), LH=(lc, ho), LL=(lc, lo)).items():
                    rt, cr = dfp_ddm(n_cell, scaled(c1, "colour") * V, scaled(c2, "orientation") * V, A, TER, SDV,
                                     arch, rule, rng=rng, a_is_separation=sep)
                    cells[name] = rt[cr == 1]
                    acc[name] = float(cr.mean())
            except RuntimeError as e:
                print(f"    ✘ S{sn} {arch}.{rule}: {e}（跳過）")
                continue
            row, _ = sic_row(cells, acc, f"{arch}.{rule}")
            # 欄名照 psi Simulation_26MAR2019.R:583 的 SFTresults
            results.append({"convention": conv, "Subject": sn, "Condition": f"{arch}.{rule}",
                            "Threshold": A, "v": V, "ter": TER, "sdv": SDV,
                            "H_Color Intensity": hc, "L_Color Intensity": lc,
                            "H_Orientation Intensity": ho, "L_Orientation Intensity": lo,
                            "D+ Statistic": row["Dplus"], "D+ Pvalue": row["p_Dplus"],
                            "D- Statistic": row["Dminus"], "D- Pvalue": row["p_Dminus"],
                            "MIC Statistic": row["MIC"], "MIC Pvalue": row["p_MIC"],
                            "Predicted": row["predicted"], "SI": row["SI"],
                            **{f"{k}_Correct": acc[k] for k in ("HH", "HL", "LH", "LL")}})
    print_table(results, ["Subject", "Condition", "D+ Pvalue", "D- Pvalue", "MIC Pvalue", "Predicted", "SI"], fmt="{:>12}")
    return results


def main():
    ap = base_parser(__doc__)
    ap.add_argument("section", choices=("convergence", "dfp", "full-experiment", "all"))
    ap.add_argument("--nsamps", type=int, default=50, help="convergence 的模擬次數（R 用 629）")
    ap.add_argument("--trials", type=int, default=300, help="每次 Psi 的試次（R 用 300）")
    ap.add_argument("--n-participants", type=int, default=10)
    ap.add_argument("--n-trials", type=int, default=100, help="DFP 每格試次（R dfptrials = 100）")
    args = ap.parse_args()
    sections = ("convergence", "dfp", "full-experiment") if args.section == "all" else (args.section,)
    conv_rows, dfp_rows, cal_rows, exp_rows = [], [], [], []
    for conv, sep in conventions(args):
        print(f"\n=== a 慣例：{conv}（simdiffT 收到的 boundary separation = {'a' if sep else '2a'}）===")
        rng = np.random.default_rng(args.seed)
        if "convergence" in sections:
            conv_rows += sec_convergence(args, conv, sep, rng)
        if "dfp" in sections:
            r, c = sec_dfp(args, conv, sep, rng)
            dfp_rows += r
            cal_rows += c
        if "full-experiment" in sections:
            exp_rows += sec_full_experiment(args, conv, sep, rng)
    if conv_rows:
        write_csv(conv_rows, os.path.join(outdir(args, "psi_convergence"), "psi_convergence.csv"))
    if cal_rows:
        print("\n=== 兩種慣例的 Psi 校準對照 ===")
        print_table(cal_rows, ["convention", "dim", "beta", "beta_true", "H", "H_true", "acc_H", "L", "L_true", "acc_L", "range_max"],
                    fmt="{:>12}")
        write_csv(cal_rows, os.path.join(outdir(args, "psi_dfp"), "calibration_by_convention.csv"))
    if dfp_rows:
        write_csv(dfp_rows, os.path.join(outdir(args, "psi_dfp"), "sic_by_convention.csv"))
    if exp_rows:
        write_csv(exp_rows, os.path.join(outdir(args, "psi_full_experiment"), "Psi_Simulation_SFTresults.csv"))


if __name__ == "__main__":
    main()
