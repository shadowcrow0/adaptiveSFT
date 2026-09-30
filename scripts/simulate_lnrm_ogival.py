"""
simulateLNRM_ogival.R 的 Python 版，四段各一個 --section，兩種 a 慣例可並排跑。

    salience         :1-81     MOC 資料 → 擬合 ogival LNRM → 反解 H / L（h_targ 8.0, l_targ 1.3）
    convergence      :83-140   每層 N 試的參數收斂（取代遺失的 post95.Rdata；N 網格稀疏化，plan D8）
    ppc              :142-300  後驗預測：各層 ecdf vs plognormalrace（varZ 當 SD，不像 R :204/213 多開根號）
    single-dfp       :308-468  五種架構各一場乾淨 DFP → survivor + SIC 圖
    full-experiment  :470-585  n 位受試者 → 兩維各擬合 → DFP → sicGroup

用法：
    python scripts/simulate_lnrm_ogival.py salience --a-convention both
    python scripts/simulate_lnrm_ogival.py full-experiment --n-participants 10 --arch PAR --rule AND
    python scripts/simulate_lnrm_ogival.py convergence --quick

預設參數全部照 simulateLNRM_ogival.R:19-40（a=3, v=2, ter=.1, sdv=.2, 10 層 × 100 試，
方位範圍 45-90、thres50 63；顏色範圍 −55-50、thres50 6；L=10）。
"""
import functools
import os

import numpy as np

from _common import (ARCHS, Timer, base_parser, conventions, outdir, plot_survivor_sic, print_table,
                     run_dfp_cells, sic_row, write_csv)
from adaptivesft.ddm import ddm_p_correct, dfp_ddm, draw_participant, moc_ddm
from adaptivesft.models import fit_lnrm, fit_lnrm0_by_level, d_numpy
from adaptivesft.race import plognormalrace_curve
from adaptivesft.salience import find_salience, summarize, targ_to_accuracy
from adaptivesft.sic import sic_group

# simulateLNRM_ogival.R:24-40
L_TARG, H_TARG, L = 1.3, 8.0, 10.0
A, V, TER, SDV = 3.0, 2.0, 0.1, 0.20
N_PER_LEVEL, N_LEVELS = 100, 10
DIMS = {"orientation": ((45.0, 90.0), 63.0), "colour": ((-55.0, 50.0), 6.0)}   # :35-36, :477-485


def scaled_levels(x_range, thres50, n_levels=N_LEVELS):
    """:38-39：strength = (x − thres50) / (x_max − thres50)。"""
    lv = np.linspace(x_range[0], x_range[1], n_levels)
    return lv, (lv - thres50) / (x_range[1] - thres50)


def fit_kwargs(args):
    return dict(tune=args.tune, draws=args.draws, chains=args.chains, random_seed=args.seed, sampler=args.sampler)


def sec_salience(args, conv, sep, rng, dim="orientation"):
    x_range, thres50 = DIMS[dim]
    levels, scaled = scaled_levels(x_range, thres50)
    data = moc_ddm(N_PER_LEVEL, A, V, TER, SDV, scaled, rng=rng, a_is_separation=sep)
    print(f"  資料：{len(data)} 試，整體正確率 {data[:, 1].mean():.3f}，"
          f"最高層理論正確率 {ddm_p_correct(A, scaled[-1] * V, sep):.3f}")
    with Timer("fit ogival"):
        tr = fit_lnrm(data, link="ogival", L=L, **fit_kwargs(args))
    res = find_salience(tr, h_targ=H_TARG, l_targ=L_TARG)
    print(summarize(res))
    high, low = res["high"]["intensity"], res["low"]["intensity"]
    if np.isfinite(high):
        print(f"  回代 DDM：H={high:.3f}(scaled) → P(correct)={ddm_p_correct(A, high * V, sep):.3f}   "
              f"L={low:.3f} → {ddm_p_correct(A, low * V, sep):.3f}")
        for name, val in (("H", high), ("L", low)):
            if not scaled.min() <= val <= scaled.max():
                print(f"  ⚠ {name} = {val:.3f} 落在校準的 scaled 範圍 [{scaled.min():.2f}, {scaled.max():.2f}] 之外"
                      f"（物理單位 {thres50 + val * (x_range[1] - thres50):.1f}，範圍 {x_range}）")
    # 心理計量圖：各層觀察正確率 vs 模型 P(correct) = Φ(2d / (varZ√2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    post = {k: tr.posterior[k].values.ravel() for k in ("slope", "midpoint", "varZ")}
    xx = np.linspace(scaled.min(), scaled.max(), 100)
    med = [np.median(targ_to_accuracy(2 * d_numpy("ogival", x, post, L), post["varZ"])) for x in xx]
    obs = [data[data[:, 2] == s, 1].mean() for s in scaled]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(xx, med, label="LNRM (ogival)")
    ax.scatter(scaled, obs, color="k", zorder=5, label="DDM data")
    ax.plot(xx, ddm_p_correct(A, xx * V, sep), ls=":", color="grey", label="DDM P(correct)")
    for val, col, lab in ((high, "green", "H"), (low, "orange", "L")):
        if np.isfinite(val):
            ax.axvline(val, ls="--", color=col, label=lab)
    ax.set_xlabel("scaled intensity")
    ax.set_ylabel("P(correct)")
    ax.set_title(f"{dim}, a = {conv}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir(args, "salience"), f"psychometric_{dim}_{conv}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"    wrote {path}")
    return {"convention": conv, "dim": dim, "high": high, "low": low,
            "acc_high": float(ddm_p_correct(A, high * V, sep)) if np.isfinite(high) else np.nan,
            "acc_low": float(ddm_p_correct(A, low * V, sep)) if np.isfinite(low) else np.nan,
            "dropped_high": res["high"]["dropped"], "dropped_low": res["low"]["dropped"],
            "slope": float(post["slope"].mean()), "midpoint": float(post["midpoint"].mean()),
            "varZ": float(post["varZ"].mean())}, tr, data, (levels, scaled)


def sec_convergence(args, conv, sep, rng):
    """:83-140。R 跑 N = 1…300 每個都擬合一次；這裡預設 log 間距 8 個點（--full 才 1…300）。"""
    x_range, thres50 = DIMS["orientation"]
    _, scaled = scaled_levels(x_range, thres50)
    Ns = [3, 5, 10, 20, 40, 80, 160, 300] if not args.quick else [5, 20, 60]
    if args.full:
        Ns = list(range(1, 301))
    from joblib import Parallel, delayed

    def one(N):
        r = np.random.default_rng(args.seed + N)
        data = moc_ddm(N, A, V, TER, SDV, scaled, rng=r, a_is_separation=sep)
        tr = fit_lnrm(data, link="ogival", L=L, chains=4, tune=args.tune, draws=args.draws,
                      random_seed=args.seed, cores=1, sampler=args.sampler)
        row = {"convention": conv, "N_per_level": N}
        for k in ("midpoint", "slope"):
            v = tr.posterior[k].values.ravel()
            row[f"{k}_mean"] = float(v.mean())
            row[f"{k}_q05"], row[f"{k}_q95"] = (float(q) for q in np.quantile(v, [0.05, 0.95]))
        return row

    with Timer(f"convergence {len(Ns)} fits"):
        rows = Parallel(n_jobs=args.jobs)(delayed(one)(N) for N in Ns)
    print_table(rows, ["N_per_level", "midpoint_mean", "midpoint_q05", "midpoint_q95", "slope_mean", "slope_q05", "slope_q95"],
                fmt="{:>14}")
    return rows


def plot_convergence(all_rows, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, k, title in zip(axes, ("midpoint", "slope"), ("Drift Difference Midpoint", "Drift Difference Slope")):
        for conv in sorted({r["convention"] for r in all_rows}):
            rows = sorted((r for r in all_rows if r["convention"] == conv), key=lambda r: r["N_per_level"])
            N = np.array([r["N_per_level"] for r in rows], float)
            for q in ("q05", "q95"):
                y = np.array([r[f"{k}_{q}"] for r in rows])
                ax.plot(N, y, ls="--", lw=0.8, label=f"{conv} {q}")
                if len(N) >= 4:
                    sm = lowess(y, N, frac=0.5, return_sorted=True)      # R :128-135 的 loess(span=.5)
                    ax.plot(sm[:, 0], sm[:, 1], lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("Trials per level")
        ax.set_title(title)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("5/95% Posterior Quantiles")
    fig.suptitle("LNRM Parameter Estimates")
    fig.tight_layout()
    path = os.path.join(outdir(args, "convergence"), "LNRM_parameter-convergence.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"    wrote {path}")


def sec_ppc(args, conv, sep, rng, tr, data, levels_scaled, n_draws=20):
    """:142-300。各層：資料 ecdf（答對 / 答錯，各乘其比例）vs 後驗子樣本的 plognormalrace。
    --fit-separate 時另做 :146-170（fit.separate）：每層各自擬合無強度項的賽跑（lnrm0），
    把各層的 d 畫在 ogival 曲線上對照。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, scaled = levels_scaled
    if args.fit_separate:
        with Timer("fit.separate: lnrm0 per level"):
            per_level = fit_lnrm0_by_level(data, tune=args.tune, draws=args.draws, chains=min(args.chains, 4),
                                           random_seed=args.seed, progressbar=False)
        post_o = {k: tr.posterior[k].values.ravel() for k in ("slope", "midpoint")}
        xx = np.linspace(scaled.min(), scaled.max(), 100)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.scatter([lv for lv, _, _ in per_level], [dd for _, dd, _ in per_level], color="k", zorder=5,
                   label="lnrm0 per level (R :150-162)")
        ax.plot(xx, [np.median(d_numpy("ogival", v, post_o, L)) for v in xx], label="ogival (lnrm2a)")
        ax.set_xlabel("scaled intensity")
        ax.set_ylabel("d = (z2 - z1) / 2")
        ax.set_title(f"fit.separate, a = {conv}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(outdir(args, "ppc"), f"fit_separate_{conv}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"    wrote {path}")
        print("    " + "  ".join(f"x={lv:.2f}: d={dd:.3f}" for lv, dd, _ in per_level))
    post = {k: tr.posterior[k].values.ravel() for k in ("mu", "slope", "midpoint", "varZ", "psi")}
    idx = rng.choice(post["mu"].size, n_draws, replace=False)          # R 的 rsamp
    tvec = np.linspace(0, 5, 200)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
    for ax, s in zip(axes.ravel(), scaled[:8]):                          # R :194 只畫前 8 層
        sub = data[data[:, 2] == s]
        cor, inc = sub[sub[:, 1] == 1, 0], sub[sub[:, 1] == 0, 0]
        for i in idx:
            d = d_numpy("ogival", s, {k: post[k][i] for k in ("slope", "midpoint")}, L)
            mux = np.array([post["mu"][i] - d, post["mu"][i] + d])     # R :206-209 的 mux
            sd = [post["varZ"][i]] * 2                                   # varZ 是 SD（不像 R :213 塞進 sigmasq）
            ax.plot(tvec, plognormalrace_curve(tvec, 0, post["psi"][i], mux, sd), color="0.8", lw=0.6)
            ax.plot(tvec, plognormalrace_curve(tvec, 1, post["psi"][i], mux, sd), color="0.8", lw=0.6)
        from adaptivesft.sic import ecdf
        ax.plot(tvec, ecdf(cor)(tvec) * len(cor) / len(sub), color="green", lw=2, label="correct")
        if len(inc):
            ax.plot(tvec, ecdf(inc)(tvec) * len(inc) / len(sub), color="red", lw=2, label="incorrect")
        ax.set_title(f"intensity {s:.2f}", fontsize=9)
        ax.set_ylim(0, 1)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"posterior predictive, a = {conv}")
    fig.tight_layout()
    path = os.path.join(outdir(args, "ppc"), f"ppc_{conv}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"    wrote {path}")


def sec_single_dfp(args, conv, sep, rng, high, low, n=250):
    """:308-468：五種架構各 250/cell。"""
    dfp = functools.partial(dfp_ddm, a=A, ter=TER, sdv=SDV, rng=rng, a_is_separation=sep)

    def call(n_, c1, c2, arch, rule):
        return dfp(n_, c1 * V, c2 * V, architecture=arch, stopping_rule=rule)

    rows = []
    for arch, rule in ARCHS:
        cells, acc = run_dfp_cells(call, high, low, n, arch, rule)
        if cells is None:
            continue
        row, s = sic_row(cells, acc, f"{arch}-{rule or ''}")
        row["convention"] = conv
        rows.append(row)
        plot_survivor_sic(cells, s, f"{arch} {rule or ''}  (a = {conv})",
                          os.path.join(outdir(args, "single_dfp"), f"lnrm_{arch}_{rule or 'x'}_{conv}.png"))
    print_table(rows, ["condition", "Dplus", "p_Dplus", "Dminus", "p_Dminus", "MIC", "p_MIC", "SI", "predicted"])
    return rows


def sec_full_experiment(args, conv, sep, rng, group_hl=None):
    """:470-585：n 位受試者，各自的 (a, v, ter, sdv)，兩維各擬合一次，DFP，sicGroup。
    group_hl=(high, low)：另跑 R 的 sft.allx 對照組（:549-565）——每個人都用同一組群體 H/L，不個別校準。"""
    n_p, n_trials = (3, 60) if args.quick else (args.n_participants, args.n_trials)
    long = {k: [] for k in ("subject", "condition", "channel1", "channel2", "correct", "rt")}
    longx = {k: [] for k in long}
    pars, sal = [], []
    for sn in range(1, n_p + 1):
        a_p, v_p, ter_p, sdv_p = draw_participant(A, V, TER, SDV, rng=rng)
        pars.append({"subject": sn, "a": a_p, "v": v_p, "ter": ter_p, "sdv": sdv_p})
        hl = {}
        for dim in ("orientation", "colour"):
            x_range, thres50 = DIMS[dim]
            _, scaled = scaled_levels(x_range, thres50)
            data = moc_ddm(N_PER_LEVEL, a_p, v_p, ter_p, sdv_p, scaled, rng=rng, a_is_separation=sep)
            with Timer(f"S{sn} {dim} fit"):
                tr = fit_lnrm(data, link="ogival", L=L, **fit_kwargs(args))
            res = find_salience(tr, h_targ=H_TARG, l_targ=L_TARG)
            hl[dim] = (res["high"]["intensity"], res["low"]["intensity"])
            sal.append({"convention": conv, "subject": sn, "dim": dim, "high": hl[dim][0], "low": hl[dim][1],
                        "dropped_high": res["high"]["dropped"], "dropped_low": res["low"]["dropped"]})
        (hc, lc), (ho, lo) = hl["colour"], hl["orientation"]
        if not all(np.isfinite([hc, lc, ho, lo])):
            print(f"  S{sn}: 反解無解，跳過 DFP")
            continue
        try:
            trials4 = [dfp_ddm(n_trials, d1 * v_p, d2 * v_p, a_p, ter_p, sdv_p, args.arch, args.rule, rng=rng,
                               a_is_separation=sep)
                       for d1, d2 in ((hc, ho), (hc, lo), (lc, ho), (lc, lo))]
        except RuntimeError as e:
            print(f"  S{sn}: DDM 拒絕抽樣失敗（{e}），H/L 漂移太大，跳過")
            continue
        for (c1, c2), (rt, cr) in zip(((2, 2), (2, 1), (1, 2), (1, 1)), trials4):
            long["subject"] += [sn] * n_trials
            long["condition"] += [f"{args.arch}.{args.rule}"] * n_trials
            long["channel1"] += [c1] * n_trials
            long["channel2"] += [c2] * n_trials
            long["correct"] += cr.tolist()
            long["rt"] += rt.tolist()
        if group_hl is not None:                                            # sft.allx：群體 H/L
            gh, gl = group_hl
            for (c1, c2), (d1, d2) in zip(((2, 2), (2, 1), (1, 2), (1, 1)), ((gh, gh), (gh, gl), (gl, gh), (gl, gl))):
                rt, cr = dfp_ddm(n_trials, d1 * v_p, d2 * v_p, a_p, ter_p, sdv_p, args.arch, args.rule, rng=rng,
                                 a_is_separation=sep)
                longx["subject"] += [sn] * n_trials
                longx["condition"] += [f"{args.arch}.{args.rule}.groupHL"] * n_trials
                longx["channel1"] += [c1] * n_trials
                longx["channel2"] += [c2] * n_trials
                longx["correct"] += cr.tolist()
                longx["rt"] += rt.tolist()
    overview, _ = sic_group(**long)
    for r in overview:
        r["convention"] = conv
    print_table(overview, ["Subject", "Selective.Influence", "Positive.SIC", "Negative.SIC", "MIC", "Predicted_by"],
                fmt="{:>16}")
    if group_hl is not None and longx["rt"]:
        ovx, _ = sic_group(**longx)
        for r in ovx:
            r["convention"] = conv
        print("    對照組（同一組群體 H/L，R 的 sft.allx）：")
        print_table(ovx, ["Subject", "Selective.Influence", "Positive.SIC", "Negative.SIC", "MIC", "Predicted_by"],
                    fmt="{:>16}")
        overview += ovx
    # printsft（:588-682）的角色：一行摘要
    n_ok = sum(1 for r in overview if r["Predicted_by"] == {"PAR.OR": "ParallelOR", "PAR.AND": "ParallelAND",
                                                              "SER.OR": "SerialOR", "SER.AND": "SerialAND"}.get(f"{args.arch}.{args.rule}"))
    print(f"    {args.arch}-{args.rule}: {n_ok}/{len(overview)} 位判成正確架構")
    return overview, pars, sal


def main():
    ap = base_parser(__doc__)
    ap.add_argument("section", choices=("salience", "convergence", "ppc", "single-dfp", "full-experiment", "all"))
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--full", action="store_true", help="convergence：N = 1…300 全跑（R 原設定，很慢）")
    ap.add_argument("--fit-separate", action="store_true", help="ppc：另做 R 的 fit.separate（每層各擬合一個 lnrm0）")
    ap.add_argument("--sampler", default="nuts", choices=("nuts", "demetropolisz"))
    ap.add_argument("--n-participants", type=int, default=10)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--arch", default="PAR")
    ap.add_argument("--rule", default="AND")
    args = ap.parse_args()
    if args.quick:
        args.tune, args.draws, args.chains = 500, 500, 4

    sections = ("salience", "single-dfp", "ppc", "convergence", "full-experiment") if args.section == "all" else (args.section,)
    summary, conv_rows, dfp_rows, exp_rows = [], [], [], []
    for conv, sep in conventions(args):
        print(f"\n=== a 慣例：{conv}（simdiffT 收到的 boundary separation = {'a' if sep else '2a'}）===")
        rng = np.random.default_rng(args.seed)
        need_fit = {"salience", "ppc", "single-dfp"} & set(sections)
        if need_fit:
            row, tr, data, ls = sec_salience(args, conv, sep, rng)
            summary.append(row)
        if "ppc" in sections:
            sec_ppc(args, conv, sep, rng, tr, data, ls)
        if "single-dfp" in sections:
            if np.isfinite(row["high"]) and np.isfinite(row["low"]):
                dfp_rows += sec_single_dfp(args, conv, sep, rng, row["high"], row["low"])
            else:
                print("  反解無解，跳過 single-dfp")
        if "convergence" in sections:
            conv_rows += sec_convergence(args, conv, sep, rng)
        if "full-experiment" in sections:
            group_hl = (row["high"], row["low"]) if need_fit and np.isfinite(row["high"]) and np.isfinite(row["low"]) else None
            ov, pars, sal = sec_full_experiment(args, conv, sep, rng, group_hl=group_hl)
            exp_rows += ov
            write_csv(pars, os.path.join(outdir(args, "full_experiment"), f"participants_{conv}.csv"))
            write_csv(sal, os.path.join(outdir(args, "full_experiment"), f"salience_{conv}.csv"))
    if summary:
        print("\n=== 兩種慣例的 H / L 對照 ===")
        print_table(summary, ["convention", "high", "acc_high", "low", "acc_low", "dropped_high", "dropped_low", "slope", "midpoint", "varZ"],
                    fmt="{:>13}")
        write_csv(summary, os.path.join(outdir(args, "salience"), "salience_by_convention.csv"))
    if dfp_rows:
        write_csv(dfp_rows, os.path.join(outdir(args, "single_dfp"), "sic_by_convention.csv"))
    if conv_rows:
        write_csv(conv_rows, os.path.join(outdir(args, "convergence"), "post95.csv"))
        plot_convergence(conv_rows, args)
    if exp_rows:
        write_csv(exp_rows, os.path.join(outdir(args, "full_experiment"), "sicGroup_overview.csv"))


if __name__ == "__main__":
    main()
