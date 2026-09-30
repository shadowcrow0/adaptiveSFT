"""
掃描：漂移差目標 (h_targ, l_targ) × 每格試次 × 五種架構 → SIC 判對率。

每位模擬受試者只擬合一次（兩維各一個 ogival LNRM），同一個後驗對每組目標各反解一次，
所以 3 組目標不用 3 倍的擬合時間。其餘照 simulateLNRM_ogival.R:470-585 的整場實驗流程。

    python scripts/power_scan.py                                   # 預設：8/1.3, 4/1, 2/.5 × 100, 250 試/格 × 10 人
    python scripts/power_scan.py --targets 6:1.5 3:0.8 --n-cell 150 --n-participants 20
    python scripts/power_scan.py --a 1.45 --v 1.6 --ter .1 --sdv .25   # 換成你自己作業的 DDM 參數

輸出：results/power_scan.csv（每列一個 目標 × n/格 × 架構）與印在終端的摘要表。
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
warnings.filterwarnings("ignore")

from adaptivesft.ddm import ddm_p_correct, dfp_ddm, draw_participant, moc_ddm   # noqa: E402
from adaptivesft.models import fit_lnrm                                            # noqa: E402
from adaptivesft.salience import find_salience                                     # noqa: E402
from adaptivesft.sic import classify, sic                                          # noqa: E402

ARCHS = (("COA", None, "Coactive"), ("PAR", "OR", "ParallelOR"), ("PAR", "AND", "ParallelAND"),
         ("SER", "OR", "SerialOR"), ("SER", "AND", "SerialAND"))
DIMS = {"orientation": ((45.0, 90.0), 63.0), "colour": ((-55.0, 50.0), 6.0)}   # simulateLNRM_ogival.R:35-36, :477-485


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=["8.0:1.3", "4.0:1.0", "2.0:0.5"], help="h_targ:l_targ …")
    ap.add_argument("--n-cell", nargs="+", type=int, default=[100, 250], help="DFP 每格試次")
    ap.add_argument("--n-participants", type=int, default=10)
    ap.add_argument("--a", type=float, default=3.0)
    ap.add_argument("--v", type=float, default=2.0)
    ap.add_argument("--ter", type=float, default=0.1)
    ap.add_argument("--sdv", type=float, default=0.2)
    ap.add_argument("--a-convention", default="separation", choices=("separation", "threshold"))
    ap.add_argument("--n-per-level", type=int, default=100)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--tune", type=int, default=800)
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--out", default="results/power_scan.csv")
    args = ap.parse_args()
    sep = args.a_convention == "separation"
    targets = [tuple(float(v) for v in t.split(":")) for t in args.targets]
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # 1. 每位受試者：抽參數、兩維各擬合一次、對每組目標反解
    people = []
    for sn in range(1, args.n_participants + 1):
        a_p, v_p, ter_p, sdv_p = draw_participant(args.a, args.v, args.ter, args.sdv, rng=rng)
        per_dim = {}
        for dim, (x_range, thres50) in DIMS.items():
            lv = np.linspace(x_range[0], x_range[1], 10)
            scaled = (lv - thres50) / (x_range[1] - thres50)
            data = moc_ddm(args.n_per_level, a_p, v_p, ter_p, sdv_p, scaled, rng=rng, a_is_separation=sep)
            tr = fit_lnrm(data, link="ogival", L=args.L, tune=args.tune, draws=args.draws, chains=4,
                          random_seed=args.seed + sn)
            per_dim[dim] = {}
            for h, l in targets:
                res = find_salience(tr, h_targ=h, l_targ=l)
                per_dim[dim][(h, l)] = dict(
                    high=res["high"]["intensity"], low=res["low"]["intensity"],
                    in_range=bool(np.isfinite(res["high"]["intensity"]) and scaled.min() <= res["high"]["intensity"] <= scaled.max()),
                    dropped=res["high"]["dropped"])
        people.append(dict(sn=sn, a=a_p, v=v_p, ter=ter_p, sdv=sdv_p, dims=per_dim))
        print(f"  S{sn:02d} fitted ({time.time() - t0:.0f} s)", flush=True)

    # 2. 每組目標 × 每格試次 × 架構：DFP → SIC → 判對？
    rows = []
    for h, l in targets:
        hl = [p["dims"] for p in people]
        n_in = sum(d["orientation"][(h, l)]["in_range"] and d["colour"][(h, l)]["in_range"] for d in hl)
        for n_cell in args.n_cell:
            for arch, rule, truth in ARCHS:
                correct, si_pass, acc_hh, acc_ll = 0, 0, [], []
                usable = 0
                for p in people:
                    do, dc = p["dims"]["orientation"][(h, l)], p["dims"]["colour"][(h, l)]
                    if not all(np.isfinite([do["high"], do["low"], dc["high"], dc["low"]])):
                        continue
                    usable += 1
                    cells = {}
                    for name, (c1, c2) in dict(HH=(dc["high"], do["high"]), HL=(dc["high"], do["low"]),
                                               LH=(dc["low"], do["high"]), LL=(dc["low"], do["low"])).items():
                        rt, cr = dfp_ddm(n_cell, c1 * p["v"], c2 * p["v"], p["a"], p["ter"], p["sdv"], arch, rule,
                                         rng=rng, a_is_separation=sep)
                        cells[name] = rt[cr == 1]
                        if name == "HH":
                            acc_hh.append(cr.mean())
                        if name == "LL":
                            acc_ll.append(cr.mean())
                    if min(len(v) for v in cells.values()) <= 10:
                        continue
                    c = classify(sic(**cells))
                    correct += c["Predicted_by"] == truth
                    si_pass += c["Selective.Influence"] == "Pass"
                rows.append(dict(h_targ=h, l_targ=l, n_cell=n_cell, arch=f"{arch}-{rule or ''}", truth=truth,
                                 n=usable, correct=correct, rate=correct / max(usable, 1), si_pass=si_pass,
                                 H_in_range=n_in, acc_HH=float(np.mean(acc_hh)) if acc_hh else np.nan,
                                 acc_LL=float(np.mean(acc_ll)) if acc_ll else np.nan))

    # 3. 摘要
    print(f"\n目標(H:L)  n/格   {'COA':>8}{'PAR-OR':>8}{'PAR-AND':>8}{'SER-OR':>8}{'SER-AND':>8}   H 在範圍內   acc HH / LL")
    for h, l in targets:
        for n_cell in args.n_cell:
            sub = {r["arch"]: r for r in rows if r["h_targ"] == h and r["l_targ"] == l and r["n_cell"] == n_cell}
            cells = "".join(f"{sub[k]['correct']:>4d}/{sub[k]['n']:<3d}" for k in ("COA-", "PAR-OR", "PAR-AND", "SER-OR", "SER-AND"))
            any_ = next(iter(sub.values()))
            print(f"{h:>4.1f}:{l:<4.1f}  {n_cell:>4d}   {cells}   {any_['H_in_range']:>2d}/{len(people)}       "
                  f"{any_['acc_HH']:.3f} / {any_['acc_LL']:.3f}")
    print(f"\n(判對 = sicGroup 決策表預測的架構 = 真實架構；DDM a={args.a} v={args.v} ter={args.ter} sdv={args.sdv}，"
          f"a 慣例 {args.a_convention}；{time.time() - t0:.0f} s)")
    import csv
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
