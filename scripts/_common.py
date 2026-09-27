"""兩支模擬腳本共用：命令列、輸出目錄、a 慣例迴圈、survivor + SIC 圖。"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptivesft.ddm import A_CONVENTIONS  # noqa: E402
from adaptivesft.sic import classify, ecdf, sic  # noqa: E402

ARCHS = (("COA", None), ("PAR", "OR"), ("PAR", "AND"), ("SER", "OR"), ("SER", "AND"))


def base_parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--a-convention", default="both", choices=("separation", "threshold", "both"),
                    help="a 的意義：separation = diffIRT 的 boundary separation（repo 字面）；"
                         "threshold = 起點到界線（psi Simulation_26MAR2019.R:117 的公式）；both = 兩種都跑")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--out", default="output", help="輸出目錄（gitignored）")
    ap.add_argument("--quick", action="store_true", help="小樣本快速跑，只驗證流程")
    return ap


def conventions(args):
    if args.a_convention == "both":
        return list(A_CONVENTIONS.items())
    return [(args.a_convention, A_CONVENTIONS[args.a_convention])]


def outdir(args, sub):
    d = os.path.join(args.out, sub)
    os.makedirs(d, exist_ok=True)
    return d


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        print(f"    [{self.label}] {time.time() - self.t0:.1f} s")


def run_dfp_cells(dfp, high, low, n, arch, rule, **kw):
    """四格 DFP → 答對試次 RT。dfp = adaptivesft.ddm.dfp_ddm 的偏函式（已綁 a, ter, sdv, rng, 慣例）。"""
    cells, acc = {}, {}
    for name, (c1, c2) in dict(HH=(high, high), HL=(high, low), LH=(low, high), LL=(low, low)).items():
        try:
            rt, cr = dfp(n, c1, c2, arch, rule, **kw)
        except RuntimeError as e:          # diffIRT 的拒絕抽樣在 a·drift 太大時會失敗（R 也一樣會 stop）
            print(f"    ✘ {arch}-{rule or ''} {name}: {e}  （drift {c1:.2f}, {c2:.2f} 太大，跳過此架構）")
            return None, None
        cells[name] = rt[cr == 1]
        acc[name] = float(cr.mean())
    return cells, acc


def sic_row(cells, acc, label):
    s = sic(**cells)
    c = classify(s)
    return {
        "condition": label,
        "Dplus": s["SICtest"]["positive"][0], "p_Dplus": s["SICtest"]["positive"][1],
        "Dminus": s["SICtest"]["negative"][0], "p_Dminus": s["SICtest"]["negative"][1],
        "MIC": s["MICtest"]["statistic"], "p_MIC": s["MICtest"]["p_value"],
        "SI": c["Selective.Influence"], "predicted": c["Predicted_by"],
        **{f"acc_{k}": v for k, v in acc.items()},
        **{f"n_{k}": len(v) for k, v in cells.items()},
    }, s


def plot_survivor_sic(cells, s, title, path, tmax=5.0):
    """simulateLNRM_ogival.R:322-345 / psi Simulation_26MAR2019.R:541-559 的兩欄圖。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(0, tmax, 2000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6))
    for name, col in (("HH", "red"), ("HL", "orange"), ("LH", "purple"), ("LL", "blue")):
        ax1.plot(t, 1 - ecdf(cells[name])(t), color=col, label=name)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("S(t)")
    ax1.legend()
    times, vals = s["SIC"]
    ax2.step(times, vals, where="post")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlim(0, tmax)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("SIC(t)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_csv(rows, path):
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"    wrote {path}")


def print_table(rows, cols, fmt="{:>10}"):
    print("    " + "".join(fmt.format(c) for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(fmt.format(f"{v:.3f}" if isinstance(v, float) else str(v)))
        print("    " + "".join(cells))
