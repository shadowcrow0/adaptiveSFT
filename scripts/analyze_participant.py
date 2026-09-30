"""
一位受試者的 DFP 資料 → SIC / MIC / dominance → 預測架構。

    python scripts/analyze_participant.py data/S01_dfp.csv
    python scripts/analyze_participant.py data/S01_dfp.csv --json results/S01.json --plot results/S01.png

csv 欄位照 adaptivesft.experiment.dfp_trial_list（subject, block, trial, condition, channel1, channel2,
c1_level, c2_level, correct, rt）；channel = 2 高 salience、1 低；rt 秒；correct 1/0。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptivesft.experiment import analyze_participant, read_rows, report   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--rt-min", type=float, default=0.15)
    ap.add_argument("--rt-max", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--json", help="把結果存成 json")
    ap.add_argument("--plot", help="survivor + SIC 圖（png）")
    args = ap.parse_args(argv)

    rows = read_rows(args.csv)
    res = analyze_participant(rows, rt_min=args.rt_min, rt_max=args.rt_max, alpha_sic=args.alpha)
    print(f"{args.csv}: {len(rows)} rows")
    print(report(res))

    if args.json:
        payload = {}
        for cond, r in res.items():
            payload[cond] = {k: v for k, v in r.items() if k != "sic"}
            if "dominance" in r:
                payload[cond]["dominance"] = [list(x) for x in r["dominance"]]
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
        print("wrote", args.json)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from adaptivesft.sic import ecdf
        conds = [c for c, r in res.items() if "sic" in r]
        fig, axes = plt.subplots(len(conds), 2, figsize=(8, 3.6 * len(conds)), squeeze=False)
        for ax_row, cond in zip(axes, conds):
            r = res[cond]
            s = r["sic"]
            tmax = max(v.max() for v in (s["SIC"][0],)) * 1.05
            t = np.linspace(0, tmax, 1000)
            cells = {k: np.array([]) for k in ("HH", "HL", "LH", "LL")}
            for k in cells:
                cells[k] = np.array([rt for rt in _cell_rts(rows, cond, k, args.rt_min, args.rt_max)])
            for name, col in (("HH", "red"), ("HL", "orange"), ("LH", "purple"), ("LL", "blue")):
                ax_row[0].plot(t, 1 - ecdf(cells[name])(t), color=col, label=name)
            ax_row[0].set_xlabel("Time (s)")
            ax_row[0].set_ylabel("S(t)")
            ax_row[0].legend()
            ax_row[1].step(s["SIC"][0], s["SIC"][1], where="post")
            ax_row[1].axhline(0, color="k", lw=0.8)
            ax_row[1].set_ylim(-0.5, 0.5)
            ax_row[1].set_xlabel("Time (s)")
            ax_row[1].set_ylabel("SIC(t)")
            ax_row[1].set_title(f"{cond}: {r['classification']['Predicted_by']}")
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        fig.savefig(args.plot, dpi=130)
        print("wrote", args.plot)
    return 0


def _cell_rts(rows, cond, name, rt_min, rt_max):
    c1, c2 = dict(HH=(2, 2), HL=(2, 1), LH=(1, 2), LL=(1, 1))[name]
    for r in rows:
        try:
            rt, c = float(r["rt"]), int(float(r["correct"]))
        except (TypeError, ValueError):
            continue
        if r["condition"] == cond and int(r["channel1"]) == c1 and int(r["channel2"]) == c2 and c == 1 and rt_min <= rt <= rt_max:
            yield rt


if __name__ == "__main__":
    raise SystemExit(main())
