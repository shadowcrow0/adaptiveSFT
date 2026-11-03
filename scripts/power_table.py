"""把 power_scan.py 的 csv 印成「目標 × 每格試次 → 各架構判對率」的表。

    python scripts/power_table.py results/power_scan_fine.csv
    python scripts/power_table.py results/power_scan.csv --arch COA-      # 只看 coactive

欄位意思：rate = 判對的比例（correct / n）；H_in_range = 反解的 H 落在刺激範圍內的人數。
"""
import argparse
import csv
from collections import OrderedDict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--arch", nargs="*", help="只印這些架構（COA- PAR-OR PAR-AND SER-OR SER-AND）")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    archs = a.arch or list(OrderedDict((r["arch"], 1) for r in rows))
    targets = list(OrderedDict(((r["h_targ"], r["l_targ"]), 1) for r in rows))
    cells = sorted({int(r["n_cell"]) for r in rows})
    for h, l in targets:
        sub = [r for r in rows if (r["h_targ"], r["l_targ"]) == (h, l)]
        n = max(int(r["n"]) for r in sub)
        hin = {int(r["n_cell"]): r["H_in_range"] for r in sub}
        print(f"\n目標 H/L = {h} / {l}   （每格 {n} 位模擬受試者；H 在範圍內：{hin[cells[0]]}/{n}）")
        print(f"  {'每格 n':>7}  " + "  ".join(f"{x:>8}" for x in archs))
        for c in cells:
            vals = {r["arch"]: float(r["rate"]) for r in sub if int(r["n_cell"]) == c}
            print(f"  {c:>7}  " + "  ".join(f"{vals.get(x, float('nan')):>8.2f}" for x in archs))
    print("\nrate = 判對比例。coactive 要 D+、D−、MIC 三個檢定同時顯著才算判對。")


if __name__ == "__main__":
    main()
