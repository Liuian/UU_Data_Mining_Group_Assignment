
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.stats.contingency_tables import mcnemar


MERGED_FILE = r"C:\Users\ankit\Desktop\Data Mining\Assignment\merged_flipped.csv" 
ID_COL, TRUTH_COL = "id", "true_label"
ALPHA = 0.05
USE_BONFERRONI = True


BETTER, WORSE, NO_DIFF, DIAG = "better", "worse", ".", "-"
LEGEND = {BETTER: "✓", WORSE: "✗", NO_DIFF: "·", DIAG: "—"}


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


df = pd.read_csv(MERGED_FILE)
pred_cols = [c for c in df.columns if c not in {ID_COL, TRUTH_COL}]
if len(pred_cols) < 2:
    raise ValueError("Need at least 2 prediction columns in the merged file.")


MAP = {"Truthful": 0, "Deceptive": 1, "truthful": 0, "deceptive": 1, "0": 0, "1": 1}
def to01(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.map(lambda x: MAP.get(str(x), x))
    return s.astype(int)

df[TRUTH_COL] = to01(df[TRUTH_COL])
for c in pred_cols:
    df[c] = to01(df[c])


num_pairs = len(pred_cols) * (len(pred_cols) - 1) // 2
alpha_eff = ALPHA / num_pairs if (USE_BONFERRONI and num_pairs > 0) else ALPHA


sym = pd.DataFrame(NO_DIFF, index=pred_cols, columns=pred_cols)
pvals = pd.DataFrame(np.nan, index=pred_cols, columns=pred_cols)
n10_mat = pd.DataFrame(0, index=pred_cols, columns=pred_cols)  # A right, B wrong
n01_mat = pd.DataFrame(0, index=pred_cols, columns=pred_cols)  # A wrong, B right

for a, b in combinations(pred_cols, 2):
    a_ok = df[a] == df[TRUTH_COL]
    b_ok = df[b] == df[TRUTH_COL]
    n01 = int(((~a_ok) &  b_ok).sum())   # A wrong, B right
    n10 = int(( a_ok   & (~b_ok)).sum()) # A right, B wrong
    exact = (n01 + n10) < 25
    res = mcnemar([[0, n01], [n10, 0]], exact=exact, correction=not exact)
    p = float(res.pvalue)

    pvals.loc[a, b] = p
    pvals.loc[b, a] = p
    n10_mat.loc[a, b] = n10
    n10_mat.loc[b, a] = n01  # swap perspective
    n01_mat.loc[a, b] = n01
    n01_mat.loc[b, a] = n10  # swap perspective

    if p < alpha_eff:
        if n10 > n01:
            sym.loc[a, b] = BETTER
            sym.loc[b, a] = WORSE
        elif n01 > n10:
            sym.loc[a, b] = WORSE
            sym.loc[b, a] = BETTER
        else:
            sym.loc[a, b] = sym.loc[b, a] = NO_DIFF
    else:
        sym.loc[a, b] = sym.loc[b, a] = NO_DIFF

for m in pred_cols:
    sym.loc[m, m] = DIAG
    pvals.loc[m, m] = 0.0
    n10_mat.loc[m, m] = 0
    n01_mat.loc[m, m] = 0


row_better = (sym == BETTER).sum(axis=1)
row_worse  = (sym == WORSE).sum(axis=1)
score = row_better - row_worse
order = score.sort_values(ascending=False).index.tolist()

sym = sym.loc[order, order]
pvals = pvals.loc[order, order]
n10_mat = n10_mat.loc[order, order]
n01_mat = n01_mat.loc[order, order]
row_better = row_better.loc[order]
row_worse = row_worse.loc[order]
score = score.loc[order]


sym_disp = sym.replace(LEGEND)


print("\nModel vs Model (ROW vs COLUMN)")
print("  ✓  = ROW significantly better than COLUMN")
print("  ✗  = ROW significantly worse than COLUMN")
print("  ·  = no significant difference")
print("  —  = same model (diagonal)")
print(f"  α = {ALPHA:.4f}  ({'Bonferroni ' if USE_BONFERRONI else ''}α_eff = {alpha_eff:.4f})\n")

print("Ranking (wins - losses):")
for m in order:
    print(f"  {m}: {int(row_better[m])} - {int(row_worse[m])} (score={int(score[m])})")

print("\nPairwise matrix:")
print(sym_disp.to_string())

sym.to_csv("mcnemar_better_matrix.csv", encoding="utf-8-sig")          
sym_disp.to_csv("mcnemar_better_matrix_pretty.csv", encoding="utf-8-sig")  
pvals.to_csv("mcnemar_pvalues.csv", encoding="utf-8-sig")


html_df = sym_disp.copy()


tooltips = pd.DataFrame("", index=html_df.index, columns=html_df.columns)
for r in html_df.index:
    for c in html_df.columns:
        if r == c:
            tip = "same model"
        else:
            tip = f"p={pvals.loc[r,c]:.4g}, n10={n10_mat.loc[r,c]} (row right, col wrong), n01={n01_mat.loc[r,c]} (row wrong, col right)"
        tooltips.loc[r, c] = tip

def color_cell(val: str) -> str:
    if val == LEGEND[BETTER]:
        return "background-color:#d9f2d9; color:#114411; font-weight:600;"   # green
    if val == LEGEND[WORSE]:
        return "background-color:#f8d6d6; color:#661111; font-weight:600;"   # red
    if val == LEGEND[NO_DIFF]:
        return "background-color:#f0f0f0; color:#333333;"                    # light gray
    if val == LEGEND[DIAG]:
        return "background-color:#000000; color:#ffffff; font-weight:600;"   # black
    return ""

styler = (html_df.style
          .applymap(color_cell)
          .set_properties(**{"text-align": "center", "white-space": "nowrap", "font-family": "Segoe UI, Arial, sans-serif"})
          .set_table_styles([
              {"selector": "th", "props": [("position", "sticky"), ("top", "0"), ("background", "#ffffff"), ("z-index", "2")]},
              {"selector": "th.col_heading", "props": [("writing-mode", "horizontal-tb")]},
              {"selector": "td, th", "props": [("padding", "6px 10px"), ("border", "1px solid #ddd")]}
          ])
          .set_caption(f"McNemar Pairwise (ROW vs COLUMN) — ✓ row better, ✗ row worse, · no diff, — same | α={ALPHA:.4f} ({'Bonferroni ' if USE_BONFERRONI else ''}α_eff={alpha_eff:.4f})")
         )


try:
    styler = styler.set_tooltips(tooltips)
except Exception:
    pass  

html_path = "mcnemar_matrix.html"
styler.to_html(html_path)

print("\nSaved files:")
print("  mcnemar_better_matrix.csv")
print("  mcnemar_better_matrix_pretty.csv")
print("  mcnemar_pvalues.csv")
print(f"  {html_path}  (open this in a browser for a colored, tooltip-rich table)")
