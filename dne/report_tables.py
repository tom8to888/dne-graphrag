"""Helpers that turn results/tables/*.csv into LaTeX tables for the report (numbers only, no text)."""
from __future__ import annotations

import pandas as pd

CRITERIA = ["comprehensiveness", "diversity", "empowerment", "directness"]


def latex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def marked_win_rates(win_rates: pd.DataFrame, sign_tests: pd.DataFrame) -> pd.DataFrame:
    """Win rate of the first system per pair and criterion, as text; '*' when both the sign test and
    the Wilcoxon test are significant after Holm correction."""
    sig = sign_tests.set_index(["pair", "criterion"])
    rows = {}
    for _, r in win_rates.iterrows():
        cells = []
        for c in CRITERIA:
            s = sig.loc[(r["pair"], c)]
            star = "*" if bool(s["significant"]) and bool(s["significant_wilcoxon"]) else ""
            cells.append(f"{r[c]:.2f}{star}")
        rows[r["pair"]] = cells
    return pd.DataFrame.from_dict(rows, orient="index", columns=CRITERIA)


def tabular(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    """A booktabs tabular; cells are used as given (escape them first if needed)."""
    align = align or "l" + "c" * (len(header) - 1)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)
