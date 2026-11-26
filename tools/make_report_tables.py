"""results/tables/*.csv -> report/tables/*.tex (booktabs tabulars; captions live in report/main.tex).

    python tools/make_report_tables.py

Re-run after notebook 05 changes. Every table in the report is written here. Most numbers come from
results/tables/*.csv; the compute, prompt-comparison and graph-size tables (sections 6, 12, 13) are
copied from the Kaggle run logs, with the source noted at each table.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dne.paths import ROOT  # noqa: E402
from dne.report_tables import CRITERIA, latex_escape, marked_win_rates, tabular  # noqa: E402

T = ROOT / "results/tables"
OUT = ROOT / "report/tables"
OUT.mkdir(parents=True, exist_ok=True)
CRIT_HEAD = ["Compr.", "Divers.", "Empow.", "Direct."]
NAMES = {"basic": "basic (12k)", "local": "local (12k)", "global": "global (12k)",
         "basic_8k": "basic (8k)", "global_8k": "global (8k)"}


def name(s: str) -> str:
    return " vs ".join(NAMES.get(p, p) for p in s.split(" vs "))


def write(stem: str, text: str) -> None:
    (OUT / f"{stem}.tex").write_text(text)
    print("wrote", f"report/tables/{stem}.tex")


def csv(stem: str, **kw) -> pd.DataFrame:
    return pd.read_csv(T / f"{stem}.csv", **kw)


# 1. Pairwise judge, 12k context: Opus and Mistral
rows = []
for judge in ("opus", "mistral"):
    m = marked_win_rates(csv(f"win_rates_{judge}"), csv(f"sign_tests_{judge}"))
    rows += [[("Opus 5.5" if judge == "opus" else "Mistral Small 24B") if i == 0 else "", name(p), *m.loc[p]]
             for i, p in enumerate(m.index)]
write("judge_12k", tabular(["Judge", "Pair (first vs second)", *CRIT_HEAD], rows, "llcccc"))

# 2. Pairwise judge, 8k context (Opus)
m = marked_win_rates(csv("win_rates_8k"), csv("sign_tests_8k"))
write("judge_8k", tabular(["Pair (first vs second)", *CRIT_HEAD], [[name(p), *m.loc[p]] for p in m.index]))

# 3. Correctness per system (test split, look-up + connecting questions)
c = pd.concat([csv("correctness_by_system"), csv("correctness_8k_by_system")]).drop_duplicates("system")
c = c.set_index("system").loc[["basic", "local", "global", "basic_8k", "global_8k"]]
write("correctness", tabular(["System", "$n$", "Mean (0--3)", "SD", "Contradicts"],
                             [[NAMES[s], str(int(r.n)), f"{r['mean']:.2f}", f"{r.sd:.2f}", str(int(r.contradicts))]
                              for s, r in c.iterrows()]))

# 4. Correctness, paired Wilcoxon with Holm
p = pd.concat([csv("correctness_paired"), csv("correctness_8k_paired")])
write("correctness_tests", tabular(
    ["Pair", "Better", "Worse", "Equal", "Mean diff.", "$p_\\text{Holm}$"],
    [[f"{NAMES[r.sys_1]} vs {NAMES[r.sys_2]}", str(r.n_first_better), str(r.n_first_worse), str(r.n_equal),
      f"{r.mean_diff:+.2f}".replace("-", "$-$"), f"{r.p_holm:.3f}"] for r in p.itertuples()], "lccccc"))

# 5. Cost per answer
a = pd.concat([csv("answers_per_system"), csv("answers_per_system_8k")]).drop_duplicates("system")
a = a.set_index("system").loc[["basic", "local", "global", "basic_8k", "global_8k"]]
write("cost_per_answer", tabular(
    ["System", "LLM calls", "Prompt tok.", "Answer tok.", "Latency (s)", "Articles"],
    [[NAMES[s], f"{r.llm_calls:.0f}", f"{r.prompt_tokens:,.0f}", f"{r.answer_tokens:.0f}", f"{r.latency_s:.0f}",
      f"{r.retrieved_articles:.1f}"] for s, r in a.iterrows()]))

# 6. Compute per stage (from the Kaggle run logs and results/spend_ledger.csv; GPU-h = 2 x T4 x wall)
spend = csv("api_spend").set_index("stage")
write("compute", tabular(["Stage", "GPU-h (T4)", "kWh", "API (USD)"], [
    ["Indexing, 200 articles (2 sessions)", "24.3", "1.56", "--"],
    ["Answers, 3 systems $\\times$ 60 q. (12k)", "13.4", "0.82", "--"],
    ["Answers, 2 systems $\\times$ 60 q. (8k)", "6.96", "0.45", "--"],
    ["Second judge (Mistral, local)", "0.93", "0.05", "--"],
    ["Opus judge + correctness", "--", "--", f"{spend.loc['total', 'usd']:.2f}"],
], "lccc"))

# 7. Human sample: agreement between raters
h = csv("human_agreement", index_col=0)
rn = {"human1": "Human 1", "human2": "Human 2", "Opus": "Opus", "Mistral": "Mistral"}
write("human_agreement", tabular(["Raters", "Verdicts", "Agreement", "$\\kappa$"],
                                 [[f"{rn[r.rater_1]} vs {rn[r.rater_2]}", str(r.verdicts), f"{r.agreement:.2f}",
                                   f"{r.kappa:.2f}"] for r in h.itertuples()]))

# 8. Human sample: wins per system
w = csv("human_wins").set_index("rater").loc[["human1", "human2", "Opus", "Mistral"]]
write("human_wins", tabular(["Rater", "basic", "local", "global", "Ties"],
                            [[rn[k], str(int(r.basic)), str(int(r.local)), str(int(r["global"])), f"{r.tie_share:.0%}".replace("%", r"\%")]
                             for k, r in w.iterrows()]))

# 9. Appendix: length bias and judge agreement per criterion
lb = pd.read_csv(T / "length_bias.csv", header=[0, 1], index_col=0)
lb8 = csv("length_bias_8k").set_index("criterion")
write("length_bias", tabular(["Criterion", "Opus (12k)", "Mistral (12k)", "Opus (8k)"],
                             [[c, f"{lb.loc[c, ('Opus 5.5', 'share_longer_won')]:.2f}",
                               f"{lb.loc[c, ('Mistral Small 24B', 'share_longer_won')]:.2f}",
                               f"{lb8.loc[c, 'share_longer_won']:.2f}"] for c in CRITERIA]))
ja = csv("judge_agreement", index_col=0)
write("judge_agreement", tabular(["Criterion", "Verdicts", "Agreement", "$\\kappa$"],
                                 [[latex_escape(c), str(int(r.n)), f"{r.agreement:.2f}", f"{r.kappa:.2f}"]
                                  for c, r in ja.iterrows()]))

# 10. Community levels of the graph
cl = csv("community_levels")
write("community_levels", tabular(["Level", "Reports", "Report tokens", "Tokens vs C0"],
                                  [[r.level, str(r.reports), f"{r.report_tokens:,}", f"{r.x_C0_tokens:.1f}"]
                                   for r in cl.itertuples()]))

# 11. Appendix: pairwise win rates and correctness per question kind (15 test questions each; descriptive only)
KINDS = {"big_picture": "big-picture", "connecting": "connecting", "lookup": "look-up"}
rows = []
for judge, label in (("opus", "Opus 5.5"), ("mistral", "Mistral Small 24B")):
    k = csv(f"win_rates_by_kind_{judge}")
    first = True
    for kind in KINDS:
        for i, r in enumerate(k[k.kind == kind].itertuples()):
            rows.append([label if first else "", KINDS[kind] if i == 0 else "", r.pair,
                         *(f"{getattr(r, c):.2f}" for c in CRITERIA)])
            first = False
write("judge_by_kind", tabular(["Judge", "Kind", "Pair (first vs second)", *CRIT_HEAD], rows, "lllcccc"))
ck = csv("correctness_by_kind").set_index("kind")
write("correctness_by_kind", tabular(
    ["Kind", "$n$", "basic", "local", "global", "Contradicts (b/l/g)"],
    [[KINDS[kd], str(int(r.n_questions)), f"{r.basic_mean:.2f}", f"{r.local_mean:.2f}", f"{r.global_mean:.2f}",
      f"{int(r.basic_contradicts)}/{int(r.local_contradicts)}/{int(r.global_contradicts)}"]
     for kd, r in ck.loc[[k for k in KINDS if k in ck.index]].iterrows()]))

# 12. Appendix: extraction-prompt comparison (recorded from the kernel logs and graph tables of the
#     prompt-comparison runs)
write("prompt_comparison", tabular(
    ["", "domain (50)", "default (50)", "domain (10)$^a$", "default (10)", "hybrid (10)"], [
        ["Entities", "1,608", "1,276", "274", "224", "877"],
        ["Relationships", "847", "1,218", "203", "247", "279"],
        ["Relationships per entity", "0.53", "0.95", "0.74", "1.10", "0.32"],
        ["Isolated entities", "41\\%", "23\\%", "21\\%", "20\\%", "71\\%"],
        ["Community reports", "38", "151", "--", "10", "34"],
        ["Extraction time (s)", "12,734$^b$", "7,096", "--", "901", "2,929"],
    ], "lccccc"))

# 13. Appendix: indexing cost against corpus and graph size (from the kernel logs; tokens from
#     data/processed/subset_200.csv: the 10- and 50-article runs are its first 10 and 50 articles)
write("graph_size", tabular(
    ["Articles", "Tokens", "Prompt", "Entities", "Relations", "Reports", "GPU-h", "kWh", "GPU-h / 100k tok."], [
        ["10", "25,789", "default", "224", "247", "10", "1.0", "0.05", "3.9"],
        ["50", "121,895", "default$^b$", "1,276", "1,218", "151", "6.9", "0.43", "5.6"],
        ["50", "121,895", "domain$^a$", "1,608", "847", "38", "7.7", "0.50", "6.3"],
        ["200", "589,594", "domain", "6,862", "4,731", "579", "24.3", "1.56", "4.1"],
    ], "rrlrrrrrr"))
