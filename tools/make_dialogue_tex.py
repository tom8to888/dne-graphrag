"""Write the demo-dialogue appendix table and a readable transcript.

    python tools/make_dialogue_tex.py

results/chat_demo.jsonl   all 4 dialogues x 3 systems (basic + local from chat run 1, global from run 2)
results/chat_demo.md      readable transcript of all of them
report/tables/demo_dialogue{,_2}.tex   dialogue 3 for the appendix (turns 1-2, 3-4): user turns, each system's rewritten
                                  question, and a verbatim excerpt of each answer (checked below)

Excerpts: list answers (turns 1 and 4) show the answer's own section headings; yes/no answers
(turns 2 and 3) quote the sentences that give the answer. Every quote is checked to occur verbatim.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dne.paths import ROOT  # noqa: E402

DIALOGUE = 3
SYSTEMS = ("basic", "local", "global")
# Turns 2 and 3: the sentences that state the answer, quoted verbatim.
QUOTES = {
    (2, "basic"): ["When dealing with an opponent who displays these emotional states, it is beneficial for "
                   "club players to employ specific interaction strategies during changeovers."],
    (2, "local"): ["Therefore, it is advisable for players to avoid engaging with emotionally charged opponents "
                   "during changeovers to preserve their mental and emotional stability, which can directly "
                   "impact their performance."],
    (2, "global"): ["Avoiding interaction with an emotionally charged opponent during changeovers is advisable "
                    "for maintaining focus and performance."],
    (3, "basic"): ["Based on the provided data, there isn't explicit guidance recommending whether to avoid "
                   "interaction or engage in light-hearted conversation during changeovers.",
                   "Ultimately, the decision should align with the individual player's comfort and "
                   "effectiveness [Data: Sources (98, 160)]."],
    (3, "local"): ["Therefore, while light-hearted conversation can be beneficial, it is advisable to maintain a "
                   "balanced and relaxed atmosphere, avoiding emotionally charged interactions that might "
                   "increase stress or negative emotions [Data: Entities (2439), Relationships (1672)]."],
    (3, "global"): ["It is generally recommended to engage in light-hearted conversation during changeovers, as "
                    "this type of interaction can be conducive to relaxation and releasing tension.",
                    "This approach is supported by several data records, indicating its positive impact on "
                    "athlete performance [Data: Reports (14, 7, 17, 32, 9, 22, 44, 4, 42, 51, 52)]."],
}


def tex(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s.replace("’", "'").replace("“", "``").replace("”", "''")


def headings(answer: str) -> list[str]:
    """Section headings of a list answer; the answer's own title (a ### above #### items) is dropped."""
    level = "####" if re.search(r"^####\s", answer, flags=re.M) else "###"
    hs = re.findall(rf"^{level}\s+(.+)$", answer, flags=re.M)
    hs = [re.sub(r"^\d+\.\s*", "", h).strip("* ").strip() for h in hs]
    return [h for h in hs if h.lower() not in {"introduction", "conclusion", "strategies to manage the situation"}]


rows = [json.loads(x) for x in (ROOT / "results/chat_demo.jsonl").read_text().splitlines() if x]

# readable transcript of all dialogues
md = ["# Demo dialogues: all three systems\n\nBasic and local from chat run 1, global from chat run 2 "
      "(same index, Qwen2.5-14B, 12k context).\n"]
for d in sorted({r["dialogue"] for r in rows}):
    for s in SYSTEMS:
        md.append(f"\n## Dialogue {d}: {s}\n")
        for r in sorted((r for r in rows if r["dialogue"] == d and r["system"] == s), key=lambda r: r["turn"]):
            md.append(f"\n**User (turn {r['turn']}):** {r['user']}\n")
            if r["turn"] > 1:
                md.append(f"\n*Rewritten:* {r['standalone']}\n")
            md.append(f"\n**{s}** ({len(r['answer'].split())} words):\n\n{r['answer']}\n")
(ROOT / "results/chat_demo.md").write_text("".join(md))

# appendix table for one dialogue
d = [r for r in rows if r["dialogue"] == DIALOGUE]
turns = sorted({r["turn"] for r in d})
def table_for(part_turns: list[int]) -> str:
  out = []
  for t in part_turns:
    by = {r["system"]: r for r in d if r["turn"] == t}
    assert set(by) == set(SYSTEMS) and not any(r["error"] for r in by.values()), (t, by.keys())
    if t != part_turns[0]:
        out.append(r"\midrule")
    # turn header: the user's question across the table
    out.append(rf"\multicolumn{{4}}{{@{{}}>{{\raggedright\arraybackslash}}p{{0.97\textwidth}}@{{}}}}"
               rf"{{\textbf{{Turn {t}.}} \textbf{{User:}} {tex(by['basic']['user'])}}} \\")
    out.append(r"\addlinespace[3pt]")
    for i, s in enumerate(SYSTEMS):
        r = by[s]
        rewritten = "--" if t == 1 else r"\emph{" + tex(r["standalone"]) + "}"
        if (t, s) in QUOTES:
            for q in QUOTES[(t, s)]:
                assert q in r["answer"], f"not verbatim: turn {t} {s}: {q[:60]}"
            excerpt = "``" + r" [\dots] ".join(tex(q) for q in QUOTES[(t, s)]) + "''"
        else:
            hs = headings(r["answer"])
            assert hs, (t, s)
            excerpt = r" \,$\cdot$\, ".join(tex(h) for h in hs)
        out.append(rf"\textsc{{{s}}} & {rewritten} & {excerpt} & {len(r['answer'].split())} \\")
        if i < len(SYSTEMS) - 1:
            out.append(r"\addlinespace[4pt]")

  body = "\n".join(out)
  L = r">{\raggedright\arraybackslash}"
  table = (r"{\renewcommand{\arraystretch}{1.1}" "\n"
           rf"\begin{{tabular}}{{@{{}}{L}p{{0.07\textwidth}}{L}p{{0.30\textwidth}}{L}p{{0.49\textwidth}}r@{{}}}}" "\n\\toprule\n"
           r"System & Question as rewritten by the system & Answer: section headings (turns 1, 4) or "
           r"the sentences that answer (turns 2, 3) & Words \\" "\n\\midrule\n"
           f"{body}\n\\bottomrule\n\\end{{tabular}}}}\n")
  return table

# split in two tables (turns 1-2, 3-4) so each fits a page with room to read
(ROOT / "report/tables/demo_dialogue.tex").write_text(table_for(turns[:2]))
(ROOT / "report/tables/demo_dialogue_2.tex").write_text(table_for(turns[2:]))

print("results/chat_demo.md, report/tables/demo_dialogue.tex: dialogue", DIALOGUE, "turns", turns)
