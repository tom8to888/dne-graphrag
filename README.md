# GraphRAG vs plain RAG on the World Tennis Coaching & Sport Science Review

IM1102 Deep Neural Engineering, assignment 2, Open Universiteit.

Does a knowledge-graph retriever (Microsoft GraphRAG, local and global search) answer coach-style questions
better than plain vector RAG when the graph is built with a small open model on free GPUs, and what does each
cost to build and to run? The report is `report/main.pdf`.

**Result in short.** On 200 coaching articles, with Qwen2.5-14B (4-bit AWQ) on two Kaggle T4 GPUs, basic RAG
beat both GraphRAG systems on almost all criteria and on correctness (1.77 against 1.37 for local and 1.13 for
global search, out of 3). Global search used about 4.6 times the prompt tokens per answer, and the index took
24.3 GPU-hours.

## Corpus

- Source: World Tennis (ex-ITF) Coaching & Sport Science Review, <https://itfcoachingreview.com/index.php/journal>
  (ISSN 1812-2302): 587 articles from 52 issues, 2009–2026. 541 have usable text (1.64M tokens).
- Licence: **CC BY 4.0**, stated per article. Attribution = author(s), title, journal, issue, year, DOI and a
  link to the licence; `data/processed/articles.csv` lists title, authors, year, issue and DOI for every article.
- Text only: figures and photos inside the PDFs may carry third-party rights and are not used.
- The extracted text is in the repository (`data/processed/text/`); the raw PDFs (~500 MB) are not, but
  `tools/harvest_itf.py` downloads them again.

## Layout

```
dne/                 library code: extraction, subset, Kaggle bundle, judge, statistics, report tables …
tools/               command-line scripts (one per stage, called by run_all.sh)
kaggle/              Kaggle GPU kernels: index/ builds the graph, query/ answers, chats and runs the second judge
kaggle/common/       vLLM start-up script, graphrag settings and prompts (prompts_domain/ = coaching extraction prompt)
notebooks/           01_corpus (corpus check and size), 05_results (all results tables)
data/processed/      extracted text (one file per article) + articles.csv + subset_200.csv
data/questions/      candidate pool, the final 60 questions with notes (questions.csv), demo dialogues
data/index/          the domain-200 graph: graphrag output tables (entities, relationships, communities, reports …)
results/             answers, judgements, correctness grades, human ratings, API spend ledger
results/tables/      every table in the report, as CSV
report/              ICML-style LaTeX report; report/tables/ is generated from results/tables/
tests/               unit tests (pytest)
```

## What is new

The knowledge graph, the community reports and the local and global search come from Microsoft's
[`graphrag`](https://github.com/microsoft/graphrag) library (version 3.3.0), used as released;
`kaggle/common/prompts/` and `kaggle/common/settings.default.yaml` are its defaults, which
`dne/graphrag_config.py` fills in for this setup. Everything else in this repository was written for
this project:

| Part | Files |
|---|---|
| GraphRAG on free Kaggle GPUs: one kernel serves Qwen2.5-14B and bge-m3 with vLLM and runs indexing, querying, the demo dialogues or the second judge | `kaggle/index/run_index.py`, `kaggle/common/serve.sh`, `dne/graphrag_config.py`, `dne/kaggle_bundle.py`, `tools/kaggle_run.py` |
| Runs that fit Kaggle's 12-hour sessions: live progress and ETA, a watchdog that stops a run that is over budget or stalled, and resuming an index from graphrag's cache in the next session | `dne/progress.py`, `dne/resume.py` |
| Basic search with a fix for graphrag 3.3's empty answers (its streaming call is never awaited), so that all three systems share the same chunks, embeddings and answer model | `dne/query_worker.py` (`basic_answer`) |
| Domain extraction prompt: ten coaching entity types and two hand-written coaching examples | `kaggle/common/prompts_domain/extract_graph.txt` |
| Conversational agent: a follow-up question is rewritten into a standalone question, then answered by any of the three systems | `dne/chat.py`, `dne/chat_worker.py` |
| Cost measurement: GPU energy from `nvidia-smi`, token counts from vLLM, graph size | `dne/power.py`, `dne/vllm_metrics.py`, `dne/graph_stats.py` |
| Evaluation: pairwise LLM judge on the four criteria of Edge et al. in both display orders, correctness grading against reference notes, a second judge from another model family, and an API spend cap | `dne/judge.py`, `dne/local_judge.py`, `dne/spend.py` |
| Statistics: sign test, Wilcoxon signed-rank test, Holm correction, Cohen's κ, length bias | `dne/stats.py`, `dne/analysis.py` |
| Blinded human sample | `dne/human_sample.py`, `tools/export_human_ratings.py` |
| Corpus: harvesting, PDF-to-text extraction, the 200-article subset, the question set | `tools/harvest_itf.py`, `dne/extract.py`, `dne/subset.py`, `dne/questions.py` |
| Report tables generated from the result files | `dne/report_tables.py`, `tools/make_report_tables.py` |

## Setup

Python 3.11. With [uv](https://docs.astral.sh/uv/):

```
uv venv ~/envs/dne_graphrag --python 3.11
VIRTUAL_ENV=~/envs/dne_graphrag uv pip install -r requirements.txt
~/envs/dne_graphrag/bin/python -m ipykernel install --user --name dne_graphrag --display-name "Python (DNE GraphRAG)"
```

The report needs a LaTeX distribution with `latexmk` (e.g. TeX Live or MacTeX).

## Reproduce

`run_all.sh` runs the study stage by stage. Set `PYTHON` to the environment's interpreter
(`export PYTHON=~/envs/dne_graphrag/bin/python`).

| Command | What it does | Needs | Time / cost |
|---|---|---|---|
| `./run_all.sh` | tests, correctness tables, results notebook, report tables, PDF, all from the result files in `results/` | nothing else | under a minute, free |
| `./run_all.sh corpus` | checks the raw corpus, extracts the text, draws the 200-article subset | `data/raw/` (below) | ~5 min |
| `./run_all.sh kaggle` | builds the index, answers the questions (12k and 8k context), runs the demo dialogues and the second judge | Kaggle account | ~50 GPU-hours over several sessions, free |
| `./run_all.sh judge --yes` | Opus judge and correctness grader (800 calls) | Anthropic API key | ~USD 16 |
| `./run_all.sh all` | all of the above, in order | all of the above | |

Without `--yes`, the judge stage prints the plan and its estimated cost and makes no calls. Both judge
scripts skip pairs already in `results/`, so rerunning them on the shipped files costs nothing.

**Raw corpus.** The journal's site is behind Cloudflare, so `tools/harvest_itf.py` fetches through a real Chrome
started with `--remote-debugging-port=9222`:

```
python tools/harvest_itf.py all      # issues → article metadata → PDFs, into data/raw/itf_ccsr/
```

**Kaggle.** Install the Kaggle CLI credentials (`~/.kaggle/kaggle.json`) and accept the T4 GPU terms. The
kernels read three private datasets that the scripts create under your username: the article texts
(`itf-ccsr-text`), the code bundle (`dne-graphrag-code`) and the finished index (`itf-ccsr-graphrag-index`).
Replace the owner in `kaggle/*/kernel-metadata.json` with your Kaggle username, and remove
`dne-graphrag-answers` (our earlier answers, used to resume) from the query kernel's `dataset_sources`; the
commands in `run_all.sh` start fresh runs (`fresh=1`). A Kaggle session ends after
12 hours and GPU time is limited to 30 hours a week, so the index takes two sessions (the second finishes the
community reports from the saved cache) and the 12k answers may need a continuation run (see the comments in
`run_all.sh`). Queues can add 1–3 hours per session.

**Anthropic API.** Put `ANTHROPIC_API_KEY=...` in `.env` (not committed). Every call is written to
`results/spend_ledger.csv`, and the scripts stop at a cap of USD 20.

**Human sample.** `python -m dne.human_sample` draws the 20 blinded answer pairs
(`results/human_sample.csv`, with the key kept apart in `results/human_sample_key.csv`). The two raters'
answers are in `results/human_ratings.csv`; `tools/export_human_ratings.py` rebuilds that file from the
rating page's export.

## Settings

| | |
|---|---|
| graphrag | 3.3.0; chunks of 1,200 tokens with 100 overlap; one gleaning round; Leiden communities on the largest connected component (default) |
| Answer and extraction model | Qwen2.5-14B-Instruct-AWQ on vLLM, 2× NVIDIA T4 (Kaggle) |
| Embeddings | BAAI/bge-m3 |
| Systems | basic (our reimplementation of graphrag's basic search), local, global at community level C0 |
| Context budget | 12,000 tokens (main run); 8,000 for basic and global, as in Edge et al. |
| Judges | Claude Opus 5.5 (API) and Mistral Small 24B Instruct 2501 AWQ (Kaggle), same prompt, both display orders |
| Questions | 60 (20 look-up, 20 connecting, 20 big-picture); dev 15, test 45; results on test |

## Main result files

| File | Contents |
|---|---|
| `results/answers.parquet`, `answers_8k.parquet` | 180 answers at 12k and 120 at 8k, with retrieved articles, tokens and latency |
| `results/judgments.jsonl`, `judgments_8k.jsonl`, `judgments_mistral.jsonl` | pairwise verdicts per criterion, with the judge's reasons |
| `results/correctness.jsonl`, `correctness_8k.jsonl` | coverage of the reference notes (0–3) and contradictions |
| `results/human_ratings.csv` | blinded ratings of the two human raters |
| `results/chat_demo.jsonl`, `chat_demo.md` | the four demo dialogues, answered by all three systems |
| `results/spend_ledger.csv` | every API call and its cost |
