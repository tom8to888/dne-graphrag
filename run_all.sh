#!/usr/bin/env bash
# Reproduce the study, stage by stage.
#
#   ./run_all.sh              tests + analysis + report from the shipped result files (offline, free, under a minute)
#   ./run_all.sh corpus       re-extract the corpus text from the raw PDFs (needs data/raw/, see README)
#   ./run_all.sh kaggle       index + answers + demo dialogue + second judge on Kaggle GPUs (hours, needs a Kaggle account)
#   ./run_all.sh judge        Opus judge + correctness through the Anthropic API (costs money, needs --yes)
#   ./run_all.sh all          corpus, kaggle, judge, analysis in order
#
# Every stage writes the files the next one reads, so a stage can be rerun on its own.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python}
LLM=Qwen/Qwen2.5-14B-Instruct-AWQ
EMBED=BAAI/bge-m3
JUDGE2=stelterlab/Mistral-Small-24B-Instruct-2501-AWQ
STAGE=${1:-analysis}
YES=${2:-}

step() { printf '\n== %s\n' "$*"; }

corpus() {
  step "Corpus: check the raw copy (587 articles), extract text, draw the 200-article subset"
  $PY tools/check_raw.py
  $PY tools/extract_all.py
  $PY tools/make_subset.py 200
}

kaggle_job() {  # <kernel dir> <out dir>: push, wait for the session to end, pull its output
  $PY tools/kaggle_run.py push "$1"
  $PY tools/kaggle_run.py wait "$1"
  $PY tools/kaggle_run.py pull "$1" "$2"
}

kaggle() {
  step "Kaggle: upload the article texts as a private dataset"
  $PY tools/kaggle_push_dataset.py

  step "Kaggle: build the GraphRAG index on the 200-article subset (domain prompt; ~11 h, stops cleanly before 12 h)"
  $PY tools/kaggle_run.py code index 200 "$LLM" "$EMBED" domain aids=data/processed/subset_200.csv stop=10.5
  kaggle_job kaggle/index data/kaggle_out/domain_200
  # The first session stops during the community reports. Finish them from the saved cache:
  $PY tools/kaggle_run.py cache data/kaggle_out/domain_200 kaggle/index
  $PY tools/kaggle_run.py code index 200 "$LLM" "$EMBED" domain aids=data/processed/subset_200.csv resume=1 finish=1 conc=5 stop=2
  kaggle_job kaggle/index data/kaggle_out/domain_200_finish
  $PY tools/kaggle_run.py index data/kaggle_out/domain_200_finish/graphrag/output
  mkdir -p data/index
  cp data/kaggle_out/domain_200_finish/graphrag/output/*.parquet data/kaggle_out/domain_200_finish/graphrag/output/stats.json data/index/

  step "Kaggle: answer the 60 questions with basic, local and global search (12k context; ~13 GPU-h)"
  $PY tools/kaggle_run.py code query 200 "$LLM" "$EMBED" domain fresh=1 qconc=3 glevel=0 qtimeout=1500 qstop=10.5
  kaggle_job kaggle/query data/kaggle_out/query
  # If the session stopped before all 180 answers (Kaggle's 12 h limit or weekly quota), continue it with
  #   ... code query ... fresh=1 resume_file=data/kaggle_out/query/answers.jsonl   and run kaggle_job again.
  cp data/kaggle_out/query/answers.parquet results/answers.parquet

  step "Kaggle: basic and global search at 8k context, as in Edge et al."
  $PY tools/kaggle_run.py code query 200 "$LLM" "$EMBED" domain ctx=8000 systems=basic,global fresh=1 qconc=3 glevel=0 qtimeout=1500 qstop=5.5
  kaggle_job kaggle/query data/kaggle_out/query_8k
  $PY -c "import pandas as pd; a = pd.read_parquet('data/kaggle_out/query_8k/answers.parquet'); a['system'] = a['system'] + '_8k'; a.to_parquet('results/answers_8k.parquet', index=False)"

  step "Kaggle: the demo dialogues (conversational agent, all three systems)"
  $PY tools/kaggle_run.py code chat 200 "$LLM" "$EMBED" domain dialogue=data/questions/dialogue.txt qtimeout=1500
  kaggle_job kaggle/query data/kaggle_out/chat
  cp data/kaggle_out/chat/chat.jsonl results/chat_demo.jsonl

  step "Kaggle: second judge (Mistral Small 24B, same prompt as the Opus judge)"
  $PY tools/kaggle_run.py code judge 200 "$JUDGE2" none domain
  kaggle_job kaggle/query data/kaggle_out/judge_mistral
  cp data/kaggle_out/judge_mistral/judgments_local.jsonl results/judgments_mistral.jsonl
}

judge() {
  step "Opus judge + correctness (12k answers: 480 calls; 8k answers: 320 calls). Without --yes: plan only, no calls"
  $PY tools/run_judge.py $YES
  $PY tools/run_judge_8k.py $YES
}

analysis() {
  step "Tests"
  $PY -m pytest -q
  step "Results notebook -> results/tables/*.csv"
  $PY -m jupyter nbconvert --to notebook --execute --inplace notebooks/05_results.ipynb
  step "Report tables, demo dialogue, PDF"
  $PY tools/make_report_tables.py
  $PY tools/make_dialogue_tex.py
  make -C report
  printf '\nDone: report/main.pdf\n'
}

case "$STAGE" in
  corpus)   corpus ;;
  kaggle)   kaggle ;;
  judge)    judge ;;
  analysis) analysis ;;
  all)      corpus; kaggle; judge; analysis ;;
  *) echo "unknown stage: $STAGE (corpus | kaggle | judge | analysis | all)"; exit 2 ;;
esac
