"""Push a Kaggle kernel, wait for it, pull its output.

    python tools/kaggle_run.py push  <kernel_dir>
    python tools/kaggle_run.py wait  <kernel_dir>
    python tools/kaggle_run.py pull  <kernel_dir> <out_dir>
    python tools/kaggle_run.py code  <mode> <n_articles> <llm_model> <embed_model> [prompt_variant] [key=value ...]
                                     # build + upload the code dataset, then wait until Kaggle has processed it
                                     # (a kernel pushed while a dataset version is pending sits QUEUED)
                                     # keys: budget=<h> stop=<h> resume=1 aids=<csv with an aid column>
                                     #       finish=1 (reports + embeddings only, on the restored tables) conc=<n>
                                     #       per_kind=<n> (mode query: benchmark on n questions per kind) qconc=<n> glevel=<n> qstop=<h> qtimeout=<s>
                                     #       ctx=<tokens> systems=<a,b> fresh=1 (no resume from the answers dataset)
                                     #       resume_file=<answers.jsonl> (fresh run: continue from its own partial answers)
                                     #       mode chat (demo dialogue): dialogue=<txt> qstop=<h> qtimeout=<s>
                                     #       mode judge (second judge, embed_model=none): jlimit=<n> jconc=<n> qstop=<h> qtimeout=<s>
    python tools/kaggle_run.py index <graphrag_output_dir>
                                     # upload a finished index (output/ incl. lancedb) as dataset itf-ccsr-graphrag-index
    python tools/kaggle_run.py cache <pulled_out_dir> <kernel_dir>
                                     # upload the stopped session's cache.tar.gz as dataset dne-graphrag-cache
                                     # and add it to the kernel's dataset_sources (then: code ... resume=1)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dne.kaggle_bundle import CODE_SLUG, build_code_bundle  # noqa: E402
from dne.kaggle_io import kaggle_username  # noqa: E402
from dne.paths import ROOT  # noqa: E402
from dne.resume import ARCHIVE, OUTPUT_ARCHIVE, pack_cache, pack_output  # noqa: E402

CACHE_SLUG = "dne-graphrag-cache"
INDEX_SLUG = "itf-ccsr-graphrag-index"

KAGGLE = str(Path(sys.executable).parent / "kaggle")


def kernel_id(kdir: str) -> str:
    return json.loads((Path(kdir) / "kernel-metadata.json").read_text())["id"]


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([KAGGLE, *args], capture_output=True, text=True)


def upload_dataset(path: Path, slug: str, user: str, msg: str) -> None:
    exists = run("datasets", "files", f"{user}/{slug}").returncode == 0
    args = (["datasets", "version", "-p", str(path), "-m", msg, "--dir-mode", "zip"]
            if exists else ["datasets", "create", "-p", str(path), "--dir-mode", "zip"])
    r = run(*args)
    print(r.stdout.strip()[-500:], r.stderr.strip()[-500:])
    for _ in range(180):
        time.sleep(10)
        if run("datasets", "status", f"{user}/{slug}").stdout.strip().endswith("ready"):
            print(f"{slug} ready")
            return
    sys.exit(f"{slug} not ready after 30 min")


def code_options(extra: list[str]) -> dict:
    """Old positional form `[budget_hours]` still works; otherwise key=value."""
    opts = {}
    for a in extra:
        if "=" not in a:
            opts["budget_hours"] = float(a)
            continue
        k, v = a.split("=", 1)
        if k == "budget":
            opts["budget_hours"] = float(v)
        elif k == "stop":
            opts["stop_after_hours"] = float(v)
        elif k == "resume":
            opts["resume"] = v.lower() in ("1", "true", "yes")
        elif k == "finish":  # finish a stopped index: restore cache + output tables, run only what is missing
            if v.lower() in ("1", "true", "yes"):
                opts.update(resume=True, restore_output=True,
                            workflows=["create_community_reports", "generate_text_embeddings"])
        elif k == "per_kind":  # query mode: benchmark on N questions per kind
            opts["per_kind"] = int(v)
        elif k == "qconc":  # query mode: queries at once
            opts["query_concurrency"] = int(v)
        elif k == "qstop":  # query mode: no new queries after this many hours (quota-limited session)
            opts["query_stop_hours"] = float(v)
        elif k == "qtimeout":  # query mode: one query is abandoned (error row, retried next run) after this many s
            opts["query_timeout_s"] = float(v)
        elif k == "jlimit":  # judge mode: trial on the first n pairs
            opts["judge_limit"] = int(v)
        elif k == "jconc":  # judge mode: calls at once
            opts["judge_concurrency"] = int(v)
        elif k == "glevel":  # query mode: community level for global search (0 = C0)
            opts["global_level"] = int(v)
        elif k == "ctx":  # query mode: final-answer context budget (12000 default; the paper used 8000)
            opts["max_context_tokens"] = int(v)
        elif k == "systems":  # query mode: answer only with these, e.g. systems=global
            opts["systems"] = v.split(",")
        elif k == "fresh":  # query mode: do not resume from the earlier answers dataset
            opts["resume_answers"] = not (v.lower() in ("1", "true", "yes"))
        elif k == "resume_file":  # query mode with fresh=1: continue from this run's own partial answers.jsonl
            opts["resume_file"] = Path(v)
        elif k == "dialogue":  # mode chat: the scripted dialogue, one user turn per line
            opts["dialogue"] = Path(v)
        elif k == "conc":
            opts["concurrent_requests"] = int(v)
        elif k == "aids":
            import pandas as pd
            opts["aids"] = [int(x) for x in pd.read_csv(v)["aid"]]
        else:
            sys.exit(f"unknown option {k}")
    return opts


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "push":
        r = run("kernels", "push", "-p", sys.argv[2], "--accelerator", "NvidiaTeslaT4")
        print(r.stdout.strip(), r.stderr.strip()[-500:])
    elif cmd == "wait":
        kid = kernel_id(sys.argv[2])
        while True:
            s = run("kernels", "status", kid).stdout.strip()
            print(time.strftime("%H:%M:%S"), s.split("status")[-1].strip(), flush=True)
            if any(w in s.lower() for w in ("complete", "error", "cancel")):
                break
            time.sleep(60)
    elif cmd == "pull":
        Path(sys.argv[3]).mkdir(parents=True, exist_ok=True)
        r = run("kernels", "output", kernel_id(sys.argv[2]), "-p", sys.argv[3], "-o")
        print(r.stdout.strip()[-500:], r.stderr.strip()[-500:])
    elif cmd == "code":
        user = kaggle_username()
        out = build_code_bundle(ROOT / "data/kaggle_code", mode=sys.argv[2], n_articles=int(sys.argv[3]),
                                llm_model=sys.argv[4], embed_model=sys.argv[5], username=user,
                                prompt_variant=sys.argv[6] if len(sys.argv) > 6 else "default",
                                **code_options(sys.argv[7:]))
        cfg = json.loads((out / "run_config.json").read_text())
        print(json.dumps({k: v for k, v in cfg.items() if k != "aids"}), "| aids:", len(cfg["aids"] or []))
        upload_dataset(out, CODE_SLUG, user, f"{sys.argv[2]} {sys.argv[3]}")
    elif cmd == "cache":
        user, src, kdir = kaggle_username(), Path(sys.argv[2]), Path(sys.argv[3])
        ds = ROOT / "data/kaggle_cache"
        ds.mkdir(parents=True, exist_ok=True)
        tar = src / ARCHIVE
        if tar.exists():
            (ds / ARCHIVE).write_bytes(tar.read_bytes())
        else:
            pack_cache(src / "graphrag", ds / ARCHIVE)
        (ds / "dataset-metadata.json").write_text(json.dumps({
            "title": "DNE GraphRAG cache", "id": f"{user}/{CACHE_SLUG}",
            "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
        if (src / OUTPUT_ARCHIVE).exists():
            (ds / OUTPUT_ARCHIVE).write_bytes((src / OUTPUT_ARCHIVE).read_bytes())
        elif (src / "graphrag/output/entities.parquet").exists():  # for finish=1
            pack_output(src / "graphrag", ds / OUTPUT_ARCHIVE)
        for f in (ARCHIVE, OUTPUT_ARCHIVE):
            if (ds / f).exists():
                print(f"{f}: {(ds / f).stat().st_size / 1e6:.1f} MB")
        upload_dataset(ds, CACHE_SLUG, user, f"cache from {src.name}")
        meta_p = kdir / "kernel-metadata.json"
        meta = json.loads(meta_p.read_text())
        if f"{user}/{CACHE_SLUG}" not in meta["dataset_sources"]:
            meta["dataset_sources"].append(f"{user}/{CACHE_SLUG}")
            meta_p.write_text(json.dumps(meta, indent=2) + "\n")
            print("added cache dataset to", meta_p)
    elif cmd == "index":
        user, src = kaggle_username(), Path(sys.argv[2])
        if not (src / "entities.parquet").exists() or not (src / "lancedb").exists():
            sys.exit(f"{src} is not a finished graphrag output dir (entities.parquet + lancedb/)")
        ds = ROOT / "data/kaggle_index"
        ds.mkdir(parents=True, exist_ok=True)
        pack_output(src.parent, ds / OUTPUT_ARCHIVE)
        print(f"{OUTPUT_ARCHIVE}: {(ds / OUTPUT_ARCHIVE).stat().st_size / 1e6:.1f} MB")
        (ds / "dataset-metadata.json").write_text(json.dumps({
            "title": "ITF CCSR GraphRAG index", "id": f"{user}/{INDEX_SLUG}",
            "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
        upload_dataset(ds, INDEX_SLUG, user, f"index from {src}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
