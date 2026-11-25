"""Package the extracted corpus as a private Kaggle dataset."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from dne.paths import PROCESSED

DATASET_SLUG = "itf-ccsr-text"


def kaggle_username() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api.get_config_value("username")


def build_dataset_dir(out: Path, slug: str = DATASET_SLUG, only_status: str = "ok") -> Path:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    arts = pd.read_csv(PROCESSED / "articles.csv")
    with (out / "articles.jsonl").open("w") as f:  # one file: the Kaggle CLI uploads files one by one
        for aid in sorted(arts.loc[arts.status == only_status, "aid"]):
            f.write(json.dumps({"aid": int(aid), "text": (PROCESSED / "text" / f"{aid}.txt").read_text()}) + "\n")
    shutil.copy2(PROCESSED / "articles.csv", out / "articles.csv")
    meta = {
        "title": "ITF CCSR text (CC BY 4.0)",
        "id": f"{kaggle_username()}/{slug}",
        "licenses": [{"name": "CC-BY-4.0"}],
        "isPrivate": True,
    }
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    return out
