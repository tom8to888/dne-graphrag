import json

import pandas as pd

import dne.kaggle_io as kio


def test_build_dataset_dir(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    (processed / "text").mkdir(parents=True)
    rows = []
    for aid, status in [(1, "ok"), (2, "ok"), (3, "ok"), (4, "too_short")]:
        if status == "ok":
            (processed / "text" / f"{aid}.txt").write_text(f"article {aid}")
        rows.append({"aid": aid, "status": status})
    pd.DataFrame(rows).to_csv(processed / "articles.csv", index=False)
    monkeypatch.setattr(kio, "PROCESSED", processed)
    monkeypatch.setattr(kio, "kaggle_username", lambda: "tester")

    out = kio.build_dataset_dir(tmp_path / "ds")

    assert not list(out.glob("*.txt"))  # one file, not 541 small uploads
    rows = [json.loads(line) for line in (out / "articles.jsonl").read_text().splitlines()]
    assert [(r["aid"], r["text"]) for r in rows] == [(1, "article 1"), (2, "article 2"), (3, "article 3")]
    assert (out / "articles.csv").exists()
    meta = json.loads((out / "dataset-metadata.json").read_text())
    assert meta["id"] == "tester/itf-ccsr-text"
    assert meta["isPrivate"] is True
    assert meta["licenses"] == [{"name": "CC-BY-4.0"}]
