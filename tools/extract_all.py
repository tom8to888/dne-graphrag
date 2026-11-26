"""Extract every article PDF to clean text; write data/processed/text/<aid>.txt + articles.csv."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import tiktoken  # noqa: E402

import json  # noqa: E402

from dne.extract import english_galley_gid, extract_article, load_meta  # noqa: E402
from dne.paths import META_DIR, PDF_DIR, PROCESSED, TEXT_DIR  # noqa: E402


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("cl100k_base")
    meta_paths = sorted(META_DIR.glob("*.json"), key=lambda p: int(p.stem))
    if not meta_paths:
        sys.exit(f"no article metadata in {META_DIR}: run tools/harvest_itf.py first (see README)")
    rows = []
    for meta_path in meta_paths:
        meta = load_meta(meta_path)
        gid = english_galley_gid(json.loads(meta_path.read_text()))
        pdf = PDF_DIR / f"{meta.aid}_{gid}.pdf" if gid else None
        if pdf is None or not pdf.exists():
            text, status = "", "no_pdf"
        else:
            text, status = extract_article(pdf, meta)
        if status == "ok":
            (TEXT_DIR / f"{meta.aid}.txt").write_text(text)
        rows.append({
            "aid": meta.aid, "title": meta.title, "authors": "; ".join(meta.authors), "year": meta.year,
            "issue": meta.issue, "section": meta.section, "language": meta.language, "doi": meta.doi,
            "status": status, "n_chars": len(text), "n_tokens": len(enc.encode(text)) if text else 0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED / "articles.csv", index=False)
    print(dict(Counter(df["status"])), f"tokens(ok)={df.loc[df.status == 'ok', 'n_tokens'].sum():,}")


if __name__ == "__main__":
    main()
