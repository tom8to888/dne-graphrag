"""Check the raw corpus copy: 587 meta, 587 PDFs, all PDFs start with %PDF, issues.json shape."""
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from dne.extract import english_galley_gid  # noqa: E402
from dne.paths import META_DIR, PDF_DIR, RAW  # noqa: E402

metas = sorted(META_DIR.glob("*.json"))
pdfs = sorted(PDF_DIR.glob("*.pdf"))
bad = [p.name for p in pdfs if p.read_bytes()[:4] != b"%PDF"]
issues = json.loads((RAW / "issues.json").read_text())
n_articles = sum(len(i["articles"]) for i in issues)
en_missing = [m.stem for m in metas
              if not (PDF_DIR / f"{m.stem}_{english_galley_gid(json.loads(m.read_text()))}.pdf").exists()]
print(f"{len(metas)} meta, {len(pdfs)} pdf files, {len(bad)} bad headers, "
      f"English galley on disk for {len(metas) - len(en_missing)}/{len(metas)}")
print(f"issues.json: {len(issues)} issues, {n_articles} articles")
if bad:
    print("bad:", bad[:10])
ok = len(metas) == 587 and len(pdfs) >= 587 and not bad and n_articles == 587
sys.exit(0 if ok else 1)
