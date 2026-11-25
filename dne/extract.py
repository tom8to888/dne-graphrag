"""PDF → clean article text for the ITF CCSR corpus."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

MIN_CHARS = 500
_REF_HEADING = re.compile(r"^\s*(references|bibliography|referencias|références)\s*$", re.I)
_PAGE_NUM = re.compile(r"^\s*\d{1,3}\s*$")
_EN_WORDS = {"the", "and", "of", "to", "in", "is", "that", "for", "with", "on", "as", "are", "be", "this",
             "it", "by", "or", "their", "should", "can", "from", "at", "an", "have", "which", "was", "were"}
EN_SHARE = 0.05


@dataclass
class ArticleMeta:
    aid: int
    title: str
    authors: list[str]
    year: int | None
    issue: str
    section: str
    language: str
    doi: str | None


def _first(meta: dict, *keys: str) -> str | None:
    for k in keys:
        v = meta.get(k)
        if isinstance(v, list) and v:
            return str(v[0]).strip()
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def load_meta(path: Path) -> ArticleMeta:
    raw = json.loads(Path(path).read_text())
    m = raw.get("meta", {})
    date = _first(m, "citation_date", "DC.Date.issued") or ""
    year = int(date[:4]) if date[:4].isdigit() else None
    authors = [" ".join(a.split()) for a in m.get("citation_author", []) if a.strip()]
    return ArticleMeta(
        aid=int(raw["aid"]),
        title=" ".join((raw.get("title") or _first(m, "citation_title") or "").split()),
        authors=authors,
        year=year,
        issue=raw.get("issue_title", ""),
        section=raw.get("section", ""),
        language=(_first(m, "citation_language", "DC.Language") or "").lower(),
        doi=_first(m, "citation_doi", "DC.Identifier.DOI"),
    )


def english_galley_gid(raw_meta: dict) -> str | None:
    """Galley id of the English PDF: label 'pdf' or 'pdf eng…' (any case/spacing); ES/FR galleys say so."""
    for g in raw_meta.get("galleys") or raw_meta.get("issue_galleys", []):
        label = " ".join(g["label"].lower().split())
        if label == "pdf" or label.startswith("pdf eng"):
            return g["url"].rstrip("/").rsplit("/", 1)[1]
    return None


def looks_english(text: str) -> bool:
    words = re.findall(r"[a-zà-ÿ]+", text.lower())
    return bool(words) and sum(w in _EN_WORDS for w in words) / len(words) >= EN_SHARE


def clean_pages(pages: list[str]) -> str:
    split = [[ln.rstrip() for ln in p.splitlines()] for p in pages]
    repeated: set[str] = set()
    if len(pages) >= 3:
        counts = Counter(ln.strip() for lines in split for ln in set(lines) if ln.strip())
        repeated = {ln for ln, c in counts.items() if c >= 0.5 * len(pages)}
    kept: list[str] = []
    for lines in split:
        for ln in lines:
            s = ln.strip()
            if s in repeated or _PAGE_NUM.match(ln):
                continue
            if _REF_HEADING.match(ln):
                break
            kept.append(ln)
        else:
            continue
        break  # references heading found: drop the rest of the document
    text = "\n".join(kept)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def article_header(meta: ArticleMeta) -> str:
    return (f"Title: {meta.title}\nAuthors: {', '.join(meta.authors)}\nYear: {meta.year}\n"
            f"Issue: {meta.issue}\nArticle id: {meta.aid}\n\n")


def extract_article(pdf: Path, meta: ArticleMeta) -> tuple[str, str]:
    with pymupdf.open(pdf) as doc:
        body = clean_pages([page.get_text("text") for page in doc])
    if len(body) < MIN_CHARS:
        status = "too_short"
    elif not looks_english(body):
        status = "not_english"
    else:
        status = "ok"
    return article_header(meta) + body, status
