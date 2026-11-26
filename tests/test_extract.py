import json

import pymupdf

from dne.extract import (ArticleMeta, article_header, clean_pages, english_galley_gid, extract_article,
                         load_meta, looks_english)

BODY = [
    "Forehand preparation starts early.",
    "Split step timing matters.",
    "Recovery footwork after the shot.",
    "Serve placement under pressure.",
]


def _meta(**kw):
    base = dict(aid=1, title="T", authors=["A B"], year=2020, issue="Vol. 1", section="Articles",
                language="en", doi=None)
    base.update(kw)
    return ArticleMeta(**base)


def test_repeated_header_removed():
    pages = [f"ITF Coaching and Sport Science Review\n{b}" for b in BODY]
    out = clean_pages(pages)
    assert "ITF Coaching and Sport Science Review" not in out
    for b in BODY:
        assert b in out


def test_page_number_lines_removed():
    out = clean_pages(["Intro text\n12\n12 players trained daily"])
    assert "12 players trained daily" in out
    assert "\n12\n" not in f"\n{out}\n"


def test_references_cut():
    out = clean_pages(["Body of the article.\nReferences\nSmith, J. (2010). A study."])
    assert "Body of the article." in out
    assert "References" not in out
    assert "Smith" not in out


def test_hyphen_rejoined():
    assert "service" in clean_pages(["A strong serv-\nice return."])


def _pdf(tmp_path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    path = tmp_path / "a.pdf"
    doc.save(path)
    return path


def test_too_short_flagged(tmp_path):
    text, status = extract_article(_pdf(tmp_path, "x" * 100), _meta())
    assert status == "too_short"


SPANISH = "El jugador de tenis debe preparar la raqueta antes de que la pelota llegue a la red. "
ENGLISH = "The player should prepare the racket early and move to the ball with small steps. "


def _lines_pdf(tmp_path, line, n=30, name="p.pdf"):
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(n):
        page.insert_text((40, 40 + i * 20), line[:80])
    path = tmp_path / name
    doc.save(path)
    return path


def test_not_english_flagged(tmp_path):
    text, status = extract_article(_lines_pdf(tmp_path, SPANISH), _meta(language="en"))
    assert status == "not_english"


def test_english_text_ok_even_if_meta_says_es(tmp_path):
    text, status = extract_article(_lines_pdf(tmp_path, ENGLISH), _meta(language="es"))
    assert status == "ok"


def test_looks_english():
    assert looks_english(ENGLISH * 5)
    assert not looks_english(SPANISH * 5)


def test_english_galley_gid():
    def rec(*labels):
        return {"galleys": [{"url": f"https://x/index.php/journal/article/view/10/{i}", "label": lab}
                            for i, lab in enumerate(labels, start=4)]}
    assert english_galley_gid(rec("pdf (Spanish)", "pdf", "pdf (French)")) == "5"
    assert english_galley_gid(rec("PDF SPA (Spanish)", "PDf Eng")) == "5"
    assert english_galley_gid(rec("pdf (Spanish)")) is None


def test_ok_article_has_header(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(30):
        page.insert_text((72, 72 + i * 20), "The backhand slice keeps the ball low and skids.")
    path = tmp_path / "long.pdf"
    doc.save(path)
    text, status = extract_article(path, _meta(aid=42, title="Slice"))
    assert status == "ok"
    assert text.startswith("Title: Slice\nAuthors: A B\nYear: 2020\nIssue: Vol. 1\nArticle id: 42\n\n")


def test_article_header_exact():
    m = _meta(aid=7, title="Heat", authors=["X Y", "Z W"], year=2015, issue="Vol. 23 No. 66 (2015)")
    assert article_header(m) == "Title: Heat\nAuthors: X Y, Z W\nYear: 2015\nIssue: Vol. 23 No. 66 (2015)\nArticle id: 7\n\n"


def test_load_meta_sample(tmp_path):
    raw = {
        "url": "https://itfcoachingreview.com/index.php/journal/article/view/10",
        "title": "Exercises to improve the head position in tennis strokes",
        "section": "Articles",
        "aid": "10",
        "issue_title": "Vol. 28 No. 82 (2020)",
        "meta": {
            "citation_author": ["Manuel  Fernández López "],
            "citation_language": ["en"],
            "citation_date": ["2020/12/31"],
            "citation_title": ["Exercises to improve the head position in tennis strokes"],
        },
    }
    p = tmp_path / "10.json"
    p.write_text(json.dumps(raw))
    m = load_meta(p)
    assert m.aid == 10
    assert m.year == 2020
    assert m.language == "en"
    assert m.authors == ["Manuel Fernández López"]
    assert m.issue == "Vol. 28 No. 82 (2020)"
