"""Harvest the World Tennis (ex-ITF) Coaching & Sport Science Review archive (CC BY 4.0).

The site sits behind Cloudflare, so plain HTTP gets 403. This drives a real Chrome over CDP
(started with --remote-debugging-port=9222, or the address in ITF_CDP) and fetches in-page.

    python harvest_itf.py issues      # archive -> issues -> article list   (data/raw/itf_ccsr/issues.json)
    python harvest_itf.py meta        # every article page -> meta/<aid>.json
    python harvest_itf.py pdf         # primary PDF galley per article -> pdf/<aid>_<gid>.pdf
    python harvest_itf.py all
    python harvest_itf.py pdf_en      # the galley labelled plain "pdf" (English) where the primary link was ES/FR

Resumable: anything already on disk is skipped. Polite: ~8 s per article page or PDF (raw fetches at ~2 s got Cloudflare-challenged).
Stops (does not work around it) if Cloudflare shows an interactive challenge.
Raw data only; never edit files under data/raw — derived text goes to data/processed.
"""
from __future__ import annotations

import base64
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "https://itfcoachingreview.com/index.php/journal"
import os
CDP = os.environ.get("ITF_CDP", "http://127.0.0.1:9222")  # a Chrome started with --remote-debugging-port
RAW = Path(__file__).resolve().parents[1] / "data/raw/itf_ccsr"
LOG = RAW / "harvest.log"


def log(msg: str) -> None:
    line = f"{datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def pause(lo: float = 1.6, hi: float = 2.6) -> None:
    time.sleep(random.uniform(lo, hi))


class Blocked(RuntimeError):
    pass


def goto(pg, url: str, ready: str) -> None:
    """Navigate and wait for `ready` (a CSS selector). Cloudflare's passive check clears on its
    own in a few seconds; an interactive one (Turnstile iframe) means stop."""
    for attempt in range(3):
        try:
            pg.goto(url, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:  # network hiccup: back off and retry
            log(f"  goto failed ({e.__class__.__name__}), retry {attempt + 1}")
            time.sleep(10 * (attempt + 1))
            continue
        for _ in range(30):
            if pg.locator(ready).count():
                return
            if pg.locator("iframe[src*='challenges.cloudflare.com']").count():
                raise Blocked(f"interactive Cloudflare challenge on {url}")
            pg.wait_for_timeout(1000)
        log(f"  {ready!r} never appeared on {url}, retry {attempt + 1}")
        time.sleep(15)
    raise RuntimeError(f"could not load {url}")


ISSUE_JS = """() => {
  const out = {title: (document.querySelector('.page_title, h1')||{}).textContent?.trim() || document.title,
               published: (document.querySelector('.published .value')||{}).textContent?.trim() || null,
               articles: []};
  for (const s of document.querySelectorAll('.obj_article_summary')) {
    const a = s.querySelector('.title a');
    if (!a) continue;
    const sec = s.closest('.section');
    out.articles.push({
      url: a.href, title: a.textContent.trim().replace(/\\s+/g, ' '),
      section: sec ? (sec.querySelector('h2,h3')||{}).textContent?.trim() || null : null,
      galleys: [...s.querySelectorAll('.galleys_links a')].map(g => ({url: g.href, label: g.textContent.trim()})),
    });
  }
  return out;
}"""

META_JS = """() => {
  const m = {};
  for (const e of document.querySelectorAll("meta[name^='citation_'], meta[name^='DC.']")) {
    (m[e.name] = m[e.name] || []).push(e.content);
  }
  const galleys = [...document.querySelectorAll('.galleys_links a, a.obj_galley_link')]
    .map(g => ({url: g.href, label: g.textContent.trim()}));
  return {meta: m, galleys};
}"""

PDF_JS = """async (url) => {
  const r = await fetch(url, {credentials: 'include'});
  const buf = new Uint8Array(await r.arrayBuffer());
  let s = ''; const CH = 0x8000;
  for (let i = 0; i < buf.length; i += CH) s += String.fromCharCode.apply(null, buf.subarray(i, i + CH));
  return {status: r.status, type: r.headers.get('content-type'), b64: btoa(s)};
}"""


def ids(url: str) -> tuple[str, str | None]:
    tail = url.split("/article/view/")[-1].split("/article/download/")[-1].strip("/").split("/")
    return tail[0], (tail[1] if len(tail) > 1 else None)


def step_issues(pg) -> None:
    out = RAW / "issues.json"
    if out.exists():
        log(f"issues.json exists ({len(json.loads(out.read_text()))} issues), skipping")
        return
    issue_urls: list[str] = []
    page = 1
    while True:
        url = f"{BASE}/issue/archive" + (f"/{page}" if page > 1 else "")
        goto(pg, url, "a[href*='/issue/view/']")
        found = pg.eval_on_selector_all("a[href*='/issue/view/']", "els=>els.map(e=>e.href)")
        new = [u for u in dict.fromkeys(found) if u not in issue_urls]
        issue_urls += new
        has_next = pg.locator(f"a[href$='/issue/archive/{page + 1}']").count() > 0
        log(f"archive page {page}: +{len(new)} issues")
        pause()
        if not has_next:
            break
        page += 1
    issues = []
    for i, u in enumerate(issue_urls, 1):
        goto(pg, u, ".obj_article_summary, .obj_issue_toc")
        d = pg.evaluate(ISSUE_JS)
        d["url"] = u
        issues.append(d)
        log(f"issue {i}/{len(issue_urls)} {d['title'][:50]!r}: {len(d['articles'])} articles")
        pause()
    out.write_text(json.dumps(issues, ensure_ascii=False, indent=1))
    log(f"wrote {out.name}: {len(issues)} issues, {sum(len(x['articles']) for x in issues)} articles")


def articles() -> list[dict]:
    issues = json.loads((RAW / "issues.json").read_text())
    seen, out = set(), []
    for iss in issues:
        for a in iss["articles"]:
            aid, _ = ids(a["url"])
            if aid not in seen:
                seen.add(aid)
                out.append({**a, "aid": aid, "issue_url": iss["url"], "issue_title": iss["title"]})
    return out


def step_meta(pg) -> None:
    d = RAW / "meta"
    d.mkdir(exist_ok=True)
    arts = articles()
    todo = [a for a in arts if not (d / f"{a['aid']}.json").exists()]
    log(f"meta: {len(arts) - len(todo)} done, {len(todo)} to go")
    for i, a in enumerate(todo, 1):
        goto(pg, a["url"], "meta[name='citation_title'], .obj_article_details")
        got = pg.evaluate(META_JS)
        rec = {**a, "issue_galleys": a.pop("galleys", []), **got, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        tmp = d / f"{a['aid']}.json.part"
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        tmp.rename(d / f"{a['aid']}.json")
        if i % 25 == 0 or i == len(todo):
            log(f"meta {i}/{len(todo)}")
        pause()


def step_pdf(pg) -> None:
    md, pd = RAW / "meta", RAW / "pdf"
    pd.mkdir(exist_ok=True)
    recs = [json.loads(p.read_text()) for p in sorted(md.glob("*.json"))]
    todo = []
    for r in recs:
        urls = r["meta"].get("citation_pdf_url") or []
        if not urls:  # fall back to the first galley link labelled pdf
            urls = [g["url"].replace("/article/view/", "/article/download/")
                    for g in (r.get("galleys") or r.get("issue_galleys", [])) if "pdf" in g["label"].lower()][:1]
        if not urls:
            log(f"  no pdf galley for {r['aid']} {r['title'][:40]!r}")
            continue
        aid, gid = ids(urls[0])
        dest = pd / f"{aid}_{gid}.pdf"
        if not dest.exists():
            todo.append((urls[0], dest))
    log(f"pdf: {len(todo)} to download")
    if todo:  # fetch from a page on the site so cookies + Cloudflare clearance apply
        goto(pg, f"{BASE}/issue/archive", "a[href*='/issue/view/']")
    total = 0
    for i, (url, dest) in enumerate(todo, 1):
        res = pg.evaluate(PDF_JS, url)
        data = base64.b64decode(res["b64"])
        if res["status"] != 200 or not data.startswith(b"%PDF"):
            log(f"  skip {url}: status {res['status']} type {res['type']} ({len(data)} B)")
            if b"challenge" in data[:5000].lower() or res["status"] in (403, 429):
                log("  looks like Cloudflare/rate limit: cooling down 120 s and reloading")
                time.sleep(120)
                goto(pg, f"{BASE}/issue/archive", "a[href*='/issue/view/']")
            pause()
            continue
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(dest)
        total += len(data)
        if i % 25 == 0 or i == len(todo):
            log(f"pdf {i}/{len(todo)} ({total / 1e6:.0f} MB this run)")
        pause(6, 8)


def english_galley_url(rec: dict) -> str | None:
    """The English galley is labelled 'pdf' or 'pdf eng' (any case); 'pdf (Spanish)', 'pdf fra (French)' … are not."""
    for g in rec.get("galleys") or rec.get("issue_galleys", []):
        label = " ".join(g["label"].lower().split())
        if label == "pdf" or label.startswith("pdf eng"):
            return g["url"].replace("/article/view/", "/article/download/")
    return None


def step_pdf_en(pg) -> None:
    """Fetch the English galley for articles whose primary PDF link pointed at the ES/FR version."""
    md, pd = RAW / "meta", RAW / "pdf"
    todo = []
    for p in sorted(md.glob("*.json")):
        r = json.loads(p.read_text())
        url = english_galley_url(r)
        if not url:
            continue
        aid, gid = ids(url)
        dest = pd / f"{aid}_{gid}.pdf"
        if not dest.exists():
            todo.append((url, dest))
    log(f"pdf_en: {len(todo)} English galleys to download")
    if todo:
        goto(pg, f"{BASE}/issue/archive", "a[href*='/issue/view/']")
    for i, (url, dest) in enumerate(todo, 1):
        res = pg.evaluate(PDF_JS, url)
        data = base64.b64decode(res["b64"])
        if res["status"] != 200 or not data.startswith(b"%PDF"):
            log(f"  skip {url}: status {res['status']} type {res['type']} ({len(data)} B)")
            if b"challenge" in data[:5000].lower() or res["status"] in (403, 429):
                log("  looks like Cloudflare/rate limit: cooling down 120 s and reloading")
                time.sleep(120)
                goto(pg, f"{BASE}/issue/archive", "a[href*='/issue/view/']")
            pause()
            continue
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(dest)
        if i % 25 == 0 or i == len(todo):
            log(f"pdf_en {i}/{len(todo)}")
        pause(6, 8)


def main() -> None:
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    RAW.mkdir(parents=True, exist_ok=True)
    log(f"=== harvest {step} ===")
    with sync_playwright() as p:
        ctx = p.chromium.connect_over_cdp(CDP).contexts[0]
        pg = ctx.new_page()
        try:
            if step in ("issues", "all"):
                step_issues(pg)
            if step in ("meta", "all"):
                step_meta(pg)
            if step in ("pdf", "all"):
                step_pdf(pg)
            if step in ("pdf_en", "all"):
                step_pdf_en(pg)
        except Blocked as e:
            log(f"STOPPED: {e} — open the site in Chrome by hand, then rerun (it resumes)")
            sys.exit(2)
        finally:
            pg.close()
    log("=== done ===")


if __name__ == "__main__":
    main()
