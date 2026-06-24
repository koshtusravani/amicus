"""
Pull SCOTUS opinions from the CourtListener v4 API.

I list SCOTUS clusters via the search API, then fetch the lead opinion's full
text. Everything is cached under data/raw/scotus, and the loader is resumable:
if I hit the rate limit it stops cleanly and re-running picks up where it left off.
"""
from __future__ import annotations
import html as html_lib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .. import config

_HTML_FIELDS = ("html_with_citations", "html", "html_columbia",
                "html_lawbox", "xml_harvard", "html_anon_2020")
_TAG_RE = re.compile(r"<[^>]+>")


class RateLimited(Exception):
    pass


def _headers() -> dict:
    token = os.environ.get("COURTLISTENER_TOKEN", "")
    return {"Authorization": f"Token {token}"} if token else {}


def _get(url: str, retries: int = 8) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_headers())
            with urllib.request.urlopen(req) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = e.headers.get("Retry-After")
                body = e.read().decode("utf-8", "ignore")[:200]
                if retry_after and retry_after.isdigit():
                    wait = min(int(retry_after), 120)
                else:
                    wait = min(2 ** attempt, 60)
                print(f"  rate limited (429), waiting ~{wait}s | {body}")
                time.sleep(wait + 1)
                continue
            body = e.read().decode("utf-8", "ignore")[:300]
            raise RuntimeError(f"HTTP {e.code} for {url}\n{body}") from None
    raise RateLimited(url)


def _pick_citation(cites: list[str]) -> str:
    for c in cites:
        if re.search(r"\bU\.?\s?S\.?\b", c):
            return c
    return cites[0] if cites else ""


def _extract_text(detail: dict) -> str:
    if detail.get("plain_text"):
        return detail["plain_text"]
    for f in _HTML_FIELDS:
        if detail.get(f):
            return html_lib.unescape(_TAG_RE.sub("", detail[f]))
    return ""


def _normalize_type(raw: str, per_curiam: bool) -> str:
    t = (raw or "").lower()
    if per_curiam:
        return "per-curiam"
    if "dissent" in t:
        return "dissent"
    if "concur" in t:
        return "concurrence"
    if any(k in t for k in ("lead", "majority", "combined")):
        return "majority"
    return t or "opinion"


def _lead_first(opinions: list[dict]) -> list[dict]:
    def rank(o: dict) -> int:
        t = (o.get("type") or "").lower()
        if o.get("per_curiam"):
            return 0
        if any(k in t for k in ("lead", "majority", "combined")):
            return 1
        if "concur" in t:
            return 3
        if "dissent" in t:
            return 4
        return 2
    return sorted(opinions, key=rank)


def _build_record(res: dict) -> dict | None:
    cluster_id = res.get("cluster_id")
    if cluster_id is None:
        return None
    cache = config.RAW_DIR / f"{cluster_id}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    year = int(res["dateFiled"][:4]) if res.get("dateFiled") else None

    opinions = []
    for o in _lead_first(res.get("opinions", []))[:config.CL_MAX_OPINIONS]:
        oid = o.get("id")
        if oid is None:
            continue
        detail = _get(f"{config.CL_BASE}/opinions/{oid}/")
        time.sleep(config.CL_SLEEP)
        text = _extract_text(detail)
        if not text.strip():
            continue
        opinions.append({
            "opinion_id": oid,
            "opinion_type": _normalize_type(o.get("type", ""), o.get("per_curiam", False)),
            "text": text,
        })

    return {
        "cluster_id": cluster_id,
        "case_name": res.get("caseName", ""),
        "citation": _pick_citation(res.get("citation") or []),
        "year": year,
        "court": res.get("court", "Supreme Court of the United States"),
        "opinions": opinions,
    }


def fetch_scotus(n_clusters: int = config.DEV_SLICE) -> list[dict]:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    params = urllib.parse.urlencode(
        {"type": "o", "court": config.CL_COURT, "order_by": config.CL_ORDER_BY}
    )
    url = f"{config.CL_BASE}/search/?{params}"

    records: list[dict] = []
    while url and len(records) < n_clusters:
        page = _get(url)
        for res in page.get("results", []):
            if len(records) >= n_clusters:
                break
            try:
                rec = _build_record(res)
            except RateLimited as e:
                print(f"\nHit the rate limit at {e}.")
                print(f"Cached {len(records)} clusters this run. Re-run later to resume.")
                return records
            if rec and rec["opinions"]:
                (config.RAW_DIR / f"{rec['cluster_id']}.json").write_text(
                    json.dumps(rec, indent=2), encoding="utf-8"
                )
                records.append(rec)
                print(f"  [{len(records)}/{n_clusters}] {rec['case_name']} ({rec['year']})")
        url = page.get("next")
        time.sleep(config.CL_SLEEP)
    return records


if __name__ == "__main__":
    recs = fetch_scotus()
    print(f"\nCached {len(recs)} clusters this run to {config.RAW_DIR}")
    if recs:
        r = recs[0]
        print(f"{r['case_name']} | {r['citation']} | {r['year']}")
        for o in r["opinions"]:
            print(f"  {o['opinion_type']}: {len(o['text'])} chars")