"""Map retrieved graphrag text units back to article ids."""
from __future__ import annotations

import pandas as pd


def _doc_of(units: pd.DataFrame) -> dict[str, str]:
    if "document_id" in units:
        return dict(zip(units["id"], units["document_id"]))
    return {u: list(d)[0] for u, d in zip(units["id"], units["document_ids"]) if len(d)}


def text_units_to_aids(unit_ids: list[str], text_units: pd.DataFrame, documents: pd.DataFrame) -> list[int]:
    doc_of = _doc_of(text_units)
    aid_of = {d: int(str(t).split(".")[0]) for d, t in zip(documents["id"], documents["title"])}
    out: list[int] = []
    for u in unit_ids:
        aid = aid_of.get(doc_of.get(u))
        if aid is not None and aid not in out:
            out.append(aid)
    return out

