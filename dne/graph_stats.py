"""Size and shape of a graphrag index (for the prompt experiment and the cost-vs-size analysis)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def graph_stats(output_dir: Path) -> dict:
    out = Path(output_dir)
    ents = pd.read_parquet(out / "entities.parquet")
    rels = pd.read_parquet(out / "relationships.parquet")
    comms = pd.read_parquet(out / "communities.parquet")
    units = pd.read_parquet(out / "text_units.parquet")
    return {
        "n_entities": len(ents),
        "n_relationships": len(rels),
        "n_communities": len(comms),
        "n_text_units": len(units),
        "entity_type_counts": {str(k): int(v) for k, v in ents["type"].value_counts().items()},
        "mean_degree": float(ents["degree"].mean()) if len(ents) else 0.0,
        "communities_per_level": {int(k): int(v) for k, v in comms["level"].value_counts().sort_index().items()},
    }
