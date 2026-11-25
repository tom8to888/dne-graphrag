import pandas as pd
import pytest

from dne.graph_stats import graph_stats


def test_graph_stats_counts(tmp_path):
    pd.DataFrame({"title": ["KICK SERVE", "SHOULDER", "PERIODISATION"],
                  "type": ["technique", "injury", "technique"], "degree": [2, 1, 1]}).to_parquet(tmp_path / "entities.parquet")
    pd.DataFrame({"source": ["KICK SERVE", "KICK SERVE"], "target": ["SHOULDER", "PERIODISATION"]}).to_parquet(
        tmp_path / "relationships.parquet")
    pd.DataFrame({"title": ["c0", "c1", "c2"], "level": [0, 0, 1]}).to_parquet(tmp_path / "communities.parquet")
    pd.DataFrame({"id": ["t1", "t2", "t3", "t4"]}).to_parquet(tmp_path / "text_units.parquet")

    s = graph_stats(tmp_path)

    assert s["n_entities"] == 3 and s["n_relationships"] == 2 and s["n_communities"] == 3
    assert s["n_text_units"] == 4
    assert s["entity_type_counts"] == {"technique": 2, "injury": 1}
    assert s["mean_degree"] == pytest.approx(4 / 3)
    assert s["communities_per_level"] == {0: 2, 1: 1}
