import pandas as pd

from dne.retrieval import hit_at_k, text_units_to_aids


def test_hit_at_k():
    assert hit_at_k([5, 7, 9], [9], k=3)
    assert not hit_at_k([5, 7, 9], [9], k=2)
    assert not hit_at_k([], [9], k=5)


def test_text_units_to_aids_dedup_order():
    units = pd.DataFrame({"id": ["u1", "u2", "u3", "u4"], "document_id": ["d2", "d1", "d2", "d3"]})
    docs = pd.DataFrame({"id": ["d1", "d2", "d3"], "title": ["10.txt", "42.txt", "7.txt"]})
    assert text_units_to_aids(["u3", "u2", "u1", "u4"], units, docs) == [42, 10, 7]


def test_text_units_to_aids_list_column():
    units = pd.DataFrame({"id": ["u1"], "document_ids": [["d1"]]})
    docs = pd.DataFrame({"id": ["d1"], "title": ["10.txt"]})
    assert text_units_to_aids(["u1", "unknown"], units, docs) == [10]
