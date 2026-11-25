import pandas as pd
import pytest

from dne.questions import assign_split, load_questions

KNOWN = {10, 11, 12}
HEADER = "id,kind,topic,split,question,gold_aids,notes\n"


def _write(tmp_path, rows):
    p = tmp_path / "q.csv"
    p.write_text(HEADER + "\n".join(rows) + "\n")
    return p


def test_valid_file_loads(tmp_path):
    p = _write(tmp_path, [
        "q1,lookup,fitness_heat,test,How to hydrate in heat?,10,",
        "q2,connecting,serve_return,dev,Serve drills and shoulder load?,10;11,",
        "q3,big_picture,mental,test,How has mental training advice changed?,,",
    ])
    df = load_questions(p, KNOWN)
    assert list(df.id) == ["q1", "q2", "q3"]
    assert df.loc[1, "gold_aids"] == [10, 11]
    assert df.loc[2, "gold_aids"] == []


@pytest.mark.parametrize("row,msg", [
    ("q1,trivia,mental,test,Q?,10,", "q1"),                       # unknown kind
    ("q1,lookup,cooking,test,Q?,10,", "q1"),                      # unknown topic
    ("q1,lookup,mental,test,Q?,99,", "q1"),                       # gold aid not in corpus
    ("q1,lookup,mental,test,Q?,,", "q1"),                         # lookup without gold
    ("q1,lookup,mental,train,Q?,10,", "q1"),                      # unknown split
])
def test_invalid_rows_rejected(tmp_path, row, msg):
    with pytest.raises(ValueError, match=msg):
        load_questions(_write(tmp_path, [row]), KNOWN)


def test_duplicate_id_rejected(tmp_path):
    p = _write(tmp_path, ["q1,big_picture,mental,test,A?,,", "q1,big_picture,mental,test,B?,,"])
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(p, KNOWN)


def test_empty_split_allowed_then_assigned(tmp_path):
    rows = [f"q{i},{k},other,,Question {i}?,{'10' if k == 'lookup' else ''}," for i, k in
            enumerate(["lookup"] * 8 + ["connecting"] * 8 + ["big_picture"] * 8)]
    df = assign_split(load_questions(_write(tmp_path, rows), KNOWN), n_dev_per_kind=5, seed=42)
    counts = df.groupby(["kind", "split"]).size()
    for k in ("lookup", "connecting", "big_picture"):
        assert counts[(k, "dev")] == 5 and counts[(k, "test")] == 3
    again = assign_split(load_questions(_write(tmp_path, rows), KNOWN), n_dev_per_kind=5, seed=42)
    assert list(df.split) == list(again.split)


def test_assign_split_keeps_given(tmp_path):
    rows = ["q1,lookup,other,test,A?,10,", "q2,lookup,other,,B?,10,"]
    df = assign_split(load_questions(_write(tmp_path, rows), KNOWN), n_dev_per_kind=5)
    assert df.set_index("id").loc["q1", "split"] == "test"
    assert df.set_index("id").loc["q2", "split"] == "dev"


def test_select_kept():
    from dne.questions import select_kept
    df = pd.DataFrame({"keep": ["x", "", "Y", "yes", "no", "1"], "id": list("abcdef")})
    assert list(select_kept(df).id) == ["a", "c", "d", "f"]
