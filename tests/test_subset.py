import pytest

from dne.subset import choose_subset

OK = list(range(1, 301))


def test_subset_nests_first_articles_and_gold():
    s = choose_subset(OK, must_include={250, 260}, n=100, nested_first=50)
    assert len(s) == 100 and s == sorted(s)
    assert set(range(1, 51)) <= set(s)  # 10 ⊂ 50 ⊂ 200: the pilot and prompt runs used the first 50
    assert {250, 260} <= set(s)


def test_subset_deterministic():
    assert choose_subset(OK, {250}, 100, seed=42) == choose_subset(OK, {250}, 100, seed=42)
    assert choose_subset(OK, {250}, 100, seed=42) != choose_subset(OK, {250}, 100, seed=7)


def test_subset_rejects_unknown_gold():
    with pytest.raises(ValueError, match="999"):
        choose_subset(OK, {999}, 100)


def test_subset_rejects_too_small_n():
    with pytest.raises(ValueError, match="n=60"):
        choose_subset(OK, set(range(200, 220)), n=60, nested_first=50)
