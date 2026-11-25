import pandas as pd

from dne.report_tables import latex_escape, marked_win_rates


def test_marked_win_rates_star_only_when_both_tests_significant():
    wr = pd.DataFrame({"pair": ["a vs b"], "comprehensiveness": [0.24], "diversity": [0.4],
                       "empowerment": [0.11], "directness": [0.5]})
    st = pd.DataFrame({
        "pair": ["a vs b"] * 4,
        "criterion": ["comprehensiveness", "diversity", "empowerment", "directness"],
        "significant": [True, True, False, True],
        "significant_wilcoxon": [True, False, True, True],
    })
    out = marked_win_rates(wr, st)
    assert out.loc["a vs b"].tolist() == ["0.24*", "0.40", "0.11", "0.50*"]


def test_latex_escape_underscore_and_percent():
    assert latex_escape("global_8k 50%") == r"global\_8k 50\%"
