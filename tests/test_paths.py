from dne.paths import ROOT, PDF_DIR, find_root


def test_find_root_from_subdir():
    assert find_root(ROOT / "notebooks") == ROOT
    assert PDF_DIR == ROOT / "data/raw/itf_ccsr/pdf"
