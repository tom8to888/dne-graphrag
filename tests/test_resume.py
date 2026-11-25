from dne.resume import find_cache_archive, pack_cache, restore_cache, session_time_up


def _cache(root):
    (root / "cache/extract_graph").mkdir(parents=True)
    (root / "cache/extract_graph/abc_v4").write_text('{"result": 1}')
    (root / "cache/community_reporting").mkdir()
    return root


def test_pack_restore_roundtrip(tmp_path):
    src = _cache(tmp_path / "run1/graphrag")
    tar = pack_cache(src, tmp_path / "ds/cache.tar.gz")
    dst = tmp_path / "run2/graphrag"
    n = restore_cache(tar, dst)
    assert n == 1
    assert (dst / "cache/extract_graph/abc_v4").read_text() == '{"result": 1}'


def test_find_cache_archive(tmp_path):
    assert find_cache_archive(tmp_path) is None
    (tmp_path / "datasets/u/dne-graphrag-cache").mkdir(parents=True)
    (tmp_path / "datasets/u/dne-graphrag-cache/cache.tar.gz").write_bytes(b"x")
    assert find_cache_archive(tmp_path).name == "cache.tar.gz"


def test_session_time_up():
    assert not session_time_up(t0=0, now=10.9 * 3600, stop_after_hours=11)
    assert session_time_up(t0=0, now=11.0 * 3600, stop_after_hours=11)


def test_find_and_restore_unpacked_cache(tmp_path):
    # Kaggle unpacks an uploaded cache.tar.gz into a cache/ folder (16/11, v8 failed on this)
    ds = tmp_path / "input/datasets/u/dne-graphrag-cache"
    _cache(ds)
    src = find_cache_archive(tmp_path / "input")
    assert src is not None and src.name == "cache"
    n = restore_cache(src, tmp_path / "work/graphrag")
    assert n == 1
    assert (tmp_path / "work/graphrag/cache/extract_graph/abc_v4").read_text() == '{"result": 1}'


def test_find_ignores_unrelated_cache_dirs(tmp_path):
    (tmp_path / "input/other/cache").mkdir(parents=True)  # no extract_graph inside
    assert find_cache_archive(tmp_path / "input") is None


def _output(root):
    (root / "output").mkdir(parents=True)
    for t in ("entities", "relationships", "communities"):
        (root / f"output/{t}.parquet").write_bytes(b"PAR1")
    (root / "output/stats.json").write_text("{}")
    return root


def test_pack_restore_output_roundtrip(tmp_path):
    from dne.resume import pack_output, restore_output
    tar = pack_output(_output(tmp_path / "run1/graphrag"), tmp_path / "ds/output.tar.gz")
    dst = tmp_path / "run2/graphrag"
    assert restore_output(tar, dst) == 3  # parquet tables restored
    assert (dst / "output/entities.parquet").read_bytes() == b"PAR1"


def test_restore_output_from_unpacked_dir(tmp_path):
    from dne.resume import restore_output
    src = _output(tmp_path / "datasets/u/dne-graphrag-cache") / "output"  # Kaggle unpacks tar.gz on upload
    assert restore_output(src, tmp_path / "run2/graphrag") == 3


def test_find_output(tmp_path):
    from dne.resume import find_output
    assert find_output(tmp_path) is None
    _output(tmp_path / "datasets/u/dne-graphrag-cache")
    assert find_output(tmp_path) == tmp_path / "datasets/u/dne-graphrag-cache/output"
    (tmp_path / "other").mkdir()
    (tmp_path / "other/output.tar.gz").write_bytes(b"x")
    assert find_output(tmp_path / "other").name == "output.tar.gz"
