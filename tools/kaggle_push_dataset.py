"""Build data/kaggle_dataset/ from the extracted text and create (or version) the private Kaggle dataset."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dne.kaggle_io import DATASET_SLUG, build_dataset_dir, kaggle_username  # noqa: E402
from dne.paths import ROOT  # noqa: E402

out = build_dataset_dir(ROOT / "data/kaggle_dataset")
ds_id = f"{kaggle_username()}/{DATASET_SLUG}"
exists = subprocess.run(["kaggle", "datasets", "files", ds_id], capture_output=True, text=True).returncode == 0
cmd = (["kaggle", "datasets", "version", "-p", str(out), "-m", "update", "--dir-mode", "zip"] if exists
       else ["kaggle", "datasets", "create", "-p", str(out), "--dir-mode", "zip"])
print("running:", " ".join(cmd[:3]), ds_id)
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-800:])
sys.exit(r.returncode)
