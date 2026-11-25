"""GPU energy from `nvidia-smi --query-gpu=timestamp,index,power.draw --format=csv,nounits -l 1`."""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def energy_kwh(power_csv: Path) -> float:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with Path(power_csv).open() as f:
        rows = csv.reader(f)
        next(rows, None)
        for row in rows:
            if len(row) < 3:
                continue
            try:
                t = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                watts = float(row[2].strip().split()[0])
            except ValueError:
                continue
            series[row[1].strip()].append((t, watts))
    joules = 0.0
    for samples in series.values():
        samples.sort()
        for (t0, w0), (t1, w1) in zip(samples, samples[1:]):
            joules += (w0 + w1) / 2 * (t1 - t0)
    return joules / 3.6e6
