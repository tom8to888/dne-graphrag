import pytest

from dne.power import energy_kwh


def test_energy_kwh_constant_power(tmp_path):
    lines = ["timestamp, index, power.draw [W]"]
    for s in range(3601):
        ts = f"2026/11/16 10:{s // 60 % 60:02d}:{s % 60:02d}.000" if s < 3600 else "2026/11/16 11:00:00.000"
        for gpu in (0, 1):
            lines.append(f"{ts}, {gpu}, 70.00")
    p = tmp_path / "power.csv"
    p.write_text("\n".join(lines))
    assert energy_kwh(p) == pytest.approx(0.14, rel=1e-3)


def test_energy_kwh_skips_bad_rows(tmp_path):
    p = tmp_path / "power.csv"
    p.write_text("timestamp, index, power.draw [W]\n2026/11/16 10:00:00.000, 0, [N/A]\n"
                 "2026/11/16 10:00:00.000, 0, 100\n2026/11/16 11:00:00.000, 0, 100\n")
    assert energy_kwh(p) == pytest.approx(0.1, rel=1e-3)
