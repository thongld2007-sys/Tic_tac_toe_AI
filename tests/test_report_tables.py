"""Test script sinh bảng T1, T3 cho report (P5.1a)."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.report_tables import (
    RESULTS,
    fmt,
    main,
    render_t1,
    render_t3,
    two_proportion_z,
    win_rates,
)


def test_fmt_uses_thin_spaces() -> None:
    assert fmt(549946) == r"549\,946"
    assert fmt(7) == "7"


def test_render_t1_has_one_row_per_entry() -> None:
    tex = render_t1([("Minimax", "10", "--"), ("Alpha-beta, natural order", "4", r"60.0\%")])
    assert tex.count(r"\\") == 3  # dòng tiêu đề + 2 dòng số liệu
    assert "Alpha-beta, natural order & 4" in tex
    assert tex.rstrip().endswith(r"\end{tabular}")


def test_render_t3_reads_summary(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text(
        "iterations,seeds,seeds_never_lose,tree_losses,minimax_games,minimax_draws\n"
        "2000,20,20,0,80,80\n10,20,0,2024,80,36\n",
        encoding="utf-8",
    )
    tex = render_t3(csv_path)
    assert r"2\,000 & 20/20 & 0 & 80/80" in tex
    assert r"10 & 0/20 & 2\,024 & 36/80" in tex


def test_two_proportion_z() -> None:
    z, p = two_proportion_z(193, 200, 177, 200)
    assert z == pytest.approx(3.037, abs=1e-3)
    assert p == pytest.approx(0.00239, abs=1e-4)
    assert two_proportion_z(200, 200, 200, 200) == (0.0, 1.0)


def test_win_rates_convert_c_to_assignment_scale() -> None:
    rates = win_rates(RESULTS / "p31b_9x9.csv")
    assert rates[(0.5, 100)] == (200, 193, 0, 7)
    assert {c for c, _ in rates} == {0.5, 1.0, 1.4, 2.0}


def test_main_writes_both_tables(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--seeds", "2", "--out-dir", str(tmp_path)]) == 0
    t1 = (tmp_path / "T1.tex").read_text(encoding="utf-8")
    assert r"Minimax & 549\,946" in t1
    assert r"natural order & 18\,297 & 96.7\%" in t1
    assert (tmp_path / "T3.tex").exists()
    assert "z=3.04" in capsys.readouterr().out


def test_result_tables_are_generated_from_csv(tmp_path: Path) -> None:
    from experiments.report_tables import write_result_tables

    written = write_result_tables(tmp_path)
    assert set(written) == {"T2.tex", "T4.tex", "T5.tex", "B2B3.tex", "T6.tex", "T7.tex", "B4.tex"}
    for name in written:
        tex = (tmp_path / name).read_text(encoding="utf-8")
        assert tex.count(r"\begin{tabular}") == 1
        assert "\t" not in tex
    t7 = (tmp_path / "T7.tex").read_text(encoding="utf-8")
    assert "100/0/0 & 50/50 & 50/50" in t7
    t2 = (tmp_path / "T2.tex").read_text(encoding="utf-8")
    assert "No pattern move ordering & 0/0/50" in t2


def test_comparison_label() -> None:
    from experiments.report_tables import comparison_label

    assert comparison_label("l2_b4") == "L2-B4"
    assert comparison_label("rootpar_k4_l2_b4") == "$K=4$"
