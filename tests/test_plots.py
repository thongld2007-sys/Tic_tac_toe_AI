"""Unit test cho script vẽ biểu đồ và tổng hợp bảng T4, T5 `experiments/plots.py` (P3.2b)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.plots import (
    R3_BUDGETS,
    R3_COEFFICIENTS,
    T4_COLUMNS,
    T5_COLUMNS,
    generate_table_t4,
    load_csv_rows,
    main,
    plot_r3_from_csv,
    write_csv_rows,
)
from src.agents.mcts import FINAL_SELECTIONS


def _make_synthetic_r3_rows(board: str) -> list[dict[str, object]]:
    """Tạo dữ liệu CSV giả lập đúng schema cho R3 (20 tổ hợp × 4 ván)."""
    rows: list[dict[str, object]] = []
    matchup = 0
    for c_unit in R3_COEFFICIENTS:
        c_code = c_unit * 2.0
        for iters in R3_BUDGETS:
            for g in range(4):
                winner = "a" if g < 3 else "b"
                rows.append(
                    {
                        "experiment": f"p31b_{board}",
                        "game": board,
                        "matchup": matchup,
                        "game_index": g,
                        "seed": 1000 + g,
                        "agent_a": f"mcts[c={c_code},iterations={iters}]",
                        "agent_b": "random",
                        "a_params": json.dumps(
                            {
                                "type": "mcts",
                                "c": c_code,
                                "budget": {"iterations": iters},
                            }
                        ),
                        "b_params": json.dumps({"type": "random"}),
                        "first_player": "a" if g % 2 == 0 else "b",
                        "opening_plies": 0,
                        "winner": winner,
                        "score_a": 1.0 if winner == "a" else 0.0,
                        "plies": 7,
                        "a_moves": 4,
                        "b_moves": 3,
                        "a_nodes": iters * 4,
                        "b_nodes": 0,
                        "a_time_s": 0.01,
                        "b_time_s": 0.001,
                        "a_max_time_s": 0.005,
                        "b_max_time_s": 0.001,
                    }
                )
            matchup += 1
    return rows


def _make_synthetic_r4_rows(board: str) -> list[dict[str, object]]:
    """Tạo dữ liệu CSV giả lập đúng schema cho R4 (vs baseline + head-to-head)."""
    rows: list[dict[str, object]] = []
    matchup = 0
    for sel in FINAL_SELECTIONS:
        for iters in (50, 100):
            for g in range(6):
                if sel == "max" and g >= 3:
                    winner, score = "b", 0.0
                elif g < 4:
                    winner, score = "a", 1.0
                else:
                    winner, score = "draw", 0.5
                rows.append(
                    {
                        "experiment": f"p32b_r4_{board}",
                        "game": board,
                        "matchup": matchup,
                        "game_index": g,
                        "seed": 2000 + g,
                        "agent_a": f"mcts[selection={sel},iterations={iters}]",
                        "agent_b": "alphabeta",
                        "a_params": json.dumps(
                            {
                                "type": "mcts",
                                "c": 0.5,
                                "selection": sel,
                                "budget": {"iterations": iters},
                            }
                        ),
                        "b_params": json.dumps({"type": "alphabeta"}),
                        "first_player": "a" if g % 2 == 0 else "b",
                        "opening_plies": 1,
                        "winner": winner,
                        "score_a": score,
                        "plies": 8,
                        "a_moves": 4,
                        "b_moves": 4,
                        "a_nodes": iters * 4,
                        "b_nodes": 100,
                        "a_time_s": 0.01,
                        "b_time_s": 0.002,
                        "a_max_time_s": 0.005,
                        "b_max_time_s": 0.001,
                    }
                )
            matchup += 1

    # Thêm 1 cặp đấu đối đầu trực tiếp (robust vs max)
    for g in range(6):
        winner = "a" if g < 4 else ("draw" if g == 4 else "b")
        score = 1.0 if winner == "a" else (0.5 if winner == "draw" else 0.0)
        rows.append(
            {
                "experiment": f"p32b_r4_{board}",
                "game": board,
                "matchup": matchup,
                "game_index": g,
                "seed": 3000 + g,
                "agent_a": "mcts[selection=robust]",
                "agent_b": "mcts[selection=max]",
                "a_params": json.dumps(
                    {
                        "type": "mcts",
                        "c": 0.5,
                        "selection": "robust",
                        "budget": {"iterations": 100},
                    }
                ),
                "b_params": json.dumps(
                    {
                        "type": "mcts",
                        "c": 0.5,
                        "selection": "max",
                        "budget": {"iterations": 100},
                    }
                ),
                "first_player": "a" if g % 2 == 0 else "b",
                "opening_plies": 1,
                "winner": winner,
                "score_a": score,
                "plies": 8,
                "a_moves": 4,
                "b_moves": 4,
                "a_nodes": 400,
                "b_nodes": 400,
                "a_time_s": 0.01,
                "b_time_s": 0.01,
                "a_max_time_s": 0.005,
                "b_max_time_s": 0.005,
            }
        )
    return rows


def test_generate_table_t4_contains_root_stats_and_e1_position(tmp_path: Path) -> None:
    """Bảng T4 trích xuất đúng các trường N, W, q, uct từ last_stats['root'] cho thế E1."""
    out_csv = tmp_path / "t4.csv"
    rows = generate_table_t4(out_csv=out_csv, num_seeds=15)
    assert out_csv.exists()
    loaded = load_csv_rows(out_csv)
    assert len(loaded) == len(rows)
    assert set(loaded[0].keys()) == set(T4_COLUMNS)

    e1_rows = [r for r in rows if r["position"] == "E1_trap"]
    assert len(e1_rows) == 5  # 5 ô trống hợp lệ ở thế E1
    opt_row = next(r for r in e1_rows if r["move"] == "(2,2)")
    assert opt_row["is_optimal"] == 1
    assert opt_row["N"] > 0
    assert -1.0 <= opt_row["q_signed"] <= 1.0
    assert 0.0 <= opt_row["q_unit"] <= 1.0


def test_summarize_r4_table_t5_and_plot_pdf_generation(tmp_path: Path) -> None:
    """Tổng hợp Bảng T5 và sinh đủ các file vector PDF (F1, F2, F3) hợp lệ."""
    res_dir = tmp_path / "results"
    fig_dir = tmp_path / "figures"

    r3_cols = list(_make_synthetic_r3_rows("3x3")[0].keys())
    write_csv_rows(res_dir / "p31b_3x3.csv", r3_cols, _make_synthetic_r3_rows("3x3"))
    write_csv_rows(res_dir / "p31b_9x9.csv", r3_cols, _make_synthetic_r3_rows("9x9"))

    r4_cols = list(_make_synthetic_r4_rows("3x3")[0].keys())
    write_csv_rows(res_dir / "p32b_r4_3x3.csv", r4_cols, _make_synthetic_r4_rows("3x3"))
    write_csv_rows(res_dir / "p32b_r4_9x9.csv", r4_cols, _make_synthetic_r4_rows("9x9"))

    code = main(["--results-dir", str(res_dir), "--out-dir", str(fig_dir), "--t4-seeds", "10"])
    assert code == 0

    for pdf_name in ("F1.pdf", "F2.pdf", "F3.pdf"):
        pdf_path = fig_dir / pdf_name
        assert pdf_path.exists()
        raw_bytes = pdf_path.read_bytes()
        assert raw_bytes.startswith(b"%PDF-")
        assert len(raw_bytes) > 1000

    t5_rows = load_csv_rows(res_dir / "p32b_r4_summary_t5.csv")
    assert len(t5_rows) > 0
    assert set(t5_rows[0].keys()) == set(T5_COLUMNS)


def test_plot_r3_from_csv_rejects_incomplete_grid(tmp_path: Path) -> None:
    """Khi file CSV thiếu tổ hợp (c, iteration), plot_r3_from_csv ném ValueError."""
    bad_csv = tmp_path / "bad.csv"
    rows = _make_synthetic_r3_rows("3x3")[:4]
    write_csv_rows(bad_csv, list(rows[0].keys()), rows)
    with pytest.raises(ValueError, match="Thiếu lưới tham số"):
        plot_r3_from_csv(bad_csv, tmp_path / "out.pdf", "Test")
