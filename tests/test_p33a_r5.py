"""Unit test cho module tổng hợp thực nghiệm R5 và Bảng T6 `experiments/p33a_r5.py` (P3.3a)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.p33a_r5 import (
    ROLLOUT_LEVELS,
    T6_E1_COLUMNS,
    T6_SPEED_COLUMNS,
    T6_SUMMARY_COLUMNS,
    benchmark_rollout_speeds,
    evaluate_e1_behavior,
    load_csv_rows,
    main,
    summarize_r5_table_t6,
    write_csv_rows,
)


def _make_synthetic_r5_rows(mode: str) -> list[dict[str, object]]:
    """Tạo dữ liệu CSV giả lập đúng schema của runner cho R5 (`equal_iter` hoặc `equal_time`)."""
    rows: list[dict[str, object]] = []
    budget = {"iterations": 200} if mode == "equal_iter" else {"time_s": 0.08}
    matchup = 0

    for idx, pol in enumerate(ROLLOUT_LEVELS):
        for g in range(6):
            if pol == "l0":
                winner = "a" if g < 3 else "b"
            else:
                winner = "a" if g < min(6, 3 + idx) else "b"
            score = 1.0 if winner == "a" else 0.0
            rows.append(
                {
                    "experiment": f"p33a_r5_{mode}",
                    "game": "gomoku9",
                    "matchup": matchup,
                    "game_index": g,
                    "seed": 1000 + matchup * 10 + g,
                    "agent_a": f"mcts[rollout={pol}]",
                    "agent_b": "mcts[rollout=l0]",
                    "a_params": json.dumps(
                        {
                            "type": "mcts",
                            "c": 1.4,
                            "rollout": pol,
                            "reuse_tree": False,
                            "use_candidates": True,
                            "budget": budget,
                        }
                    ),
                    "b_params": json.dumps(
                        {
                            "type": "mcts",
                            "c": 1.4,
                            "rollout": "l0",
                            "reuse_tree": False,
                            "use_candidates": True,
                            "budget": budget,
                        }
                    ),
                    "first_player": "a" if g % 2 == 0 else "b",
                    "opening_plies": 2,
                    "winner": winner,
                    "score_a": score,
                    "plies": 16,
                    "a_moves": 8,
                    "b_moves": 8,
                    "a_nodes": 1608 - idx * 100,
                    "b_nodes": 1608,
                    "a_time_s": 0.30 + idx * 0.15,
                    "b_time_s": 0.30,
                    "a_max_time_s": 0.05 + idx * 0.02,
                    "b_max_time_s": 0.05,
                }
            )
        matchup += 1

    # Thêm 1 cặp đấu đối đầu trực tiếp (l4 vs l3)
    for g in range(6):
        winner = "a" if g < 4 else ("draw" if g == 4 else "b")
        score = 1.0 if winner == "a" else (0.5 if winner == "draw" else 0.0)
        rows.append(
            {
                "experiment": f"p33a_r5_{mode}",
                "game": "gomoku9",
                "matchup": matchup,
                "game_index": g,
                "seed": 2000 + g,
                "agent_a": "mcts[rollout=l4]",
                "agent_b": "mcts[rollout=l3]",
                "a_params": json.dumps(
                    {
                        "type": "mcts",
                        "c": 1.4,
                        "rollout": "l4",
                        "reuse_tree": False,
                        "use_candidates": True,
                        "budget": budget,
                    }
                ),
                "b_params": json.dumps(
                    {
                        "type": "mcts",
                        "c": 1.4,
                        "rollout": "l3",
                        "reuse_tree": False,
                        "use_candidates": True,
                        "budget": budget,
                    }
                ),
                "first_player": "a" if g % 2 == 0 else "b",
                "opening_plies": 2,
                "winner": winner,
                "score_a": score,
                "plies": 16,
                "a_moves": 8,
                "b_moves": 8,
                "a_nodes": 1608,
                "b_nodes": 1208,
                "a_time_s": 0.64,
                "b_time_s": 0.64,
                "a_max_time_s": 0.08,
                "b_max_time_s": 0.08,
            }
        )
    return rows


def test_summarize_r5_table_t6_computes_ci_and_sims_per_second(tmp_path: Path) -> None:
    """Bảng T6 tổng hợp đúng khoảng tin cậy Wilson/bootstrap, kiểm định z và tốc độ sims/s."""
    iter_csv = tmp_path / "p33a_r5_equal_iter.csv"
    time_csv = tmp_path / "p33a_r5_equal_time.csv"
    out_csv = tmp_path / "p33a_r5_summary_t6.csv"

    iter_rows = _make_synthetic_r5_rows("equal_iter")
    time_rows = _make_synthetic_r5_rows("equal_time")
    cols = list(iter_rows[0].keys())
    write_csv_rows(iter_csv, cols, iter_rows)
    write_csv_rows(time_csv, cols, time_rows)

    summary = summarize_r5_table_t6(iter_csv, time_csv, out_csv=out_csv)
    assert out_csv.exists()
    loaded = load_csv_rows(out_csv)
    assert len(loaded) == len(summary) == 12
    assert set(loaded[0].keys()) == set(T6_SUMMARY_COLUMNS)

    for r in summary:
        assert (
            0.0 <= float(r["win_ci_low"]) <= float(r["win_rate"]) <= float(r["win_ci_high"]) <= 1.0
        )
        assert (
            0.0
            <= float(r["score_ci_low"])
            <= float(r["mean_score"])
            <= float(r["score_ci_high"])
            <= 1.0
        )
        assert float(r["in_game_sims_per_s"]) > 0.0
        assert (
            float(r["sims_per_s_ci_low"])
            <= float(r["in_game_sims_per_s"])
            <= float(r["sims_per_s_ci_high"])
        )

    # Kiểm tra dòng head_to_head (l4 vs l3) dùng bootstrap ghép cặp: diff_mean = 2 * mean_score - 1
    h2h_rows = [r for r in summary if r["category"] == "head_to_head"]
    assert len(h2h_rows) == 2
    for hr in h2h_rows:
        expected_diff = 2.0 * float(hr["mean_score"]) - 1.0
        assert float(hr["diff_mean"]) == pytest.approx(expected_diff, abs=1e-3)
        assert float(hr["diff_ci_low"]) == pytest.approx(
            2.0 * float(hr["score_ci_low"]) - 1.0, abs=1e-3
        )
        assert float(hr["diff_ci_high"]) == pytest.approx(
            2.0 * float(hr["score_ci_high"]) - 1.0, abs=1e-3
        )


def test_evaluate_e1_behavior_shows_tactical_rollouts_solve_e1(tmp_path: Path) -> None:
    """Trên thế bẫy E1 3×3, L1–L4 chọn đúng ô (2, 2) 100% ngay ở mức ngân sách thấp."""
    out_csv = tmp_path / "e1.csv"
    rows = evaluate_e1_behavior(out_csv=out_csv, num_seeds=8)
    assert out_csv.exists()
    loaded = load_csv_rows(out_csv)
    assert len(loaded) == len(rows)
    assert set(loaded[0].keys()) == set(T6_E1_COLUMNS)

    e1_low = {
        r["rollout"]: r for r in rows if r["position"] == "E1_3x3" and int(r["iterations"]) == 10
    }
    assert set(e1_low.keys()) == set(ROLLOUT_LEVELS)
    for pol in ("l1", "l2", "l3", "l4"):
        assert float(e1_low[pol]["optimal_rate"]) == pytest.approx(1.0)
        assert float(e1_low[pol]["opt_ci_high"]) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="num_seeds"):
        evaluate_e1_behavior(num_seeds=0)


def test_benchmark_rollout_speeds_and_cli_main(tmp_path: Path) -> None:
    """Hàm benchmark_rollout_speeds và CLI main chạy đúng, ghi đủ 3 bảng CSV của P3.3a."""
    res_dir = tmp_path / "results"
    iter_rows = _make_synthetic_r5_rows("equal_iter")
    time_rows = _make_synthetic_r5_rows("equal_time")
    cols = list(iter_rows[0].keys())
    write_csv_rows(res_dir / "p33a_r5_equal_iter.csv", cols, iter_rows)
    write_csv_rows(res_dir / "p33a_r5_equal_time.csv", cols, time_rows)

    code = main(
        [
            "--results-dir",
            str(res_dir),
            "--e1-seeds",
            "4",
            "--speed-iterations",
            "30",
            "--speed-seeds",
            "2",
        ]
    )
    assert code == 0

    for fname, expected_cols in (
        ("p33a_r5_summary_t6.csv", T6_SUMMARY_COLUMNS),
        ("p33a_r5_e1_t6.csv", T6_E1_COLUMNS),
        ("p33a_r5_speed_t6.csv", T6_SPEED_COLUMNS),
    ):
        fpath = res_dir / fname
        assert fpath.exists()
        loaded = load_csv_rows(fpath)
        assert len(loaded) > 0
        assert set(loaded[0].keys()) == set(expected_cols)

    with pytest.raises(ValueError, match="iterations"):
        benchmark_rollout_speeds(iterations=0)
    with pytest.raises(ValueError, match="num_seeds"):
        benchmark_rollout_speeds(num_seeds=0)
