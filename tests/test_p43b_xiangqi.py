"""Unit test cho thực nghiệm Cờ tướng P4.3b (`experiments/p43b_xiangqi.py`)."""

from __future__ import annotations

from pathlib import Path

import yaml

from experiments.p43b_xiangqi import (
    DEFAULT_CSV_IN,
    DEFAULT_SUMMARY_OUT,
    SUMMARY_FIELDS,
    bootstrap_mean_ci,
    build_summary_rows,
    load_csv_rows,
    one_sample_score_z_test,
    wilson_ci_95,
)
from experiments.run import normalize_config


def test_p43b_yaml_config_valid() -> None:
    """File cấu hình configs/p43b_xiangqi_exp.yaml hợp lệ và bung đủ 6 cặp đấu × 60 ván."""
    cfg_path = Path("configs/p43b_xiangqi_exp.yaml")
    assert cfg_path.exists()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = normalize_config(raw)
    assert cfg["name"] == "p43b_xiangqi_exp"
    assert cfg["games"] == 60
    assert cfg["swap_first"] is True
    assert cfg["opening_plies"] == 2
    assert len(cfg["matchups"]) == 6


def test_p43b_statistical_helpers_and_synthetic_summary() -> None:
    """Kiểm tra hàm thống kê và pipeline tổng hợp tách Đỏ / Đen."""
    lo, hi = wilson_ci_95(45, 60)
    assert 0.60 < lo < 0.75 < hi < 0.86

    b_lo, b_hi = bootstrap_mean_ci([1.0] * 45 + [0.0] * 15, n_boot=500, seed=42)
    assert 0.60 <= b_lo <= 0.75 <= b_hi <= 0.90

    z_val, p_val = one_sample_score_z_test([1.0] * 50 + [0.0] * 10, mu0=0.5)
    assert z_val > 4.0
    assert p_val < 1e-4

    synthetic_rows = []
    for g in range(6):
        synthetic_rows.append(
            {
                "matchup": "0",
                "agent_a": "alphabeta_xiangqi[time_s=0.1]",
                "agent_b": "mcts_xiangqi[time_s=0.1]",
                "a_params": '{"budget": {"time_s": 0.1}}',
                "b_params": '{"budget": {"time_s": 0.1}}',
                "first_player": "a" if g % 2 == 0 else "b",
                "score_a": "1.0" if g < 5 else "0.5",
                "plies": "50",
                "a_moves": "25",
                "b_moves": "25",
                "a_nodes": "5000",
                "b_nodes": "2500",
                "a_time_s": "2.0",
                "b_time_s": "2.5",
            }
        )
    summary = build_summary_rows(synthetic_rows, seed=42)
    assert len(summary) == 1
    row = summary[0]
    assert set(row.keys()) == set(SUMMARY_FIELDS)
    assert row["red_games"] == 3
    assert row["black_games"] == 3
    assert row["a_time_pct"] == 80.0
    assert row["b_time_pct"] == 100.0


def test_committed_p43b_csvs_reproducible() -> None:
    """Kiểm tra 2 file CSV kết quả P4.3b đã commit đầy đủ và tái tạo khớp 100%."""
    assert DEFAULT_CSV_IN.exists()
    assert DEFAULT_SUMMARY_OUT.exists()

    game_rows = load_csv_rows(DEFAULT_CSV_IN)
    assert len(game_rows) == 360

    committed = load_csv_rows(DEFAULT_SUMMARY_OUT)
    assert len(committed) == 6

    recomputed = build_summary_rows(game_rows, seed=2026)
    assert len(recomputed) == 6
    for rec, com in zip(recomputed, committed, strict=True):
        assert rec["comparison"] == com["comparison"]
        assert int(rec["games"]) == int(com["games"]) == 60
        assert int(rec["wins_a"]) == int(com["wins_a"])
        assert float(rec["score_a"]) == float(com["score_a"])
        assert float(rec["red_score_a"]) == float(com["red_score_a"])
        assert float(rec["black_score_a"]) == float(com["black_score_a"])
