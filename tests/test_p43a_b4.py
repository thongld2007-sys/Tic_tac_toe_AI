"""Unit test cho thực nghiệm P4.3a — Bonus B4 (Nước bắt buộc mở rộng & Song song hóa theo gốc)."""

from __future__ import annotations

from pathlib import Path

import yaml

from experiments.p43a_b4 import (
    DEFAULT_CSV_IN,
    DEFAULT_SPEED_OUT,
    DEFAULT_SUMMARY_OUT,
    SPEED_FIELDS,
    SUMMARY_FIELDS,
    bootstrap_mean_ci,
    build_speed_rows,
    build_summary_rows,
    load_csv_rows,
    one_sample_score_z_test,
    two_sample_score_z_test,
    wilson_ci_95,
)
from experiments.run import normalize_config


def test_p43a_yaml_config_valid() -> None:
    """File cấu hình configs/p43a_b4_9x9.yaml hợp lệ và bung đủ 12 cặp đấu × 200 ván."""
    cfg_path = Path("configs/p43a_b4_9x9.yaml")
    assert cfg_path.exists()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = normalize_config(raw)
    assert cfg["name"] == "p43a_b4_9x9"
    assert cfg["games"] == 200
    assert len(cfg["matchups"]) == 12


def test_statistical_helpers_p43a() -> None:
    """Kiểm tra Wilson CI, bootstrap CI và kiểm định z một mẫu / hai mẫu."""
    lo, hi = wilson_ci_95(80, 100)
    assert 0.70 < lo < 0.80 < hi < 0.88

    b_lo, b_hi = bootstrap_mean_ci([1.0] * 80 + [0.0] * 20, n_boot=500, seed=42)
    assert 0.70 <= b_lo <= 0.80 <= b_hi <= 0.90

    z1, p1 = one_sample_score_z_test([1.0] * 85 + [0.0] * 15, mu0=0.5)
    assert z1 > 5.0
    assert p1 < 1e-5

    z2, p2 = two_sample_score_z_test([1.0] * 85 + [0.0] * 15, [1.0] * 50 + [0.0] * 50)
    assert z2 > 4.0
    assert p2 < 1e-4


def test_build_summary_and_speed_rows_synthetic(tmp_path: Path) -> None:
    """Kiểm tra pipeline tạo bảng summary và speed từ dữ liệu mẫu nhỏ."""
    synthetic_rows = []
    for base_tag in ("l2", "l3", "l4"):
        after_tag = f"{base_tag}_b4"
        for i in range(4):
            synthetic_rows.append(
                {
                    "agent_a": f"mcts(c=1.0,rollout={after_tag},iterations=200)",
                    "agent_b": f"mcts(c=1.0,rollout={base_tag},iterations=200)",
                    "score_a": "1.0" if i < 3 else "0.0",
                    "a_moves": "10",
                    "a_nodes": "2010",
                    "a_time_s": "1.0",
                    "b_moves": "10",
                    "b_nodes": "2010",
                    "b_time_s": "0.5",
                }
            )
            synthetic_rows.append(
                {
                    "agent_a": f"mcts(c=1.0,rollout={after_tag},time_s=0.08)",
                    "agent_b": f"mcts(c=1.0,rollout={base_tag},time_s=0.08)",
                    "score_a": "1.0" if i < 3 else "0.0",
                    "a_moves": "10",
                    "a_nodes": "1510",
                    "a_time_s": "0.8",
                    "b_moves": "10",
                    "b_nodes": "3010",
                    "b_time_s": "0.8",
                }
            )
            synthetic_rows.append(
                {
                    "agent_a": f"mcts(c=1.0,rollout={after_tag},iterations=200)",
                    "agent_b": "mcts(c=1.0,rollout=l0,iterations=200)",
                    "score_a": "1.0",
                    "a_moves": "10",
                    "a_nodes": "2010",
                    "a_time_s": "1.0",
                    "b_moves": "10",
                    "b_nodes": "2010",
                    "b_time_s": "0.2",
                }
            )
            synthetic_rows.append(
                {
                    "agent_a": f"mcts(c=1.0,rollout={base_tag},iterations=200)",
                    "agent_b": "mcts(c=1.0,rollout=l0,iterations=200)",
                    "score_a": "1.0" if i < 2 else "0.0",
                    "a_moves": "10",
                    "a_nodes": "2010",
                    "a_time_s": "0.5",
                    "b_moves": "10",
                    "b_nodes": "2010",
                    "b_time_s": "0.2",
                }
            )

    summary = build_summary_rows(synthetic_rows, root_parallel_games=2, seed=42)
    assert len(summary) == 13
    for row in summary:
        assert set(row.keys()) == set(SUMMARY_FIELDS)

    speed = build_speed_rows(
        num_rollouts=20,
        mcts_trials=1,
        mcts_iterations=20,
        trap_trials=2,
        trap_iterations=20,
        seed=42,
    )
    assert len(speed) == 14
    for srow in speed:
        assert set(srow.keys()) == set(SPEED_FIELDS)
        assert float(srow["mcts_sims_per_s_after"]) > 0.0
        assert float(srow["mcts_sims_per_s_before"]) > 0.0


def test_committed_p43a_csvs_reproducible() -> None:
    """Kiểm tra 3 file CSV kết quả P4.3a đã commit đầy đủ và tái tạo khớp 9 dòng từ CSV ván."""
    assert DEFAULT_CSV_IN.exists()
    assert DEFAULT_SUMMARY_OUT.exists()
    assert DEFAULT_SPEED_OUT.exists()

    game_rows = load_csv_rows(DEFAULT_CSV_IN)
    assert len(game_rows) == 2400

    committed_summary = load_csv_rows(DEFAULT_SUMMARY_OUT)
    assert len(committed_summary) == 13

    recomputed_9 = build_summary_rows(game_rows, root_parallel_games=0, seed=2026)
    assert len(recomputed_9) == 9
    for rec, com in zip(recomputed_9, committed_summary[:9], strict=True):
        assert rec["comparison"] == com["comparison"]
        assert int(rec["wins_after"]) == int(com["wins_after"])
        assert float(rec["score_after"]) == float(com["score_after"])
        assert float(rec["score_diff"]) == float(com["score_diff"])
        assert float(rec["score_after"]) > float(rec["score_before"])

    committed_speed = load_csv_rows(DEFAULT_SPEED_OUT)
    assert len(committed_speed) == 14
    for srow in committed_speed:
        assert float(srow["open_three_solve_after"]) > float(srow["open_three_solve_before"])
