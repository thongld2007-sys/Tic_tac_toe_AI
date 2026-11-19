"""Unit test cho thực nghiệm P4.3a — Bonus B4 (Nước bắt buộc mở rộng & Song song hóa theo gốc)."""

from __future__ import annotations

from pathlib import Path

import yaml

from experiments.p43a_b4 import (
    DEFAULT_CSV_IN,
    DEFAULT_ROOTPAR_CSV_IN,
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
from experiments.run import normalize_config, run_experiment
from src.agents.base import Budget
from src.agents.mcts_parallel import RootParallelMCTSAgent, shutdown_root_process_pools
from src.games.gomoku import Gomoku


def test_p43a_yaml_configs_valid() -> None:
    """Hai file cấu hình P4.3a hợp lệ và bung đủ 12 + 4 cặp đấu × 200 ván."""
    cfg_path = Path("configs/p43a_b4_9x9.yaml")
    assert cfg_path.exists()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = normalize_config(raw)
    assert cfg["name"] == "p43a_b4_9x9"
    assert cfg["games"] == 200
    assert len(cfg["matchups"]) == 12

    rp_path = Path("configs/p43a_b4_rootpar_9x9.yaml")
    assert rp_path.exists()
    rp_raw = yaml.safe_load(rp_path.read_text(encoding="utf-8"))
    rp_cfg = normalize_config(rp_raw)
    assert rp_cfg["name"] == "p43a_b4_rootpar_9x9"
    assert rp_cfg["games"] == 200
    assert len(rp_cfg["matchups"]) == 4


def test_mcts_parallel_in_experiment_runner_and_uct_variant() -> None:
    """Runner nhận diện `mcts_parallel`, hỗ trợ `uct_variant`, callable rollout và `close()`."""
    yaml_text = """
name: test_rootpar_runner
seed: 42
games: 2
swap_first: true
opening_plies: 2
workers: 1
game:
  type: gomoku
  size: 9
matchups:
  - a:
      type: mcts_parallel
      num_workers: 2
      uct_variant: ucb1_tuned
      rollout: l2_b4
      c: 1.0
      use_candidates: true
      candidate_radius: 2
      split_iterations: true
      executor: thread
      budget:
        iterations: 40
    b:
      type: mcts_parallel
      num_workers: 1
      rollout: l2_b4
      c: 1.0
      use_candidates: true
      candidate_radius: 2
      budget:
        iterations: 20
"""
    cfg = normalize_config(yaml.safe_load(yaml_text))
    rows = run_experiment(cfg, workers=1)
    assert len(rows) == 2

    # Kiểm tra truyền hàm callable trực tiếp vào rollout và đóng pool
    state = Gomoku(size=9)
    state.play((4, 4))
    custom_agent = RootParallelMCTSAgent(
        num_workers=2,
        uct_variant="ucb1_tuned",
        rollout=lambda s, rng: 0.0,
        split_iterations=True,
        seed=7,
    )
    assert custom_agent.get_params()["uct_variant"] == "ucb1_tuned"
    mv = custom_agent.select_move(state, Budget(iterations=20))
    assert mv in state.legal_moves()
    custom_agent.close()
    shutdown_root_process_pools()


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


def test_build_summary_and_speed_rows_synthetic() -> None:
    """Kiểm tra pipeline tạo bảng summary và speed từ dữ liệu mẫu nhỏ."""
    synthetic_rows = []
    for base_tag in ("l2", "l3", "l4"):
        after_tag = f"{base_tag}_b4"
        for i in range(4):
            synthetic_rows.append(
                {
                    "agent_a": f"mcts[c=1.0,rollout={after_tag},iterations=200]",
                    "agent_b": f"mcts[c=1.0,rollout={base_tag},iterations=200]",
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
                    "agent_a": f"mcts[c=1.0,rollout={after_tag},time_s=0.08]",
                    "agent_b": f"mcts[c=1.0,rollout={base_tag},time_s=0.08]",
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
                    "agent_a": f"mcts[c=1.0,rollout={after_tag},iterations=200]",
                    "agent_b": "mcts[c=1.0,rollout=l0,iterations=200]",
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
                    "agent_a": f"mcts[c=1.0,rollout={base_tag},iterations=200]",
                    "agent_b": "mcts[c=1.0,rollout=l0,iterations=200]",
                    "score_a": "1.0" if i < 2 else "0.0",
                    "a_moves": "10",
                    "a_nodes": "2010",
                    "a_time_s": "0.5",
                    "b_moves": "10",
                    "b_nodes": "2010",
                    "b_time_s": "0.2",
                }
            )

    synthetic_rootpar = []
    for k in (2, 4):
        for i in range(4):
            synthetic_rootpar.append(
                {
                    "agent_a": f"mcts_parallel[num_workers={k},rollout=l2_b4,time_s=0.04]",
                    "agent_b": "mcts_parallel[num_workers=1,rollout=l2_b4,time_s=0.04]",
                    "a_params": f'{{"num_workers": {k}}}',
                    "b_params": '{"num_workers": 1}',
                    "score_a": "1.0" if i < 3 else "0.0",
                    "a_moves": "10",
                    "a_nodes": "3010",
                    "a_time_s": "0.4",
                    "b_moves": "10",
                    "b_nodes": "1510",
                    "b_time_s": "0.4",
                }
            )
            synthetic_rootpar.append(
                {
                    "agent_a": f"mcts_parallel[num_workers={k},rollout=l2_b4,iterations=120]",
                    "agent_b": "mcts_parallel[num_workers=1,rollout=l2_b4,iterations=120]",
                    "a_params": f'{{"num_workers": {k}}}',
                    "b_params": '{"num_workers": 1}',
                    "score_a": "1.0" if i < 3 else "0.0",
                    "a_moves": "10",
                    "a_nodes": "2410",
                    "a_time_s": "0.4",
                    "b_moves": "10",
                    "b_nodes": "1210",
                    "b_time_s": "0.4",
                }
            )

    summary = build_summary_rows(synthetic_rows, rootpar_csv_rows=synthetic_rootpar, seed=42)
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
    shutdown_root_process_pools()


def test_committed_p43a_csvs_reproducible() -> None:
    """Kiểm tra các file CSV kết quả P4.3a đã commit đầy đủ và tái tạo khớp toàn bộ 13 dòng."""
    assert DEFAULT_CSV_IN.exists()
    assert DEFAULT_ROOTPAR_CSV_IN.exists()
    assert DEFAULT_SUMMARY_OUT.exists()
    assert DEFAULT_SPEED_OUT.exists()

    game_rows = load_csv_rows(DEFAULT_CSV_IN)
    assert len(game_rows) == 2400

    rootpar_rows = load_csv_rows(DEFAULT_ROOTPAR_CSV_IN)
    assert len(rootpar_rows) == 800

    committed_summary = load_csv_rows(DEFAULT_SUMMARY_OUT)
    assert len(committed_summary) == 13

    recomputed = build_summary_rows(game_rows, rootpar_csv_rows=rootpar_rows, seed=2026)
    assert len(recomputed) == 13
    for rec, com in zip(recomputed, committed_summary, strict=True):
        assert rec["comparison"] == com["comparison"]
        assert int(rec["games"]) == 200
        assert int(rec["wins_after"]) == int(com["wins_after"])
        assert float(rec["score_after"]) == float(com["score_after"])
        assert float(rec["score_diff"]) == float(com["score_diff"])
        assert float(rec["score_after"]) > float(rec["score_before"])

    committed_speed = load_csv_rows(DEFAULT_SPEED_OUT)
    assert len(committed_speed) == 14
    for srow in committed_speed:
        assert float(srow["open_three_solve_after"]) > float(srow["open_three_solve_before"])
