"""Kiểm thử đơn vị cho các agent Cờ tướng (P4.2b): AlphaBetaXiangqi và XiangqiMCTSAgent."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.run import ConfigError, build_agent, normalize_config, run_experiment, write_csv
from src.agents.base import Budget
from src.agents.mcts import ROLLOUT_POLICIES, MCTSAgent, compute_move_prior, resolve_rollout
from src.agents.random_agent import RandomAgent
from src.agents.xiangqi_agents import (
    PIECE_VALUES,
    AlphaBetaXiangqi,
    XiangqiMCTSAgent,
    evaluate_xiangqi,
    rollout_xiangqi_l4,
    score_xiangqi_move,
)
from src.games.tictactoe import TicTacToe
from src.games.xiangqi import BLACK, RED, ROOK, Xiangqi, parse_iccs

# ---------------------------------------------------------------------------
# Hàm lượng giá vật chất + vị trí (PST)
# ---------------------------------------------------------------------------


def test_evaluate_xiangqi_start_position_is_symmetric() -> None:
    st = Xiangqi()
    assert evaluate_xiangqi(st, RED) == pytest.approx(0.0)
    assert evaluate_xiangqi(st, BLACK) == pytest.approx(0.0)
    assert evaluate_xiangqi(st) == pytest.approx(0.0)


def test_evaluate_xiangqi_material_and_perspective() -> None:
    # Đỏ hơn Đen đúng 1 Xe (Đen mất Xe a9)
    fen = "1nbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    st = Xiangqi(fen)
    red_score = evaluate_xiangqi(st, RED)
    black_score = evaluate_xiangqi(st, BLACK)
    assert red_score > PIECE_VALUES[ROOK] * 0.8
    assert black_score == pytest.approx(-red_score)


def test_evaluate_xiangqi_crossed_pawn_worth_more_than_uncrossed() -> None:
    # Hai thế cờ chỉ khác vị trí Tốt Đỏ: chưa qua sông (e3) vs đã qua sông (e6)
    uncrossed = Xiangqi("4k4/9/9/9/9/9/4P4/9/9/4K4 w - - 0 1")
    crossed = Xiangqi("4k4/9/9/4P4/9/9/9/9/9/4K4 w - - 0 1")
    assert evaluate_xiangqi(crossed, RED) > evaluate_xiangqi(uncrossed, RED)


def test_evaluate_xiangqi_validation() -> None:
    st = Xiangqi()
    with pytest.raises(TypeError):
        evaluate_xiangqi(TicTacToe())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evaluate_xiangqi(st, 0)


# ---------------------------------------------------------------------------
# AlphaBetaXiangqi
# ---------------------------------------------------------------------------


def test_alphabeta_xiangqi_validation_and_params() -> None:
    agent = AlphaBetaXiangqi(
        default_depth=2,
        quiescence_depth=1,
        use_move_ordering=True,
        use_iterative_deepening=False,
        use_tt=True,
    )
    assert agent.get_params() == {
        "default_depth": 2,
        "quiescence_depth": 1,
        "use_move_ordering": True,
        "use_iterative_deepening": False,
        "use_tt": True,
    }
    agent.clear_table()

    with pytest.raises(ValueError):
        AlphaBetaXiangqi(default_depth=0)
    with pytest.raises(ValueError):
        AlphaBetaXiangqi(quiescence_depth=-1)
    with pytest.raises(ValueError):
        AlphaBetaXiangqi(use_tt="yes")  # type: ignore[arg-type]

    st = Xiangqi()
    with pytest.raises(TypeError):
        agent.select_move(TicTacToe(), Budget(depth=1))
    with pytest.raises(ValueError):
        agent.select_move(st, Budget(depth=0))
    with pytest.raises(ValueError):
        agent.select_move(st, Budget(depth=2.5))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        agent.select_move(st, Budget(time_s=0.0))


def test_alphabeta_xiangqi_preserves_state_and_stats() -> None:
    st = Xiangqi()
    before_fen = st.to_fen()
    before_hash = st.hash()

    agent = AlphaBetaXiangqi(default_depth=2, quiescence_depth=1)
    move = agent.select_move(st, Budget(depth=2))

    assert move in st.legal_moves()
    assert st.to_fen() == before_fen
    assert st.hash() == before_hash
    assert agent.last_value is not None
    assert agent.last_stats["nodes"] >= 1
    assert agent.last_stats["depth"] == 2
    assert agent.last_stats["time_s"] >= 0.0


def test_alphabeta_xiangqi_finds_mate_in_one() -> None:
    # Tướng Đen ở e9 bị kẹt hàng trên; Xe Đỏ a8 -> a9 chiếu bí (vì Tướng Đỏ ở e0 giữ cột e)
    # và Xe Đỏ b8 chặn hàng 8.
    fen = "4k4/RR7/9/9/9/9/9/9/9/3K5 w - - 0 1"
    st = Xiangqi(fen)
    agent = AlphaBetaXiangqi(default_depth=2, quiescence_depth=0)
    mv = agent.select_move(st, Budget(depth=2))
    st.play(mv)
    assert st.is_terminal()
    assert st.winner() == RED


def test_alphabeta_xiangqi_captures_free_rook() -> None:
    # Xe Đỏ e2 có thể ăn không Xe Đen e7
    fen = "3k5/9/4r4/9/9/9/9/4R4/9/5K3 w - - 0 1"
    st = Xiangqi(fen)
    agent = AlphaBetaXiangqi(default_depth=2, quiescence_depth=1)
    mv = agent.select_move(st, Budget(depth=2))
    assert mv == parse_iccs("e2e7")


def test_alphabeta_xiangqi_time_budget() -> None:
    st = Xiangqi()
    agent = AlphaBetaXiangqi(default_depth=2, quiescence_depth=1)
    mv = agent.select_move(st, Budget(time_s=0.08))
    assert mv in st.legal_moves()
    assert agent.last_stats["nodes"] >= 1


# ---------------------------------------------------------------------------
# Rollout L4 và XiangqiMCTSAgent
# ---------------------------------------------------------------------------


def test_rollout_xiangqi_l4_registration_and_fresh_subprocess() -> None:
    assert "xiangqi_l4" in ROLLOUT_POLICIES
    name, fn = resolve_rollout("xiangqi_l4")
    assert name == "xiangqi_l4"
    assert fn is rollout_xiangqi_l4

    # Kiểm tra trong tiến trình Python mới (chưa import src.agents.xiangqi_agents trước)
    code = (
        "from src.agents.mcts import resolve_rollout; "
        "name, fn = resolve_rollout('xiangqi_l4'); "
        "assert name == 'xiangqi_l4' and callable(fn)"
    )
    proc = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_rollout_xiangqi_l4_preserves_state_and_bounds() -> None:
    st = Xiangqi()
    before_fen = st.to_fen()
    before_hash = st.hash()
    rng = random.Random(42)

    val = rollout_xiangqi_l4(st, rng, cutoff_depth=4)
    assert -1.0 <= val <= 1.0
    assert st.to_fen() == before_fen
    assert st.hash() == before_hash

    val0 = rollout_xiangqi_l4(st, rng, cutoff_depth=0)
    assert val0 == pytest.approx(0.0)

    with pytest.raises(TypeError):
        rollout_xiangqi_l4(TicTacToe(), rng)
    with pytest.raises(ValueError):
        rollout_xiangqi_l4(st, rng, cutoff_depth=-1)
    with pytest.raises(ValueError):
        rollout_xiangqi_l4(st, rng, max_moves=-1)
    with pytest.raises(ValueError):
        rollout_xiangqi_l4(st, rng, max_moves=True)
    with pytest.raises(ValueError):
        rollout_xiangqi_l4(st, rng, tau=0.0)


def test_mcts_xiangqi_finds_mate_and_supports_progressive_bias() -> None:
    # Thế chiếu bí 1 nước
    fen = "4k4/RR7/9/9/9/9/9/9/9/3K5 w - - 0 1"
    st = Xiangqi(fen)
    agent = XiangqiMCTSAgent(c=1.0, rollout_depth=4, seed=7)
    mv = agent.select_move(st, Budget(iterations=60))
    st.play(mv)
    assert st.is_terminal()
    assert st.winner() == RED

    # Kiểm tra tương thích với progressive_bias (compute_move_prior trên Xiangqi) và reuse_tree
    # Đảm bảo ăn Xe có prior lớn hơn hẳn ăn Tốt (không bị bão hòa tanh ở ~1.0)
    st_cap = Xiangqi("3k5/9/4r4/9/4R2p1/9/9/9/9/5K3 w - - 0 1")
    prior_rook = compute_move_prior(st_cap, parse_iccs("e5e7"), RED)
    prior_pawn = compute_move_prior(st_cap, parse_iccs("e5h5"), RED)
    assert 0.0 < prior_pawn < prior_rook < 1.0
    assert prior_rook - prior_pawn > 0.5

    st2 = Xiangqi()
    prior = compute_move_prior(st2, st2.legal_moves()[0], RED)
    assert 0.0 <= prior <= 1.0
    assert isinstance(score_xiangqi_move(st2, st2.legal_moves()[0]), float)

    pb_agent = XiangqiMCTSAgent(c=1.0, uct_variant="progressive_bias", reuse_tree=True, seed=11)
    mv1 = pb_agent.select_move(st2, Budget(iterations=20))
    assert mv1 in st2.legal_moves()
    assert len(pb_agent.last_stats["root"]) >= 1
    st2.play(mv1)
    st2.play(st2.legal_moves()[0])
    mv2 = pb_agent.select_move(st2, Budget(iterations=20))
    assert mv2 in st2.legal_moves()

    # MCTSAgent chuẩn truyền rollout="xiangqi_l4" cũng phải chạy được
    plain = MCTSAgent(rollout="xiangqi_l4", rollout_depth=3, use_candidates=False, seed=13)
    assert plain.select_move(Xiangqi(), Budget(iterations=15)) in Xiangqi().legal_moves()


# ---------------------------------------------------------------------------
# Tích hợp runner và tiêu chí thắng Random
# ---------------------------------------------------------------------------


def test_runner_builds_and_runs_xiangqi_agents(tmp_path: Path) -> None:
    ab, ab_budget = build_agent({"type": "alphabeta_xiangqi", "budget": {"depth": 2}}, seed=1)
    assert isinstance(ab, AlphaBetaXiangqi)
    assert ab_budget.depth == 2

    mcts, mcts_budget = build_agent(
        {"type": "mcts_xiangqi", "c": 1.0, "budget": {"iterations": 20}}, seed=2
    )
    assert isinstance(mcts, XiangqiMCTSAgent)
    assert mcts_budget.iterations == 20

    with pytest.raises(ConfigError):
        build_agent({"type": "alphabeta_xiangqi", "bad_param": 1}, seed=1)
    with pytest.raises(ConfigError):
        build_agent({"type": "mcts_xiangqi", "bad_param": 1}, seed=1)

    cfg = {
        "name": "smoke_xiangqi_agents",
        "seed": 123,
        "games": 2,
        "swap_first": True,
        "opening_plies": 0,
        "game": {"type": "xiangqi", "max_plies": 40},
        "matchups": [
            {
                "a": {"type": "alphabeta_xiangqi", "default_depth": 1, "quiescence_depth": 0},
                "b": {"type": "mcts_xiangqi", "rollout_depth": 2, "budget": {"iterations": 10}},
            }
        ],
    }
    out_csv = tmp_path / "out.csv"
    rows = run_experiment(normalize_config(cfg), workers=1)
    write_csv(rows, out_csv)
    assert len(rows) == 2
    assert out_csv.exists()


def test_both_xiangqi_agents_beat_random_consistently() -> None:
    """Kiểm tra nhanh cả AlphaBetaXiangqi và XiangqiMCTSAgent đều thắng Random áp đảo (>= 95%)."""
    n_games = 20

    for label, make_agent, budget in (
        (
            "alphabeta_xiangqi",
            lambda i: AlphaBetaXiangqi(default_depth=2, quiescence_depth=1),
            Budget(depth=2),
        ),
        (
            "mcts_xiangqi",
            lambda i: XiangqiMCTSAgent(c=1.0, rollout_depth=4, seed=500 + i),
            Budget(iterations=80),
        ),
    ):
        wins = 0
        for i in range(n_games):
            st = Xiangqi(max_plies=240)
            agent = make_agent(i)
            rnd = RandomAgent(seed=900 + i)
            agent_side = RED if i % 2 == 0 else BLACK
            while not st.is_terminal():
                if st.current_player() == agent_side:
                    mv = agent.select_move(st, budget)
                else:
                    mv = rnd.select_move(st, Budget())
                st.play(mv)
            if st.winner() == agent_side:
                wins += 1
        assert wins / n_games >= 0.95, f"{label} chỉ thắng {wins}/{n_games} trước RandomAgent"
