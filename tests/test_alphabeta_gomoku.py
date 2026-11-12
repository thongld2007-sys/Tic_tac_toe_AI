"""Unit test cho AlphaBetaGomoku — P1.3b (R1.3–R1.4).

Chủ sở hữu: TV3.  Review: TV1.

Kiểm tra:
  - Tuân thủ giao kèo Agent (state bất biến sau select_move, ghi đủ last_stats).
  - Bàn trống chọn đúng tâm bàn.
  - Phát hiện thắng ngay (liên tiếp và đứt quãng XXX_X).
  - Phát hiện chặn ngay khi đối thủ có 4 quân đe dọa thắng, có xét luật block_two_ends.
  - Tạo nước đôi / OPEN_FOUR từ OPEN_THREE.
  - Hiệu quả cắt tỉa của sắp xếp nước (move ordering) và bảng chuyển vị (TT).
  - Iterative deepening tuân thủ ngân sách thời gian (time_s) và khôi phục state khi hết giờ.
  - Tích hợp cấu hình YAML với experiments/run.py.
  - Điều kiện xong P1.3b: <= 1 s/nước ở độ sâu >= 3 trên bàn 15×15, thắng Random >= 99% (200 ván).
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from experiments.run import build_agent, normalize_config
from src.agents.alphabeta_gomoku import AlphaBetaGomoku
from src.agents.base import Budget
from src.agents.random_agent import RandomAgent
from src.games.base import PLAYER_O, PLAYER_X, Move
from src.games.gomoku import Gomoku
from src.games.tictactoe import TicTacToe


def play_moves(state: Gomoku, x_moves: list[Move], o_moves: list[Move]) -> Gomoku:
    for i in range(max(len(x_moves), len(o_moves))):
        if i < len(x_moves):
            state.play(x_moves[i])
        if i < len(o_moves):
            state.play(o_moves[i])
    return state


def test_empty_board_selects_center_and_records_stats() -> None:
    state = Gomoku(size=15)
    agent = AlphaBetaGomoku()
    move = agent.select_move(state, Budget(depth=3))
    assert move == (7, 7)
    assert agent.last_stats["nodes"] >= 1
    assert agent.last_stats["depth"] >= 1
    assert agent.last_stats["time_s"] >= 0.0
    assert state.move_count() == 0


def test_select_move_leaves_state_unchanged() -> None:
    state = Gomoku(size=9)
    play_moves(state, x_moves=[(4, 4), (4, 5)], o_moves=[(3, 4), (5, 4)])
    board_before = state.board.copy()
    hash_before = state.hash()
    player_before = state.current_player()
    last_before = state.last_move()
    count_before = state.move_count()

    agent = AlphaBetaGomoku()
    move = agent.select_move(state, Budget(depth=3))

    assert move in state.legal_moves()
    assert np.array_equal(state.board, board_before)
    assert state.hash() == hash_before
    assert state.current_player() == player_before
    assert state.last_move() == last_before
    assert state.move_count() == count_before


def test_takes_immediate_winning_move_contiguous_and_split() -> None:
    # 1. Bốn quân liên tiếp: X ở (4,2)..(4,5), phải đánh (4,1) hoặc (4,6) để thắng ngay
    state = Gomoku(size=9)
    play_moves(
        state,
        x_moves=[(4, 2), (4, 3), (4, 4), (4, 5)],
        o_moves=[(0, 0), (0, 1), (0, 2), (0, 3)],
    )
    agent = AlphaBetaGomoku()
    move = agent.select_move(state, Budget(depth=2))
    assert move in {(4, 1), (4, 6)}

    # 2. Mẫu đứt quãng XXX_X: X ở (4,2), (4,3), (4,4), (4,6) -> phải đi (4,5) thắng ngay
    split_state = Gomoku(size=9)
    play_moves(
        split_state,
        x_moves=[(4, 2), (4, 3), (4, 4), (4, 6)],
        o_moves=[(1, 1), (1, 2), (1, 3), (1, 4)],
    )
    assert agent.select_move(split_state, Budget(depth=2)) == (4, 5)


def test_blocks_opponent_immediate_winning_threat() -> None:
    # O có 4 quân ở (2, 1), (2, 2), (2, 3), (2, 4), đầu (2, 0) bị X chặn -> X bắt buộc chặn (2, 5)
    state = Gomoku(size=9)
    play_moves(
        state,
        x_moves=[(2, 0), (4, 4), (5, 5), (6, 6)],
        o_moves=[(2, 1), (2, 2), (2, 3), (2, 4)],
    )
    assert state.current_player() == PLAYER_X
    agent = AlphaBetaGomoku()
    move = agent.select_move(state, Budget(depth=3))
    assert move == (2, 5)


def test_respects_block_two_ends_rule_when_evaluating_threat() -> None:
    # Với block_two_ends=True: O có O O _ O O ở (2,1), (2,2), (2,4), (2,5), nhưng X đã chặn cả
    # hai đầu tại (2,0) và (2,6) -> ô (2,3) của O nếu đánh vào sẽ thành XOOOOOX (bị chặn 2 đầu,
    # không thắng). Trong khi đó X có _XXX_ ở hàng 5 -> X phải ưu tiên tạo OPEN_FOUR ở hàng 5.
    state = Gomoku(size=9, block_two_ends=True)
    play_moves(
        state,
        x_moves=[(2, 0), (2, 6), (5, 3), (5, 4), (5, 5)],
        o_moves=[(2, 1), (2, 2), (2, 4), (2, 5), (0, 0)],
    )
    agent = AlphaBetaGomoku()
    move = agent.select_move(state, Budget(depth=2))
    assert move in {(5, 2), (5, 6)}


def test_creates_open_four_from_open_three() -> None:
    state = Gomoku(size=9)
    play_moves(
        state,
        x_moves=[(4, 3), (4, 4), (4, 5)],
        o_moves=[(1, 1), (1, 2), (7, 7)],
    )
    agent = AlphaBetaGomoku()
    move = agent.select_move(state, Budget(depth=2))
    assert move in {(4, 2), (4, 6)}


def test_move_ordering_and_tt_reduce_search_nodes() -> None:
    state = Gomoku(size=9)
    play_moves(
        state,
        x_moves=[(4, 4), (4, 5), (3, 4)],
        o_moves=[(3, 3), (4, 3), (5, 4)],
    )

    ordered_agent = AlphaBetaGomoku(
        max_candidates=12,
        use_move_ordering=True,
        use_iterative_deepening=False,
        use_tt=True,
    )
    unordered_agent = AlphaBetaGomoku(
        max_candidates=12,
        use_move_ordering=False,
        use_iterative_deepening=False,
        use_tt=False,
    )

    ordered_agent.select_move(state, Budget(depth=3))
    unordered_agent.select_move(state, Budget(depth=3))

    assert ordered_agent.last_stats["nodes"] < unordered_agent.last_stats["nodes"]
    assert ordered_agent.last_stats["cutoffs"] > 0

    # Gọi lại trên cùng thế cờ khi đã có TT phải duyệt ít nút hơn hẳn lần đầu
    first_nodes = ordered_agent.last_stats["nodes"]
    ordered_agent.select_move(state, Budget(depth=3))
    assert ordered_agent.last_stats["nodes"] < first_nodes


def test_iterative_deepening_respects_time_budget_and_restores_state() -> None:
    state = Gomoku(size=15)
    play_moves(
        state,
        x_moves=[(7, 7), (7, 8), (6, 7), (8, 8)],
        o_moves=[(6, 6), (7, 6), (8, 7), (6, 8)],
    )
    board_before = state.board.copy()
    hash_before = state.hash()

    agent = AlphaBetaGomoku()
    move = agent.select_move(state, Budget(depth=6, time_s=0.08))

    assert move in state.legal_moves()
    assert agent.last_stats["time_s"] <= 0.25
    assert agent.last_stats["depth"] >= 1
    assert np.array_equal(state.board, board_before)
    assert state.hash() == hash_before


def test_runner_yaml_integration() -> None:
    yaml_text = """
name: test_ab_gomoku
seed: 2026
games: 2
game: {type: gomoku, size: 9}
matchups:
  - a:
      type: alphabeta_gomoku
      radius: 2
      default_depth: 2
      max_candidates: 12
      use_tt: [true, false]
      budget: {depth: 2}
    b: {type: random}
"""
    cfg = normalize_config(yaml.safe_load(yaml_text))
    assert len(cfg["matchups"]) == 2
    agent, budget = build_agent(cfg["matchups"][0]["a"], seed=42)
    assert isinstance(agent, AlphaBetaGomoku)
    assert agent.get_params()["max_candidates"] == 12
    assert budget == Budget(depth=2)


def test_depth_3_on_15x15_completes_under_one_second() -> None:
    """Điều kiện xong P1.3b (1): <= 1 s/nước ở độ sâu >= 3 trên bàn 15×15."""
    state = Gomoku(size=15)
    play_moves(
        state,
        x_moves=[(7, 7), (7, 8), (8, 7), (6, 8), (8, 6)],
        o_moves=[(6, 6), (6, 7), (7, 6), (9, 5), (5, 9)],
    )
    agent = AlphaBetaGomoku()
    move = agent.select_move(state, Budget(depth=3))

    assert move in state.legal_moves()
    assert agent.last_stats["depth"] == 3
    assert agent.last_stats["time_s"] <= 1.0


def test_wins_at_least_99_percent_against_random_in_200_games() -> None:
    """Điều kiện xong P1.3b (2): Thắng Random >= 99% trong 200 ván (100 đi trước, 100 đi sau)."""
    agent = AlphaBetaGomoku(default_depth=2, max_candidates=12)
    wins = 0
    total_games = 200

    for game_idx in range(total_games):
        rng_agent = RandomAgent(seed=10_000 + game_idx)
        state = Gomoku(size=9)
        ai_player = PLAYER_X if game_idx < total_games // 2 else PLAYER_O
        agent.clear_table()

        while not state.is_terminal():
            if state.current_player() == ai_player:
                move = agent.select_move(state, Budget(depth=2))
            else:
                move = rng_agent.select_move(state, Budget())
            state.play(move)

        if state.winner() == ai_player:
            wins += 1

    win_rate = wins / total_games
    assert win_rate >= 0.99, f"Tỷ lệ thắng Random chỉ đạt {win_rate:.1%} ({wins}/{total_games})"


def test_works_on_generic_gamestate_and_resets_tt_on_rule_change() -> None:
    """Chạy được trên TicTacToe (chỉ qua GameState) và tự làm sạch TT khi đổi cấu hình bàn."""
    ttt = TicTacToe()
    for mv in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        ttt.play(mv)
    agent = AlphaBetaGomoku(max_candidates=None)
    assert agent.select_move(ttt, Budget(depth=2)) == (0, 2)
    assert agent.last_value is not None and agent.last_value > 0

    # Khi đổi sang bàn Gomoku có luật khác, TT cũ được tự động xóa để không đụng độ hash
    g1 = Gomoku(size=9, block_two_ends=False)
    g1.play((4, 4))
    agent.select_move(g1, Budget(depth=2))
    assert len(agent._tt) > 0

    g2 = Gomoku(size=9, block_two_ends=True)
    g2.play((4, 4))
    agent.select_move(g2, Budget(depth=1))
    assert agent._tt_game_sig == (9, 5, True)


def test_invalid_arguments_and_terminal_state_raise() -> None:
    with pytest.raises(ValueError):
        AlphaBetaGomoku(radius=0)
    with pytest.raises(ValueError):
        AlphaBetaGomoku(radius=True)
    with pytest.raises(ValueError):
        AlphaBetaGomoku(default_depth=0)
    with pytest.raises(ValueError):
        AlphaBetaGomoku(max_candidates=0)
    with pytest.raises(ValueError):
        AlphaBetaGomoku(use_tt=1)  # type: ignore[arg-type]

    state = Gomoku(size=9)
    with pytest.raises(ValueError):
        AlphaBetaGomoku().select_move(state, Budget(depth=0))
    with pytest.raises(ValueError):
        AlphaBetaGomoku().select_move(state, Budget(time_s=0.0))

    play_moves(
        state,
        x_moves=[(4, 2), (4, 3), (4, 4), (4, 5), (4, 6)],
        o_moves=[(0, 0), (0, 1), (0, 2), (0, 3)],
    )
    assert state.is_terminal()
    with pytest.raises(ValueError):
        AlphaBetaGomoku().select_move(state, Budget(depth=2))
