"""Unit test cho bàn cờ Caro (Gomoku) — P1.2.

Chủ sở hữu: TV2.  Review: TV3.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.games.base import DRAW, EMPTY, PLAYER_O, PLAYER_X, Move
from src.games.gomoku import Gomoku


def play_sequence(
    state: Gomoku,
    x_moves: list[Move],
    o_moves: list[Move],
) -> Gomoku:
    """Cho X và O lần lượt đánh các nước trong `x_moves` và `o_moves`."""
    for i in range(max(len(x_moves), len(o_moves))):
        if i < len(x_moves):
            state.play(x_moves[i])
        if i < len(o_moves):
            state.play(o_moves[i])
    return state


# ---------------------------------------------------------------------------
# Khởi tạo và tham số kích thước
# ---------------------------------------------------------------------------


def test_default_initial_state() -> None:
    state = Gomoku()
    assert state.size == 15
    assert state.win_length == 5
    assert not state.block_two_ends
    assert state.board.shape == (15, 15)
    assert state.board.dtype == np.int8
    assert not state.board.any()
    assert state.current_player() == PLAYER_X
    assert state.winner() is None
    assert not state.is_terminal()
    assert state.last_move() is None
    assert state.move_count() == 0
    assert len(state.legal_moves()) == 225
    assert state.candidate_moves() == [(7, 7)]


@pytest.mark.parametrize("size", [5, 9, 11, 15, 19])
def test_parametric_board_sizes(size: int) -> None:
    state = Gomoku(size=size)
    assert state.board.shape == (size, size)
    assert len(state.legal_moves()) == size * size
    assert state.candidate_moves() == [(size // 2, size // 2)]


@pytest.mark.parametrize(("size", "win_length"), [(4, 5), (2, 2), (5, 2), (0, 5)])
def test_invalid_size_or_win_length_raises(size: int, win_length: int) -> None:
    with pytest.raises(ValueError):
        Gomoku(size=size, win_length=win_length)


# ---------------------------------------------------------------------------
# Luật thắng >= 5 quân theo 4 hướng và nối giữa
# ---------------------------------------------------------------------------


def test_x_wins_horizontal() -> None:
    state = Gomoku(size=9)
    play_sequence(
        state,
        x_moves=[(4, 2), (4, 3), (4, 4), (4, 5), (4, 6)],
        o_moves=[(0, 0), (0, 1), (0, 2), (0, 3)],
    )
    assert state.winner() == PLAYER_X
    assert state.is_terminal()
    assert state.legal_moves() == []
    assert state.candidate_moves() == []


def test_o_wins_vertical() -> None:
    state = Gomoku(size=9)
    play_sequence(
        state,
        x_moves=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
        o_moves=[(2, 5), (3, 5), (4, 5), (5, 5), (6, 5)],
    )
    assert state.winner() == PLAYER_O
    assert state.is_terminal()


def test_x_wins_main_diagonal() -> None:
    state = Gomoku(size=9)
    play_sequence(
        state,
        x_moves=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
        o_moves=[(0, 8), (1, 8), (2, 8), (3, 8)],
    )
    assert state.winner() == PLAYER_X


def test_x_wins_anti_diagonal() -> None:
    state = Gomoku(size=9)
    play_sequence(
        state,
        x_moves=[(1, 7), (2, 6), (3, 5), (4, 4), (5, 3)],
        o_moves=[(0, 0), (0, 1), (0, 2), (0, 3)],
    )
    assert state.winner() == PLAYER_X


def test_win_by_connecting_middle_gap_and_overline() -> None:
    # X đánh 3 quân bên trái và 3 quân bên phải, sau đó đánh vào ô trống ở giữa tạo 7 quân (>= 5).
    state = Gomoku(size=9)
    play_sequence(
        state,
        x_moves=[(4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7)],
        o_moves=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    )
    assert state.winner() is None
    state.play((4, 4))
    assert state.winner() == PLAYER_X


# ---------------------------------------------------------------------------
# Test biên và góc bàn cờ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_line",
    [
        pytest.param([(0, c) for c in range(5)], id="top-edge-from-top-left-corner"),
        pytest.param([(8, c) for c in range(4, 9)], id="bottom-edge-to-bottom-right-corner"),
        pytest.param([(r, 0) for r in range(5)], id="left-edge-from-top-left-corner"),
        pytest.param([(r, 8) for r in range(4, 9)], id="right-edge-to-bottom-right-corner"),
        pytest.param([(r, 8 - r) for r in range(5)], id="anti-diag-from-top-right-corner"),
        pytest.param([(4 + i, i) for i in range(5)], id="main-diag-on-corner-of-length-5"),
    ],
)
def test_wins_along_board_edges_and_corners(x_line: list[Move]) -> None:
    state = Gomoku(size=9)
    o_fillers = [(2, 1), (2, 2), (6, 6), (6, 7)]
    play_sequence(state, x_moves=x_line, o_moves=o_fillers)
    assert state.winner() == PLAYER_X


def test_candidate_moves_at_corners_and_edges_stay_in_bounds() -> None:
    state = Gomoku(size=9)
    state.play((0, 0))
    assert state.candidate_moves(radius=1) == [(0, 1), (1, 0), (1, 1)]
    assert len(state.candidate_moves(radius=2)) == 8

    state_br = Gomoku(size=9)
    state_br.play((8, 8))
    assert state_br.candidate_moves(radius=1) == [(7, 7), (7, 8), (8, 7)]


# ---------------------------------------------------------------------------
# Luật chặn hai đầu (block_two_ends)
# ---------------------------------------------------------------------------


def test_five_blocked_both_ends_does_not_win_when_rule_enabled() -> None:
    # Hàng 4: O X X X X X O (O ở cột 1 và cột 7, X đánh cuối vào cột 4)
    x_moves = [(4, 2), (4, 3), (4, 5), (4, 6), (4, 4)]
    o_moves = [(4, 1), (4, 7), (0, 0), (0, 1)]

    free_rule = Gomoku(size=9, block_two_ends=False)
    play_sequence(free_rule, x_moves, o_moves)
    assert free_rule.winner() == PLAYER_X

    blocked_rule = Gomoku(size=9, block_two_ends=True)
    play_sequence(blocked_rule, x_moves, o_moves)
    assert blocked_rule.winner() is None
    assert not blocked_rule.is_terminal()


def test_overline_blocked_both_ends_does_not_win_when_rule_enabled() -> None:
    # Hàng 4: O X X X X X X O (6 quân X bị chặn cả 2 đầu bởi O ở cột 1 và cột 8)
    state = Gomoku(size=9, block_two_ends=True)
    play_sequence(
        state,
        x_moves=[(4, 2), (4, 3), (4, 4), (4, 6), (4, 7), (4, 5)],
        o_moves=[(4, 1), (4, 8), (0, 0), (0, 1), (0, 2)],
    )
    assert state.winner() is None


def test_five_blocked_only_one_end_still_wins_under_blocked_rule() -> None:
    # Hàng 4: O X X X X X . (chỉ chặn đầu trái)
    state = Gomoku(size=9, block_two_ends=True)
    play_sequence(
        state,
        x_moves=[(4, 2), (4, 3), (4, 4), (4, 5), (4, 6)],
        o_moves=[(4, 1), (0, 0), (0, 1), (0, 2)],
    )
    assert state.winner() == PLAYER_X


def test_board_edge_is_not_considered_a_block() -> None:
    # Hàng 0 sát góc trái: X X X X X O (cột 0..4 là X, cột 5 là O) -> biên không tính là chặn.
    state = Gomoku(size=9, block_two_ends=True)
    play_sequence(
        state,
        x_moves=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        o_moves=[(0, 5), (2, 0), (2, 1), (2, 2)],
    )
    assert state.winner() == PLAYER_X


def test_intersection_wins_if_at_least_one_direction_is_unblocked() -> None:
    # Nước (4, 4) tạo 5 quân ngang bị chặn 2 đầu (O ở (4, 1) và (4, 7)),
    # nhưng đồng thời tạo 5 quân dọc mở hai đầu ((2, 4)..(6, 4)).
    state = Gomoku(size=9, block_two_ends=True)
    play_sequence(
        state,
        x_moves=[(4, 2), (4, 3), (4, 5), (4, 6), (2, 4), (3, 4), (5, 4), (6, 4), (4, 4)],
        o_moves=[(4, 1), (4, 7), (0, 0), (0, 1), (0, 2), (0, 3), (8, 0), (8, 1)],
    )
    assert state.winner() == PLAYER_X


# ---------------------------------------------------------------------------
# Hòa và thắng ở nước cuối cùng
# ---------------------------------------------------------------------------


def test_full_board_without_five_is_draw() -> None:
    # Mẫu bàn 5×5 (13 quân X, 12 quân O) không có 5 quân cùng màu trên bất kỳ hàng, cột, chéo nào:
    #   X X O X X   (4 X, 1 O)
    #   O O X O O   (1 X, 4 O)
    #   X X O X X   (4 X, 1 O)
    #   O O X O O   (1 X, 4 O)
    #   X X O X O   (3 X, 2 O) -> tổng 13 X, 12 O
    target = np.array(
        [
            [PLAYER_X, PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_X],
            [PLAYER_O, PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_O],
            [PLAYER_X, PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_X],
            [PLAYER_O, PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_O],
            [PLAYER_X, PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O],
        ],
        dtype=np.int8,
    )
    x_moves = [(int(r), int(c)) for r, c in zip(*np.nonzero(target == PLAYER_X), strict=True)]
    o_moves = [(int(r), int(c)) for r, c in zip(*np.nonzero(target == PLAYER_O), strict=True)]
    assert len(x_moves) == 13 and len(o_moves) == 12

    state = Gomoku(size=5)
    play_sequence(state, x_moves, o_moves)
    assert state.winner() == DRAW
    assert state.is_terminal()
    assert state.legal_moves() == []
    assert state.candidate_moves() == []


def test_win_on_last_cell_of_board_is_not_draw() -> None:
    # Bàn 5×5 mà chỉ hàng 0 có 5 quân X, các hàng/cột/chéo khác đều pha trộn X và O:
    #   X X X X X   (5 X, 0 O) - ô (0, 2) được đánh ở nước thứ 25 cuối cùng
    #   O O X O O   (1 X, 4 O)
    #   X X O X O   (3 X, 2 O)
    #   O O X O O   (1 X, 4 O)
    #   X X O X O   (3 X, 2 O) -> tổng 13 X, 12 O
    target = np.array(
        [
            [PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X, PLAYER_X],
            [PLAYER_O, PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_O],
            [PLAYER_X, PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_O, PLAYER_X, PLAYER_O, PLAYER_O],
            [PLAYER_X, PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_O],
        ],
        dtype=np.int8,
    )
    x_moves = [
        (int(r), int(c))
        for r, c in zip(*np.nonzero(target == PLAYER_X), strict=True)
        if (r, c) != (0, 2)
    ] + [(0, 2)]
    o_moves = [(int(r), int(c)) for r, c in zip(*np.nonzero(target == PLAYER_O), strict=True)]

    state = Gomoku(size=5)
    play_sequence(state, x_moves, o_moves)
    assert state.move_count() == 25
    assert state.winner() == PLAYER_X


# ---------------------------------------------------------------------------
# Zobrist hash, play/undo và clone
# ---------------------------------------------------------------------------


def test_play_and_undo_preserve_hash_and_state_across_random_walk() -> None:
    rng = np.random.default_rng(2026)
    state = Gomoku(size=9)
    snapshots: list[tuple[np.ndarray, int, int, Move | None, int]] = []

    for _step in range(30):
        if state.is_terminal():
            break
        snapshots.append(
            (
                state.board.copy(),
                state.current_player(),
                state.hash(),
                state.last_move(),
                state.move_count(),
            )
        )
        moves = state.legal_moves()
        move = moves[int(rng.integers(len(moves)))]
        state.play(move)

    while snapshots:
        board_before, player_before, hash_before, last_before, count_before = snapshots.pop()
        state.undo()
        assert np.array_equal(state.board, board_before)
        assert state.current_player() == player_before
        assert state.hash() == hash_before
        assert state.last_move() == last_before
        assert state.move_count() == count_before
        assert state.winner() is None


def test_zobrist_hash_is_transposition_invariant() -> None:
    # Hai thứ tự nước đi khác nhau tạo ra cùng một thế cờ trên bàn 9×9:
    first = Gomoku(size=9)
    play_sequence(first, x_moves=[(4, 4), (4, 5), (3, 3)], o_moves=[(4, 3), (5, 5)])

    second = Gomoku(size=9)
    play_sequence(second, x_moves=[(3, 3), (4, 4), (4, 5)], o_moves=[(5, 5), (4, 3)])

    assert np.array_equal(first.board, second.board)
    assert first.current_player() == second.current_player()
    assert first.hash() == second.hash()


def test_zobrist_hash_distinguishes_positions_and_players() -> None:
    empty = Gomoku(size=9)
    x_center = Gomoku(size=9)
    x_center.play((4, 4))
    x_corner = Gomoku(size=9)
    x_corner.play((0, 0))

    assert len({empty.hash(), x_center.hash(), x_corner.hash()}) == 3


def test_undo_reopens_won_game() -> None:
    state = Gomoku(size=9)
    play_sequence(
        state,
        x_moves=[(4, 2), (4, 3), (4, 4), (4, 5), (4, 6)],
        o_moves=[(0, 0), (0, 1), (0, 2), (0, 3)],
    )
    assert state.winner() == PLAYER_X
    state.undo()
    assert state.winner() is None
    assert not state.is_terminal()
    assert state.board[4, 6] == EMPTY
    assert state.current_player() == PLAYER_X
    assert state.last_move() == (0, 3)


def test_clone_is_independent() -> None:
    state = Gomoku(size=9, block_two_ends=True)
    state.play((4, 4))
    state.play((4, 5))
    copy = state.clone()

    assert copy.hash() == state.hash()
    assert copy.last_move() == state.last_move()
    assert copy.move_count() == state.move_count()
    assert copy.block_two_ends == state.block_two_ends

    copy.play((3, 3))
    assert state.board[3, 3] == EMPTY
    assert state.move_count() == 2
    assert copy.move_count() == 3


# ---------------------------------------------------------------------------
# candidate_moves và kiểm tra ngoại lệ
# ---------------------------------------------------------------------------


def test_candidate_moves_radius_around_single_and_multiple_stones() -> None:
    state = Gomoku(size=9)
    state.play((4, 4))
    cands_r1 = state.candidate_moves(radius=1)
    cands_r2 = state.candidate_moves(radius=2)
    assert len(cands_r1) == 8
    assert len(cands_r2) == 24
    assert (4, 4) not in cands_r1
    assert (4, 4) not in cands_r2

    state.play((4, 5))
    cands_two_stones = state.candidate_moves(radius=1)
    # Khung 3×4 quanh (4,4) và (4,5) có 12 ô, trừ 2 ô đã có quân còn 10 ô trống.
    assert len(cands_two_stones) == 10
    assert (4, 4) not in cands_two_stones and (4, 5) not in cands_two_stones


def test_invalid_moves_and_invalid_operations_raise() -> None:
    state = Gomoku(size=9)
    with pytest.raises(ValueError):
        state.undo()
    with pytest.raises(ValueError):
        state.candidate_moves(radius=0)

    for bad_move in [(-1, 0), (0, -1), (9, 0), (0, 9)]:
        with pytest.raises(ValueError):
            state.play(bad_move)

    state.play((4, 4))
    with pytest.raises(ValueError):
        state.play((4, 4))

    # Khi ván đã kết thúc, play() phải từ chối
    won = Gomoku(size=9)
    play_sequence(
        won,
        x_moves=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        o_moves=[(1, 0), (1, 1), (1, 2), (1, 3)],
    )
    with pytest.raises(ValueError):
        won.play((5, 5))
