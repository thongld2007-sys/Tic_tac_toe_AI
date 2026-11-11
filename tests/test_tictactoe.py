"""Test luật Tic-Tac-Toe 3×3."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from src.games.base import DRAW, EMPTY, PLAYER_O, PLAYER_X, Move
from src.games.tictactoe import LINES, TicTacToe

ALL_CELLS: list[Move] = [(r, c) for r in range(3) for c in range(3)]


def play_all(moves: list[Move]) -> TicTacToe:
    state = TicTacToe()
    for m in moves:
        state.play(m)
    return state


def fillers(line: tuple[Move, ...], count: int) -> list[Move]:
    """`count` ô nằm ngoài `line` và không tự tạo thành một đường thắng."""
    outside = [cell for cell in ALL_CELLS if cell not in line]
    for combo in combinations(outside, count):
        if not any(set(other) <= set(combo) for other in LINES):
            return list(combo)
    raise AssertionError("Không tìm được ô đệm")


def test_initial_state() -> None:
    state = TicTacToe()
    assert state.board.shape == (3, 3)
    assert state.board.dtype == np.int8
    assert not state.board.any()
    assert state.current_player() == PLAYER_X
    assert state.winner() is None
    assert not state.is_terminal()
    assert state.legal_moves() == ALL_CELLS
    assert state.candidate_moves() == state.legal_moves()


def test_there_are_eight_distinct_lines() -> None:
    assert len(LINES) == 8
    assert len({frozenset(line) for line in LINES}) == 8


@pytest.mark.parametrize("line", LINES)
def test_x_wins_on_each_line(line: tuple[Move, ...]) -> None:
    o1, o2 = fillers(line, 2)
    state = play_all([line[0], o1, line[1], o2])
    assert state.winner() is None

    state.play(line[2])
    assert state.winner() == PLAYER_X
    assert state.is_terminal()
    assert state.legal_moves() == []


@pytest.mark.parametrize("line", LINES)
def test_o_wins_on_each_line(line: tuple[Move, ...]) -> None:
    x1, x2, x3 = fillers(line, 3)
    state = play_all([x1, line[0], x2, line[1], x3])
    assert state.winner() is None

    state.play(line[2])
    assert state.winner() == PLAYER_O
    assert state.is_terminal()


def test_full_board_without_line_is_draw() -> None:
    # X O X
    # X O O
    # O X X
    state = play_all([(0, 0), (0, 1), (0, 2), (1, 1), (1, 0), (1, 2), (2, 1), (2, 0), (2, 2)])
    assert state.winner() == DRAW
    assert state.is_terminal()
    assert state.legal_moves() == []


def test_win_on_ninth_move_is_not_a_draw() -> None:
    # X O X
    # O X O
    # O X X   (X hoàn thành đường chéo ở nước thứ 9)
    state = play_all([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (2, 0), (2, 2)])
    assert state.winner() == PLAYER_X


def test_players_alternate() -> None:
    state = TicTacToe()
    state.play((1, 1))
    assert state.board[1, 1] == PLAYER_X
    assert state.current_player() == PLAYER_O
    state.play((0, 0))
    assert state.board[0, 0] == PLAYER_O
    assert state.current_player() == PLAYER_X
    assert (1, 1) not in state.legal_moves()
    assert len(state.legal_moves()) == 7


@pytest.mark.parametrize("move", [(-1, 0), (0, 3), (3, 3)])
def test_play_rejects_move_off_board(move: Move) -> None:
    with pytest.raises(ValueError):
        TicTacToe().play(move)


def test_play_rejects_occupied_cell() -> None:
    state = play_all([(1, 1)])
    with pytest.raises(ValueError):
        state.play((1, 1))
    assert state.current_player() == PLAYER_O


def test_play_rejects_move_after_game_over() -> None:
    state = play_all([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
    assert state.winner() == PLAYER_X
    with pytest.raises(ValueError):
        state.play((2, 2))


def test_undo_restores_board_player_and_hash() -> None:
    state = play_all([(0, 0), (1, 1)])
    board_before, hash_before = state.board.copy(), state.hash()

    state.play((2, 2))
    assert state.hash() != hash_before
    state.undo()

    assert np.array_equal(state.board, board_before)
    assert state.current_player() == PLAYER_X
    assert state.hash() == hash_before


def test_undo_reopens_finished_game() -> None:
    state = play_all([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
    state.undo()
    assert state.winner() is None
    assert state.board[0, 2] == EMPTY
    assert (0, 2) in state.legal_moves()


def test_undo_without_moves_raises() -> None:
    with pytest.raises(ValueError):
        TicTacToe().undo()


def test_hash_depends_on_position_not_move_order() -> None:
    first = play_all([(0, 0), (1, 1), (2, 2)])
    second = play_all([(2, 2), (1, 1), (0, 0)])
    assert np.array_equal(first.board, second.board)
    assert first.hash() == second.hash()


def test_hash_distinguishes_positions() -> None:
    x_corner = play_all([(0, 0)])
    x_center = play_all([(1, 1)])
    assert len({TicTacToe().hash(), x_corner.hash(), x_center.hash()}) == 3


def test_clone_is_independent_and_keeps_history() -> None:
    state = play_all([(0, 0), (1, 1)])
    copy = state.clone()
    assert copy.hash() == state.hash()

    copy.play((2, 2))
    assert state.board[2, 2] == EMPTY
    assert state.current_player() == PLAYER_X

    copy.undo()
    copy.undo()
    assert copy.board[1, 1] == EMPTY
    assert state.board[1, 1] == PLAYER_O


def test_full_game_tree_matches_known_counts() -> None:
    """Duyệt toàn bộ cây trò chơi bằng play()/undo() và so với số ván đã biết."""
    counts = {PLAYER_X: 0, PLAYER_O: 0, DRAW: 0}
    state = TicTacToe()

    def visit() -> None:
        result = state.winner()
        if result is not None:
            counts[result] += 1
            return
        for m in state.legal_moves():
            state.play(m)
            visit()
            state.undo()

    visit()
    assert counts == {PLAYER_X: 131_184, PLAYER_O: 77_904, DRAW: 46_080}
    assert not state.board.any()
    assert state.hash() == TicTacToe().hash()
