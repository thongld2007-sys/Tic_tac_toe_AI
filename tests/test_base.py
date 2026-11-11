"""Test cho interface GameState, Agent và Budget, dùng một trò chơi giả tối giản."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.base import Agent, Budget
from src.games.base import DRAW, EMPTY, PLAYER_O, PLAYER_X, GameState, Move


class FillGame(GameState):
    """Hai người lần lượt lấp bàn 2×2; lấp đầy thì hòa. Chỉ để thử interface."""

    def __init__(self) -> None:
        self.board = np.zeros((2, 2), dtype=np.int8)
        self._player = PLAYER_X
        self._history: list[Move] = []

    def legal_moves(self) -> list[Move]:
        rows, cols = np.nonzero(self.board == EMPTY)
        return [(int(r), int(c)) for r, c in zip(rows, cols, strict=True)]

    def play(self, m: Move) -> None:
        self.board[m] = self._player
        self._history.append(m)
        self._player = -self._player

    def undo(self) -> None:
        self.board[self._history.pop()] = EMPTY
        self._player = -self._player

    def winner(self) -> int | None:
        return DRAW if not (self.board == EMPTY).any() else None

    def current_player(self) -> int:
        return self._player

    def hash(self) -> int:
        return hash((self.board.tobytes(), self._player))

    def clone(self) -> FillGame:
        other = FillGame()
        other.board = self.board.copy()
        other._player = self._player
        other._history = list(self._history)
        return other


class FirstMoveAgent(Agent):
    name = "first-move"

    def select_move(self, state: GameState, budget: Budget) -> Move:
        moves = state.legal_moves()
        self.last_stats = {"nodes": len(moves), "time_s": 0.0}
        return moves[0]


def test_abstract_classes_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        GameState()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        Agent()  # type: ignore[abstract]


def test_candidate_moves_defaults_to_legal_moves() -> None:
    state = FillGame()
    assert state.candidate_moves() == state.legal_moves()
    assert state.candidate_moves(radius=1) == state.legal_moves()


def test_play_alternates_players_and_undo_restores_state() -> None:
    state = FillGame()
    board_before, hash_before = state.board.copy(), state.hash()

    state.play((0, 1))
    assert state.board[0, 1] == PLAYER_X
    assert state.current_player() == PLAYER_O
    assert state.hash() != hash_before

    state.undo()
    assert np.array_equal(state.board, board_before)
    assert state.current_player() == PLAYER_X
    assert state.hash() == hash_before


def test_is_terminal_follows_winner() -> None:
    state = FillGame()
    assert state.winner() is None
    assert not state.is_terminal()

    for move in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        state.play(move)
    assert state.winner() == DRAW
    assert state.is_terminal()
    assert state.legal_moves() == []


def test_clone_is_independent() -> None:
    state = FillGame()
    state.play((0, 0))
    copy = state.clone()
    assert copy.hash() == state.hash()

    copy.play((1, 1))
    assert state.board[1, 1] == EMPTY
    assert state.current_player() == PLAYER_O


def test_budget_defaults_to_unlimited() -> None:
    budget = Budget()
    assert (budget.iterations, budget.depth, budget.time_s) == (None, None, None)
    assert Budget(iterations=1000).iterations == 1000


def test_agent_selects_legal_move_and_records_stats() -> None:
    state = FillGame()
    agent = FirstMoveAgent()
    assert agent.last_stats == {}

    move = agent.select_move(state, Budget(depth=1))
    assert move in state.legal_moves()
    assert agent.last_stats["nodes"] == 4


def test_agents_do_not_share_last_stats() -> None:
    first, second = FirstMoveAgent(), FirstMoveAgent()
    first.select_move(FillGame(), Budget())
    assert second.last_stats == {}
