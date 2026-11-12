"""Test RandomAgent."""

from __future__ import annotations

from collections import Counter

import pytest

from src.agents.base import Budget
from src.agents.random_agent import RandomAgent
from src.games.tictactoe import TicTacToe


def test_selects_a_legal_move_and_leaves_state_untouched() -> None:
    state = TicTacToe()
    state.play((1, 1))
    agent = RandomAgent(seed=0)

    move = agent.select_move(state, Budget())
    assert move in state.legal_moves()
    assert state.move_count() == 1
    assert agent.last_stats["nodes"] == 1
    assert agent.last_stats["time_s"] >= 0


def test_same_seed_gives_same_moves() -> None:
    def moves(seed: int) -> list[tuple[int, int]]:
        agent = RandomAgent(seed=seed)
        return [agent.select_move(TicTacToe(), Budget()) for _ in range(20)]

    assert moves(3) == moves(3)
    assert moves(3) != moves(4)


def test_every_legal_move_can_be_chosen() -> None:
    agent = RandomAgent(seed=1)
    counts = Counter(agent.select_move(TicTacToe(), Budget()) for _ in range(900))
    assert len(counts) == 9
    assert min(counts.values()) > 60  # kỳ vọng 100 mỗi ô


def test_rejects_finished_game() -> None:
    state = TicTacToe()
    for move in [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]:
        state.play(move)
    with pytest.raises(ValueError):
        RandomAgent(seed=0).select_move(state, Budget())
