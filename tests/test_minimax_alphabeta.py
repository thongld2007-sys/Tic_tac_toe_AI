"""Test Minimax và Alpha-Beta trên Tic-Tac-Toe 3×3 (P1.1c)."""

from __future__ import annotations

import random
from collections.abc import Callable
from functools import partial

import numpy as np
import pytest

from src.agents.alphabeta import ORDERINGS, AlphaBetaAgent, lines_through
from src.agents.base import Agent, Budget
from src.agents.minimax import MinimaxAgent, terminal_value
from src.games.base import DRAW, PLAYER_O, PLAYER_X, GameState, Move
from src.games.gomoku import Gomoku
from src.games.tictactoe import TicTacToe

FULL = Budget()

# Số nút của cây trò chơi 3×3 đầy đủ, tính cả nút gốc.
FULL_TREE_NODES = 549_946
# Số nút Alpha-Beta thăm từ bàn trống, cửa sổ (-inf, +inf), thứ tự legal_moves().
ALPHABETA_NATURAL_NODES = 18_297
# Số thế 3×3 chưa kết thúc có thể đi tới từ bàn trống.
NON_TERMINAL_POSITIONS = 4_520

AgentFactory = Callable[[], MinimaxAgent | AlphaBetaAgent]
AGENT_FACTORIES: list[AgentFactory] = [
    MinimaxAgent,
    *(partial(AlphaBetaAgent, ordering) for ordering in ORDERINGS),
]


def position(moves: list[Move]) -> TicTacToe:
    state = TicTacToe()
    for m in moves:
        state.play(m)
    return state


def all_non_terminal_lines() -> list[list[Move]]:
    """Một dãy nước dẫn tới mỗi thế chưa kết thúc, theo thứ tự duyệt cố định."""
    found: dict[int, list[Move]] = {}
    state = TicTacToe()
    path: list[Move] = []

    def visit() -> None:
        if state.is_terminal() or state.hash() in found:
            return
        found[state.hash()] = list(path)
        for m in state.legal_moves():
            state.play(m)
            path.append(m)
            visit()
            path.pop()
            state.undo()

    visit()
    return list(found.values())


def play_game(x_agent: Agent, o_agent: Agent) -> int:
    state = TicTacToe()
    agents = {PLAYER_X: x_agent, PLAYER_O: o_agent}
    while not state.is_terminal():
        state.play(agents[state.current_player()].select_move(state, FULL))
    result = state.winner()
    assert result is not None
    return result


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int) -> None:
        super().__init__()
        self._rng = random.Random(seed)

    def select_move(self, state: GameState, budget: Budget) -> Move:
        return self._rng.choice(state.legal_moves())


# ---------------------------------------------------------------------------
# Minimax
# ---------------------------------------------------------------------------


def test_minimax_value_of_empty_board_is_draw() -> None:
    state = TicTacToe()
    agent = MinimaxAgent()
    move = agent.select_move(state, FULL)

    assert agent.last_value == 0
    assert move == (0, 0)  # mọi nước đầu đều hòa, lấy nước đầu tiên của legal_moves()
    assert agent.last_stats["nodes"] == FULL_TREE_NODES
    assert agent.last_stats["depth"] == 9
    assert agent.last_stats["time_s"] > 0
    assert "cutoffs" not in agent.last_stats
    # state phải y như lúc đưa vào
    assert not state.board.any()
    assert state.move_count() == 0
    assert state.hash() == TicTacToe().hash()


def test_terminal_value_is_from_side_to_move() -> None:
    x_won = position([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])  # X thắng, tới lượt O
    assert x_won.winner() == PLAYER_X
    assert terminal_value(x_won) == -1

    with pytest.raises(ValueError):
        terminal_value(TicTacToe())


# ---------------------------------------------------------------------------
# Alpha-Beta từ bàn trống
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_alphabeta_value_of_empty_board_is_draw(ordering: str) -> None:
    state = TicTacToe()
    agent = AlphaBetaAgent(ordering, seed=0)
    move = agent.select_move(state, FULL)

    assert agent.last_value == 0
    assert move in state.legal_moves()
    assert agent.name == f"alphabeta-{ordering}"
    assert agent.last_stats["depth"] == 9
    assert 0 < agent.last_stats["cutoffs"] < agent.last_stats["nodes"] < FULL_TREE_NODES
    assert not state.board.any()
    assert state.hash() == TicTacToe().hash()


def test_alphabeta_node_counts_from_empty_board() -> None:
    nodes: dict[str, int] = {}
    moves: dict[str, Move] = {}
    for ordering in ORDERINGS:
        agent = AlphaBetaAgent(ordering, seed=0)
        moves[ordering] = agent.select_move(TicTacToe(), FULL)
        nodes[ordering] = agent.last_stats["nodes"]

    assert nodes["natural"] == ALPHABETA_NATURAL_NODES
    assert moves["natural"] == (0, 0)
    # Thử ô giữa trước giúp cắt tỉa sớm hơn.
    assert nodes["heuristic"] == 7_275
    assert moves["heuristic"] == (1, 1)
    assert nodes["heuristic"] < nodes["random"] <= nodes["natural"]


def test_random_ordering_is_reproducible_with_seed() -> None:
    def run(seed: int) -> tuple[Move, int]:
        agent = AlphaBetaAgent("random", seed=seed)
        move = agent.select_move(TicTacToe(), FULL)
        return move, agent.last_stats["nodes"]

    assert run(7) == run(7)
    assert len({run(seed)[1] for seed in range(5)}) > 1


def test_heuristic_ordering_tries_center_then_corners_then_edges() -> None:
    assert lines_through((1, 1), 3) == 4
    assert {lines_through(m, 3) for m in [(0, 0), (0, 2), (2, 0), (2, 2)]} == {3}
    assert {lines_through(m, 3) for m in [(0, 1), (1, 0), (1, 2), (2, 1)]} == {2}

    agent = AlphaBetaAgent("heuristic")
    assert agent._ordered_moves(TicTacToe()) == [
        (1, 1),
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2),
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    ]


# ---------------------------------------------------------------------------
# Alpha-Beta = Minimax trên 1 000 thế
# ---------------------------------------------------------------------------


def test_alphabeta_matches_minimax_on_1000_positions() -> None:
    lines = all_non_terminal_lines()
    assert len(lines) == NON_TERMINAL_POSITIONS
    sample = random.Random(2026).sample(lines, 1000)

    minimax = MinimaxAgent()
    agents = {ordering: AlphaBetaAgent(ordering, seed=1) for ordering in ORDERINGS}
    # Giá trị Minimax của các thế con đã tính, theo hash: mỗi thế chỉ duyệt cả cây một lần.
    child_values: dict[int, int] = {}

    for line in sample:
        state = position(line)
        board_before, hash_before = state.board.copy(), state.hash()

        expected_move = minimax.select_move(state, FULL)
        expected_value = minimax.last_value
        minimax_nodes = minimax.last_stats["nodes"]

        for ordering, agent in agents.items():
            move = agent.select_move(state, FULL)
            assert agent.last_value == expected_value, (ordering, line)
            assert agent.last_stats["nodes"] <= minimax_nodes, (ordering, line)
            assert move in state.legal_moves()
            if ordering == "natural":
                assert move == expected_move, line

            # Nước được chọn phải thật sự đạt giá trị đã báo.
            state.play(move)
            if state.is_terminal():
                achieved = -terminal_value(state)
            else:
                key = state.hash()
                if key not in child_values:
                    minimax.select_move(state, FULL)
                    assert minimax.last_value is not None
                    child_values[key] = minimax.last_value
                achieved = -child_values[key]
            state.undo()
            assert achieved == expected_value, (ordering, line)

        assert np.array_equal(state.board, board_before)
        assert state.hash() == hash_before


# ---------------------------------------------------------------------------
# Chơi thật
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_agent", AGENT_FACTORIES)
def test_agents_see_forced_win_and_forced_block(make_agent: AgentFactory) -> None:
    # X X .
    # O O .      X đi: thắng ngay ở (0, 2)
    # . . .
    winning = position([(0, 0), (1, 0), (0, 1), (1, 1)])
    agent = make_agent()
    agent.select_move(winning, FULL)
    assert agent.last_value == 1

    # X X .
    # . O .      O đi: chỉ (0, 2) là không thua
    # . . .
    must_block = position([(0, 0), (1, 1), (0, 1)])
    agent = make_agent()
    assert agent.select_move(must_block, FULL) == (0, 2)
    assert agent.last_value == 0


def test_perfect_play_against_itself_is_a_draw() -> None:
    assert play_game(AlphaBetaAgent("heuristic"), AlphaBetaAgent("natural")) == DRAW
    assert play_game(MinimaxAgent(), AlphaBetaAgent("heuristic")) == DRAW


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_alphabeta_never_loses_to_random(ordering: str) -> None:
    results_as_x = [play_game(AlphaBetaAgent(ordering, seed=s), RandomAgent(s)) for s in range(20)]
    results_as_o = [play_game(RandomAgent(s), AlphaBetaAgent(ordering, seed=s)) for s in range(20)]

    assert PLAYER_O not in results_as_x
    assert PLAYER_X not in results_as_o
    assert results_as_x.count(PLAYER_X) > 10  # đi trước, gặp Random thì phần lớn là thắng


# ---------------------------------------------------------------------------
# Ngân sách, lỗi đầu vào, interface chung
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_agent", [MinimaxAgent, AlphaBetaAgent])
def test_depth_budget_limits_search(make_agent: AgentFactory) -> None:
    agent = make_agent()
    agent.select_move(TicTacToe(), Budget(depth=1))
    assert agent.last_stats["nodes"] == 1 + 9
    assert agent.last_stats["depth"] == 1
    assert agent.last_value == 0

    agent.select_move(TicTacToe(), Budget(depth=2))
    assert agent.last_stats["depth"] == 2
    assert agent.last_stats["nodes"] <= 1 + 9 + 9 * 8


@pytest.mark.parametrize("make_agent", [MinimaxAgent, AlphaBetaAgent])
@pytest.mark.parametrize("depth", [0, -1])
def test_depth_budget_below_one_is_rejected(make_agent: AgentFactory, depth: int) -> None:
    state = TicTacToe()
    with pytest.raises(ValueError):
        make_agent().select_move(state, Budget(depth=depth))
    assert state.move_count() == 0


@pytest.mark.parametrize("make_agent", [MinimaxAgent, AlphaBetaAgent])
def test_select_move_rejects_finished_game(make_agent: AgentFactory) -> None:
    finished = position([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
    with pytest.raises(ValueError):
        make_agent().select_move(finished, FULL)


def test_alphabeta_rejects_unknown_ordering() -> None:
    with pytest.raises(ValueError):
        AlphaBetaAgent("best-first")


def test_stats_are_reset_between_calls() -> None:
    agent = AlphaBetaAgent("natural")
    agent.select_move(TicTacToe(), FULL)
    first = agent.last_stats["nodes"]

    late = position([(1, 1), (0, 0), (2, 2), (0, 2), (0, 1), (2, 1)])
    agent.select_move(late, FULL)
    assert agent.last_stats["nodes"] < first
    assert agent.last_stats["depth"] <= 3


def test_agents_only_need_the_gamestate_interface() -> None:
    """Gomoku(3, 3) là cùng một trò chơi, cài bằng lớp khác: kết quả phải giống hệt."""
    agent = AlphaBetaAgent("natural")
    state = Gomoku(size=3, win_length=3)
    move = agent.select_move(state, FULL)

    assert agent.last_value == 0
    assert move == (0, 0)
    assert agent.last_stats["nodes"] == ALPHABETA_NATURAL_NODES
    assert state.move_count() == 0
