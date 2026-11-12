"""Test script kiểm chứng "MCTS không bao giờ thua" (P2.1)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

import experiments.verify_never_lose as vnl
from experiments.verify_never_lose import (
    DETAIL_COLUMNS,
    SUMMARY_COLUMNS,
    TRAP_COLUMNS,
    Solver,
    board_text,
    collect_traps,
    find_threshold,
    main,
    play_against,
    run_verification,
    summarize,
    walk_response_tree,
)
from src.agents.alphabeta import AlphaBetaAgent
from src.games.base import GameState, Move
from src.games.tictactoe import TicTacToe


def position(moves: list[Move]) -> TicTacToe:
    state = TicTacToe()
    for m in moves:
        state.play(m)
    return state


def perfect_decision(
    state: GameState, iterations: int, seed: int, c: float, selection: str
) -> Move:
    values = Solver().move_values(state)
    return max(values, key=lambda m: values[m])


def first_move_decision(
    state: GameState, iterations: int, seed: int, c: float, selection: str
) -> Move:
    return state.legal_moves()[0]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def test_solver_values() -> None:
    solver = Solver()
    assert solver.value(TicTacToe()) == 0
    # X X . / O O . / . . .  — X đi và thắng ngay
    assert solver.value(position([(0, 0), (1, 0), (0, 1), (1, 1)])) == 1
    # X thắng rồi, tới lượt O
    assert solver.value(position([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])) == -1
    # X ở góc, O đi cạnh (0, 1): X thắng nếu chơi đúng
    assert solver.value(position([(0, 0), (0, 1)])) == 1


def test_solver_leaves_state_untouched_and_agrees_with_alphabeta() -> None:
    solver = Solver()
    agent = AlphaBetaAgent("natural")
    for moves in ([], [(1, 1)], [(0, 0), (1, 1)], [(0, 0), (0, 1)], [(1, 1), (0, 1), (0, 0)]):
        state = position(list(moves))
        before = state.hash()
        value = solver.value(state)
        assert state.hash() == before
        agent.select_move(state, vnl.Budget())
        assert agent.last_value == value


def test_move_values_identify_the_only_saving_move() -> None:
    # X X . / . O . / . . .  — O phải chặn ở (0, 2)
    values = Solver().move_values(position([(0, 0), (1, 1), (0, 1)]))
    assert values[(0, 2)] == 0
    assert all(v == -1 for m, v in values.items() if m != (0, 2))


def test_board_text() -> None:
    assert board_text(TicTacToe()) == ".../.../..."
    assert board_text(position([(0, 0), (1, 1), (2, 2)])) == "X../.O./..X"


# ---------------------------------------------------------------------------
# Duyệt cây phản ứng
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["X", "O"])
def test_perfect_player_never_loses_in_the_response_tree(
    role: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vnl, "mcts_decision", perfect_decision)
    result = walk_response_tree(role, iterations=1, seed=0)
    assert result["tree_losses"] == 0
    assert result["losing_moves"] == 0
    assert result["missed_wins"] == 0
    assert result["traps"] == []
    assert result["tree_wins"] > 0  # đối thủ đi sai thì người chơi hoàn hảo thắng
    assert result["tree_draws"] > 0
    assert result["terminal_positions"] == result["tree_wins"] + result["tree_draws"]


@pytest.mark.parametrize("role", ["X", "O"])
def test_weak_player_is_caught_with_traps(role: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vnl, "mcts_decision", first_move_decision)
    result = walk_response_tree(role, iterations=1, seed=0)
    assert result["tree_losses"] > 0
    assert result["losing_moves"] > 0
    assert len(result["traps"]) == result["losing_moves"]
    for trap in result["traps"]:
        assert trap["role"] == role
        assert trap["mcts_move"] not in trap["good_moves"].split()
        assert trap["good_moves"]
        assert len(trap["board"]) == 11


def test_trap_is_a_real_blunder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mỗi thế bẫy: thế đang không thua, nước đã đi dẫn tới thế thua, nước đúng thì không."""
    monkeypatch.setattr(vnl, "mcts_decision", first_move_decision)
    solver = Solver()
    for trap in walk_response_tree("O", iterations=1, seed=0)["traps"]:
        state = TicTacToe()
        cells = trap["board"].replace("/", "")
        xs = [(i // 3, i % 3) for i, ch in enumerate(cells) if ch == "X"]
        os_ = [(i // 3, i % 3) for i, ch in enumerate(cells) if ch == "O"]
        assert len(xs) == len(os_) + 1  # tới lượt O
        for index, x_move in enumerate(xs):
            state.play(x_move)
            if index < len(os_):
                state.play(os_[index])
        if state.is_terminal():
            continue  # thứ tự dựng lại ngẫu nhiên có thể kết thúc sớm; bỏ qua thế đó
        assert board_text(state) == trap["board"]
        values = solver.move_values(state)
        r, c = (int(v) for v in trap["mcts_move"].split(","))
        assert values[(r, c)] == -1
        assert max(values.values()) >= 0


def test_walk_with_real_mcts_is_deterministic_and_consistent() -> None:
    first = walk_response_tree("X", iterations=20, seed=123)
    second = walk_response_tree("X", iterations=20, seed=123)
    assert first == second
    assert first["decisions"] > 0
    # Có thế thua trong cây khi và chỉ khi MCTS đã đi ít nhất một "nước thua".
    assert (first["tree_losses"] > 0) == (first["losing_moves"] > 0)


def test_mcts_with_few_iterations_does_lose_somewhere() -> None:
    """Lý do phải tìm ngưỡng I*: với rất ít iteration, MCTS thua được."""
    losses = sum(walk_response_tree("O", iterations=5, seed=s)["tree_losses"] for s in range(3))
    assert losses > 0


# ---------------------------------------------------------------------------
# Đấu Minimax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["X", "O"])
def test_play_against_perfect_opponent_never_wins(role: str) -> None:
    result = play_against(AlphaBetaAgent("natural"), role, 30, seed=1, c=1.4, selection="robust")
    assert result in ("draw", "loss")


def test_mcts_with_enough_iterations_draws_against_minimax() -> None:
    for role in ("X", "O"):
        result = play_against(
            AlphaBetaAgent("natural"), role, 2000, seed=7, c=vnl.DEFAULT_C, selection="robust"
        )
        assert result == "draw"


# ---------------------------------------------------------------------------
# Tổng hợp
# ---------------------------------------------------------------------------


def fake_row(iterations: int, seed_index: int, role: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "iterations": iterations,
        "seed_index": seed_index,
        "role": role,
        "decisions": 10,
        "tree_losses": 0,
        "losing_moves": 0,
        "missed_wins": 0,
        "vs_minimax": "draw",
        "vs_minimax_random": "draw",
        "traps": [],
    }
    row.update(overrides)
    return row


def test_summarize_counts_clean_seeds_over_both_roles() -> None:
    rows = [
        fake_row(100, 0, "X"),
        fake_row(100, 0, "O", tree_losses=3, losing_moves=2),
        fake_row(100, 1, "X"),
        fake_row(100, 1, "O", vs_minimax="loss"),
        fake_row(500, 0, "X"),
        fake_row(500, 0, "O"),
    ]
    low, high = summarize(rows)
    assert (low["iterations"], low["seeds"], low["seeds_never_lose"]) == (100, 2, 1)
    assert (low["tree_losses"], low["losing_moves"], low["decisions"]) == (3, 2, 40)
    assert (low["minimax_games"], low["minimax_draws"], low["minimax_losses"]) == (8, 7, 1)
    assert low["never_lose"] == 0
    assert (high["seeds"], high["seeds_never_lose"], high["never_lose"]) == (1, 1, 1)


def test_tree_clean_but_minimax_loss_does_not_pass() -> None:
    (entry,) = summarize([fake_row(100, 0, "X", vs_minimax_random="loss"), fake_row(100, 0, "O")])
    assert entry["seeds_never_lose"] == 1
    assert entry["never_lose"] == 0


def test_threshold_requires_every_larger_level_to_pass() -> None:
    def levels(*flags: tuple[int, int]) -> list[dict[str, Any]]:
        return [{"iterations": i, "never_lose": ok} for i, ok in flags]

    assert find_threshold(levels((10, 0), (100, 1), (1000, 1))) == 100
    assert find_threshold(levels((10, 1), (100, 0), (500, 1), (1000, 1))) == 500
    assert find_threshold(levels((10, 1), (100, 1))) == 10
    assert find_threshold(levels((10, 1), (100, 0))) is None
    assert find_threshold([]) is None


def test_collect_traps_merges_identical_traps_across_seeds() -> None:
    trap = {"role": "O", "board": "X../.../...", "mcts_move": "0,1", "good_moves": "1,1"}
    other = {"role": "O", "board": "..X/.../...", "mcts_move": "0,1", "good_moves": "1,1"}
    rows = [
        fake_row(10, 0, "O", traps=[dict(trap), dict(other)]),
        fake_row(10, 1, "O", traps=[dict(trap)]),
        fake_row(50, 0, "O", traps=[dict(trap)]),
    ]
    traps = collect_traps(rows)
    assert [(t["iterations"], t["board"], t["seeds"]) for t in traps] == [
        (10, "X../.../...", 2),
        (10, "..X/.../...", 1),
        (50, "X../.../...", 1),
    ]


# ---------------------------------------------------------------------------
# Cả lần chạy
# ---------------------------------------------------------------------------


def test_run_verification_rows_are_complete_and_reproducible() -> None:
    rows = run_verification([5, 10], seeds=2, workers=1)
    assert [(r["iterations"], r["seed_index"], r["role"]) for r in rows] == [
        (i, s, role) for i in (5, 10) for s in range(2) for role in ("O", "X")
    ]
    assert all(set(DETAIL_COLUMNS) <= set(row) for row in rows)
    assert len({row["seed"] for row in rows}) == len(rows)

    def stable(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in item.items() if k != "time_s"} for item in items]

    assert stable(run_verification([5, 10], seeds=2, workers=2)) == stable(rows)


def test_main_writes_three_csv_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["--iterations", "5", "--seeds", "2", "--workers", "1", "--out-dir", str(tmp_path)]
    )
    assert code == 0
    assert "I*" in capsys.readouterr().out

    def read(name: str) -> tuple[tuple[str, ...], list[dict[str, str]]]:
        with open(tmp_path / name, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return tuple(reader.fieldnames or ()), list(reader)

    columns, detail = read("verify_never_lose.csv")
    assert columns == DETAIL_COLUMNS
    assert len(detail) == 4
    columns, summary = read("verify_never_lose_summary.csv")
    assert columns == SUMMARY_COLUMNS
    assert [row["iterations"] for row in summary] == ["5"]
    columns, traps = read("verify_never_lose_traps.csv")
    assert columns == TRAP_COLUMNS
    assert sum(int(t["seeds"]) for t in traps) == sum(int(r["losing_moves"]) for r in detail)


def test_main_rejects_bad_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--seeds", "0"]) == 2
    assert main(["--iterations", "0"]) == 2
    assert main(["--workers", "0"]) == 2
    assert "Lỗi" in capsys.readouterr().err
