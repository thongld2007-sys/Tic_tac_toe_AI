"""Test runner thực nghiệm (P3.1a)."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import pytest

from experiments.run import (
    CSV_COLUMNS,
    ConfigError,
    agent_label,
    build_agent,
    build_game,
    derive_seed,
    expand_agent,
    game_label,
    load_config,
    main,
    normalize_config,
    play_game,
    run_experiment,
    summarize,
    write_csv,
)
from src.agents.alphabeta import AlphaBetaAgent
from src.agents.mcts import MCTSAgent
from src.agents.random_agent import RandomAgent

REPO_ROOT = Path(__file__).resolve().parent.parent
TIME_COLUMNS = {"a_time_s", "b_time_s", "a_max_time_s", "b_max_time_s"}


def config(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "name": "test",
        "seed": 7,
        "games": 6,
        "game": {"type": "tictactoe"},
        "matchups": [{"a": {"type": "random"}, "b": {"type": "random"}}],
    }
    raw.update(overrides)
    return normalize_config(raw)


def without_times(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if k not in TIME_COLUMNS} for row in rows]


# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------


def test_defaults_are_filled_in() -> None:
    cfg = normalize_config(
        {
            "name": "x",
            "game": {"type": "tictactoe"},
            "matchups": [{"a": {"type": "random"}, "b": {"type": "minimax"}}],
        }
    )
    assert cfg["seed"] == 0
    assert cfg["games"] == 50
    assert cfg["swap_first"] is True
    assert cfg["opening_plies"] == 0
    assert cfg["workers"] is None
    assert cfg["matchups"] == [{"a": {"type": "random"}, "b": {"type": "minimax"}}]


def test_list_values_expand_to_every_combination() -> None:
    variants = expand_agent({"type": "mcts", "c": [0.5, 1.4], "budget": {"iterations": [100, 200]}})
    assert variants == [
        {"type": "mcts", "c": 0.5, "budget": {"iterations": 100}},
        {"type": "mcts", "c": 0.5, "budget": {"iterations": 200}},
        {"type": "mcts", "c": 1.4, "budget": {"iterations": 100}},
        {"type": "mcts", "c": 1.4, "budget": {"iterations": 200}},
    ]
    assert expand_agent({"type": "random"}) == [{"type": "random"}]


def test_matchups_are_the_product_of_both_sides() -> None:
    cfg = config(
        matchups=[
            {
                "a": {
                    "type": "mcts",
                    "c": [0.5, 1.0, 1.4, 2.0],
                    "budget": {"iterations": [10, 20]},
                },
                "b": {"type": "alphabeta", "ordering": ["natural", "heuristic"]},
            },
            {"a": {"type": "random"}, "b": {"type": "random"}},
        ]
    )
    assert len(cfg["matchups"]) == 4 * 2 * 2 + 1
    assert cfg["matchups"][0] == {
        "a": {"type": "mcts", "c": 0.5, "budget": {"iterations": 10}},
        "b": {"type": "alphabeta", "ordering": "natural"},
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"games": 0},
        {"games": "many"},
        {"games": True},
        {"games": 2.5},
        {"seed": "abc"},
        {"seed": True},
        {"swap_first": "yes"},
        {"swap_first": 1},
        {"opening_plies": -1},
        {"opening_plies": False},
        {"workers": 0},
        {"workers": True},
        {"matchups": []},
        {"matchups": [{"a": {"type": "random"}}]},
        {"matchups": [{"a": {"type": "random"}, "b": {"type": "nobody"}}]},
        {"matchups": [{"a": {"type": "random"}, "b": {"type": "mcts", "cc": 1.4}}]},
        {"matchups": [{"a": {"type": "random"}, "b": {"type": "mcts", "c": []}}]},
        {"matchups": [{"a": {"type": "random", "budget": {"x": 5}}, "b": {"type": "random"}}]},
        {"matchups": [{"a": {"type": "alphabeta", "ordering": "best"}, "b": {"type": "random"}}]},
        {"game": {"type": "chess"}},
        {"game": {"type": "gomoku", "board": 9}},
        {"game": {"type": "gomoku", "size": 3}},
        {"typo": 1},
    ],
)
def test_bad_config_is_rejected_before_running(overrides: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        config(**overrides)


def test_missing_required_field_is_rejected() -> None:
    with pytest.raises(ConfigError):
        normalize_config({"name": "x", "game": {"type": "tictactoe"}})
    with pytest.raises(ConfigError):
        normalize_config(["not", "a", "mapping"])


def test_labels() -> None:
    assert agent_label({"type": "random"}) == "random"
    spec = {"type": "mcts", "c": 1.4, "budget": {"iterations": 1000}}
    assert agent_label(spec) == "mcts[c=1.4,iterations=1000]"
    assert game_label({"type": "tictactoe"}) == "tictactoe"
    assert game_label({"type": "gomoku", "size": 9}) == "gomoku9"
    assert game_label({"type": "gomoku"}) == "gomoku15"
    assert game_label({"type": "gomoku", "size": 9, "block_two_ends": True}) == "gomoku9-block"


def test_build_agent_passes_params_seed_and_budget() -> None:
    agent, budget = build_agent(
        {"type": "mcts", "c": 0.5, "selection": "max", "budget": {"iterations": 30}}, seed=11
    )
    assert isinstance(agent, MCTSAgent)
    assert (agent.c, agent.selection, agent.seed) == (0.5, "max", 11)
    assert (budget.iterations, budget.depth, budget.time_s) == (30, None, None)

    agent, budget = build_agent({"type": "alphabeta", "ordering": "heuristic"}, seed=1)
    assert isinstance(agent, AlphaBetaAgent)
    assert agent.ordering == "heuristic"
    assert budget.depth is None

    agent, _ = build_agent({"type": "random"}, seed=5)
    assert isinstance(agent, RandomAgent)
    assert agent.seed == 5


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def test_derive_seed_is_stable_and_spread_out() -> None:
    # Giá trị cố định: nếu đổi cách suy ra seed thì mọi kết quả cũ không còn tái lập được.
    assert derive_seed(2026, 0, 0) == 5615496007769692471
    assert derive_seed(2026, 0, 0) != derive_seed(2026, 0, 1) != derive_seed(2026, 1, 0)
    seeds = {derive_seed(1, m, g) for m in range(20) for g in range(50)}
    assert len(seeds) == 1000
    assert all(0 <= s < 2**63 for s in seeds)


# ---------------------------------------------------------------------------
# Một ván
# ---------------------------------------------------------------------------


def test_row_has_exactly_the_csv_columns() -> None:
    row = play_game(config(), 0, 0)
    assert tuple(row) == CSV_COLUMNS


def test_row_is_consistent() -> None:
    cfg = config(games=40)
    for index in range(40):
        row = play_game(cfg, 0, index)
        assert row["experiment"] == "test"
        assert row["game"] == "tictactoe"
        assert row["first_player"] == ("a" if index % 2 == 0 else "b")
        assert row["winner"] in ("a", "b", "draw")
        assert row["score_a"] == {"a": 1.0, "draw": 0.5, "b": 0.0}[row["winner"]]
        assert 5 <= row["plies"] <= 9
        assert row["a_moves"] + row["b_moves"] == row["plies"]
        first, second = ("a", "b") if row["first_player"] == "a" else ("b", "a")
        assert row[f"{first}_moves"] - row[f"{second}_moves"] in (0, 1)
        assert row["a_nodes"] == row["a_moves"]  # RandomAgent: 1 nút mỗi nước
        assert row["a_time_s"] >= row["a_max_time_s"] >= 0
        assert json.loads(row["a_params"]) == {"type": "random"}


def test_swap_first_can_be_turned_off() -> None:
    cfg = config(swap_first=False)
    assert {play_game(cfg, 0, i)["first_player"] for i in range(6)} == {"a"}


def test_same_seed_gives_same_game() -> None:
    cfg = config()
    assert without_times([play_game(cfg, 0, 3)]) == without_times([play_game(cfg, 0, 3)])

    other = config(seed=8)
    games = without_times([play_game(cfg, 0, i) for i in range(6)])
    other_games = without_times([play_game(other, 0, i) for i in range(6)])
    assert [g["plies"] for g in games] != [g["plies"] for g in other_games] or [
        g["winner"] for g in games
    ] != [g["winner"] for g in other_games]


def test_perfect_players_always_draw() -> None:
    cfg = config(
        games=4,
        matchups=[
            {
                "a": {"type": "alphabeta", "ordering": "heuristic"},
                "b": {"type": "alphabeta", "ordering": "natural"},
            }
        ],
    )
    rows = run_experiment(cfg, workers=1)
    assert [row["winner"] for row in rows] == ["draw"] * 4
    assert all(row["plies"] == 9 for row in rows)
    assert all(row["a_nodes"] > row["a_moves"] for row in rows)


def test_alphabeta_never_loses_to_random_through_the_runner() -> None:
    cfg = config(
        games=30,
        matchups=[{"a": {"type": "alphabeta", "ordering": "heuristic"}, "b": {"type": "random"}}],
    )
    summary = summarize(run_experiment(cfg, workers=1))
    assert summary[0]["b_wins"] == 0
    assert summary[0]["a_wins"] > 15


def test_mcts_runs_with_iteration_budget() -> None:
    cfg = config(
        games=4,
        matchups=[
            {
                "a": {"type": "mcts", "c": 1.4, "budget": {"iterations": 50}},
                "b": {"type": "random"},
            }
        ],
    )
    rows = run_experiment(cfg, workers=1)
    assert rows[0]["agent_a"] == "mcts[c=1.4,iterations=50]"
    assert all(row["a_nodes"] > row["a_moves"] for row in rows)
    assert without_times(rows) == without_times(run_experiment(cfg, workers=1))


def test_opening_plies_are_shared_by_each_swapped_pair() -> None:
    cfg = config(game={"type": "gomoku", "size": 7}, opening_plies=4, games=4)
    rows = [play_game(cfg, 0, i) for i in range(4)]
    assert all(row["opening_plies"] == 4 for row in rows)
    assert all(row["plies"] >= 4 + 9 for row in rows)
    assert all(row["a_moves"] + row["b_moves"] == row["plies"] - 4 for row in rows)
    assert rows[0]["game"] == "gomoku7"


def test_gomoku_opening_stays_near_the_centre() -> None:
    """Nước mở đầu lấy từ candidate_moves(): quân đầu ở tâm, các quân sau quanh quân đã có."""
    cfg = config(game={"type": "gomoku", "size": 15}, opening_plies=4)
    for group in range(30):
        state = build_game(cfg["game"])
        rng = random.Random(derive_seed(cfg["seed"], 0, "opening", group))
        for _ in range(4):
            state.play(rng.choice(state.candidate_moves()))
        stones = [(r, c) for r in range(15) for c in range(15) if state.board[r, c] != 0]
        assert (7, 7) in stones
        # 4 quân, mỗi quân cách một quân trước đó không quá 2 ô: không ra khỏi vùng quanh tâm.
        assert all(1 <= r <= 13 and 1 <= c <= 13 for r, c in stones)

    rows = [play_game(cfg, 0, i) for i in range(2)]
    assert all(row["opening_plies"] == 4 for row in rows)


def test_gomoku_game_can_be_played_to_the_end() -> None:
    cfg = config(game={"type": "gomoku", "size": 5, "block_two_ends": True}, games=3)
    rows = run_experiment(cfg, workers=1)
    assert all(9 <= row["plies"] <= 25 for row in rows)
    assert rows[0]["game"] == "gomoku5-block"


# ---------------------------------------------------------------------------
# Cả thực nghiệm
# ---------------------------------------------------------------------------


def test_rows_are_sorted_and_complete() -> None:
    cfg = config(
        games=5,
        matchups=[
            {"a": {"type": "random"}, "b": {"type": "random"}},
            {"a": {"type": "random"}, "b": {"type": "alphabeta", "ordering": "heuristic"}},
        ],
    )
    rows = run_experiment(cfg, workers=1)
    assert [(r["matchup"], r["game_index"]) for r in rows] == [
        (m, g) for m in range(2) for g in range(5)
    ]


def test_parallel_run_matches_serial_run() -> None:
    cfg = config(games=24)
    serial = run_experiment(cfg, workers=1)
    parallel = run_experiment(cfg, workers=2)
    assert without_times(parallel) == without_times(serial)


def test_summarize_counts_results_per_matchup() -> None:
    rows = [
        {"matchup": 0, "agent_a": "x", "agent_b": "y", "winner": "a"},
        {"matchup": 0, "agent_a": "x", "agent_b": "y", "winner": "draw"},
        {"matchup": 0, "agent_a": "x", "agent_b": "y", "winner": "a"},
        {"matchup": 1, "agent_a": "x", "agent_b": "z", "winner": "b"},
    ]
    first, second = summarize(rows)
    assert (first["games"], first["a_wins"], first["draws"], first["b_wins"]) == (3, 2, 1, 0)
    assert first["score_a"] == pytest.approx(2.5 / 3)
    assert (second["agent_b"], second["b_wins"], second["score_a"]) == ("z", 1, 0.0)


def test_write_csv_round_trips(tmp_path: Path) -> None:
    rows = run_experiment(config(), workers=1)
    target = write_csv(rows, tmp_path / "nested" / "out.csv")

    with open(target, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
        loaded = list(reader)

    assert len(loaded) == len(rows)
    assert loaded[0]["winner"] == rows[0]["winner"]
    assert int(loaded[0]["seed"]) == rows[0]["seed"]
    assert json.loads(loaded[0]["a_params"]) == {"type": "random"}


# ---------------------------------------------------------------------------
# Dòng lệnh và tiêu chí nghiệm thu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["smoke_random_3x3.yaml", "example_mcts_3x3.yaml"])
def test_shipped_configs_are_valid(name: str) -> None:
    cfg = load_config(REPO_ROOT / "configs" / name)
    assert cfg["matchups"]


def test_cli_runs_200_random_games_in_under_a_minute(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "smoke.csv"
    start = time.perf_counter()
    code = main([str(REPO_ROOT / "configs" / "smoke_random_3x3.yaml"), "--out", str(out)])
    elapsed = time.perf_counter() - start

    assert code == 0
    assert elapsed < 60
    with open(out, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 200
    assert sum(row["first_player"] == "a" for row in rows) == 100
    assert {row["winner"] for row in rows} == {"a", "b", "draw"}
    assert "200" in capsys.readouterr().out


@pytest.mark.parametrize("workers", ["0", "-1"])
def test_cli_rejects_non_positive_workers(
    workers: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "never.csv"
    smoke = str(REPO_ROOT / "configs" / "smoke_random_3x3.yaml")
    assert main([smoke, "--workers", workers, "--out", str(out)]) == 2
    assert "--workers" in capsys.readouterr().err
    assert not out.exists()


def test_run_experiment_rejects_non_positive_workers() -> None:
    with pytest.raises(ValueError):
        run_experiment(config(), workers=0)


def test_cli_reports_bad_seed_as_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad_seed.yaml"
    lines = [
        "name: x",
        "seed: abc",
        "game: {type: tictactoe}",
        "matchups:",
        "  - a: {type: random}",
        "    b: {type: random}",
    ]
    bad.write_text("\n".join(lines), encoding="utf-8")
    assert main([str(bad)]) == 2
    assert "seed" in capsys.readouterr().err


def test_cli_reports_bad_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\ngame: {type: tictactoe}\n", encoding="utf-8")
    assert main([str(bad)]) == 2
    assert "matchups" in capsys.readouterr().err
    assert main([str(tmp_path / "missing.yaml")]) == 2
