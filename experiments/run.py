"""Runner thực nghiệm: đọc cấu hình YAML, cho các agent đấu nhau, ghi mỗi ván một dòng CSV.

Cách chạy (từ thư mục gốc của repo):

    python -m experiments.run configs/smoke_random_3x3.yaml
    python -m experiments.run configs/smoke_random_3x3.yaml --workers 4 --out ket_qua.csv

Chủ sở hữu: TV1. Schema CSV là `CSV_COLUMNS` bên dưới; đổi schema phải báo TV2 (script vẽ hình).

Cấu hình YAML
-------------
    name: ten_thuc_nghiem          # bắt buộc; cũng là tên file CSV mặc định
    seed: 2026                     # seed gốc, mọi seed khác suy ra từ đây
    games: 200                     # số ván cho MỖI cặp đấu
    swap_first: true               # true: ván chẵn A đi trước, ván lẻ B đi trước
    opening_plies: 0               # số nước mở đầu ngẫu nhiên (lấy từ candidate_moves)
    workers: 4                     # số tiến trình; bỏ trống thì dùng số CPU - 1
    game: {type: gomoku, size: 9}  # hoặc {type: tictactoe}
    matchups:
      - a: {type: mcts, c: [0.5, 1.4], budget: {iterations: [100, 1000]}}
        b: {type: random}

Giá trị nào của agent là một danh sách thì được bung thành mọi tổ hợp: ví dụ trên cho
2 × 2 = 4 cặp đấu, mỗi cặp `games` ván.

Nước mở đầu
-----------
`opening_plies` nước đầu được chọn ngẫu nhiên từ `state.candidate_moves()`: trên 3×3 đó là mọi
ô trống; trên caro quân đầu nằm ở tâm bàn và các quân sau nằm quanh những quân đã đánh, nên thế
mở đầu không bị rải ra các góc. Vì quân đầu của caro luôn ở tâm, cần `opening_plies >= 2` mới
có thế mở đầu khác nhau.

Tái lập kết quả
---------------
Seed của ván, của nước mở đầu và của từng agent đều suy ra từ (seed gốc, số thứ tự cặp đấu,
số thứ tự ván) bằng SHA-256, nên không phụ thuộc số tiến trình hay thứ tự chạy. Chỉ các cột
thời gian là thay đổi giữa các lần chạy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from src.agents.alphabeta import AlphaBetaAgent
from src.agents.alphabeta_gomoku import AlphaBetaGomoku
from src.agents.base import Agent, Budget
from src.agents.mcts import MCTSAgent
from src.agents.minimax import MinimaxAgent
from src.agents.random_agent import RandomAgent
from src.games.base import PLAYER_O, PLAYER_X, GameState
from src.games.gomoku import Gomoku
from src.games.tictactoe import TicTacToe

RESULTS_DIR = Path("experiments") / "results"

# Schema CSV: mỗi dòng là một ván. "a" và "b" là hai agent của cặp đấu, không phải X và O.
CSV_COLUMNS: tuple[str, ...] = (
    "experiment",  # tên thực nghiệm (trường `name` của cấu hình)
    "game",  # nhãn trò chơi, ví dụ "tictactoe", "gomoku9"
    "matchup",  # số thứ tự cặp đấu sau khi bung tổ hợp, từ 0
    "game_index",  # số thứ tự ván trong cặp đấu, từ 0
    "seed",  # seed của ván
    "agent_a",  # nhãn agent A, ví dụ "mcts[c=1.4,iterations=1000]"
    "agent_b",  # nhãn agent B
    "a_params",  # JSON: toàn bộ tham số và ngân sách của A
    "b_params",  # JSON: toàn bộ tham số và ngân sách của B
    "first_player",  # "a" hoặc "b": ai cầm X (đi trước)
    "opening_plies",  # số nước mở đầu ngẫu nhiên đã đánh trước khi agent vào cuộc
    "winner",  # "a", "b" hoặc "draw"
    "score_a",  # điểm của A: thắng 1, hòa 0.5, thua 0
    "plies",  # tổng số nước của ván, tính cả nước mở đầu
    "a_moves",  # số nước A đã chọn
    "b_moves",  # số nước B đã chọn
    "a_nodes",  # tổng last_stats["nodes"] của A trong cả ván
    "b_nodes",  # tổng last_stats["nodes"] của B trong cả ván
    "a_time_s",  # tổng thời gian nghĩ của A (runner đo), giây
    "b_time_s",  # tổng thời gian nghĩ của B (runner đo), giây
    "a_max_time_s",  # nước lâu nhất của A, giây
    "b_max_time_s",  # nước lâu nhất của B, giây
)

BUDGET_FIELDS = ("iterations", "depth", "time_s")
CONFIG_FIELDS = (
    "name",
    "seed",
    "games",
    "swap_first",
    "opening_plies",
    "workers",
    "game",
    "matchups",
)

AgentBuilder = Callable[[Mapping[str, Any], int], Agent]
GameBuilder = Callable[[Mapping[str, Any]], GameState]


class ConfigError(ValueError):
    """Cấu hình YAML sai hoặc thiếu trường."""


# ---------------------------------------------------------------------------
# Đăng ký trò chơi và agent
# ---------------------------------------------------------------------------


def _check_params(kind: str, params: Mapping[str, Any], allowed: Iterable[str]) -> None:
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        raise ConfigError(f"{kind}: tham số không hợp lệ {unknown}; cho phép {sorted(allowed)}")


def _build_tictactoe(params: Mapping[str, Any]) -> GameState:
    _check_params("game tictactoe", params, ())
    return TicTacToe()


def _build_gomoku(params: Mapping[str, Any]) -> GameState:
    _check_params("game gomoku", params, ("size", "win_length", "block_two_ends"))
    return Gomoku(**params)


def _build_random(params: Mapping[str, Any], seed: int) -> Agent:
    _check_params("agent random", params, ())
    return RandomAgent(seed=seed)


def _build_minimax(params: Mapping[str, Any], seed: int) -> Agent:
    _check_params("agent minimax", params, ())
    return MinimaxAgent()


def _build_alphabeta(params: Mapping[str, Any], seed: int) -> Agent:
    _check_params("agent alphabeta", params, ("ordering",))
    return AlphaBetaAgent(seed=seed, **params)


ALPHABETA_GOMOKU_PARAMS = (
    "radius",
    "default_depth",
    "max_candidates",
    "use_candidates",
    "use_immediate_tactics",
    "use_move_ordering",
    "use_iterative_deepening",
    "use_tt",
)


def _build_alphabeta_gomoku(params: Mapping[str, Any], seed: int) -> Agent:
    _check_params("agent alphabeta_gomoku", params, ALPHABETA_GOMOKU_PARAMS)
    return AlphaBetaGomoku(**params)


MCTS_PARAMS = (
    "c",
    "selection",
    "rollout",
    "rollout_depth",
    "use_candidates",
    "candidate_radius",
    "reuse_tree",
)


def _build_mcts(params: Mapping[str, Any], seed: int) -> Agent:
    _check_params("agent mcts", params, MCTS_PARAMS)
    return MCTSAgent(seed=seed, **params)


GAME_BUILDERS: dict[str, GameBuilder] = {
    "tictactoe": _build_tictactoe,
    "gomoku": _build_gomoku,
}

# Muốn thêm agent mới (ví dụ Alpha-Beta caro, MCTS với rollout khác) thì thêm một dòng ở đây.
# Hàm dựng nhận (tham số từ YAML, seed của agent) và phải nằm ở cấp module để chạy song song.
AGENT_BUILDERS: dict[str, AgentBuilder] = {
    "random": _build_random,
    "minimax": _build_minimax,
    "alphabeta": _build_alphabeta,
    "alphabeta_gomoku": _build_alphabeta_gomoku,
    "mcts": _build_mcts,
}


def build_game(spec: Mapping[str, Any]) -> GameState:
    params = dict(spec)
    kind = params.pop("type", None)
    if kind not in GAME_BUILDERS:
        raise ConfigError(f"game.type phải thuộc {sorted(GAME_BUILDERS)}, nhận được {kind!r}")
    try:
        return GAME_BUILDERS[kind](params)
    except ConfigError:
        raise
    except (TypeError, ValueError) as error:  # ví dụ size nhỏ hơn win_length
        raise ConfigError(f"game {kind}: {error}") from error


def game_label(spec: Mapping[str, Any]) -> str:
    """Nhãn ngắn của trò chơi cho cột `game`: "tictactoe", "gomoku9", "gomoku15-block"."""
    label = str(spec.get("type"))
    if label == "gomoku":
        label += str(spec.get("size", 15))
        if spec.get("win_length", 5) != 5:
            label += f"-win{spec['win_length']}"
        if spec.get("block_two_ends", False):
            label += "-block"
    return label


def build_agent(spec: Mapping[str, Any], seed: int) -> tuple[Agent, Budget]:
    """Dựng agent và ngân sách của nó từ một đặc tả đã bung (không còn danh sách)."""
    params = dict(spec)
    kind = params.pop("type", None)
    if kind not in AGENT_BUILDERS:
        raise ConfigError(f"agent.type phải thuộc {sorted(AGENT_BUILDERS)}, nhận được {kind!r}")
    budget_spec = dict(params.pop("budget", None) or {})
    _check_params(f"budget của agent {kind}", budget_spec, BUDGET_FIELDS)
    try:
        return AGENT_BUILDERS[kind](params, seed), Budget(**budget_spec)
    except ConfigError:
        raise
    except (TypeError, ValueError) as error:  # ví dụ ordering lạ, c âm
        raise ConfigError(f"agent {kind}: {error}") from error


def agent_label(spec: Mapping[str, Any]) -> str:
    """Nhãn dễ đọc của một agent đã bung: "mcts[c=1.4,iterations=1000]"."""
    params = {k: v for k, v in spec.items() if k not in ("type", "budget")}
    params.update(spec.get("budget") or {})
    if not params:
        return str(spec.get("type"))
    inner = ",".join(f"{key}={params[key]}" for key in sorted(params))
    return f"{spec.get('type')}[{inner}]"


# ---------------------------------------------------------------------------
# Đọc và bung cấu hình
# ---------------------------------------------------------------------------


def expand_agent(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Bung mọi giá trị dạng danh sách của một agent thành các tổ hợp cụ thể."""
    if not isinstance(spec, Mapping) or "type" not in spec:
        raise ConfigError(f"Mỗi agent phải là một mục có trường `type`, nhận được {spec!r}")
    budget = spec.get("budget") or {}
    if not isinstance(budget, Mapping):
        raise ConfigError(f"budget phải là một mục khóa-giá trị, nhận được {budget!r}")

    # Mỗi trục là (thuộc budget hay không, tên, các giá trị).
    axes: list[tuple[bool, str, list[Any]]] = []
    for in_budget, source in ((False, spec), (True, budget)):
        for key, value in source.items():
            if key == "budget" and not in_budget:
                continue
            values = list(value) if isinstance(value, list) else [value]
            if not values:
                raise ConfigError(f"Danh sách giá trị của `{key}` đang rỗng")
            axes.append((in_budget, key, values))

    variants: list[dict[str, Any]] = []
    for combo in itertools.product(*(values for _, _, values in axes)):
        variant: dict[str, Any] = {}
        variant_budget: dict[str, Any] = {}
        for (in_budget, key, _), value in zip(axes, combo, strict=True):
            (variant_budget if in_budget else variant)[key] = value
        if variant_budget:
            variant["budget"] = variant_budget
        variants.append(variant)
    return variants


def load_config(path: str | Path) -> dict[str, Any]:
    """Đọc file YAML, kiểm tra, điền mặc định và bung `matchups` thành danh sách cặp cụ thể."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return normalize_config(raw)


def _is_int(value: object) -> bool:
    """Số nguyên thật sự: trong Python `True` cũng là `int`, nên phải loại bool riêng."""
    return isinstance(value, int) and not isinstance(value, bool)


def normalize_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConfigError("Cấu hình phải là một mục khóa-giá trị ở cấp cao nhất")
    _check_params("cấu hình", raw, CONFIG_FIELDS)
    for key in ("name", "game", "matchups"):
        if key not in raw:
            raise ConfigError(f"Cấu hình thiếu trường `{key}`")

    seed = raw.get("seed", 0)
    games = raw.get("games", 50)
    swap_first = raw.get("swap_first", True)
    opening_plies = raw.get("opening_plies", 0)
    workers = raw.get("workers")
    if not _is_int(seed):
        raise ConfigError(f"seed phải là số nguyên, nhận được {seed!r}")
    if not _is_int(games) or games < 1:
        raise ConfigError(f"games phải là số nguyên >= 1, nhận được {games!r}")
    if not isinstance(swap_first, bool):
        raise ConfigError(f"swap_first phải là true hoặc false, nhận được {swap_first!r}")
    if not _is_int(opening_plies) or opening_plies < 0:
        raise ConfigError(f"opening_plies phải là số nguyên >= 0, nhận được {opening_plies!r}")
    if workers is not None and (not _is_int(workers) or workers < 1):
        raise ConfigError(f"workers phải là số nguyên >= 1, nhận được {workers!r}")
    if not isinstance(raw["game"], Mapping):
        raise ConfigError("game phải là một mục có trường `type`")
    if not isinstance(raw["matchups"], list) or not raw["matchups"]:
        raise ConfigError("matchups phải là một danh sách không rỗng")

    game = dict(raw["game"])
    build_game(game)  # báo lỗi cấu hình trò chơi ngay, trước khi chạy

    matchups: list[dict[str, Any]] = []
    for entry in raw["matchups"]:
        if not isinstance(entry, Mapping) or set(entry) != {"a", "b"}:
            raise ConfigError(f"Mỗi cặp đấu phải có đúng hai trường `a` và `b`: {entry!r}")
        for a_spec, b_spec in itertools.product(expand_agent(entry["a"]), expand_agent(entry["b"])):
            build_agent(a_spec, 0)  # báo lỗi cấu hình agent ngay
            build_agent(b_spec, 0)
            matchups.append({"a": a_spec, "b": b_spec})

    return {
        "name": str(raw["name"]),
        "seed": seed,
        "games": games,
        "swap_first": swap_first,
        "opening_plies": opening_plies,
        "workers": workers,
        "game": game,
        "matchups": matchups,
    }


# ---------------------------------------------------------------------------
# Chơi một ván
# ---------------------------------------------------------------------------


def derive_seed(*parts: object) -> int:
    """Seed 63 bit suy ra tất định từ các thành phần, giống nhau trên mọi máy và mọi tiến trình."""
    text = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 1


def play_game(config: Mapping[str, Any], matchup: int, game_index: int) -> dict[str, Any]:
    """Chơi một ván của cặp đấu `matchup` và trả về một dòng theo `CSV_COLUMNS`."""
    specs = config["matchups"][matchup]
    seed = derive_seed(config["seed"], matchup, game_index)
    a_first = not (config["swap_first"] and game_index % 2 == 1)

    state = build_game(config["game"])

    # Hai ván liền nhau (đổi bên) dùng chung một thế mở đầu để so sánh công bằng.
    opening_group = game_index // 2 if config["swap_first"] else game_index
    opening_rng = random.Random(derive_seed(config["seed"], matchup, "opening", opening_group))
    opening_done = 0
    while opening_done < config["opening_plies"] and not state.is_terminal():
        state.play(opening_rng.choice(state.candidate_moves()))
        opening_done += 1

    sides: dict[str, tuple[Agent, Budget]] = {
        side: build_agent(specs[side], derive_seed(seed, side)) for side in ("a", "b")
    }
    side_of_player = {PLAYER_X: "a", PLAYER_O: "b"} if a_first else {PLAYER_X: "b", PLAYER_O: "a"}

    moves = {"a": 0, "b": 0}
    nodes = {"a": 0, "b": 0}
    total_time = {"a": 0.0, "b": 0.0}
    max_time = {"a": 0.0, "b": 0.0}

    while not state.is_terminal():
        side = side_of_player[state.current_player()]
        agent, budget = sides[side]
        start = time.perf_counter()
        move = agent.select_move(state, budget)
        elapsed = time.perf_counter() - start
        state.play(move)  # nước không hợp lệ sẽ ném ValueError và dừng cả lần chạy

        moves[side] += 1
        nodes[side] += int(agent.last_stats.get("nodes", 0))
        total_time[side] += elapsed
        max_time[side] = max(max_time[side], elapsed)

    result = state.winner()
    assert result is not None
    winner = "draw" if result == 0 else side_of_player[result]
    score_a = {"a": 1.0, "draw": 0.5, "b": 0.0}[winner]

    return {
        "experiment": config["name"],
        "game": game_label(config["game"]),
        "matchup": matchup,
        "game_index": game_index,
        "seed": seed,
        "agent_a": agent_label(specs["a"]),
        "agent_b": agent_label(specs["b"]),
        "a_params": json.dumps(specs["a"], sort_keys=True),
        "b_params": json.dumps(specs["b"], sort_keys=True),
        "first_player": "a" if a_first else "b",
        "opening_plies": opening_done,
        "winner": winner,
        "score_a": score_a,
        "plies": state.move_count(),
        "a_moves": moves["a"],
        "b_moves": moves["b"],
        "a_nodes": nodes["a"],
        "b_nodes": nodes["b"],
        "a_time_s": round(total_time["a"], 6),
        "b_time_s": round(total_time["b"], 6),
        "a_max_time_s": round(max_time["a"], 6),
        "b_max_time_s": round(max_time["b"], 6),
    }


def _play_chunk(args: tuple[Mapping[str, Any], Sequence[tuple[int, int]]]) -> list[dict[str, Any]]:
    """Chạy một lô ván trong tiến trình con. Phải ở cấp module để pickle được trên Windows."""
    config, tasks = args
    return [play_game(config, matchup, game_index) for matchup, game_index in tasks]


# ---------------------------------------------------------------------------
# Chạy cả thực nghiệm
# ---------------------------------------------------------------------------


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def run_experiment(config: Mapping[str, Any], workers: int | None = None) -> list[dict[str, Any]]:
    """Chạy mọi ván của cấu hình đã chuẩn hóa. Kết quả xếp theo (matchup, game_index)."""
    tasks = [(m, g) for m in range(len(config["matchups"])) for g in range(config["games"])]
    if workers is None:
        workers = config.get("workers") or default_workers()
    if not _is_int(workers) or workers < 1:
        raise ValueError(f"workers phải là số nguyên >= 1, nhận được {workers!r}")
    workers = min(workers, len(tasks))

    if workers == 1:
        rows = _play_chunk((config, tasks))
    else:
        # Lô nhỏ để chia việc đều; lô xen kẽ để mỗi tiến trình nhận đủ loại cặp đấu.
        chunk_count = min(len(tasks), workers * 8)
        chunks = [tasks[i::chunk_count] for i in range(chunk_count)]
        jobs = [(config, chunk) for chunk in chunks]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = [row for part in pool.map(_play_chunk, jobs) for row in part]

    rows.sort(key=lambda row: (row["matchup"], row["game_index"]))
    return rows


def write_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return target


def summarize(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Gộp theo cặp đấu: số ván thắng / hòa / thua của A và tỷ lệ điểm của A."""
    groups: dict[int, dict[str, Any]] = {}
    for row in rows:
        entry = groups.setdefault(
            row["matchup"],
            {
                "matchup": row["matchup"],
                "agent_a": row["agent_a"],
                "agent_b": row["agent_b"],
                "games": 0,
                "a_wins": 0,
                "draws": 0,
                "b_wins": 0,
            },
        )
        entry["games"] += 1
        entry[{"a": "a_wins", "draw": "draws", "b": "b_wins"}[row["winner"]]] += 1
    for entry in groups.values():
        entry["score_a"] = (entry["a_wins"] + 0.5 * entry["draws"]) / entry["games"]
    return [groups[key] for key in sorted(groups)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm đấu agent từ cấu hình YAML.")
    parser.add_argument("config", help="đường dẫn file YAML, ví dụ configs/smoke_random_3x3.yaml")
    parser.add_argument("--workers", type=int, help="số tiến trình (mặc định: theo cấu hình)")
    parser.add_argument("--out", help="file CSV đầu ra (mặc định: experiments/results/<name>.csv)")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.workers is not None and args.workers < 1:
            raise ConfigError(f"--workers phải >= 1, nhận được {args.workers}")
    except (OSError, yaml.YAMLError, ConfigError) as error:
        print(f"Lỗi cấu hình: {error}", file=sys.stderr)
        return 2

    total = len(config["matchups"]) * config["games"]
    print(f"{config['name']}: {len(config['matchups'])} cặp đấu × {config['games']} ván = {total}")

    start = time.perf_counter()
    rows = run_experiment(config, workers=args.workers)
    elapsed = time.perf_counter() - start
    target = write_csv(rows, args.out or RESULTS_DIR / f"{config['name']}.csv")

    for entry in summarize(rows):
        print(
            f"  [{entry['matchup']}] {entry['agent_a']} vs {entry['agent_b']}: "
            f"A thắng {entry['a_wins']}, hòa {entry['draws']}, B thắng {entry['b_wins']} "
            f"(điểm A {entry['score_a']:.3f})"
        )
    print(f"Xong {total} ván trong {elapsed:.1f} s -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
