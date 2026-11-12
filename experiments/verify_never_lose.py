"""Kiểm chứng "MCTS-UCT không bao giờ thua" trên Tic-Tac-Toe 3×3 (P2.1) → bảng T3 và ngưỡng I*.

Cách chạy (từ thư mục gốc của repo):

    python -m experiments.verify_never_lose
    python -m experiments.verify_never_lose --iterations 100 1000 --seeds 5 --workers 4

Chủ sở hữu: TV1.

Phương pháp
-----------
MCTS là thuật toán ngẫu nhiên, nên "không bao giờ thua" được kiểm chứng bằng thực nghiệm, theo
từng mức iteration I và từng seed, với hai phép thử:

1. **Duyệt cây phản ứng.** Cố định MCTS ở một bên (X hoặc O). Tới lượt MCTS thì lấy đúng nước
   MCTS chọn; tới lượt đối thủ thì thử MỌI nước hợp lệ. Nếu không có thế kết thúc nào là MCTS
   thua thì không đối thủ nào, dù chơi cách gì, thắng được MCTS với seed đó. Phép thử này mạnh
   hơn đấu với một đối thủ cụ thể.

   Để "nước MCTS chọn" là một hàm của thế cờ, mỗi lần quyết định dùng một MCTSAgent mới, seed
   suy ra từ (seed, hash của thế cờ), không tái dùng cây. Thế trùng nhau chỉ được xét một lần.

2. **Đấu Minimax hoàn hảo.** Một MCTSAgent chơi trọn ván (cách dùng thông thường) với
   Alpha-Beta thứ tự `natural` (đã kiểm chứng là đi giống hệt Minimax) và với Alpha-Beta thứ tự
   `random` (hoàn hảo nhưng chọn ngẫu nhiên giữa các nước tốt nhất), ở cả hai vai đi trước và
   đi sau.

Mỗi quyết định của MCTS còn được so với giá trị chính xác của thế cờ (giải toàn bộ 3×3):
  - "nước thua" (thế bẫy): thế đang không thua mà MCTS đi vào thế thua nếu đối thủ chơi đúng;
  - "bỏ lỡ thắng": thế đang thắng mà MCTS đi vào thế chỉ còn hòa.

I* là mức iteration nhỏ nhất trong lưới sao cho ở mức đó VÀ mọi mức lớn hơn, cả 20 seed đều
không có thế thua nào trong cây phản ứng và mọi ván với Minimax đều hòa.

Kết quả ghi vào experiments/results/:
  verify_never_lose.csv          mỗi dòng là một (iterations, seed, vai của MCTS)
  verify_never_lose_summary.csv  bảng T3: mỗi dòng là một mức iteration
  verify_never_lose_traps.csv    các thế bẫy: thế cờ, nước MCTS đã đi, nước đúng
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from experiments.run import default_workers, derive_seed
from src.agents.alphabeta import AlphaBetaAgent
from src.agents.base import Agent, Budget
from src.agents.mcts import DEFAULT_C, FINAL_SELECTIONS, MCTSAgent
from src.games.base import PLAYER_O, PLAYER_X, GameState, Move
from src.games.tictactoe import TicTacToe

RESULTS_DIR = Path("experiments") / "results"
DEFAULT_ITERATIONS = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
DEFAULT_SEEDS = 20
ROLES = {"X": PLAYER_X, "O": PLAYER_O}

DETAIL_COLUMNS: tuple[str, ...] = (
    "iterations",  # số iteration cho mỗi nước của MCTS
    "seed_index",  # số thứ tự seed, từ 0
    "seed",  # seed thực dùng
    "role",  # vai của MCTS: "X" (đi trước) hoặc "O" (đi sau)
    "c",  # hằng số khám phá
    "selection",  # cách chọn nước cuối
    "decisions",  # số thế khác nhau mà MCTS phải chọn nước trong cây phản ứng
    "terminal_positions",  # số thế kết thúc khác nhau trong cây phản ứng
    "tree_wins",  # trong đó MCTS thắng
    "tree_draws",  # trong đó hòa
    "tree_losses",  # trong đó MCTS thua (phải bằng 0 thì mới "không thua")
    "losing_moves",  # số quyết định là "nước thua" (thế bẫy)
    "missed_wins",  # số quyết định "bỏ lỡ thắng"
    "vs_minimax",  # kết quả ván với Alpha-Beta natural (= Minimax): "win" / "draw" / "loss"
    "vs_minimax_random",  # kết quả ván với Alpha-Beta random: "win" / "draw" / "loss"
    "time_s",  # thời gian chạy của dòng này, giây
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "iterations",
    "seeds",  # số seed đã thử
    "seeds_never_lose",  # số seed không có thế thua nào trong cây phản ứng, ở cả hai vai
    "tree_losses",  # tổng số thế thua trong cây phản ứng (mọi seed, hai vai)
    "losing_moves",  # tổng số "nước thua"
    "missed_wins",  # tổng số "bỏ lỡ thắng"
    "decisions",  # tổng số quyết định đã kiểm tra
    "minimax_games",  # số ván đấu Minimax hoàn hảo
    "minimax_draws",  # số ván hòa trong đó
    "minimax_losses",  # số ván MCTS thua trong đó
    "never_lose",  # 1 nếu mức này đạt: không thế thua nào và mọi ván với Minimax đều hòa
)

TRAP_COLUMNS: tuple[str, ...] = (
    "iterations",
    "role",  # vai của MCTS
    "board",  # thế cờ trước nước của MCTS, ba hàng cách nhau bởi "/", ô trống là "."
    "mcts_move",  # nước MCTS đã đi, dạng "hàng,cột"
    "good_moves",  # các nước giữ được giá trị của thế cờ, cách nhau bởi dấu cách
    "seeds",  # số seed mắc đúng nước thua này
)


# ---------------------------------------------------------------------------
# Giải chính xác 3×3
# ---------------------------------------------------------------------------


class Solver:
    """Giá trị chính xác của mọi thế cờ, theo góc nhìn của người sắp đi: 1 / 0 / -1."""

    def __init__(self) -> None:
        self._values: dict[int, int] = {}

    def value(self, state: GameState) -> int:
        key = state.hash()
        cached = self._values.get(key)
        if cached is not None:
            return cached

        result = state.winner()
        if result is not None:
            value = result * state.current_player()
        else:
            value = -2
            for m in state.legal_moves():
                state.play(m)
                value = max(value, -self.value(state))
                state.undo()
                if value == 1:
                    break
        self._values[key] = value
        return value

    def move_values(self, state: GameState) -> dict[Move, int]:
        """Giá trị của từng nước hợp lệ, theo góc nhìn của người sắp đi."""
        values: dict[Move, int] = {}
        for m in state.legal_moves():
            state.play(m)
            values[m] = -self.value(state)
            state.undo()
        return values


def board_text(state: GameState) -> str:
    symbols = {0: ".", PLAYER_X: "X", PLAYER_O: "O"}
    return "/".join("".join(symbols[int(v)] for v in row) for row in state.board)


# ---------------------------------------------------------------------------
# Phép thử 1: duyệt cây phản ứng
# ---------------------------------------------------------------------------


def mcts_decision(
    state: GameState, iterations: int, seed: int, c: float, selection: str
) -> Move:
    """Nước MCTS chọn ở `state`, là hàm tất định của (seed, thế cờ)."""
    agent = MCTSAgent(c=c, selection=selection, seed=derive_seed(seed, state.hash()))
    return agent.select_move(state, Budget(iterations=iterations))


def walk_response_tree(
    role: str,
    iterations: int,
    seed: int,
    c: float = DEFAULT_C,
    selection: str = "robust",
    solver: Solver | None = None,
) -> dict[str, Any]:
    """Duyệt mọi cách đối thủ có thể đi khi MCTS giữ vai `role`. Trả về số đếm và các thế bẫy."""
    mcts_player = ROLES[role]
    solver = solver or Solver()
    state = TicTacToe()
    seen: set[int] = set()
    counts = {"decisions": 0, "win": 0, "draw": 0, "loss": 0, "losing_moves": 0, "missed_wins": 0}
    traps: list[dict[str, Any]] = []

    def visit() -> None:
        key = state.hash()
        if key in seen:
            return
        seen.add(key)

        result = state.winner()
        if result is not None:
            outcome = "draw" if result == 0 else ("win" if result == mcts_player else "loss")
            counts[outcome] += 1
            return

        if state.current_player() != mcts_player:
            for m in state.legal_moves():
                state.play(m)
                visit()
                state.undo()
            return

        counts["decisions"] += 1
        move = mcts_decision(state, iterations, seed, c, selection)
        values = solver.move_values(state)
        best = max(values.values())
        if values[move] < best:
            if values[move] == -1:
                counts["losing_moves"] += 1
                traps.append(
                    {
                        "role": role,
                        "board": board_text(state),
                        "mcts_move": f"{move[0]},{move[1]}",
                        "good_moves": " ".join(
                            f"{m[0]},{m[1]}" for m, v in values.items() if v == best
                        ),
                    }
                )
            else:
                counts["missed_wins"] += 1
        state.play(move)
        visit()
        state.undo()

    visit()
    return {
        "decisions": counts["decisions"],
        "terminal_positions": counts["win"] + counts["draw"] + counts["loss"],
        "tree_wins": counts["win"],
        "tree_draws": counts["draw"],
        "tree_losses": counts["loss"],
        "losing_moves": counts["losing_moves"],
        "missed_wins": counts["missed_wins"],
        "traps": traps,
    }


# ---------------------------------------------------------------------------
# Phép thử 2: đấu Minimax hoàn hảo
# ---------------------------------------------------------------------------


def play_against(
    opponent: Agent, role: str, iterations: int, seed: int, c: float, selection: str
) -> str:
    """Một ván giữa MCTS (vai `role`) và `opponent`. Trả về kết quả của MCTS."""
    mcts_player = ROLES[role]
    mcts = MCTSAgent(c=c, selection=selection, seed=seed)
    mcts_budget, full = Budget(iterations=iterations), Budget()
    state = TicTacToe()
    while not state.is_terminal():
        if state.current_player() == mcts_player:
            state.play(mcts.select_move(state, mcts_budget))
        else:
            state.play(opponent.select_move(state, full))
    result = state.winner()
    assert result is not None
    return "draw" if result == 0 else ("win" if result == mcts_player else "loss")


# ---------------------------------------------------------------------------
# Một dòng kết quả, và cả lần chạy
# ---------------------------------------------------------------------------


def verify_one(task: Mapping[str, Any]) -> dict[str, Any]:
    """Chạy cả hai phép thử cho một (iterations, seed, vai). Ở cấp module để chạy song song."""
    start = time.perf_counter()
    iterations, role = task["iterations"], task["role"]
    c, selection = task["c"], task["selection"]
    seed = derive_seed(task["base_seed"], "never-lose", iterations, task["seed_index"], role)

    row = walk_response_tree(role, iterations, seed, c, selection)
    row.update(
        {
            "iterations": iterations,
            "seed_index": task["seed_index"],
            "seed": seed,
            "role": role,
            "c": round(c, 6),
            "selection": selection,
            "vs_minimax": play_against(
                AlphaBetaAgent("natural"), role, iterations, seed, c, selection
            ),
            "vs_minimax_random": play_against(
                AlphaBetaAgent("random", seed=derive_seed(seed, "opponent")),
                role,
                iterations,
                seed,
                c,
                selection,
            ),
        }
    )
    row["time_s"] = round(time.perf_counter() - start, 3)
    return row


def run_verification(
    iterations: Sequence[int],
    seeds: int = DEFAULT_SEEDS,
    base_seed: int = 2026,
    c: float = DEFAULT_C,
    selection: str = "robust",
    workers: int | None = None,
) -> list[dict[str, Any]]:
    """Chạy mọi (iterations, seed, vai). Kết quả xếp theo (iterations, seed_index, vai)."""
    tasks = [
        {
            "iterations": i,
            "seed_index": s,
            "role": role,
            "base_seed": base_seed,
            "c": c,
            "selection": selection,
        }
        for i in iterations
        for s in range(seeds)
        for role in ROLES
    ]
    workers = min(workers or default_workers(), len(tasks))
    if workers <= 1:
        rows = [verify_one(task) for task in tasks]
    else:
        # Việc nặng (iteration lớn) xếp trước để các tiến trình chia tải đều hơn.
        ordered = sorted(tasks, key=lambda t: -t["iterations"])
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(verify_one, ordered))
    rows.sort(key=lambda r: (r["iterations"], r["seed_index"], r["role"]))
    return rows


def summarize(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Bảng T3: gộp theo mức iteration."""
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["iterations"], []).append(row)

    summary: list[dict[str, Any]] = []
    for iterations in sorted(groups):
        group = groups[iterations]
        seed_ids = sorted({r["seed_index"] for r in group})
        clean_seeds = sum(
            all(r["tree_losses"] == 0 for r in group if r["seed_index"] == s) for s in seed_ids
        )
        games = [r[col] for r in group for col in ("vs_minimax", "vs_minimax_random")]
        tree_losses = sum(r["tree_losses"] for r in group)
        summary.append(
            {
                "iterations": iterations,
                "seeds": len(seed_ids),
                "seeds_never_lose": clean_seeds,
                "tree_losses": tree_losses,
                "losing_moves": sum(r["losing_moves"] for r in group),
                "missed_wins": sum(r["missed_wins"] for r in group),
                "decisions": sum(r["decisions"] for r in group),
                "minimax_games": len(games),
                "minimax_draws": games.count("draw"),
                "minimax_losses": games.count("loss"),
                "never_lose": int(tree_losses == 0 and games.count("draw") == len(games)),
            }
        )
    return summary


def find_threshold(summary: Sequence[Mapping[str, Any]]) -> int | None:
    """I*: mức iteration nhỏ nhất mà từ đó trở lên mọi mức trong lưới đều đạt `never_lose`."""
    threshold: int | None = None
    for entry in sorted(summary, key=lambda e: e["iterations"], reverse=True):
        if not entry["never_lose"]:
            break
        threshold = entry["iterations"]
    return threshold


def collect_traps(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Gộp các thế bẫy giống nhau và đếm số seed mắc phải."""
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        for trap in row["traps"]:
            key = (row["iterations"], trap["role"], trap["board"], trap["mcts_move"])
            entry = merged.setdefault(
                key, {"iterations": row["iterations"], **trap, "seeds": 0}
            )
            entry["seeds"] += 1
    return sorted(
        merged.values(), key=lambda t: (t["iterations"], -t["seeds"], t["role"], t["board"])
    )


def write_csv(rows: Iterable[Mapping[str, Any]], columns: Sequence[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Kiểm chứng "MCTS không bao giờ thua" trên 3×3.')
    parser.add_argument("--iterations", type=int, nargs="+", default=list(DEFAULT_ITERATIONS))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--c", type=float, default=DEFAULT_C)
    parser.add_argument("--selection", choices=FINAL_SELECTIONS, default="robust")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--name", default="verify_never_lose", help="tiền tố tên file kết quả")
    args = parser.parse_args(argv)

    bad_workers = args.workers is not None and args.workers < 1
    if args.seeds < 1 or min(args.iterations) < 1 or bad_workers:
        print("Lỗi: --seeds, --iterations và --workers phải >= 1", file=sys.stderr)
        return 2

    iterations = sorted(set(args.iterations))
    start = time.perf_counter()
    rows = run_verification(
        iterations, args.seeds, args.base_seed, args.c, args.selection, args.workers
    )
    elapsed = time.perf_counter() - start

    summary = summarize(rows)
    traps = collect_traps(rows)
    out_dir = Path(args.out_dir)
    write_csv(rows, DETAIL_COLUMNS, out_dir / f"{args.name}.csv")
    write_csv(summary, SUMMARY_COLUMNS, out_dir / f"{args.name}_summary.csv")
    write_csv(traps, TRAP_COLUMNS, out_dir / f"{args.name}_traps.csv")

    print(f"c = {args.c:.4f}, selection = {args.selection}, {args.seeds} seed, hai vai X và O")
    print("iteration | seed không thua | thế thua | nước thua | Minimax: hòa/ván | đạt")
    for entry in summary:
        print(
            f"{entry['iterations']:>9} | {entry['seeds_never_lose']:>7}/{entry['seeds']:<7} |"
            f" {entry['tree_losses']:>8} | {entry['losing_moves']:>9} |"
            f" {entry['minimax_draws']:>7}/{entry['minimax_games']:<8} |"
            f" {'có' if entry['never_lose'] else 'không'}"
        )
    threshold = find_threshold(summary)
    if threshold is None:
        print("I*: chưa đạt ở mức iteration lớn nhất đã thử")
    else:
        print(f"I* = {threshold}")
    print(f"Xong trong {elapsed:.1f} s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
