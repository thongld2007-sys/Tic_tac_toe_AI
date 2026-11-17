"""Script sinh toàn bộ biểu đồ vector PDF và bảng tổng hợp T4, T5 từ CSV thực nghiệm (P3.2b).

Chủ sở hữu: TV2.  Review: TV3.

Cách chạy từ thư mục gốc của repo:
    python -m experiments.plots
    make figures

Đầu vào:
  - `experiments/results/p31b_3x3.csv` & `p31b_9x9.csv` (R3: khảo sát c × iteration).
  - `experiments/results/p32b_r4_3x3.csv` & `p32b_r4_9x9.csv` (R4: so sánh 3 quy tắc chọn
    nước cuối tại gốc `robust`, `max`, `max_robust`).

Đầu ra:
  - `report/figures/F1.pdf`: Biểu đồ tỷ lệ thắng theo c × iteration trên Tic-Tac-Toe 3×3.
  - `report/figures/F2.pdf`: Biểu đồ tỷ lệ thắng theo c × iteration trên Caro 9×9.
  - `report/figures/F3.pdf`: Biểu đồ so sánh 3 tiêu chuẩn chọn nước cuối (R4) trên 3×3 và 9×9.
  - `experiments/results/p32b_r4_root_t4.csv`: Bảng T4 — thống kê nút gốc (N, W, W/N, UCT)
    trích xuất từ `agent.last_stats["root"]` tại thế bẫy E1, thế bẫy đường chéo 3×3 và Caro 9×9.
  - `experiments/results/p32b_r4_summary_t5.csv`: Bảng T5 — tổng hợp kết quả R4 kèm khoảng
    tin cậy Wilson 95%, kiểm định hai tỷ lệ z-test và bootstrap CI cho hiệu điểm số.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib.pyplot as plt

from experiments.p31b_figures import plot as plot_p31b_curve
from experiments.stats import (
    bootstrap_diff_ci,
    summarize_matchup,
    two_proportion_ztest,
    wilson_ci,
)
from src.agents.base import Budget, RootStat
from src.agents.mcts import FINAL_SELECTIONS, UCT_VARIANTS, MCTSAgent, select_final_child
from src.games.base import GameState, Move
from src.games.gomoku import Gomoku
from src.games.tictactoe import TicTacToe

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
FIGURES_DIR = ROOT / "report" / "figures"

R3_BUDGETS: tuple[int, ...] = (100, 500, 1000, 2000, 5000)
R3_COEFFICIENTS: tuple[float, ...] = (0.5, 1.0, 1.4, 2.0)

SELECTION_LABELS: dict[str, str] = {
    "robust": "Robust (argmax N)",
    "max": "Max (argmax W/N)",
    "max_robust": "Max-Robust (LCB)",
}
SELECTION_MARKERS: dict[str, str] = {
    "robust": "o",
    "max": "s",
    "max_robust": "^",
}

UCT_VARIANT_LABELS: dict[str, str] = {
    "standard": "Standard UCT",
    "no_sqrt": "No-Sqrt (B2)",
    "no_log": "No-Log (B2)",
    "sqrt_n": "Sqrt-N / PUCT (B2)",
    "ucb1_tuned": "UCB1-Tuned (B3)",
    "progressive_bias": "Progressive Bias (B3)",
}
UCT_VARIANT_MARKERS: dict[str, str] = {
    "standard": "o",
    "no_sqrt": "s",
    "no_log": "^",
    "sqrt_n": "v",
    "ucb1_tuned": "D",
    "progressive_bias": "P",
}

B2_B3_COLUMNS: tuple[str, ...] = (
    "board",
    "category",
    "variant_a",
    "variant_b",
    "iterations",
    "games",
    "wins",
    "draws",
    "losses",
    "win_rate",
    "win_ci_low",
    "win_ci_high",
    "non_loss_rate",
    "non_loss_ci_low",
    "non_loss_ci_high",
    "mean_score",
    "score_ci_low",
    "score_ci_high",
    "vs_std_z",
    "vs_std_p",
    "vs_std_diff_mean",
    "vs_std_diff_ci_low",
    "vs_std_diff_ci_high",
)

T4_COLUMNS: tuple[str, ...] = (
    "position",
    "board",
    "iterations",
    "c",
    "seed",
    "move",
    "is_optimal",
    "N",
    "W",
    "q_signed",
    "q_unit",
    "uct",
    "picked_robust",
    "picked_max",
    "picked_max_robust",
    "robust_opt_rate_200",
    "max_opt_rate_200",
    "max_robust_opt_rate_200",
    "disagree_rate_200",
)

T5_COLUMNS: tuple[str, ...] = (
    "board",
    "category",
    "selection_a",
    "selection_b",
    "iterations",
    "games",
    "wins",
    "draws",
    "losses",
    "win_rate",
    "win_ci_low",
    "win_ci_high",
    "non_loss_rate",
    "non_loss_ci_low",
    "non_loss_ci_high",
    "mean_score",
    "score_ci_low",
    "score_ci_high",
    "vs_max_z",
    "vs_max_p",
    "vs_max_diff_mean",
    "vs_max_diff_ci_low",
    "vs_max_diff_ci_high",
)


# ---------------------------------------------------------------------------
# Đọc CSV và tổng hợp thống kê R4 (Bảng T4 & Bảng T5)
# ---------------------------------------------------------------------------


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Đọc toàn bộ các dòng của một file CSV dưới dạng list[dict]."""
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Ghi danh sách từ điển ra file CSV theo đúng thứ tự cột `columns`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in columns})


def _build_tactical_positions() -> list[tuple[str, str, GameState, set[Move], int, float]]:
    """Khởi tạo các thế cờ chiến thuật tiêu biểu để trích xuất bảng T4 (`last_stats['root']`).

    Trả về danh sách `(position_name, board_name, state, optimal_moves, iterations, c)`:
      1. `E1_trap`: Thế bẫy E1 trên bàn 3×3 (`O(0,0), X(0,1), O(1,1), X(2,0)`), tới lượt X
         bắt buộc phải đánh `(2, 2)` để chặn đường chéo của O.
      2. `3x3_diagonal_fork`: Thế bẫy khai cuộc đường chéo `X(0,0), O(1,1), X(2,2)` (phát hiện
         trong P2.1), tới lượt O bắt buộc phải đi vào 1 trong 4 ô cạnh `(0,1), (1,0), (1,2), (2,1)`
         và tránh 2 ô góc bẫy `(0,2), (2,0)`.
      3. `Gomoku9_open_three`: Thế cờ trên bàn Caro 9×9 khi X đang có chuỗi 3 quân hở hai đầu
         tại hàng 4 `(4,3)-(4,4)-(4,5)`, tới lượt O cần chặn ngay ở `(4,2)` hoặc `(4,6)`.
    """
    e1 = TicTacToe()
    for mv in [(0, 1), (0, 0), (2, 0), (1, 1)]:
        e1.play(mv)

    fork_3x3 = TicTacToe()
    for mv in [(0, 0), (1, 1), (2, 2)]:
        fork_3x3.play(mv)

    gmk9 = Gomoku(size=9)
    for mv in [(4, 4), (3, 4), (4, 3), (3, 5), (4, 5)]:
        gmk9.play(mv)

    return [
        ("E1_trap", "3x3", e1, {(2, 2)}, 60, 0.5),
        ("3x3_diagonal_fork", "3x3", fork_3x3, {(0, 1), (1, 0), (1, 2), (2, 1)}, 50, 0.5),
        ("Gomoku9_open_three", "9x9", gmk9, {(4, 2), (4, 6)}, 80, 0.5),
    ]


def _find_illustrative_seed(
    state: GameState,
    optimal_moves: set[Move],
    iterations: int,
    c: float,
    *,
    use_candidates: bool,
    num_seeds: int = 200,
) -> tuple[int, dict[str, float]]:
    """Quét `num_seeds` seed để đo tần suất chọn nước tối ưu và tìm 1 seed minh họa rõ nhất.

    Ưu tiên chọn seed mà tại đó `robust` chọn đúng nước tối ưu còn `max` bị đánh lừa bởi
    một nhánh ít lượt thăm nhưng `W/N` cao ngẫu nhiên (nếu có); nếu cả hai cùng chọn đúng,
    chọn seed `0`.
    """
    opt_counts = {sel: 0 for sel in FINAL_SELECTIONS}
    disagree_count = 0
    chosen_seed = 0
    found_divergence = False

    for seed in range(num_seeds):
        agent = MCTSAgent(
            c=c,
            selection="robust",
            rollout="l0",
            use_candidates=use_candidates,
            seed=seed,
        )
        agent.select_move(state, Budget(iterations=iterations))
        root = agent.last_root
        assert root is not None

        picks = {sel: select_final_child(root, sel, c).move for sel in FINAL_SELECTIONS}
        for sel, mv in picks.items():
            if mv in optimal_moves:
                opt_counts[sel] += 1
        if picks["robust"] != picks["max"]:
            disagree_count += 1
            if (
                not found_divergence
                and picks["robust"] in optimal_moves
                and picks["max"] not in optimal_moves
            ):
                chosen_seed = seed
                found_divergence = True

    rates = {
        "robust_opt_rate_200": round(opt_counts["robust"] / num_seeds, 4),
        "max_opt_rate_200": round(opt_counts["max"] / num_seeds, 4),
        "max_robust_opt_rate_200": round(opt_counts["max_robust"] / num_seeds, 4),
        "disagree_rate_200": round(disagree_count / num_seeds, 4),
    }
    return chosen_seed, rates


def generate_table_t4(
    out_csv: Path | None = None,
    *,
    num_seeds: int = 200,
) -> list[dict[str, Any]]:
    """Sinh Bảng T4 (`p32b_r4_root_t4.csv`) từ trường `agent.last_stats['root']`."""
    rows: list[dict[str, Any]] = []
    for pos_name, board_name, state, opt_moves, iters, c_val in _build_tactical_positions():
        use_cand = isinstance(state, Gomoku)
        seed, rates = _find_illustrative_seed(
            state,
            opt_moves,
            iters,
            c_val,
            use_candidates=use_cand,
            num_seeds=num_seeds,
        )
        agent = MCTSAgent(
            c=c_val,
            selection="robust",
            rollout="l0",
            use_candidates=use_cand,
            seed=seed,
        )
        agent.select_move(state, Budget(iterations=iters))
        root_node = agent.last_root
        assert root_node is not None
        picks = {sel: select_final_child(root_node, sel, c_val).move for sel in FINAL_SELECTIONS}

        root_stats: list[RootStat] = agent.last_stats.get("root", [])
        for item in root_stats:
            mv = (int(item["move"][0]), int(item["move"][1]))
            q_signed = float(item["q"])
            q_unit = (q_signed + 1.0) * 0.5
            rows.append(
                {
                    "position": pos_name,
                    "board": board_name,
                    "iterations": iters,
                    "c": c_val,
                    "seed": seed,
                    "move": f"({mv[0]},{mv[1]})",
                    "is_optimal": int(mv in opt_moves),
                    "N": int(item["N"]),
                    "W": round(float(item["W"]), 2),
                    "q_signed": round(q_signed, 4),
                    "q_unit": round(q_unit, 4),
                    "uct": round(float(item["uct"]), 4),
                    "picked_robust": int(mv == picks["robust"]),
                    "picked_max": int(mv == picks["max"]),
                    "picked_max_robust": int(mv == picks["max_robust"]),
                    **rates,
                }
            )

    if out_csv is not None:
        write_csv_rows(out_csv, T4_COLUMNS, rows)
    return rows


def _paired_win_loss_ztest(wins: int, losses: int) -> tuple[float, float]:
    """Kiểm định z ghép cặp (McNemar/sign test) cho số ván thắng-thua: z = (W - L) / sqrt(W + L)."""
    decisive = wins + losses
    if decisive <= 0:
        return 0.0, 1.0
    z_stat = (wins - losses) / math.sqrt(decisive)
    p_val = 2.0 * (1.0 - NormalDist(0.0, 1.0).cdf(abs(z_stat)))
    return float(z_stat), max(0.0, min(1.0, float(p_val)))


def summarize_r4_table_t5(
    csv_3x3: Path,
    csv_9x9: Path,
    out_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Tổng hợp Bảng T5 (`p32b_r4_summary_t5.csv`) từ kết quả chạy R4 trên 3×3 và 9×9."""
    summary_rows: list[dict[str, Any]] = []

    for board_label, path in (("3x3", csv_3x3), ("9x9", csv_9x9)):
        raw_rows = load_csv_rows(path)
        by_matchup: dict[int, list[dict[str, str]]] = defaultdict(list)
        for r in raw_rows:
            by_matchup[int(r["matchup"])].append(r)

        # Nhóm các cặp đấu vs baseline theo (selection, iterations) để đối chiếu với 'max'
        vs_baseline_groups: dict[tuple[str, int], list[dict[str, str]]] = {}
        h2h_groups: list[list[dict[str, str]]] = []

        for m_id in sorted(by_matchup.keys()):
            m_rows = by_matchup[m_id]
            a_params = json.loads(m_rows[0]["a_params"])
            b_params = json.loads(m_rows[0]["b_params"])
            sel_a = str(a_params.get("selection", "robust"))
            iters_a = int(a_params["budget"]["iterations"])

            # Phân biệt đấu với baseline cố định và đấu đối đầu trực tiếp (head-to-head)
            is_h2h = (
                b_params.get("type") == "mcts"
                and float(b_params.get("c", -1.0)) == float(a_params.get("c", -2.0))
                and str(b_params.get("selection", "")) != sel_a
            )
            if is_h2h:
                h2h_groups.append(m_rows)
            else:
                vs_baseline_groups[(sel_a, iters_a)] = m_rows

        budgets = sorted({it for _, it in vs_baseline_groups.keys()})
        for iters in budgets:
            max_rows = vs_baseline_groups.get(("max", iters), [])
            max_scores = [float(r["score_a"]) for r in max_rows]
            max_non_losses = sum(r["winner"] in ("a", "draw") for r in max_rows)
            max_wins = sum(r["winner"] == "a" for r in max_rows)

            for sel in FINAL_SELECTIONS:
                if (sel, iters) not in vs_baseline_groups:
                    continue
                rows = vs_baseline_groups[(sel, iters)]
                scores = [float(r["score_a"]) for r in rows]
                stats = summarize_matchup(scores, seed=2026)
                n_games = int(stats["games"])
                wins = int(stats["wins"])
                draws = int(stats["draws"])
                losses = int(stats["losses"])
                non_losses = wins + draws
                nl_low, nl_high = wilson_ci(non_losses, n_games)

                if sel != "max" and max_rows:
                    # Trên 3×3 đấu Alpha-Beta, so tỷ lệ bất bại; trên 9×9 so tỷ lệ thắng
                    if board_label == "3x3":
                        z_stat, p_val = two_proportion_ztest(
                            non_losses, n_games, max_non_losses, len(max_rows)
                        )
                    else:
                        z_stat, p_val = two_proportion_ztest(wins, n_games, max_wins, len(max_rows))
                    diff_mean = (sum(scores) / len(scores)) - (sum(max_scores) / len(max_scores))
                    diff_lo, diff_hi = bootstrap_diff_ci(scores, max_scores, seed=2026)
                else:
                    z_stat, p_val = 0.0, 1.0
                    diff_mean, diff_lo, diff_hi = 0.0, 0.0, 0.0

                b_type = json.loads(rows[0]["b_params"]).get("type", "baseline")
                summary_rows.append(
                    {
                        "board": board_label,
                        "category": "vs_baseline",
                        "selection_a": sel,
                        "selection_b": str(b_type),
                        "iterations": iters,
                        "games": n_games,
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                        "win_rate": round(float(stats["win_rate"]), 4),
                        "win_ci_low": round(float(stats["win_wilson_low"]), 4),
                        "win_ci_high": round(float(stats["win_wilson_high"]), 4),
                        "non_loss_rate": round(non_losses / n_games, 4),
                        "non_loss_ci_low": round(nl_low, 4),
                        "non_loss_ci_high": round(nl_high, 4),
                        "mean_score": round(float(stats["score_mean"]), 4),
                        "score_ci_low": round(float(stats["score_boot_low"]), 4),
                        "score_ci_high": round(float(stats["score_boot_high"]), 4),
                        "vs_max_z": round(z_stat, 4),
                        "vs_max_p": round(p_val, 4),
                        "vs_max_diff_mean": round(diff_mean, 4),
                        "vs_max_diff_ci_low": round(diff_lo, 4),
                        "vs_max_diff_ci_high": round(diff_hi, 4),
                    }
                )

        # Tổng hợp các cặp đấu đối đầu trực tiếp (head-to-head)
        for rows in h2h_groups:
            a_params = json.loads(rows[0]["a_params"])
            b_params = json.loads(rows[0]["b_params"])
            sel_a = str(a_params.get("selection", "robust"))
            sel_b = str(b_params.get("selection", "max"))
            iters = int(a_params["budget"]["iterations"])

            scores_a = [float(r["score_a"]) for r in rows]
            scores_b = [1.0 - s for s in scores_a]
            stats = summarize_matchup(scores_a, seed=2026)
            n_games = int(stats["games"])
            wins = int(stats["wins"])
            draws = int(stats["draws"])
            losses = int(stats["losses"])
            non_losses = wins + draws
            nl_low, nl_high = wilson_ci(non_losses, n_games)
            z_stat, p_val = _paired_win_loss_ztest(wins, losses)
            diff_mean = (sum(scores_a) / len(scores_a)) - (sum(scores_b) / len(scores_b))
            diff_lo, diff_hi = bootstrap_diff_ci(scores_a, scores_b, seed=2026, paired=True)

            summary_rows.append(
                {
                    "board": board_label,
                    "category": "head_to_head",
                    "selection_a": sel_a,
                    "selection_b": sel_b,
                    "iterations": iters,
                    "games": n_games,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "win_rate": round(float(stats["win_rate"]), 4),
                    "win_ci_low": round(float(stats["win_wilson_low"]), 4),
                    "win_ci_high": round(float(stats["win_wilson_high"]), 4),
                    "non_loss_rate": round(non_losses / n_games, 4),
                    "non_loss_ci_low": round(nl_low, 4),
                    "non_loss_ci_high": round(nl_high, 4),
                    "mean_score": round(float(stats["score_mean"]), 4),
                    "score_ci_low": round(float(stats["score_boot_low"]), 4),
                    "score_ci_high": round(float(stats["score_boot_high"]), 4),
                    "vs_max_z": round(z_stat, 4),
                    "vs_max_p": round(p_val, 4),
                    "vs_max_diff_mean": round(diff_mean, 4),
                    "vs_max_diff_ci_low": round(diff_lo, 4),
                    "vs_max_diff_ci_high": round(diff_hi, 4),
                }
            )

    if out_csv is not None:
        write_csv_rows(out_csv, T5_COLUMNS, summary_rows)
    return summary_rows


def summarize_b2_b3_table(
    csv_3x3: Path,
    csv_9x9: Path,
    out_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Tổng hợp kết quả thực nghiệm B2 (biến thể UCT) & B3 (cải tiến UCT) từ CSV (`P4.2a`)."""
    summary_rows: list[dict[str, Any]] = []

    for board_label, path in (("3x3", csv_3x3), ("9x9", csv_9x9)):
        raw_rows = load_csv_rows(path)
        by_matchup: dict[int, list[dict[str, str]]] = defaultdict(list)
        for r in raw_rows:
            by_matchup[int(r["matchup"])].append(r)

        vs_baseline_groups: dict[tuple[str, int], list[dict[str, str]]] = {}
        h2h_groups: list[list[dict[str, str]]] = []

        for m_id in sorted(by_matchup.keys()):
            m_rows = by_matchup[m_id]
            a_params = json.loads(m_rows[0]["a_params"])
            b_params = json.loads(m_rows[0]["b_params"])
            var_a = str(a_params.get("uct_variant", "standard"))
            iters_a = int(a_params["budget"]["iterations"])

            is_h2h_3x3 = (
                board_label == "3x3"
                and b_params.get("type") == "mcts"
                and int(b_params.get("budget", {}).get("iterations", -1)) == iters_a
                and str(b_params.get("uct_variant", "standard")) != var_a
            )
            if is_h2h_3x3:
                h2h_groups.append(m_rows)
            else:
                vs_baseline_groups[(var_a, iters_a)] = m_rows
                # Trên 9×9, ở mức N = 80 đấu với standard(N=80) cũng đồng thời là head-to-head
                if (
                    board_label == "9x9"
                    and b_params.get("type") == "mcts"
                    and int(b_params.get("budget", {}).get("iterations", -1)) == iters_a
                    and var_a != "standard"
                ):
                    h2h_groups.append(m_rows)

        budgets = sorted({it for _, it in vs_baseline_groups.keys()})
        for iters in budgets:
            std_rows = vs_baseline_groups.get(("standard", iters), [])
            std_scores = [float(r["score_a"]) for r in std_rows]
            std_non_losses = sum(r["winner"] in ("a", "draw") for r in std_rows)
            std_wins = sum(r["winner"] == "a" for r in std_rows)

            for var in UCT_VARIANTS:
                if (var, iters) not in vs_baseline_groups:
                    continue
                rows = vs_baseline_groups[(var, iters)]
                scores = [float(r["score_a"]) for r in rows]
                stats = summarize_matchup(scores, seed=2026)
                n_games = int(stats["games"])
                wins = int(stats["wins"])
                draws = int(stats["draws"])
                losses = int(stats["losses"])
                non_losses = wins + draws
                nl_low, nl_high = wilson_ci(non_losses, n_games)

                if var != "standard" and std_rows:
                    if board_label == "3x3":
                        z_stat, p_val = two_proportion_ztest(
                            non_losses, n_games, std_non_losses, len(std_rows)
                        )
                    else:
                        z_stat, p_val = two_proportion_ztest(wins, n_games, std_wins, len(std_rows))
                    diff_mean = (sum(scores) / len(scores)) - (sum(std_scores) / len(std_scores))
                    diff_lo, diff_hi = bootstrap_diff_ci(scores, std_scores, seed=2026)
                else:
                    z_stat, p_val = 0.0, 1.0
                    diff_mean, diff_lo, diff_hi = 0.0, 0.0, 0.0

                b_type = json.loads(rows[0]["b_params"]).get("type", "baseline")
                summary_rows.append(
                    {
                        "board": board_label,
                        "category": "vs_baseline",
                        "variant_a": var,
                        "variant_b": str(b_type),
                        "iterations": iters,
                        "games": n_games,
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                        "win_rate": round(float(stats["win_rate"]), 4),
                        "win_ci_low": round(float(stats["win_wilson_low"]), 4),
                        "win_ci_high": round(float(stats["win_wilson_high"]), 4),
                        "non_loss_rate": round(non_losses / n_games, 4),
                        "non_loss_ci_low": round(nl_low, 4),
                        "non_loss_ci_high": round(nl_high, 4),
                        "mean_score": round(float(stats["score_mean"]), 4),
                        "score_ci_low": round(float(stats["score_boot_low"]), 4),
                        "score_ci_high": round(float(stats["score_boot_high"]), 4),
                        "vs_std_z": round(z_stat, 4),
                        "vs_std_p": round(p_val, 4),
                        "vs_std_diff_mean": round(diff_mean, 4),
                        "vs_std_diff_ci_low": round(diff_lo, 4),
                        "vs_std_diff_ci_high": round(diff_hi, 4),
                    }
                )

        for rows in h2h_groups:
            a_params = json.loads(rows[0]["a_params"])
            b_params = json.loads(rows[0]["b_params"])
            var_a = str(a_params.get("uct_variant", "standard"))
            var_b = str(b_params.get("uct_variant", "standard"))
            iters = int(a_params["budget"]["iterations"])

            scores_a = [float(r["score_a"]) for r in rows]
            scores_b = [1.0 - s for s in scores_a]
            stats = summarize_matchup(scores_a, seed=2026)
            n_games = int(stats["games"])
            wins = int(stats["wins"])
            draws = int(stats["draws"])
            losses = int(stats["losses"])
            non_losses = wins + draws
            nl_low, nl_high = wilson_ci(non_losses, n_games)
            z_stat, p_val = _paired_win_loss_ztest(wins, losses)
            diff_mean = (sum(scores_a) / len(scores_a)) - (sum(scores_b) / len(scores_b))
            diff_lo, diff_hi = bootstrap_diff_ci(scores_a, scores_b, seed=2026, paired=True)

            summary_rows.append(
                {
                    "board": board_label,
                    "category": "head_to_head",
                    "variant_a": var_a,
                    "variant_b": var_b,
                    "iterations": iters,
                    "games": n_games,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "win_rate": round(float(stats["win_rate"]), 4),
                    "win_ci_low": round(float(stats["win_wilson_low"]), 4),
                    "win_ci_high": round(float(stats["win_wilson_high"]), 4),
                    "non_loss_rate": round(non_losses / n_games, 4),
                    "non_loss_ci_low": round(nl_low, 4),
                    "non_loss_ci_high": round(nl_high, 4),
                    "mean_score": round(float(stats["score_mean"]), 4),
                    "score_ci_low": round(float(stats["score_boot_low"]), 4),
                    "score_ci_high": round(float(stats["score_boot_high"]), 4),
                    "vs_std_z": round(z_stat, 4),
                    "vs_std_p": round(p_val, 4),
                    "vs_std_diff_mean": round(diff_mean, 4),
                    "vs_std_diff_ci_low": round(diff_lo, 4),
                    "vs_std_diff_ci_high": round(diff_hi, 4),
                }
            )

    if out_csv is not None:
        write_csv_rows(out_csv, B2_B3_COLUMNS, summary_rows)
    return summary_rows


# ---------------------------------------------------------------------------
# Vẽ biểu đồ vector PDF (F1, F2, F3)
# ---------------------------------------------------------------------------


def plot_r3_from_csv(csv_path: Path, out_pdf: Path, title: str) -> None:
    """Vẽ biểu đồ khảo sát c × iteration (F1 hoặc F2) từ `csv_path` ra `out_pdf`."""
    groups: dict[tuple[float, int], list[dict[str, str]]] = defaultdict(list)
    for row in load_csv_rows(csv_path):
        params = json.loads(row["a_params"])
        key = round(float(params["c"]) / 2.0, 4), int(params["budget"]["iterations"])
        groups[key].append(row)

    expected = {(c, n) for c in R3_COEFFICIENTS for n in R3_BUDGETS}
    if set(groups.keys()) != expected:
        raise ValueError(f"Thiếu lưới tham số (c, iteration) trong {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    for c, marker in zip(R3_COEFFICIENTS, ["o", "s", "^", "D"], strict=True):
        rates: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for n in R3_BUDGETS:
            rows = groups[(c, n)]
            wins = sum(r["winner"] == "a" for r in rows)
            low, high = wilson_ci(wins, len(rows))
            rate = wins / len(rows)
            rates.append(rate * 100.0)
            lows.append(max(0.0, rate - low) * 100.0)
            highs.append(max(0.0, high - rate) * 100.0)
        ax.errorbar(
            R3_BUDGETS,
            rates,
            yerr=[lows, highs],
            label=f"c = {c:.1f}",
            marker=marker,
            capsize=2,
            linewidth=1.3,
            markersize=4,
        )
    ax.set_xscale("log")
    ax.set_xticks(R3_BUDGETS, labels=[str(n) for n in R3_BUDGETS])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set(title=title, xlabel="Iterations per MCTS move (log scale)", ylabel="MCTS wins (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=9)
    fig.text(
        0.5,
        -0.045,
        "200 games per point · 95% Wilson intervals · c on the [0, 1] scale",
        ha="center",
        fontsize=9,
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def plot_r4_selection_figure(
    t5_rows: Sequence[Mapping[str, Any]],
    out_pdf: Path,
) -> None:
    """Vẽ biểu đồ vector PDF `F3.pdf` so sánh 3 quy tắc chọn nước cuối (R4) trên 3×3 và 9×9."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), layout="constrained")

    configs = [
        (
            axes[0],
            "3x3",
            "non_loss_rate",
            "non_loss_ci_low",
            "non_loss_ci_high",
            "Tic-Tac-Toe 3×3 · vs Perfect Alpha-Beta",
            "Non-losing rate: Win + Draw (%)",
        ),
        (
            axes[1],
            "9x9",
            "win_rate",
            "win_ci_low",
            "win_ci_high",
            "Gomoku 9×9 · vs Baseline MCTS (N=60)",
            "Win rate (%)",
        ),
    ]

    for ax, board_label, rate_col, lo_col, hi_col, subtitle, ylabel in configs:
        sub_rows = [
            r for r in t5_rows if r["board"] == board_label and r["category"] == "vs_baseline"
        ]
        budgets = sorted({int(r["iterations"]) for r in sub_rows})
        by_key = {(str(r["selection_a"]), int(r["iterations"])): r for r in sub_rows}

        for sel in FINAL_SELECTIONS:
            xs: list[int] = []
            rates: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for n in budgets:
                if (sel, n) not in by_key:
                    continue
                item = by_key[(sel, n)]
                rate = float(item[rate_col])
                lo = float(item[lo_col])
                hi = float(item[hi_col])
                xs.append(n)
                rates.append(rate * 100.0)
                lows.append(max(0.0, rate - lo) * 100.0)
                highs.append(max(0.0, hi - rate) * 100.0)
            if xs:
                ax.errorbar(
                    xs,
                    rates,
                    yerr=[lows, highs],
                    label=SELECTION_LABELS[sel],
                    marker=SELECTION_MARKERS[sel],
                    capsize=2,
                    linewidth=1.3,
                    markersize=4,
                )

        ax.set_xscale("log")
        if budgets:
            ax.set_xticks(budgets, labels=[str(n) for n in budgets])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set(
            title=subtitle,
            xlabel="Iterations per MCTS move (log scale)",
            ylabel=ylabel,
        )
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8.5)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def plot_b2_b3_variants_figure(
    b2_rows: Sequence[Mapping[str, Any]],
    out_pdf: Path,
) -> None:
    """Vẽ biểu đồ vector PDF `F4.pdf` so sánh các biến thể (B2) & cải tiến UCT (B3)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), layout="constrained")

    configs = [
        (
            axes[0],
            "3x3",
            "non_loss_rate",
            "non_loss_ci_low",
            "non_loss_ci_high",
            "Tic-Tac-Toe 3×3 · UCT Variants & Improvements vs Alpha-Beta",
            "Non-losing rate: Win + Draw (%)",
        ),
        (
            axes[1],
            "9x9",
            "win_rate",
            "win_ci_low",
            "win_ci_high",
            "Gomoku 9×9 · UCT Variants & Improvements vs Standard UCT (N=80)",
            "Win rate (%)",
        ),
    ]

    for ax, board_label, rate_col, lo_col, hi_col, subtitle, ylabel in configs:
        sub_rows = [
            r for r in b2_rows if r["board"] == board_label and r["category"] == "vs_baseline"
        ]
        budgets = sorted({int(r["iterations"]) for r in sub_rows})
        by_key = {(str(r["variant_a"]), int(r["iterations"])): r for r in sub_rows}

        for var in UCT_VARIANTS:
            xs: list[int] = []
            rates: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for n in budgets:
                if (var, n) not in by_key:
                    continue
                item = by_key[(var, n)]
                rate = float(item[rate_col])
                lo = float(item[lo_col])
                hi = float(item[hi_col])
                xs.append(n)
                rates.append(rate * 100.0)
                lows.append(max(0.0, rate - lo) * 100.0)
                highs.append(max(0.0, hi - rate) * 100.0)
            if xs:
                ax.errorbar(
                    xs,
                    rates,
                    yerr=[lows, highs],
                    label=UCT_VARIANT_LABELS[var],
                    marker=UCT_VARIANT_MARKERS[var],
                    capsize=2,
                    linewidth=1.3,
                    markersize=4,
                )

        ax.set_xscale("log")
        if budgets:
            ax.set_xticks(budgets, labels=[str(n) for n in budgets])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set(
            title=subtitle,
            xlabel="Iterations per MCTS move (log scale)",
            ylabel=ylabel,
        )
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def generate_all(
    *,
    results_dir: Path = RESULTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    t4_seeds: int = 200,
) -> dict[str, Path]:
    """Sinh toàn bộ các hình PDF (`F1.pdf`, `F2.pdf`, `F3.pdf`) và bảng CSV (`T4`, `T5`)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    # 1. Hình F1 và F2 (R3: khảo sát c × iteration)
    p31b_3x3 = results_dir / "p31b_3x3.csv"
    p31b_9x9 = results_dir / "p31b_9x9.csv"
    if results_dir == RESULTS_DIR and figures_dir == FIGURES_DIR:
        if p31b_3x3.exists():
            plot_p31b_curve("3x3", "F1", "Tic-Tac-Toe 3×3 · MCTS vs Random")
            outputs["F1"] = figures_dir / "F1.pdf"
        if p31b_9x9.exists():
            plot_p31b_curve("9x9", "F2", "Gomoku 9×9 · MCTS vs Random")
            outputs["F2"] = figures_dir / "F2.pdf"
    else:
        if p31b_3x3.exists():
            f1_path = figures_dir / "F1.pdf"
            plot_r3_from_csv(p31b_3x3, f1_path, "Tic-Tac-Toe 3×3 · MCTS vs Random")
            outputs["F1"] = f1_path
        if p31b_9x9.exists():
            f2_path = figures_dir / "F2.pdf"
            plot_r3_from_csv(p31b_9x9, f2_path, "Gomoku 9×9 · MCTS vs Random")
            outputs["F2"] = f2_path

    # 2. Bảng T4 (thống kê W, N, W/N, UCT tại nút gốc trên các thế cờ chiến thuật)
    t4_path = results_dir / "p32b_r4_root_t4.csv"
    generate_table_t4(out_csv=t4_path, num_seeds=t4_seeds)
    outputs["T4"] = t4_path

    # 3. Bảng T5 và Hình F3 (R4: so sánh 3 quy tắc chọn nước cuối tại gốc)
    r4_3x3 = results_dir / "p32b_r4_3x3.csv"
    r4_9x9 = results_dir / "p32b_r4_9x9.csv"
    if r4_3x3.exists() and r4_9x9.exists():
        t5_path = results_dir / "p32b_r4_summary_t5.csv"
        t5_rows = summarize_r4_table_t5(r4_3x3, r4_9x9, out_csv=t5_path)
        outputs["T5"] = t5_path

        f3_path = figures_dir / "F3.pdf"
        plot_r4_selection_figure(t5_rows, f3_path)
        outputs["F3"] = f3_path

    # 4. Bảng tổng hợp và Hình F4 (P4.2a: Bonus B2 biến thể UCT & B3 cải tiến UCT)
    b2_3x3 = results_dir / "p42a_b2_b3_3x3.csv"
    b2_9x9 = results_dir / "p42a_b2_b3_9x9.csv"
    if b2_3x3.exists() and b2_9x9.exists():
        b2_summary_path = results_dir / "p42a_b2_b3_summary.csv"
        b2_rows = summarize_b2_b3_table(b2_3x3, b2_9x9, out_csv=b2_summary_path)
        outputs["B2_B3"] = b2_summary_path

        f4_path = figures_dir / "F4.pdf"
        plot_b2_b3_variants_figure(b2_rows, f4_path)
        outputs["F4"] = f4_path

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh cho `python -m experiments.plots`."""
    parser = argparse.ArgumentParser(
        description="Sinh toàn bộ hình PDF vector (F1, F2, F3) và bảng T4, T5 từ CSV (P3.2b)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Thư mục chứa các file CSV kết quả thực nghiệm (mặc định: experiments/results).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Thư mục ghi các file hình PDF (mặc định: report/figures).",
    )
    parser.add_argument(
        "--t4-seeds",
        type=int,
        default=200,
        help="Số lượng seed để thống kê tần suất chọn nước trong Bảng T4 (mặc định: 200).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào chính khi chạy `python -m experiments.plots` hoặc `make figures`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    outputs = generate_all(
        results_dir=args.results_dir,
        figures_dir=args.out_dir,
        t4_seeds=args.t4_seeds,
    )
    for name, path in sorted(outputs.items()):
        print(f"[{name}] -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
