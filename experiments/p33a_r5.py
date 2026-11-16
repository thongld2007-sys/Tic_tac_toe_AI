"""Tổng hợp thực nghiệm R5 (so sánh L0–L4, tốc độ sims/s và thế bẫy E1) → Bảng T6 (P3.3a).

Chủ sở hữu: TV3.  Review: TV1.

Cách chạy từ thư mục gốc của repo:
    python -m experiments.run configs/p33a_r5_equal_iter.yaml --workers 8
    python -m experiments.run configs/p33a_r5_equal_time.yaml --workers 4
    python -m experiments.p33a_r5

Đầu vào:
  - `experiments/results/p33a_r5_equal_iter.csv`: đấu L0–L4 cùng 200 iteration trên Caro 9×9.
  - `experiments/results/p33a_r5_equal_time.csv`: đấu L0–L4 cùng 0.08 s/nước trên Caro 9×9.

Đầu ra (phục vụ Bảng T6 trong báo cáo):
  - `experiments/results/p33a_r5_summary_t6.csv`: Bảng T6 phần thực chiến — tỷ lệ thắng kèm khoảng
    tin cậy Wilson 95%, điểm trung bình kèm bootstrap CI 95%, kiểm định z hai tỷ lệ, hiệu điểm
    bootstrap CI 95% và tốc độ mô phỏng/giây thực tế trong ván (`in_game_sims_per_s` kèm CI 95%).
  - `experiments/results/p33a_r5_e1_t6.csv`: Bảng T6 phần thế bẫy E1 (R5.3) — tần suất chọn đúng
    nước tối ưu (kèm Wilson 95% CI) và thống kê nút gốc (`N`, `W`, `W/N`, `UCT`) của L0–L4 trên
    thế bẫy E1 3×3 và hai thế bẫy chiến thuật trên bàn Caro 9×9.
  - `experiments/results/p33a_r5_speed_t6.csv`: Bảng T6 phần tốc độ độc lập — `rollout_sims_per_s`
    và `mcts_sims_per_s` (kèm khoảng tin cậy bootstrap 95%) của L0–L4 trên bàn Caro 9×9 và 15×15.
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

from experiments.stats import (
    bootstrap_ci,
    bootstrap_diff_ci,
    summarize_matchup,
    two_proportion_ztest,
    wilson_ci,
)
from src.agents.base import Budget, RootStat
from src.agents.mcts import MCTSAgent
from src.agents.rollouts import measure_sims_per_second
from src.games.base import GameState, Move
from src.games.gomoku import Gomoku
from src.games.tictactoe import TicTacToe

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"

ROLLOUT_LEVELS: tuple[str, ...] = ("l0", "l1", "l2", "l3", "l4")

T6_SUMMARY_COLUMNS: tuple[str, ...] = (
    "mode",
    "category",
    "rollout_a",
    "rollout_b",
    "budget_spec",
    "games",
    "wins",
    "draws",
    "losses",
    "win_rate",
    "win_ci_low",
    "win_ci_high",
    "mean_score",
    "score_ci_low",
    "score_ci_high",
    "sims_per_move",
    "time_per_move_ms",
    "in_game_sims_per_s",
    "sims_per_s_ci_low",
    "sims_per_s_ci_high",
    "z_stat",
    "p_value",
    "diff_mean",
    "diff_ci_low",
    "diff_ci_high",
)

T6_E1_COLUMNS: tuple[str, ...] = (
    "position",
    "board",
    "rollout",
    "iterations",
    "c",
    "seeds",
    "optimal_picks",
    "optimal_rate",
    "opt_ci_low",
    "opt_ci_high",
    "mean_opt_visit_share",
    "mean_opt_N",
    "mean_opt_W",
    "mean_opt_q",
    "mean_opt_uct",
)

T6_SPEED_COLUMNS: tuple[str, ...] = (
    "board",
    "size",
    "rollout",
    "iterations",
    "seeds",
    "rollout_sims_per_s",
    "rollout_sps_ci_low",
    "rollout_sps_ci_high",
    "mcts_sims_per_s",
    "mcts_sps_ci_low",
    "mcts_sps_ci_high",
    "mean_time_ms",
    "mean_depth",
)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Đọc toàn bộ dòng của một file CSV dưới dạng `list[dict[str, str]]`."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Ghi danh sách từ điển ra file CSV theo đúng thứ tự cột `columns`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in columns})


def _paired_win_loss_ztest(wins: int, losses: int) -> tuple[float, float]:
    """Kiểm định z ghép cặp (McNemar/sign test) cho số ván thắng-thua: z = (W - L) / sqrt(W + L)."""
    decisive = wins + losses
    if decisive <= 0:
        return 0.0, 1.0
    z_stat = (wins - losses) / math.sqrt(decisive)
    p_val = 2.0 * (1.0 - NormalDist(0.0, 1.0).cdf(abs(z_stat)))
    return float(z_stat), max(0.0, min(1.0, float(p_val)))


def _extract_budget_label(params: Mapping[str, Any]) -> tuple[str, str]:
    """Trích xuất `(mode, budget_spec)` từ `a_params` trong CSV của runner."""
    budget = params.get("budget") or {}
    if "iterations" in budget and budget["iterations"] is not None:
        return "equal_iter", f"{int(budget['iterations'])}_iters"
    if "time_s" in budget and budget["time_s"] is not None:
        return "equal_time", f"{float(budget['time_s']):g}_s"
    return "unknown", "default"


def _compute_game_sims_metrics(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    """Tính số mô phỏng/nước, thời gian/nước (ms) và tốc độ `sims/s` kèm bootstrap CI 95%."""
    total_moves = sum(int(r["a_moves"]) for r in rows)
    total_nodes = sum(int(r["a_nodes"]) for r in rows)
    total_time = sum(float(r["a_time_s"]) for r in rows)
    total_sims = max(0, total_nodes - total_moves)

    sims_per_move = (total_sims / total_moves) if total_moves > 0 else 0.0
    time_per_move_ms = (1000.0 * total_time / total_moves) if total_moves > 0 else 0.0

    per_game_sps: list[float] = []
    for r in rows:
        t_g = float(r["a_time_s"])
        m_g = int(r["a_moves"])
        n_g = int(r["a_nodes"])
        s_g = max(0, n_g - m_g)
        if t_g > 0.0:
            per_game_sps.append(s_g / t_g)

    if per_game_sps:
        in_game_sps = sum(per_game_sps) / len(per_game_sps)
        sps_ci = bootstrap_ci(per_game_sps, seed=2026)
        sps_lo, sps_hi = sps_ci.low, sps_ci.high
    else:
        in_game_sps, sps_lo, sps_hi = 0.0, 0.0, 0.0

    return {
        "sims_per_move": sims_per_move,
        "time_per_move_ms": time_per_move_ms,
        "in_game_sims_per_s": in_game_sps,
        "sims_per_s_ci_low": sps_lo,
        "sims_per_s_ci_high": sps_hi,
    }


def summarize_r5_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Tổng hợp một file CSV thực nghiệm R5 (`equal_iter` hoặc `equal_time`) thành các dòng T6."""
    raw_rows = load_csv_rows(csv_path)
    by_matchup: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in raw_rows:
        by_matchup[int(r["matchup"])].append(r)

    vs_l0_groups: dict[str, list[dict[str, str]]] = {}
    h2h_groups: list[list[dict[str, str]]] = []

    for m_id in sorted(by_matchup.keys()):
        m_rows = by_matchup[m_id]
        a_params = json.loads(m_rows[0]["a_params"])
        b_params = json.loads(m_rows[0]["b_params"])
        pol_a = str(a_params.get("rollout", "l0")).lower()
        pol_b = str(b_params.get("rollout", "l0")).lower()

        if pol_b == "l0" and pol_a not in vs_l0_groups:
            vs_l0_groups[pol_a] = m_rows
        else:
            h2h_groups.append(m_rows)

    out_rows: list[dict[str, Any]] = []
    l0_rows = vs_l0_groups.get("l0", [])
    l0_scores = [float(r["score_a"]) for r in l0_rows]
    l0_wins = sum(r["winner"] == "a" for r in l0_rows)

    for pol in ROLLOUT_LEVELS:
        if pol not in vs_l0_groups:
            continue
        rows = vs_l0_groups[pol]
        a_params = json.loads(rows[0]["a_params"])
        mode, budget_spec = _extract_budget_label(a_params)
        scores = [float(r["score_a"]) for r in rows]
        stats = summarize_matchup(scores, seed=2026)
        n_games = int(stats["games"])
        wins = int(stats["wins"])
        draws = int(stats["draws"])
        losses = int(stats["losses"])
        sim_metrics = _compute_game_sims_metrics(rows)

        if pol != "l0" and l0_rows:
            z_stat, p_val = two_proportion_ztest(wins, n_games, l0_wins, len(l0_rows))
            diff_mean = (sum(scores) / len(scores)) - (sum(l0_scores) / len(l0_scores))
            diff_lo, diff_hi = bootstrap_diff_ci(scores, l0_scores, seed=2026, paired=False)
        else:
            z_stat, p_val = 0.0, 1.0
            diff_mean, diff_lo, diff_hi = 0.0, 0.0, 0.0

        out_rows.append(
            {
                "mode": mode,
                "category": "vs_l0",
                "rollout_a": pol,
                "rollout_b": "l0",
                "budget_spec": budget_spec,
                "games": n_games,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate": round(float(stats["win_rate"]), 4),
                "win_ci_low": round(float(stats["win_wilson_low"]), 4),
                "win_ci_high": round(float(stats["win_wilson_high"]), 4),
                "mean_score": round(float(stats["score_mean"]), 4),
                "score_ci_low": round(float(stats["score_boot_low"]), 4),
                "score_ci_high": round(float(stats["score_boot_high"]), 4),
                "sims_per_move": round(sim_metrics["sims_per_move"], 1),
                "time_per_move_ms": round(sim_metrics["time_per_move_ms"], 2),
                "in_game_sims_per_s": round(sim_metrics["in_game_sims_per_s"], 1),
                "sims_per_s_ci_low": round(sim_metrics["sims_per_s_ci_low"], 1),
                "sims_per_s_ci_high": round(sim_metrics["sims_per_s_ci_high"], 1),
                "z_stat": round(z_stat, 4),
                "p_value": round(p_val, 4),
                "diff_mean": round(diff_mean, 4),
                "diff_ci_low": round(diff_lo, 4),
                "diff_ci_high": round(diff_hi, 4),
            }
        )

    for rows in h2h_groups:
        a_params = json.loads(rows[0]["a_params"])
        b_params = json.loads(rows[0]["b_params"])
        pol_a = str(a_params.get("rollout", "l0")).lower()
        pol_b = str(b_params.get("rollout", "l0")).lower()
        mode, budget_spec = _extract_budget_label(a_params)

        scores_a = [float(r["score_a"]) for r in rows]
        scores_b = [1.0 - s for s in scores_a]
        stats = summarize_matchup(scores_a, seed=2026)
        n_games = int(stats["games"])
        wins = int(stats["wins"])
        draws = int(stats["draws"])
        losses = int(stats["losses"])
        sim_metrics = _compute_game_sims_metrics(rows)

        z_stat, p_val = _paired_win_loss_ztest(wins, losses)
        diff_mean = (sum(scores_a) / len(scores_a)) - (sum(scores_b) / len(scores_b))
        diff_lo, diff_hi = bootstrap_diff_ci(scores_a, scores_b, seed=2026, paired=True)

        out_rows.append(
            {
                "mode": mode,
                "category": "head_to_head",
                "rollout_a": pol_a,
                "rollout_b": pol_b,
                "budget_spec": budget_spec,
                "games": n_games,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate": round(float(stats["win_rate"]), 4),
                "win_ci_low": round(float(stats["win_wilson_low"]), 4),
                "win_ci_high": round(float(stats["win_wilson_high"]), 4),
                "mean_score": round(float(stats["score_mean"]), 4),
                "score_ci_low": round(float(stats["score_boot_low"]), 4),
                "score_ci_high": round(float(stats["score_boot_high"]), 4),
                "sims_per_move": round(sim_metrics["sims_per_move"], 1),
                "time_per_move_ms": round(sim_metrics["time_per_move_ms"], 2),
                "in_game_sims_per_s": round(sim_metrics["in_game_sims_per_s"], 1),
                "sims_per_s_ci_low": round(sim_metrics["sims_per_s_ci_low"], 1),
                "sims_per_s_ci_high": round(sim_metrics["sims_per_s_ci_high"], 1),
                "z_stat": round(z_stat, 4),
                "p_value": round(p_val, 4),
                "diff_mean": round(diff_mean, 4),
                "diff_ci_low": round(diff_lo, 4),
                "diff_ci_high": round(diff_hi, 4),
            }
        )

    return out_rows


def summarize_r5_table_t6(
    equal_iter_csv: Path,
    equal_time_csv: Path,
    out_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Kết hợp hai file CSV (`equal_iter` và `equal_time`) thành Bảng T6 tổng hợp."""
    rows: list[dict[str, Any]] = []
    if equal_iter_csv.exists():
        rows.extend(summarize_r5_csv(equal_iter_csv))
    if equal_time_csv.exists():
        rows.extend(summarize_r5_csv(equal_time_csv))
    if out_csv is not None:
        write_csv_rows(out_csv, T6_SUMMARY_COLUMNS, rows)
    return rows


def _build_e1_positions() -> list[tuple[str, str, GameState, set[Move], tuple[int, ...], float]]:
    """Khởi tạo các thế cờ bẫy E1 (3×3 và Caro 9×9) để đánh giá hành vi L0–L4 (R5.3)."""
    # 1. Thế bẫy E1 chuẩn trên Tic-Tac-Toe 3×3: X(0,1), O(0,0), X(2,0), O(1,1) -> X phải chặn (2, 2)
    e1_3x3 = TicTacToe()
    for mv in [(0, 1), (0, 0), (2, 0), (1, 1)]:
        e1_3x3.play(mv)

    # 2. Thế bẫy tương đương E1 trên Caro 9×9: O có 4 quân (4,2)-(4,5) bị chặn 1 đầu ở (4,1),
    # X có 3 quân cột 6 tại (2,6), (3,6), (5,6) -> X bắt buộc đi (4, 6) để chặn thua và phản công.
    e1_gmk9 = Gomoku(size=9)
    for mv in [(4, 1), (4, 2), (2, 6), (4, 3), (3, 6), (4, 4), (5, 6), (4, 5)]:
        e1_gmk9.play(mv)

    # 3. Thế bẫy chuỗi 3 hở hai đầu trên Caro 9×9: O có (4,3)-(4,4)-(4,5) hở hai đầu,
    # X có (2,2), (3,2), (6,6) -> X cần chặn sớm ở (4, 2) hoặc (4, 6).
    open3_gmk9 = Gomoku(size=9)
    for mv in [(2, 2), (4, 3), (3, 2), (4, 4), (6, 6), (4, 5)]:
        open3_gmk9.play(mv)

    return [
        ("E1_3x3", "3x3", e1_3x3, {(2, 2)}, (10, 15, 25, 50, 100), 1.4),
        ("E1_gomoku9_four", "9x9", e1_gmk9, {(4, 6)}, (35, 60, 100), 1.4),
        ("Gomoku9_open_three", "9x9", open3_gmk9, {(4, 2), (4, 6)}, (60, 120), 1.4),
    ]


def evaluate_e1_behavior(
    out_csv: Path | None = None,
    *,
    num_seeds: int = 200,
) -> list[dict[str, Any]]:
    """Đánh giá tần suất chọn nước tối ưu và thống kê nút gốc của L0–L4 trên thế bẫy E1 (R5.3)."""
    if num_seeds < 1:
        raise ValueError(f"num_seeds phải >= 1, nhận được {num_seeds}")

    rows: list[dict[str, Any]] = []
    for pos_name, board_name, state, opt_moves, budgets, c_val in _build_e1_positions():
        use_cand = isinstance(state, Gomoku)
        for iters in budgets:
            for pol in ROLLOUT_LEVELS:
                opt_picks = 0
                sum_share = 0.0
                sum_n = 0.0
                sum_w = 0.0
                sum_q = 0.0
                sum_uct = 0.0

                for seed in range(num_seeds):
                    agent = MCTSAgent(
                        c=c_val,
                        selection="robust",
                        rollout=pol,
                        use_candidates=use_cand,
                        candidate_radius=2,
                        reuse_tree=False,
                        seed=seed,
                    )
                    chosen = agent.select_move(state, Budget(iterations=iters))
                    if chosen in opt_moves:
                        opt_picks += 1

                    root_stats: list[RootStat] = agent.last_stats.get("root", [])
                    opt_items = [
                        item
                        for item in root_stats
                        if (int(item["move"][0]), int(item["move"][1])) in opt_moves
                    ]
                    if opt_items:
                        best_opt = max(opt_items, key=lambda x: (int(x["N"]), float(x["q"])))
                        opt_n_total = sum(int(x["N"]) for x in opt_items)
                        sum_share += opt_n_total / iters
                        sum_n += float(best_opt["N"])
                        sum_w += float(best_opt["W"])
                        sum_q += float(best_opt["q"])
                        sum_uct += float(best_opt["uct"])

                opt_rate = opt_picks / num_seeds
                ci_low, ci_high = wilson_ci(opt_picks, num_seeds)
                rows.append(
                    {
                        "position": pos_name,
                        "board": board_name,
                        "rollout": pol,
                        "iterations": iters,
                        "c": c_val,
                        "seeds": num_seeds,
                        "optimal_picks": opt_picks,
                        "optimal_rate": round(opt_rate, 4),
                        "opt_ci_low": round(ci_low, 4),
                        "opt_ci_high": round(ci_high, 4),
                        "mean_opt_visit_share": round(sum_share / num_seeds, 4),
                        "mean_opt_N": round(sum_n / num_seeds, 2),
                        "mean_opt_W": round(sum_w / num_seeds, 2),
                        "mean_opt_q": round(sum_q / num_seeds, 4),
                        "mean_opt_uct": round(sum_uct / num_seeds, 4),
                    }
                )

    if out_csv is not None:
        write_csv_rows(out_csv, T6_E1_COLUMNS, rows)
    return rows


def benchmark_rollout_speeds(
    out_csv: Path | None = None,
    *,
    iterations: int = 400,
    num_seeds: int = 10,
) -> list[dict[str, Any]]:
    """Đo tốc độ `rollout_sims_per_s` và `mcts_sims_per_s` của L0–L4 trên bàn 9×9 và 15×15."""
    if iterations < 1:
        raise ValueError(f"iterations phải >= 1, nhận được {iterations}")
    if num_seeds < 1:
        raise ValueError(f"num_seeds phải >= 1, nhận được {num_seeds}")

    rows: list[dict[str, Any]] = []
    for size in (9, 15):
        state = Gomoku(size=size)
        mid = size // 2
        for mv in [
            (mid, mid),
            (mid, mid + 1),
            (mid + 1, mid),
            (mid - 1, mid + 1),
            (mid, mid - 1),
            (mid - 1, mid),
        ]:
            state.play(mv)

        board_label = f"{size}x{size}"
        for pol in ROLLOUT_LEVELS:
            rollout_sps_samples: list[float] = []
            mcts_sps_samples: list[float] = []
            time_ms_samples: list[float] = []
            depth_samples: list[float] = []

            for s in range(num_seeds):
                metrics = measure_sims_per_second(
                    state,
                    rollout=pol,
                    iterations=iterations,
                    use_candidates=True,
                    candidate_radius=2,
                    reuse_tree=False,
                    seed=100 + s,
                )
                rollout_sps_samples.append(metrics["rollout_sims_per_s"])
                mcts_sps_samples.append(metrics["mcts_sims_per_s"])
                time_ms_samples.append(metrics["mcts_time_s"] * 1000.0)
                depth_samples.append(metrics["depth"])

            r_ci = bootstrap_ci(rollout_sps_samples, seed=2026)
            m_ci = bootstrap_ci(mcts_sps_samples, seed=2026)
            rows.append(
                {
                    "board": board_label,
                    "size": size,
                    "rollout": pol,
                    "iterations": iterations,
                    "seeds": num_seeds,
                    "rollout_sims_per_s": round(sum(rollout_sps_samples) / num_seeds, 1),
                    "rollout_sps_ci_low": round(r_ci.low, 1),
                    "rollout_sps_ci_high": round(r_ci.high, 1),
                    "mcts_sims_per_s": round(sum(mcts_sps_samples) / num_seeds, 1),
                    "mcts_sps_ci_low": round(m_ci.low, 1),
                    "mcts_sps_ci_high": round(m_ci.high, 1),
                    "mean_time_ms": round(sum(time_ms_samples) / num_seeds, 2),
                    "mean_depth": round(sum(depth_samples) / num_seeds, 2),
                }
            )

    if out_csv is not None:
        write_csv_rows(out_csv, T6_SPEED_COLUMNS, rows)
    return rows


def generate_all_p33a(
    *,
    results_dir: Path = RESULTS_DIR,
    e1_seeds: int = 200,
    speed_iterations: int = 400,
    speed_seeds: int = 10,
) -> dict[str, Path]:
    """Sinh toàn bộ các bảng CSV phục vụ Bảng T6 (P3.3a)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    eq_iter_csv = results_dir / "p33a_r5_equal_iter.csv"
    eq_time_csv = results_dir / "p33a_r5_equal_time.csv"
    if eq_iter_csv.exists() or eq_time_csv.exists():
        t6_summary_path = results_dir / "p33a_r5_summary_t6.csv"
        summarize_r5_table_t6(eq_iter_csv, eq_time_csv, out_csv=t6_summary_path)
        outputs["T6_summary"] = t6_summary_path

    t6_e1_path = results_dir / "p33a_r5_e1_t6.csv"
    evaluate_e1_behavior(out_csv=t6_e1_path, num_seeds=e1_seeds)
    outputs["T6_e1"] = t6_e1_path

    t6_speed_path = results_dir / "p33a_r5_speed_t6.csv"
    benchmark_rollout_speeds(
        out_csv=t6_speed_path,
        iterations=speed_iterations,
        num_seeds=speed_seeds,
    )
    outputs["T6_speed"] = t6_speed_path

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh cho `python -m experiments.p33a_r5`."""
    parser = argparse.ArgumentParser(
        description=(
            "Tổng hợp kết quả R5 (so sánh L0–L4 cùng iteration, cùng thời gian, E1 và sims/s) "
            "-> Bảng T6 (P3.3a)."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Thư mục chứa các file CSV thực nghiệm (mặc định: experiments/results).",
    )
    parser.add_argument(
        "--e1-seeds",
        type=int,
        default=200,
        help="Số lượng seed đánh giá trên mỗi thế bẫy E1 (mặc định: 200).",
    )
    parser.add_argument(
        "--speed-iterations",
        type=int,
        default=400,
        help="Số iteration mỗi lần đo tốc độ sims/s (mặc định: 400).",
    )
    parser.add_argument(
        "--speed-seeds",
        type=int,
        default=10,
        help="Số seed lặp lại khi đo tốc độ sims/s (mặc định: 10).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào chính khi chạy `python -m experiments.p33a_r5`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    outputs = generate_all_p33a(
        results_dir=args.results_dir,
        e1_seeds=args.e1_seeds,
        speed_iterations=args.speed_iterations,
        speed_seeds=args.speed_seeds,
    )
    for name, path in sorted(outputs.items()):
        print(f"[{name}] -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
