r"""Phân tích thực nghiệm và đo tốc độ cho P4.3a — Bonus B4 (Nước bắt buộc mở rộng & Root Parallel).

Chủ sở hữu: TV3.  Review: TV1.

Tiêu chí nghiệm thu P4.3a:
  - Có số mô phỏng/giây (`sims_per_s`) và điểm (`score` / `win_rate`) trước và sau khi áp dụng:
      1. **Nước bắt buộc mở rộng (Extended Forced Moves)**: `l2` -> `l2_b4`, `l3` -> `l3_b4`,
         `l4` -> `l4_b4` (ưu tiên 4 tầng: thắng 1 bước -> chặn thắng 1 bước / dừng sớm khi bị
         dọa đôi -> tạo Bốn hở hai đầu / nước đôi 4-4 dừng sớm -> chặn đối thủ tạo Bốn hở hai đầu).
      2. **Song song hóa theo gốc (Root Parallelization)**: `RootParallelMCTSAgent` với
         $K = 1$ (trước) so với $K \in \{2, 4\}$ (sau) cây MCTS độc lập chạy song song từ gốc
         rồi gộp thống kê $(N, W)$ tại các nút con của gốc.

Đầu ra:
  - `experiments/results/p43a_b4_summary.csv`: Bảng tổng hợp điểm số / tỷ lệ thắng Trước vs Sau
    (kèm khoảng tin cậy Wilson 95%, bootstrap 95%, kiểm định $z$ và tốc độ mở nút trong ván).
  - `experiments/results/p43a_b4_speed.csv`: Bảng đo tốc độ mô phỏng (`rollout_sims_per_s`,
    `mcts_sims_per_s`, hệ số tăng tốc `speed_ratio`) và tỷ lệ giải đúng thế cờ chiến thuật
    `Gomoku9_open_three` Trước vs Sau trên bàn 9×9 và 15×15.

Cách chạy:
    python -m experiments.p43a_b4
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.agents.base import Agent, Budget
from src.agents.mcts import MCTSAgent, resolve_rollout
from src.agents.rollouts import RootParallelMCTSAgent
from src.games.base import PLAYER_O, PLAYER_X
from src.games.gomoku import Gomoku

DEFAULT_CSV_IN = Path("experiments/results/p43a_b4_9x9.csv")
DEFAULT_SUMMARY_OUT = Path("experiments/results/p43a_b4_summary.csv")
DEFAULT_SPEED_OUT = Path("experiments/results/p43a_b4_speed.csv")

SUMMARY_FIELDS: tuple[str, ...] = (
    "category",
    "comparison",
    "agent_after",
    "agent_before",
    "budget",
    "games",
    "wins_after",
    "draws",
    "losses_after",
    "win_rate_after",
    "wilson_ci_low",
    "wilson_ci_high",
    "score_after",
    "score_before",
    "score_diff",
    "boot_ci_low",
    "boot_ci_high",
    "z_stat",
    "p_value",
    "significant_005",
    "new_nodes_per_s_after",
    "new_nodes_per_s_before",
    "speed_ratio",
    "ms_per_move_after",
    "ms_per_move_before",
)

SPEED_FIELDS: tuple[str, ...] = (
    "category",
    "board_size",
    "variant_before",
    "variant_after",
    "workers",
    "mode",
    "rollout_sims_per_s_before",
    "rollout_sims_per_s_after",
    "rollout_speed_ratio",
    "mcts_sims_per_s_before",
    "mcts_sims_per_s_after",
    "mcts_speed_ratio",
    "ms_per_move_before",
    "ms_per_move_after",
    "open_three_solve_before",
    "open_three_solve_after",
    "open_three_solve_diff",
)


def wilson_ci_95(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Khoảng tin cậy Wilson 95% cho tỷ lệ nhị thức `successes / total`."""
    if total <= 0:
        return 0.0, 0.0
    n = float(total)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 2026,
) -> tuple[float, float]:
    """Khoảng tin cậy bootstrap 95% (phân vị 2.5% – 97.5%) cho trung bình mẫu."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = max(0, int(0.025 * n_boot))
    hi_idx = min(n_boot - 1, int(0.975 * n_boot))
    return means[lo_idx], means[hi_idx]


def one_sample_score_z_test(scores: Sequence[float], mu0: float = 0.5) -> tuple[float, float]:
    """Kiểm định z hai phía cho điểm trung bình `scores` so với mốc cân bằng `mu0 = 0.5`."""
    n = len(scores)
    if n <= 1:
        return 0.0, 1.0
    mean_val = sum(scores) / n
    var_val = sum((x - mean_val) ** 2 for x in scores) / (n - 1)
    if var_val <= 1e-12:
        if abs(mean_val - mu0) <= 1e-12:
            return 0.0, 1.0
        return (math.copysign(99.0, mean_val - mu0), 0.0)
    se = math.sqrt(var_val / n)
    z = (mean_val - mu0) / se
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p


def two_sample_score_z_test(
    scores_after: Sequence[float],
    scores_before: Sequence[float],
) -> tuple[float, float]:
    """Kiểm định z hai phía so sánh trung bình hai mẫu điểm `scores_after` và `scores_before`."""
    n1 = len(scores_after)
    n2 = len(scores_before)
    if n1 <= 1 or n2 <= 1:
        return 0.0, 1.0
    m1 = sum(scores_after) / n1
    m2 = sum(scores_before) / n2
    v1 = sum((x - m1) ** 2 for x in scores_after) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in scores_before) / (n2 - 1)
    se2 = (v1 / n1) + (v2 / n2)
    if se2 <= 1e-12:
        if abs(m1 - m2) <= 1e-12:
            return 0.0, 1.0
        return (math.copysign(99.0, m1 - m2), 0.0)
    z = (m1 - m2) / math.sqrt(se2)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Đọc toàn bộ các dòng từ file CSV kết quả ván đấu."""
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _extract_rollout_tag(agent_label: str) -> str:
    """Trích xuất tên chính sách rollout từ nhãn agent của runner."""
    for tag in ("l2_b4", "l3_b4", "l4_b4", "l0", "l1", "l2", "l3", "l4"):
        if f"rollout={tag}" in agent_label:
            return tag
    return agent_label


def _compute_side_speed(rows: Sequence[Mapping[str, Any]], prefix: str) -> tuple[float, float]:
    """Trả về `(new_nodes_per_s, ms_per_move)` của bên `prefix` (`'a'` hoặc `'b'`)."""
    total_moves = sum(int(r[f"{prefix}_moves"]) for r in rows)
    total_nodes = sum(int(r[f"{prefix}_nodes"]) for r in rows)
    total_time = sum(float(r[f"{prefix}_time_s"]) for r in rows)
    new_nodes = max(0, total_nodes - total_moves)
    nodes_per_s = (new_nodes / total_time) if total_time > 0.0 else 0.0
    ms_per_move = (total_time * 1000.0 / total_moves) if total_moves > 0 else 0.0
    return nodes_per_s, ms_per_move


def _play_head_to_head_agents(
    agent_after: Agent,
    budget_after: Budget,
    agent_before: Agent,
    budget_before: Budget,
    *,
    games: int = 40,
    size: int = 9,
    opening_plies: int = 2,
    seed: int = 2026,
) -> dict[str, Any]:
    """Chạy `games` ván đối đầu trực tiếp (đổi bên đi trước, khai cuộc ngẫu nhiên 2 ply)."""
    wins_after = 0
    draws = 0
    losses_after = 0
    scores_after: list[float] = []
    iters_after = 0
    time_after = 0.0
    moves_after = 0
    iters_before = 0
    time_before = 0.0
    moves_before = 0

    for g_idx in range(games):
        pair_idx = g_idx // 2
        after_is_x = (g_idx % 2) == 0
        open_rng = random.Random(seed + 10007 * pair_idx)
        state = Gomoku(size=size, win_length=5, block_two_ends=False)
        for _ in range(opening_plies):
            if state.is_terminal():
                break
            state.play(open_rng.choice(state.legal_moves()))

        while not state.is_terminal():
            cur = state.current_player()
            is_after_turn = (cur == PLAYER_X and after_is_x) or (cur == PLAYER_O and not after_is_x)
            if is_after_turn:
                mv = agent_after.select_move(state, budget_after)
                st = agent_after.last_stats
                iters_after += int(st.get("iterations", max(0, int(st.get("nodes", 1)) - 1)))
                time_after += float(st.get("time_s", 0.0))
                moves_after += 1
            else:
                mv = agent_before.select_move(state, budget_before)
                st = agent_before.last_stats
                iters_before += int(st.get("iterations", max(0, int(st.get("nodes", 1)) - 1)))
                time_before += float(st.get("time_s", 0.0))
                moves_before += 1
            state.play(mv)

        winner = state.winner()
        after_sign = PLAYER_X if after_is_x else PLAYER_O
        if winner == 0 or winner is None:
            draws += 1
            scores_after.append(0.5)
        elif winner == after_sign:
            wins_after += 1
            scores_after.append(1.0)
        else:
            losses_after += 1
            scores_after.append(0.0)

    sims_s_after = (iters_after / time_after) if time_after > 0.0 else 0.0
    sims_s_before = (iters_before / time_before) if time_before > 0.0 else 0.0
    ms_after = (time_after * 1000.0 / moves_after) if moves_after > 0 else 0.0
    ms_before = (time_before * 1000.0 / moves_before) if moves_before > 0 else 0.0
    return {
        "games": games,
        "wins_after": wins_after,
        "draws": draws,
        "losses_after": losses_after,
        "scores_after": scores_after,
        "sims_s_after": sims_s_after,
        "sims_s_before": sims_s_before,
        "ms_after": ms_after,
        "ms_before": ms_before,
    }


def build_summary_rows(
    csv_rows: Sequence[Mapping[str, Any]],
    *,
    root_parallel_games: int = 40,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Tạo bảng tổng hợp `p43a_b4_summary.csv` từ CSV ván đấu và các cặp đấu Root Parallel."""
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in csv_rows:
        tag_a = _extract_rollout_tag(str(row["agent_a"]))
        tag_b = _extract_rollout_tag(str(row["agent_b"]))
        has_time = "time_s=" in str(row["agent_a"])
        budget_tag = "time_s=0.08" if has_time else "iterations=200"
        key = (tag_a, tag_b, budget_tag)
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []

    # 1 & 2. Đối đầu trực tiếp Trước vs Sau (cùng iteration và cùng thời gian)
    for budget_tag, cat_name in (
        ("iterations=200", "extended_forced_eq_iter"),
        ("time_s=0.08", "extended_forced_eq_time"),
    ):
        for base_tag in ("l2", "l3", "l4"):
            after_tag = f"{base_tag}_b4"
            key = (after_tag, base_tag, budget_tag)
            rows = grouped.get(key, [])
            if not rows:
                continue
            n = len(rows)
            wins_a = sum(1 for r in rows if float(r["score_a"]) == 1.0)
            draws = sum(1 for r in rows if float(r["score_a"]) == 0.5)
            losses_a = n - wins_a - draws
            scores_a = [float(r["score_a"]) for r in rows]
            mean_after = sum(scores_a) / n
            mean_before = 1.0 - mean_after
            w_lo, w_hi = wilson_ci_95(wins_a, n)
            b_lo, b_hi = bootstrap_mean_ci(scores_a, seed=seed)
            z_val, p_val = one_sample_score_z_test(scores_a, mu0=0.5)
            nps_after, ms_after = _compute_side_speed(rows, "a")
            nps_before, ms_before = _compute_side_speed(rows, "b")
            ratio = (nps_after / nps_before) if nps_before > 0.0 else 0.0

            summary_rows.append(
                {
                    "category": cat_name,
                    "comparison": f"{after_tag}_vs_{base_tag}",
                    "agent_after": after_tag,
                    "agent_before": base_tag,
                    "budget": budget_tag,
                    "games": n,
                    "wins_after": wins_a,
                    "draws": draws,
                    "losses_after": losses_a,
                    "win_rate_after": round(wins_a / n, 4),
                    "wilson_ci_low": round(w_lo, 4),
                    "wilson_ci_high": round(w_hi, 4),
                    "score_after": round(mean_after, 4),
                    "score_before": round(mean_before, 4),
                    "score_diff": round(mean_after - mean_before, 4),
                    "boot_ci_low": round(b_lo, 4),
                    "boot_ci_high": round(b_hi, 4),
                    "z_stat": round(z_val, 4),
                    "p_value": round(p_val, 6),
                    "significant_005": p_val < 0.05,
                    "new_nodes_per_s_after": round(nps_after, 1),
                    "new_nodes_per_s_before": round(nps_before, 1),
                    "speed_ratio": round(ratio, 4),
                    "ms_per_move_after": round(ms_after, 2),
                    "ms_per_move_before": round(ms_before, 2),
                }
            )

    # 3. Đấu với mốc cơ sở L0 (so sánh điểm đạt được trước vs sau khi thêm nước bắt buộc mở rộng)
    for base_tag in ("l2", "l3", "l4"):
        after_tag = f"{base_tag}_b4"
        rows_after = grouped.get((after_tag, "l0", "iterations=200"), [])
        rows_before = grouped.get((base_tag, "l0", "iterations=200"), [])
        if not rows_after or not rows_before:
            continue
        n = len(rows_after)
        wins_a = sum(1 for r in rows_after if float(r["score_a"]) == 1.0)
        draws = sum(1 for r in rows_after if float(r["score_a"]) == 0.5)
        losses_a = n - wins_a - draws
        scores_after = [float(r["score_a"]) for r in rows_after]
        scores_before = [float(r["score_a"]) for r in rows_before]
        mean_after = sum(scores_after) / len(scores_after)
        mean_before = sum(scores_before) / len(scores_before)
        w_lo, w_hi = wilson_ci_95(wins_a, n)
        b_lo, b_hi = bootstrap_mean_ci(scores_after, seed=seed)
        z_val, p_val = two_sample_score_z_test(scores_after, scores_before)
        nps_after, ms_after = _compute_side_speed(rows_after, "a")
        nps_before, ms_before = _compute_side_speed(rows_before, "a")
        ratio = (nps_after / nps_before) if nps_before > 0.0 else 0.0

        summary_rows.append(
            {
                "category": "extended_forced_vs_l0",
                "comparison": f"{after_tag}_vs_l0_ref_{base_tag}",
                "agent_after": f"{after_tag}_vs_l0",
                "agent_before": f"{base_tag}_vs_l0",
                "budget": "iterations=200",
                "games": n,
                "wins_after": wins_a,
                "draws": draws,
                "losses_after": losses_a,
                "win_rate_after": round(wins_a / n, 4),
                "wilson_ci_low": round(w_lo, 4),
                "wilson_ci_high": round(w_hi, 4),
                "score_after": round(mean_after, 4),
                "score_before": round(mean_before, 4),
                "score_diff": round(mean_after - mean_before, 4),
                "boot_ci_low": round(b_lo, 4),
                "boot_ci_high": round(b_hi, 4),
                "z_stat": round(z_val, 4),
                "p_value": round(p_val, 6),
                "significant_005": p_val < 0.05,
                "new_nodes_per_s_after": round(nps_after, 1),
                "new_nodes_per_s_before": round(nps_before, 1),
                "speed_ratio": round(ratio, 4),
                "ms_per_move_after": round(ms_after, 2),
                "ms_per_move_before": round(ms_before, 2),
            }
        )

    # 4. Song song hóa theo gốc (RootParallelMCTSAgent): K=2, 4 (sau) đấu K=1 (trước)
    if root_parallel_games > 0:
        for workers_k in (2, 4):
            # 4a. Cùng ngân sách thời gian thực (time_s = 0.04 s/nước, mỗi worker chạy hết 0.04 s)
            ag_after_t = RootParallelMCTSAgent(
                num_workers=workers_k,
                rollout="l2_b4",
                c=1.0,
                use_candidates=True,
                candidate_radius=2,
                split_iterations=False,
                executor="process",
                seed=seed + 100 * workers_k,
            )
            ag_before_t = RootParallelMCTSAgent(
                num_workers=1,
                rollout="l2_b4",
                c=1.0,
                use_candidates=True,
                candidate_radius=2,
                split_iterations=False,
                executor="process",
                seed=seed + 100 * workers_k + 1,
            )
            res_t = _play_head_to_head_agents(
                ag_after_t,
                Budget(time_s=0.04),
                ag_before_t,
                Budget(time_s=0.04),
                games=root_parallel_games,
                seed=seed + 500 * workers_k,
            )
            n_t = int(res_t["games"])
            scores_t = list(res_t["scores_after"])
            mean_after_t = sum(scores_t) / n_t
            mean_before_t = 1.0 - mean_after_t
            w_lo_t, w_hi_t = wilson_ci_95(int(res_t["wins_after"]), n_t)
            b_lo_t, b_hi_t = bootstrap_mean_ci(scores_t, seed=seed)
            z_t, p_t = one_sample_score_z_test(scores_t, mu0=0.5)
            s_after_t = float(res_t["sims_s_after"])
            s_before_t = float(res_t["sims_s_before"])
            summary_rows.append(
                {
                    "category": "root_parallel_eq_time",
                    "comparison": f"rootpar_k{workers_k}_vs_k1_time004",
                    "agent_after": f"rootpar_k{workers_k}_l2_b4",
                    "agent_before": "rootpar_k1_l2_b4",
                    "budget": "time_s=0.04",
                    "games": n_t,
                    "wins_after": int(res_t["wins_after"]),
                    "draws": int(res_t["draws"]),
                    "losses_after": int(res_t["losses_after"]),
                    "win_rate_after": round(int(res_t["wins_after"]) / n_t, 4),
                    "wilson_ci_low": round(w_lo_t, 4),
                    "wilson_ci_high": round(w_hi_t, 4),
                    "score_after": round(mean_after_t, 4),
                    "score_before": round(mean_before_t, 4),
                    "score_diff": round(mean_after_t - mean_before_t, 4),
                    "boot_ci_low": round(b_lo_t, 4),
                    "boot_ci_high": round(b_hi_t, 4),
                    "z_stat": round(z_t, 4),
                    "p_value": round(p_t, 6),
                    "significant_005": p_t < 0.05,
                    "new_nodes_per_s_after": round(s_after_t, 1),
                    "new_nodes_per_s_before": round(s_before_t, 1),
                    "speed_ratio": round(s_after_t / s_before_t, 4) if s_before_t > 0 else 0.0,
                    "ms_per_move_after": round(float(res_t["ms_after"]), 2),
                    "ms_per_move_before": round(float(res_t["ms_before"]), 2),
                }
            )

            # 4b. Cùng số lượt mô phỏng/worker (120 iter/worker vs K=1 chạy 120 iter)
            ag_after_i = RootParallelMCTSAgent(
                num_workers=workers_k,
                rollout="l2_b4",
                c=1.0,
                use_candidates=True,
                candidate_radius=2,
                split_iterations=False,
                executor="process",
                seed=seed + 200 * workers_k,
            )
            ag_before_i = RootParallelMCTSAgent(
                num_workers=1,
                rollout="l2_b4",
                c=1.0,
                use_candidates=True,
                candidate_radius=2,
                split_iterations=False,
                executor="process",
                seed=seed + 200 * workers_k + 1,
            )
            res_i = _play_head_to_head_agents(
                ag_after_i,
                Budget(iterations=120),
                ag_before_i,
                Budget(iterations=120),
                games=root_parallel_games,
                seed=seed + 700 * workers_k,
            )
            n_i = int(res_i["games"])
            scores_i = list(res_i["scores_after"])
            mean_after_i = sum(scores_i) / n_i
            mean_before_i = 1.0 - mean_after_i
            w_lo_i, w_hi_i = wilson_ci_95(int(res_i["wins_after"]), n_i)
            b_lo_i, b_hi_i = bootstrap_mean_ci(scores_i, seed=seed)
            z_i, p_i = one_sample_score_z_test(scores_i, mu0=0.5)
            s_after_i = float(res_i["sims_s_after"])
            s_before_i = float(res_i["sims_s_before"])
            summary_rows.append(
                {
                    "category": "root_parallel_per_worker_iter",
                    "comparison": f"rootpar_k{workers_k}_vs_k1_iter120",
                    "agent_after": f"rootpar_k{workers_k}_l2_b4",
                    "agent_before": "rootpar_k1_l2_b4",
                    "budget": "iterations=120/worker",
                    "games": n_i,
                    "wins_after": int(res_i["wins_after"]),
                    "draws": int(res_i["draws"]),
                    "losses_after": int(res_i["losses_after"]),
                    "win_rate_after": round(int(res_i["wins_after"]) / n_i, 4),
                    "wilson_ci_low": round(w_lo_i, 4),
                    "wilson_ci_high": round(w_hi_i, 4),
                    "score_after": round(mean_after_i, 4),
                    "score_before": round(mean_before_i, 4),
                    "score_diff": round(mean_after_i - mean_before_i, 4),
                    "boot_ci_low": round(b_lo_i, 4),
                    "boot_ci_high": round(b_hi_i, 4),
                    "z_stat": round(z_i, 4),
                    "p_value": round(p_i, 6),
                    "significant_005": p_i < 0.05,
                    "new_nodes_per_s_after": round(s_after_i, 1),
                    "new_nodes_per_s_before": round(s_before_i, 1),
                    "speed_ratio": round(s_after_i / s_before_i, 4) if s_before_i > 0 else 0.0,
                    "ms_per_move_after": round(float(res_i["ms_after"]), 2),
                    "ms_per_move_before": round(float(res_i["ms_before"]), 2),
                }
            )

    return summary_rows


def _make_midgame_state(size: int) -> Gomoku:
    """Khởi tạo thế cờ trung cuộc chuẩn (6 quân quanh tâm) trên bàn `size × size`."""
    c = size // 2
    state = Gomoku(size=size, win_length=5, block_two_ends=False)
    for mv in (
        (c, c),
        (c - 1, c - 1),
        (c, c - 1),
        (c - 1, c),
        (c + 1, c + 1),
        (c - 1, c + 1),
    ):
        state.play(mv)
    return state


def _make_open_three_trap(size: int) -> tuple[Gomoku, set[tuple[int, int]]]:
    """Thế cờ bẫy Ba hở hai đầu chuẩn `Gomoku_open_three` (tương ứng P3.3a): X phải chặn 2 đầu."""
    c = size // 2
    state = Gomoku(size=size, win_length=5, block_two_ends=False)
    for mv in (
        (c - 2, c - 2),
        (c, c - 1),
        (c - 1, c - 2),
        (c, c),
        (c + 2, c + 2),
        (c, c + 1),
    ):
        state.play(mv)
    return state, {(c, c - 2), (c, c + 2)}


def _bench_rollout_speed(state: Gomoku, rollout_name: str, num_rollouts: int, seed: int) -> float:
    """Đo `rollout_sims_per_s` thuần túy của chính sách `rollout_name`."""
    _, fn = resolve_rollout(rollout_name)
    rng = random.Random(seed)
    depth = 8 if rollout_name in ("l4", "l4_b4") else None
    t0 = time.perf_counter()
    if depth is not None:
        for _ in range(num_rollouts):
            fn(state, rng, max_moves=depth)  # type: ignore[call-arg]
    else:
        for _ in range(num_rollouts):
            fn(state, rng)
    elapsed = time.perf_counter() - t0
    return (num_rollouts / elapsed) if elapsed > 0.0 else 0.0


def _bench_mcts_speed(
    state: Gomoku,
    rollout_name: str,
    *,
    trials: int,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Đo `(mcts_sims_per_s, ms_per_move)` của MCTSAgent đơn luồng."""
    depth = 8 if rollout_name in ("l4", "l4_b4") else None
    total_iters = 0
    t0 = time.perf_counter()
    for t in range(trials):
        ag = MCTSAgent(
            c=1.0,
            rollout=rollout_name,
            rollout_depth=depth,
            use_candidates=True,
            candidate_radius=2,
            reuse_tree=False,
            seed=seed + t,
        )
        ag.select_move(state, Budget(iterations=iterations))
        total_iters += iterations
    elapsed = time.perf_counter() - t0
    sims_s = (total_iters / elapsed) if elapsed > 0.0 else 0.0
    ms_move = (elapsed * 1000.0 / trials) if trials > 0 else 0.0
    return sims_s, ms_move


def _bench_trap_solve_rate(
    trap_state: Gomoku,
    optimal_moves: set[tuple[int, int]],
    rollout_name: str,
    *,
    trials: int,
    iterations: int,
    seed: int,
) -> float:
    """Đo tỷ lệ chọn đúng nước chặn Ba hở hai đầu của MCTSAgent qua `trials` hạt giống."""
    depth = 8 if rollout_name in ("l4", "l4_b4") else None
    solved = 0
    for t in range(trials):
        ag = MCTSAgent(
            c=1.0,
            rollout=rollout_name,
            rollout_depth=depth,
            use_candidates=True,
            candidate_radius=2,
            reuse_tree=False,
            seed=seed + 1000 + t,
        )
        mv = ag.select_move(trap_state, Budget(iterations=iterations))
        if mv in optimal_moves:
            solved += 1
    return solved / trials if trials > 0 else 0.0


def build_speed_rows(
    *,
    num_rollouts: int = 500,
    mcts_trials: int = 8,
    mcts_iterations: int = 200,
    trap_trials: int = 40,
    trap_iterations: int = 60,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Tạo bảng đo tốc độ mô phỏng (`sims_per_s`) và tỷ lệ giải bẫy Trước vs Sau."""
    rows: list[dict[str, Any]] = []

    # Phần 1: Nước bắt buộc mở rộng (Extended Forced Moves) trên bàn 9×9 và 15×15
    for size in (9, 15):
        mid_state = _make_midgame_state(size)
        trap_state, opt_moves = _make_open_three_trap(size)
        for base_tag in ("l2", "l3", "l4"):
            after_tag = f"{base_tag}_b4"
            r_before = _bench_rollout_speed(mid_state, base_tag, num_rollouts, seed)
            r_after = _bench_rollout_speed(mid_state, after_tag, num_rollouts, seed)
            m_before, ms_before = _bench_mcts_speed(
                mid_state,
                base_tag,
                trials=mcts_trials,
                iterations=mcts_iterations,
                seed=seed,
            )
            m_after, ms_after = _bench_mcts_speed(
                mid_state,
                after_tag,
                trials=mcts_trials,
                iterations=mcts_iterations,
                seed=seed,
            )
            solve_before = _bench_trap_solve_rate(
                trap_state,
                opt_moves,
                base_tag,
                trials=trap_trials,
                iterations=trap_iterations,
                seed=seed,
            )
            solve_after = _bench_trap_solve_rate(
                trap_state,
                opt_moves,
                after_tag,
                trials=trap_trials,
                iterations=trap_iterations,
                seed=seed,
            )
            rows.append(
                {
                    "category": "extended_forced_moves",
                    "board_size": f"{size}x{size}",
                    "variant_before": base_tag,
                    "variant_after": after_tag,
                    "workers": 1,
                    "mode": "single_tree",
                    "rollout_sims_per_s_before": round(r_before, 1),
                    "rollout_sims_per_s_after": round(r_after, 1),
                    "rollout_speed_ratio": round(r_after / r_before, 4) if r_before > 0 else 0.0,
                    "mcts_sims_per_s_before": round(m_before, 1),
                    "mcts_sims_per_s_after": round(m_after, 1),
                    "mcts_speed_ratio": round(m_after / m_before, 4) if m_before > 0 else 0.0,
                    "ms_per_move_before": round(ms_before, 2),
                    "ms_per_move_after": round(ms_after, 2),
                    "open_three_solve_before": round(solve_before, 4),
                    "open_three_solve_after": round(solve_after, 4),
                    "open_three_solve_diff": round(solve_after - solve_before, 4),
                }
            )

    # Phần 2: Song song hóa theo gốc (Root Parallelization) với K=1 (trước) vs K=2, 4 (sau)
    for size in (9, 15):
        mid_state = _make_midgame_state(size)
        trap_state, opt_moves = _make_open_three_trap(size)
        r_l2b4 = _bench_rollout_speed(mid_state, "l2_b4", num_rollouts, seed)

        # Đo mốc K=1 (Before)
        ag_k1 = RootParallelMCTSAgent(
            num_workers=1,
            rollout="l2_b4",
            c=1.0,
            use_candidates=True,
            candidate_radius=2,
            split_iterations=False,
            executor="process",
            seed=seed,
        )
        # Warm-up và đo K=1
        t0_k1 = time.perf_counter()
        iters_k1 = 0
        for _ in range(mcts_trials):
            ag_k1.select_move(mid_state, Budget(iterations=mcts_iterations))
            iters_k1 += int(ag_k1.last_stats["iterations"])
        dt_k1 = time.perf_counter() - t0_k1
        m_before_k1 = (iters_k1 / dt_k1) if dt_k1 > 0.0 else 0.0
        ms_before_k1 = (dt_k1 * 1000.0 / mcts_trials) if mcts_trials > 0 else 0.0

        solved_k1 = 0
        for t in range(trap_trials):
            ag_t1 = RootParallelMCTSAgent(
                num_workers=1,
                rollout="l2_b4",
                c=1.0,
                use_candidates=True,
                candidate_radius=2,
                split_iterations=False,
                executor="process",
                seed=seed + 3000 + t,
            )
            if ag_t1.select_move(trap_state, Budget(iterations=trap_iterations)) in opt_moves:
                solved_k1 += 1
        solve_rate_k1 = solved_k1 / trap_trials if trap_trials > 0 else 0.0

        for workers_k in (2, 4):
            for split_flag, mode_label in (
                (False, "unsplit_iter_per_worker"),
                (True, "split_total_iter"),
            ):
                ag_k = RootParallelMCTSAgent(
                    num_workers=workers_k,
                    rollout="l2_b4",
                    c=1.0,
                    use_candidates=True,
                    candidate_radius=2,
                    split_iterations=split_flag,
                    executor="process",
                    seed=seed,
                )
                # Warm-up pool
                ag_k.select_move(mid_state, Budget(iterations=20))

                t0_k = time.perf_counter()
                iters_k = 0
                for _ in range(mcts_trials):
                    ag_k.select_move(mid_state, Budget(iterations=mcts_iterations))
                    iters_k += int(ag_k.last_stats["iterations"])
                dt_k = time.perf_counter() - t0_k
                m_after_k = (iters_k / dt_k) if dt_k > 0.0 else 0.0
                ms_after_k = (dt_k * 1000.0 / mcts_trials) if mcts_trials > 0 else 0.0

                solved_k = 0
                eff_trap_iters = trap_iterations * workers_k if split_flag else trap_iterations
                for t in range(trap_trials):
                    ag_tk = RootParallelMCTSAgent(
                        num_workers=workers_k,
                        rollout="l2_b4",
                        c=1.0,
                        use_candidates=True,
                        candidate_radius=2,
                        split_iterations=split_flag,
                        executor="process",
                        seed=seed + 3000 + t,
                    )
                    if (
                        ag_tk.select_move(trap_state, Budget(iterations=eff_trap_iters))
                        in opt_moves
                    ):
                        solved_k += 1
                solve_rate_k = solved_k / trap_trials if trap_trials > 0 else 0.0

                rows.append(
                    {
                        "category": "root_parallelization",
                        "board_size": f"{size}x{size}",
                        "variant_before": "rootpar_k1_l2_b4",
                        "variant_after": f"rootpar_k{workers_k}_l2_b4",
                        "workers": workers_k,
                        "mode": mode_label,
                        "rollout_sims_per_s_before": round(r_l2b4, 1),
                        "rollout_sims_per_s_after": round(r_l2b4 * workers_k, 1),
                        "rollout_speed_ratio": round(float(workers_k), 4),
                        "mcts_sims_per_s_before": round(m_before_k1, 1),
                        "mcts_sims_per_s_after": round(m_after_k, 1),
                        "mcts_speed_ratio": (
                            round(m_after_k / m_before_k1, 4) if m_before_k1 > 0 else 0.0
                        ),
                        "ms_per_move_before": round(ms_before_k1, 2),
                        "ms_per_move_after": round(ms_after_k, 2),
                        "open_three_solve_before": round(solve_rate_k1, 4),
                        "open_three_solve_after": round(solve_rate_k, 4),
                        "open_three_solve_diff": round(solve_rate_k - solve_rate_k1, 4),
                    }
                )

    return rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Ghi danh sách từ điển ra file CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tổng hợp kết quả P4.3a (Bonus B4: Nước bắt buộc mở rộng & Root Parallel)."
    )
    parser.add_argument(
        "--csv-in",
        type=Path,
        default=DEFAULT_CSV_IN,
        help="Đường dẫn CSV ván đấu đầu vào (mặc định: experiments/results/p43a_b4_9x9.csv).",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_SUMMARY_OUT,
        help="Đường dẫn CSV tổng hợp điểm trước/sau (experiments/results/p43a_b4_summary.csv).",
    )
    parser.add_argument(
        "--speed-out",
        type=Path,
        default=DEFAULT_SPEED_OUT,
        help="Đường dẫn CSV tốc độ sims/s trước/sau (experiments/results/p43a_b4_speed.csv).",
    )
    parser.add_argument(
        "--root-parallel-games",
        type=int,
        default=40,
        help="Số ván đối đầu trực tiếp cho mỗi cặp RootParallelMCTSAgent (mặc định: 40).",
    )
    args = parser.parse_args(argv)

    csv_rows = load_csv_rows(args.csv_in)
    summary_rows = build_summary_rows(
        csv_rows,
        root_parallel_games=args.root_parallel_games,
    )
    write_csv(args.summary_out, SUMMARY_FIELDS, summary_rows)

    speed_rows = build_speed_rows()
    write_csv(args.speed_out, SPEED_FIELDS, speed_rows)

    print(f"Đã ghi {len(summary_rows)} dòng vào {args.summary_out}")
    print(f"Đã ghi {len(speed_rows)} dòng vào {args.speed_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
