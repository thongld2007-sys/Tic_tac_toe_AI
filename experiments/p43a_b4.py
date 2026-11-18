r"""Phân tích thực nghiệm và đo tốc độ cho P4.3a — Bonus B4 (Nước bắt buộc mở rộng & Root Parallel).

Chủ sở hữu: TV3.  Review: TV1.

Tiêu chí nghiệm thu P4.3a:
  - Có số mô phỏng/giây (`sims_per_s`) và điểm (`score` / `win_rate`) trước và sau khi áp dụng:
      1. **Nước bắt buộc mở rộng (Extended Forced Moves)**: `l2` -> `l2_b4`, `l3` -> `l3_b4`,
         `l4` -> `l4_b4` (ưu tiên 4 tầng: thắng 1 bước -> chặn thắng 1 bước / dừng sớm khi bị
         dọa đôi -> tạo Bốn hở hai đầu / nước đôi 4-4 dừng sớm -> chặn đối thủ tạo Bốn hở hai đầu).
      2. **Song song hóa theo gốc (Root Parallelization)**: `RootParallelMCTSAgent`
         (`mcts_parallel`) với $K = 1$ (trước) so với $K \in \{2, 4\}$ (sau) cây MCTS độc lập
         chạy song song từ gốc rồi gộp thống kê $(N, W)$ tại các nút con của gốc.

Đầu vào (sinh trực tiếp từ `experiments/run.py`):
  - `experiments/results/p43a_b4_9x9.csv`: 12 cặp đấu × 200 ván = 2.400 ván đánh giá `l2_b4`,
    `l3_b4`, `l4_b4`.
  - `experiments/results/p43a_b4_rootpar_9x9.csv`: 4 cặp đấu × 200 ván = 800 ván đánh giá
    `mcts_parallel` ($K = 2, 4$ đấu với $K = 1$ ở cùng thời gian thực và cùng số vòng lặp/worker).

Đầu ra:
  - `experiments/results/p43a_b4_summary.csv`: Bảng tổng hợp điểm số / tỷ lệ thắng Trước vs Sau
    cho cả 13 phép so sánh (kèm khoảng tin cậy Wilson 95%, bootstrap 95%, kiểm định $z$ và tốc độ
    mở nút thực tế trong ván đấu).
  - `experiments/results/p43a_b4_speed.csv`: Bảng đo tốc độ mô phỏng thực tế (`rollout_sims_per_s`,
    `mcts_sims_per_s`, hệ số tăng tốc `speed_ratio`) và tỷ lệ giải đúng thế cờ chiến thuật
    `Gomoku_open_three` Trước vs Sau trên bàn 9×9 và 15×15.
    Lưu ý kỹ thuật cho báo cáo: ở chế độ `split_total_iter` (chia tổng ngân sách
    $T_{\mathrm{total}}$ vòng lặp cho $K$ worker), mỗi cây con chỉ có kích thước
    $T_{\mathrm{total}} / K$ nên độ sâu cây nhỏ hơn và tỷ lệ vòng lặp nằm ở pha mở rộng gốc
    $O(1)$ cao hơn so với 1 cây đơn kích thước $T_{\mathrm{total}}$ (vốn phải duyệt
    `select_child_uct` qua nhiều tầng hơn). Nhờ vậy trên bàn 15×15, hệ số tăng tốc
    `mcts_speed_ratio` của chế độ `split_total_iter` có thể vượt nhẹ $K$.

Cách chạy:
    python -m experiments.run configs/p43a_b4_9x9.yaml --workers 4
    python -m experiments.run configs/p43a_b4_rootpar_9x9.yaml --workers 1
    python -m experiments.p43a_b4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.agents.base import Budget
from src.agents.mcts import MCTSAgent, resolve_rollout
from src.agents.mcts_parallel import (
    RootParallelMCTSAgent,
    measure_parallel_rollouts_per_second,
)
from src.games.gomoku import Gomoku

DEFAULT_CSV_IN = Path("experiments/results/p43a_b4_9x9.csv")
DEFAULT_ROOTPAR_CSV_IN = Path("experiments/results/p43a_b4_rootpar_9x9.csv")
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


def _extract_num_workers(row: Mapping[str, Any], side: str) -> int:
    """Trích xuất `num_workers` từ cột `a_params`/`b_params` hoặc nhãn `agent_a`/`agent_b`."""
    params_key = f"{side}_params"
    if params_key in row and row[params_key]:
        try:
            parsed = json.loads(str(row[params_key]))
            if "num_workers" in parsed:
                return int(parsed["num_workers"])
        except (ValueError, TypeError):
            pass
    label = str(row.get(f"agent_{side}", ""))
    for k in (8, 4, 2, 1):
        if f"num_workers={k}" in label:
            return k
    return 1


def _compute_side_speed(rows: Sequence[Mapping[str, Any]], prefix: str) -> tuple[float, float]:
    """Trả về `(new_nodes_per_s, ms_per_move)` của bên `prefix` (`'a'` hoặc `'b'`)."""
    total_moves = sum(int(r[f"{prefix}_moves"]) for r in rows)
    total_nodes = sum(int(r[f"{prefix}_nodes"]) for r in rows)
    total_time = sum(float(r[f"{prefix}_time_s"]) for r in rows)
    new_nodes = max(0, total_nodes - total_moves)
    nodes_per_s = (new_nodes / total_time) if total_time > 0.0 else 0.0
    ms_per_move = (total_time * 1000.0 / total_moves) if total_moves > 0 else 0.0
    return nodes_per_s, ms_per_move


def _summarize_direct_matchup_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    category: str,
    comparison: str,
    agent_after: str,
    agent_before: str,
    budget: str,
    seed: int = 2026,
) -> dict[str, Any]:
    """Tổng hợp thống kê cho một cặp đấu đối đầu trực tiếp (A = sau B4, B = trước B4)."""
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
    return {
        "category": category,
        "comparison": comparison,
        "agent_after": agent_after,
        "agent_before": agent_before,
        "budget": budget,
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


def build_summary_rows(
    csv_rows: Sequence[Mapping[str, Any]],
    *,
    rootpar_csv_rows: Sequence[Mapping[str, Any]] | None = None,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Tạo bảng tổng hợp `p43a_b4_summary.csv` từ CSV ván đấu của `experiments/run.py`."""
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    rootpar_grouped: dict[tuple[int, int, str], list[Mapping[str, Any]]] = {}

    for row in csv_rows:
        label_a = str(row["agent_a"])
        if "mcts_parallel" in label_a or "num_workers=" in label_a:
            k_a = _extract_num_workers(row, "a")
            k_b = _extract_num_workers(row, "b")
            b_tag = "time_s=0.04" if "time_s=" in label_a else "iterations=120/worker"
            rootpar_grouped.setdefault((k_a, k_b, b_tag), []).append(row)
            continue
        tag_a = _extract_rollout_tag(label_a)
        tag_b = _extract_rollout_tag(str(row["agent_b"]))
        has_time = "time_s=" in label_a
        budget_tag = "time_s=0.08" if has_time else "iterations=200"
        key = (tag_a, tag_b, budget_tag)
        grouped.setdefault(key, []).append(row)

    if rootpar_csv_rows is not None:
        for row in rootpar_csv_rows:
            label_a = str(row["agent_a"])
            k_a = _extract_num_workers(row, "a")
            k_b = _extract_num_workers(row, "b")
            b_tag = "time_s=0.04" if "time_s=" in label_a else "iterations=120/worker"
            rootpar_grouped.setdefault((k_a, k_b, b_tag), []).append(row)

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
            summary_rows.append(
                _summarize_direct_matchup_rows(
                    rows,
                    category=cat_name,
                    comparison=f"{after_tag}_vs_{base_tag}",
                    agent_after=after_tag,
                    agent_before=base_tag,
                    budget=budget_tag,
                    seed=seed,
                )
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

    # 4. Song song hóa theo gốc (RootParallelMCTSAgent) từ dữ liệu CSV của experiments/run.py
    for workers_k in (2, 4):
        rows_time = rootpar_grouped.get((workers_k, 1, "time_s=0.04"), [])
        if rows_time:
            summary_rows.append(
                _summarize_direct_matchup_rows(
                    rows_time,
                    category="root_parallel_eq_time",
                    comparison=f"rootpar_k{workers_k}_vs_k1_time004",
                    agent_after=f"rootpar_k{workers_k}_l2_b4",
                    agent_before="rootpar_k1_l2_b4",
                    budget="time_s=0.04",
                    seed=seed,
                )
            )
        rows_iter = rootpar_grouped.get((workers_k, 1, "iterations=120/worker"), [])
        if rows_iter:
            summary_rows.append(
                _summarize_direct_matchup_rows(
                    rows_iter,
                    category="root_parallel_per_worker_iter",
                    comparison=f"rootpar_k{workers_k}_vs_k1_iter120",
                    agent_after=f"rootpar_k{workers_k}_l2_b4",
                    agent_before="rootpar_k1_l2_b4",
                    budget="iterations=120/worker",
                    seed=seed,
                )
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


def _bench_rootpar_trap_solve_rate(
    trap_state: Gomoku,
    optimal_moves: set[tuple[int, int]],
    *,
    num_workers: int,
    split_iterations: bool,
    iterations: int,
    trials: int,
    seed: int,
) -> float:
    """Đo tỷ lệ giải đúng thế bẫy của `RootParallelMCTSAgent` ở tổng ngân sách `iterations`."""
    solved = 0
    for t in range(trials):
        ag = RootParallelMCTSAgent(
            num_workers=num_workers,
            rollout="l2_b4",
            c=1.0,
            use_candidates=True,
            candidate_radius=2,
            split_iterations=split_iterations,
            executor="process",
            seed=seed + 3000 + t,
        )
        if ag.select_move(trap_state, Budget(iterations=iterations)) in optimal_moves:
            solved += 1
    return solved / trials if trials > 0 else 0.0


def build_speed_rows(
    *,
    num_rollouts: int = 500,
    mcts_trials: int = 8,
    mcts_iterations: int = 200,
    trap_trials: int = 40,
    trap_iterations: int = 80,
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
        r_before_k1 = measure_parallel_rollouts_per_second(
            mid_state,
            "l2_b4",
            num_workers=1,
            num_rollouts_per_worker=num_rollouts,
            seed=seed,
        )

        # Đo tốc độ MCTS của mốc K=1 (Before) ở ngân sách mcts_iterations
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
        t0_k1 = time.perf_counter()
        iters_k1 = 0
        for _ in range(mcts_trials):
            ag_k1.select_move(mid_state, Budget(iterations=mcts_iterations))
            iters_k1 += int(ag_k1.last_stats["iterations"])
        dt_k1 = time.perf_counter() - t0_k1
        m_before_k1 = (iters_k1 / dt_k1) if dt_k1 > 0.0 else 0.0
        ms_before_k1 = (dt_k1 * 1000.0 / mcts_trials) if mcts_trials > 0 else 0.0

        # Tỷ lệ giải bẫy của mốc K=1 ở tổng ngân sách `mcts_iterations` (dùng cho split_total_iter)
        solve_k1_split_total = _bench_rootpar_trap_solve_rate(
            trap_state,
            opt_moves,
            num_workers=1,
            split_iterations=False,
            iterations=mcts_iterations,
            trials=trap_trials,
            seed=seed,
        )

        for workers_k in (2, 4):
            # Ở chế độ unsplit, mỗi worker chạy `trap_iterations` vòng (tổng K * trap_iterations);
            # mốc K=1 đối chiếu cũng chạy đúng `K * trap_iterations` vòng để cùng tổng ngân sách.
            solve_k1_unsplit_total = _bench_rootpar_trap_solve_rate(
                trap_state,
                opt_moves,
                num_workers=1,
                split_iterations=False,
                iterations=trap_iterations * workers_k,
                trials=trap_trials,
                seed=seed,
            )

            for split_flag, mode_label in (
                (False, "unsplit_iter_per_worker"),
                (True, "split_total_iter"),
            ):
                # Đo thực tế thông lượng rollout song song qua ProcessPoolExecutor
                r_after_k = measure_parallel_rollouts_per_second(
                    mid_state,
                    "l2_b4",
                    num_workers=workers_k,
                    num_rollouts_per_worker=num_rollouts,
                    split_total=split_flag,
                    seed=seed,
                )

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

                if not split_flag:
                    # Cùng tổng ngân sách `K * trap_iterations` giữa K=1 và K worker
                    s_before = solve_k1_unsplit_total
                    s_after = _bench_rootpar_trap_solve_rate(
                        trap_state,
                        opt_moves,
                        num_workers=workers_k,
                        split_iterations=False,
                        iterations=trap_iterations,
                        trials=trap_trials,
                        seed=seed,
                    )
                else:
                    # Cùng tổng ngân sách `mcts_iterations` giữa K=1 và K worker (mỗi worker chia K)
                    s_before = solve_k1_split_total
                    s_after = _bench_rootpar_trap_solve_rate(
                        trap_state,
                        opt_moves,
                        num_workers=workers_k,
                        split_iterations=True,
                        iterations=mcts_iterations,
                        trials=trap_trials,
                        seed=seed,
                    )

                rows.append(
                    {
                        "category": "root_parallelization",
                        "board_size": f"{size}x{size}",
                        "variant_before": "rootpar_k1_l2_b4",
                        "variant_after": f"rootpar_k{workers_k}_l2_b4",
                        "workers": workers_k,
                        "mode": mode_label,
                        "rollout_sims_per_s_before": round(r_before_k1, 1),
                        "rollout_sims_per_s_after": round(r_after_k, 1),
                        "rollout_speed_ratio": (
                            round(r_after_k / r_before_k1, 4) if r_before_k1 > 0 else 0.0
                        ),
                        "mcts_sims_per_s_before": round(m_before_k1, 1),
                        "mcts_sims_per_s_after": round(m_after_k, 1),
                        "mcts_speed_ratio": (
                            round(m_after_k / m_before_k1, 4) if m_before_k1 > 0 else 0.0
                        ),
                        "ms_per_move_before": round(ms_before_k1, 2),
                        "ms_per_move_after": round(ms_after_k, 2),
                        "open_three_solve_before": round(s_before, 4),
                        "open_three_solve_after": round(s_after, 4),
                        "open_three_solve_diff": round(s_after - s_before, 4),
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
        help="Đường dẫn CSV ván đấu B4 đầu vào (mặc định: experiments/results/p43a_b4_9x9.csv).",
    )
    parser.add_argument(
        "--rootpar-csv-in",
        type=Path,
        default=DEFAULT_ROOTPAR_CSV_IN,
        help="Đường dẫn CSV ván đấu Root Parallel (experiments/results/p43a_b4_rootpar_9x9.csv).",
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
    args = parser.parse_args(argv)

    csv_rows = load_csv_rows(args.csv_in)
    rootpar_rows = load_csv_rows(args.rootpar_csv_in) if args.rootpar_csv_in.exists() else []
    summary_rows = build_summary_rows(csv_rows, rootpar_csv_rows=rootpar_rows)
    write_csv(args.summary_out, SUMMARY_FIELDS, summary_rows)

    speed_rows = build_speed_rows()
    write_csv(args.speed_out, SPEED_FIELDS, speed_rows)

    print(f"Đã ghi {len(summary_rows)} dòng vào {args.summary_out}")
    print(f"Đã ghi {len(speed_rows)} dòng vào {args.speed_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
