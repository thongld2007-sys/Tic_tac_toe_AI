r"""Tổng hợp kết quả thực nghiệm Cờ tướng cho P4.3b (Alpha-Beta PST vs MCTS-L4).

Chủ sở hữu: TV3.  Review: TV1.

Đầu vào (sinh trực tiếp từ `experiments/run.py`):
  - `experiments/results/p43b_xiangqi_exp.csv`: 6 cặp đấu × 60 ván = 360 ván Cờ tướng
    (`configs/p43b_xiangqi_exp.yaml`, đổi bên Đỏ/Đen với 2 nước khai cuộc ngẫu nhiên dùng chung).

Đầu ra:
  - `experiments/results/p43b_xiangqi_summary.csv`: Bảng tổng hợp tỷ lệ thắng, điểm số,
    khoảng tin cậy Wilson 95%, bootstrap 95%, kiểm định $z$ hai phía so với mốc cân bằng 0,5,
    tách riêng kết quả khi bên A cầm **Đỏ (đi trước)** và cầm **Đen (đi sau)**, cùng số nút/nước,
    thời gian trung bình (ms/nước) và tỷ lệ sử dụng ngân sách thời gian.

Cách chạy:
    python -m experiments.run configs/p43b_xiangqi_exp.yaml --workers 6
    python -m experiments.p43b_xiangqi
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_CSV_IN = Path("experiments/results/p43b_xiangqi_exp.csv")
DEFAULT_SUMMARY_OUT = Path("experiments/results/p43b_xiangqi_summary.csv")

SUMMARY_FIELDS: tuple[str, ...] = (
    "matchup",
    "comparison",
    "agent_a",
    "agent_b",
    "budget",
    "games",
    "wins_a",
    "draws",
    "losses_a",
    "win_rate_a",
    "wilson_ci_low",
    "wilson_ci_high",
    "score_a",
    "boot_ci_low",
    "boot_ci_high",
    "z_stat",
    "p_value",
    "significant_005",
    "red_games",
    "red_wins_a",
    "red_draws",
    "red_score_a",
    "black_games",
    "black_wins_a",
    "black_draws",
    "black_score_a",
    "avg_plies",
    "a_nodes_per_move",
    "b_nodes_per_move",
    "a_ms_per_move",
    "b_ms_per_move",
    "a_time_pct",
    "b_time_pct",
)

_COMPARISON_LABELS: dict[int, tuple[str, str, str, str]] = {
    0: ("ab_vs_mcts_t010", "alphabeta_xiangqi", "mcts_xiangqi", "time_s=0.10"),
    1: ("ab_vs_mcts_t020", "alphabeta_xiangqi", "mcts_xiangqi", "time_s=0.20"),
    2: ("ab_d2q2_vs_ab_d1q0", "ab_d2_q2", "ab_d1_q0", "depth=2 vs 1"),
    3: ("mcts_i200_vs_mcts_i50", "mcts_i200", "mcts_i50", "iters=200 vs 50"),
    4: ("ab_d2q2_vs_random", "ab_d2_q2", "random", "depth=2"),
    5: ("mcts_i100_vs_random", "mcts_i100", "random", "iters=100"),
}


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
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
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


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Đọc toàn bộ các dòng từ file CSV kết quả ván đấu."""
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _extract_time_budget(params_json: str) -> float | None:
    try:
        parsed = json.loads(params_json)
    except (ValueError, TypeError):
        return None
    budget = parsed.get("budget")
    if isinstance(budget, Mapping) and budget.get("time_s") is not None:
        return float(budget["time_s"])
    return None


def build_summary_rows(
    csv_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Tổng hợp toàn bộ các cặp đấu Cờ tướng thành bảng `p43b_xiangqi_summary.csv`."""
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in csv_rows:
        m_idx = int(row["matchup"])
        grouped.setdefault(m_idx, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for m_idx in sorted(grouped):
        rows = grouped[m_idx]
        n = len(rows)
        comp_name, ag_a, ag_b, budget_str = _COMPARISON_LABELS.get(
            m_idx,
            (
                f"matchup_{m_idx}",
                str(rows[0]["agent_a"]),
                str(rows[0]["agent_b"]),
                "custom",
            ),
        )

        scores_a = [float(r["score_a"]) for r in rows]
        wins_a = sum(1 for s in scores_a if s == 1.0)
        draws = sum(1 for s in scores_a if s == 0.5)
        losses_a = n - wins_a - draws
        mean_score = sum(scores_a) / n
        w_lo, w_hi = wilson_ci_95(wins_a, n)
        b_lo, b_hi = bootstrap_mean_ci(scores_a, seed=seed + m_idx)
        z_val, p_val = one_sample_score_z_test(scores_a, mu0=0.5)

        red_rows = [r for r in rows if str(r["first_player"]) == "a"]
        black_rows = [r for r in rows if str(r["first_player"]) == "b"]
        red_n = len(red_rows)
        black_n = len(black_rows)
        red_wins = sum(1 for r in red_rows if float(r["score_a"]) == 1.0)
        red_draws = sum(1 for r in red_rows if float(r["score_a"]) == 0.5)
        red_score = sum(float(r["score_a"]) for r in red_rows) / red_n if red_n else 0.0
        black_wins = sum(1 for r in black_rows if float(r["score_a"]) == 1.0)
        black_draws = sum(1 for r in black_rows if float(r["score_a"]) == 0.5)
        black_score = sum(float(r["score_a"]) for r in black_rows) / black_n if black_n else 0.0

        avg_plies = sum(int(r["plies"]) for r in rows) / n
        a_moves = sum(int(r["a_moves"]) for r in rows)
        b_moves = sum(int(r["b_moves"]) for r in rows)
        a_nodes = sum(int(r["a_nodes"]) for r in rows)
        b_nodes = sum(int(r["b_nodes"]) for r in rows)
        a_time = sum(float(r["a_time_s"]) for r in rows)
        b_time = sum(float(r["b_time_s"]) for r in rows)

        a_npm = (a_nodes / a_moves) if a_moves > 0 else 0.0
        b_npm = (b_nodes / b_moves) if b_moves > 0 else 0.0
        a_ms = (a_time * 1000.0 / a_moves) if a_moves > 0 else 0.0
        b_ms = (b_time * 1000.0 / b_moves) if b_moves > 0 else 0.0

        a_t_budget = _extract_time_budget(str(rows[0].get("a_params", "")))
        b_t_budget = _extract_time_budget(str(rows[0].get("b_params", "")))
        a_time_pct = (a_ms / (a_t_budget * 10.0)) if a_t_budget else 0.0
        b_time_pct = (b_ms / (b_t_budget * 10.0)) if b_t_budget else 0.0

        summary_rows.append(
            {
                "matchup": m_idx,
                "comparison": comp_name,
                "agent_a": ag_a,
                "agent_b": ag_b,
                "budget": budget_str,
                "games": n,
                "wins_a": wins_a,
                "draws": draws,
                "losses_a": losses_a,
                "win_rate_a": round(wins_a / n, 4),
                "wilson_ci_low": round(w_lo, 4),
                "wilson_ci_high": round(w_hi, 4),
                "score_a": round(mean_score, 4),
                "boot_ci_low": round(b_lo, 4),
                "boot_ci_high": round(b_hi, 4),
                "z_stat": round(z_val, 4),
                "p_value": round(p_val, 6),
                "significant_005": p_val < 0.05,
                "red_games": red_n,
                "red_wins_a": red_wins,
                "red_draws": red_draws,
                "red_score_a": round(red_score, 4),
                "black_games": black_n,
                "black_wins_a": black_wins,
                "black_draws": black_draws,
                "black_score_a": round(black_score, 4),
                "avg_plies": round(avg_plies, 1),
                "a_nodes_per_move": round(a_npm, 1),
                "b_nodes_per_move": round(b_npm, 1),
                "a_ms_per_move": round(a_ms, 2),
                "b_ms_per_move": round(b_ms, 2),
                "a_time_pct": round(a_time_pct, 1),
                "b_time_pct": round(b_time_pct, 1),
            }
        )

    return summary_rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Ghi danh sách từ điển ra file CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tổng hợp kết quả thực nghiệm Cờ tướng P4.3b (Alpha-Beta PST vs MCTS-L4)."
    )
    parser.add_argument(
        "--csv-in",
        type=Path,
        default=DEFAULT_CSV_IN,
        help="Đường dẫn CSV ván đấu Cờ tướng (experiments/results/p43b_xiangqi_exp.csv).",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_SUMMARY_OUT,
        help="Đường dẫn CSV bảng tổng hợp P4.3b (experiments/results/p43b_xiangqi_summary.csv).",
    )
    args = parser.parse_args(argv)

    csv_rows = load_csv_rows(args.csv_in)
    summary_rows = build_summary_rows(csv_rows)
    write_csv(args.summary_out, SUMMARY_FIELDS, summary_rows)
    print(f"Đã ghi {len(summary_rows)} dòng vào {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
