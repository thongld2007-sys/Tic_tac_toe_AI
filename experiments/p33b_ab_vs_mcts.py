"""Tổng hợp P3.3b: Alpha-Beta vs MCTS cùng thời gian (0.5 / 1 / 2 s) trên 15×15 → Bảng T7.

Chủ sở hữu: TV3.  Review: TV1.

Cách chạy từ thư mục gốc của repo:
    python -m experiments.run configs/p33b_ab_vs_mcts.yaml --workers 8
    python -m experiments.p33b_ab_vs_mcts

Đầu vào:
  - `experiments/results/p33b_ab_vs_mcts.csv`: kết quả đấu từng ván trên bàn Caro 15×15 ở ba mức
    ngân sách thời gian `0.5`, `1.0`, `2.0` s/nước (`swap_first=True`, `opening_plies=2`).

Đầu ra (phục vụ Bảng T7 trong báo cáo):
  - `experiments/results/p33b_ab_vs_mcts_t7.csv`: tổng hợp toàn phần và tách riêng theo bên đi
    trước (`overall`, `ab_first`, `mcts_first`) cho từng mức thời gian và từng biến thể MCTS, kèm
    khoảng tin cậy Wilson 95%, bootstrap CI 95% cho điểm trung bình, kiểm định z thắng-thua ghép
    cặp và thống kê thời gian / số nút / độ dài ván.
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

from experiments.stats import bootstrap_diff_ci, summarize_matchup, wilson_ci

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"

SPLIT_MODES: tuple[str, ...] = ("overall", "ab_first", "mcts_first")

T7_COLUMNS: tuple[str, ...] = (
    "matchup",
    "game",
    "time_s",
    "mcts_rollout",
    "mcts_c",
    "first_side",
    "games",
    "ab_wins",
    "draws",
    "mcts_wins",
    "ab_win_rate",
    "ab_win_ci_low",
    "ab_win_ci_high",
    "mcts_win_rate",
    "mcts_win_ci_low",
    "mcts_win_ci_high",
    "ab_mean_score",
    "ab_score_ci_low",
    "ab_score_ci_high",
    "z_stat",
    "p_value",
    "diff_mean",
    "diff_ci_low",
    "diff_ci_high",
    "avg_plies",
    "ab_nodes_per_move",
    "mcts_nodes_per_move",
    "ab_avg_time_ms",
    "mcts_avg_time_ms",
    "ab_max_time_s",
    "mcts_max_time_s",
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
    """Kiểm định z ghép cặp (sign test) cho số ván thắng-thua: z = (W - L) / sqrt(W + L)."""
    decisive = wins + losses
    if decisive <= 0:
        return 0.0, 1.0
    z_stat = (wins - losses) / math.sqrt(decisive)
    p_val = 2.0 * (1.0 - NormalDist(0.0, 1.0).cdf(abs(z_stat)))
    return float(z_stat), max(0.0, min(1.0, float(p_val)))


def _summarize_slice(
    rows: Sequence[Mapping[str, str]],
    *,
    matchup_id: int,
    game_label: str,
    time_s: float,
    mcts_rollout: str,
    mcts_c: float,
    first_side: str,
) -> dict[str, Any]:
    """Tính đầy đủ thống kê Bảng T7 cho một lát cắt (`overall`, `ab_first`, hoặc `mcts_first`)."""
    scores_ab = [float(r["score_a"]) for r in rows]
    scores_mcts = [1.0 - s for s in scores_ab]
    stats = summarize_matchup(scores_ab, seed=2026)

    n_games = int(stats["games"])
    ab_wins = int(stats["wins"])
    draws = int(stats["draws"])
    mcts_wins = int(stats["losses"])

    mcts_ci = wilson_ci(mcts_wins, n_games)
    z_stat, p_val = _paired_win_loss_ztest(ab_wins, mcts_wins)
    diff_mean = (sum(scores_ab) / n_games) - (sum(scores_mcts) / n_games)
    diff_lo, diff_hi = bootstrap_diff_ci(scores_ab, scores_mcts, seed=2026, paired=True)

    total_plies = sum(int(r["plies"]) for r in rows)
    total_ab_moves = sum(int(r["a_moves"]) for r in rows)
    total_mcts_moves = sum(int(r["b_moves"]) for r in rows)
    total_ab_nodes = sum(int(r["a_nodes"]) for r in rows)
    total_mcts_nodes = sum(int(r["b_nodes"]) for r in rows)
    total_ab_time = sum(float(r["a_time_s"]) for r in rows)
    total_mcts_time = sum(float(r["b_time_s"]) for r in rows)
    ab_max_t = max(float(r["a_max_time_s"]) for r in rows)
    mcts_max_t = max(float(r["b_max_time_s"]) for r in rows)

    return {
        "matchup": matchup_id,
        "game": game_label,
        "time_s": round(time_s, 3),
        "mcts_rollout": mcts_rollout,
        "mcts_c": round(mcts_c, 4),
        "first_side": first_side,
        "games": n_games,
        "ab_wins": ab_wins,
        "draws": draws,
        "mcts_wins": mcts_wins,
        "ab_win_rate": round(float(stats["win_rate"]), 4),
        "ab_win_ci_low": round(float(stats["win_wilson_low"]), 4),
        "ab_win_ci_high": round(float(stats["win_wilson_high"]), 4),
        "mcts_win_rate": round(mcts_wins / n_games, 4),
        "mcts_win_ci_low": round(mcts_ci.low, 4),
        "mcts_win_ci_high": round(mcts_ci.high, 4),
        "ab_mean_score": round(float(stats["score_mean"]), 4),
        "ab_score_ci_low": round(float(stats["score_boot_low"]), 4),
        "ab_score_ci_high": round(float(stats["score_boot_high"]), 4),
        "z_stat": round(z_stat, 4),
        "p_value": round(p_val, 4),
        "diff_mean": round(diff_mean, 4),
        "diff_ci_low": round(diff_lo, 4),
        "diff_ci_high": round(diff_hi, 4),
        "avg_plies": round(total_plies / n_games, 2),
        "ab_nodes_per_move": round(total_ab_nodes / max(1, total_ab_moves), 1),
        "mcts_nodes_per_move": round(total_mcts_nodes / max(1, total_mcts_moves), 1),
        "ab_avg_time_ms": round(1000.0 * total_ab_time / max(1, total_ab_moves), 2),
        "mcts_avg_time_ms": round(1000.0 * total_mcts_time / max(1, total_mcts_moves), 2),
        "ab_max_time_s": round(ab_max_t, 4),
        "mcts_max_time_s": round(mcts_max_t, 4),
    }


def summarize_p33b_table_t7(
    csv_path: Path,
    out_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Tổng hợp kết quả Alpha-Beta vs MCTS thành Bảng T7 (tổng + tách theo đi trước/sau)."""
    raw_rows = load_csv_rows(csv_path)
    if not raw_rows:
        raise ValueError(f"File CSV {csv_path} không có dữ liệu")

    by_matchup: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        by_matchup[int(row["matchup"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for m_id in sorted(by_matchup.keys()):
        m_rows = by_matchup[m_id]
        a_params = json.loads(m_rows[0]["a_params"])
        b_params = json.loads(m_rows[0]["b_params"])

        if a_params.get("type") != "alphabeta_gomoku" or b_params.get("type") != "mcts":
            raise ValueError(
                f"Cặp đấu {m_id} phải là alphabeta_gomoku vs mcts, "
                f"nhận được {a_params.get('type')!r} vs {b_params.get('type')!r}"
            )

        game_label = str(m_rows[0]["game"])
        time_s = float((a_params.get("budget") or {}).get("time_s", 0.0))
        mcts_rollout = str(b_params.get("rollout", "l0")).lower()
        mcts_c = float(b_params.get("c", 1.4142))

        slices: dict[str, list[dict[str, str]]] = {
            "overall": m_rows,
            "ab_first": [r for r in m_rows if r["first_player"] == "a"],
            "mcts_first": [r for r in m_rows if r["first_player"] == "b"],
        }
        for split_mode in SPLIT_MODES:
            sub = slices[split_mode]
            if not sub:
                continue
            summary_rows.append(
                _summarize_slice(
                    sub,
                    matchup_id=m_id,
                    game_label=game_label,
                    time_s=time_s,
                    mcts_rollout=mcts_rollout,
                    mcts_c=mcts_c,
                    first_side=split_mode,
                )
            )

    if out_csv is not None:
        write_csv_rows(out_csv, T7_COLUMNS, summary_rows)
    return summary_rows


def build_arg_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh cho `python -m experiments.p33b_ab_vs_mcts`."""
    parser = argparse.ArgumentParser(
        description=(
            "Tổng hợp kết quả Alpha-Beta vs MCTS (0.5 / 1 / 2 s trên Caro 15×15) "
            "tách theo đi trước / đi sau -> Bảng T7 (P3.3b)."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULTS_DIR / "p33b_ab_vs_mcts.csv",
        help="File CSV đầu vào sinh bởi runner (mặc định: p33b_ab_vs_mcts.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "p33b_ab_vs_mcts_t7.csv",
        help="File CSV Bảng T7 đầu ra (mặc định: experiments/results/p33b_ab_vs_mcts_t7.csv).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào chính khi chạy `python -m experiments.p33b_ab_vs_mcts`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    rows = summarize_p33b_table_t7(args.csv, out_csv=args.out)
    print(f"[T7] {len(rows)} dòng -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
