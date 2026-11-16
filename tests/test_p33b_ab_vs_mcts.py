"""Unit test cho module tổng hợp thực nghiệm P3.3b và Bảng T7 `experiments/p33b_ab_vs_mcts.py`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.p33b_ab_vs_mcts import (
    T7_COLUMNS,
    load_csv_rows,
    main,
    summarize_p33b_table_t7,
    write_csv_rows,
)


def _make_synthetic_p33b_rows() -> list[dict[str, object]]:
    """Tạo dữ liệu CSV giả lập đúng schema cho P3.3b ở 3 mức ngân sách (0.5, 1.0, 2.0 s)."""
    rows: list[dict[str, object]] = []
    matchup = 0
    for time_s in (0.5, 1.0, 2.0):
        for rollout in ("l2", "l4"):
            for g in range(10):
                first_player = "a" if g % 2 == 0 else "b"
                # AB thắng mọi ván khi đi trước, thắng 4/5 khi đi sau (MCTS thắng 1 ván đi trước)
                if first_player == "b" and g == 9:
                    winner, score = "b", 0.0
                else:
                    winner, score = "a", 1.0
                rows.append(
                    {
                        "experiment": "p33b_ab_vs_mcts",
                        "game": "gomoku15",
                        "matchup": matchup,
                        "game_index": g,
                        "seed": 5000 + matchup * 100 + g,
                        "agent_a": f"alphabeta_gomoku[time_s={time_s}]",
                        "agent_b": f"mcts[c=0.5,rollout={rollout},time_s={time_s}]",
                        "a_params": json.dumps(
                            {
                                "type": "alphabeta_gomoku",
                                "radius": 2,
                                "max_candidates": 16,
                                "budget": {"time_s": time_s},
                            }
                        ),
                        "b_params": json.dumps(
                            {
                                "type": "mcts",
                                "c": 0.5,
                                "rollout": rollout,
                                "selection": "robust",
                                "use_candidates": True,
                                "candidate_radius": 2,
                                "reuse_tree": True,
                                "budget": {"time_s": time_s},
                            }
                        ),
                        "first_player": first_player,
                        "opening_plies": 2,
                        "winner": winner,
                        "score_a": score,
                        "plies": 14,
                        "a_moves": 7,
                        "b_moves": 7,
                        "a_nodes": int(4000 * time_s),
                        "b_nodes": int(9000 * time_s),
                        "a_time_s": 7 * time_s * 0.85,
                        "b_time_s": 7 * time_s,
                        "a_max_time_s": time_s * 0.88,
                        "b_max_time_s": time_s * 1.001,
                    }
                )
            matchup += 1
    return rows


def test_summarize_p33b_table_t7_splits_by_first_mover(tmp_path: Path) -> None:
    """Bảng T7 tách đúng 3 lát cắt (overall, ab_first, mcts_first) và tính đúng khoảng tin cậy."""
    in_csv = tmp_path / "p33b_ab_vs_mcts.csv"
    out_csv = tmp_path / "p33b_ab_vs_mcts_t7.csv"
    synth = _make_synthetic_p33b_rows()
    write_csv_rows(in_csv, list(synth[0].keys()), synth)

    code = main(["--csv", str(in_csv), "--out", str(out_csv)])
    assert code == 0
    assert out_csv.exists()

    loaded = load_csv_rows(out_csv)
    # 6 cặp đấu × 3 lát cắt (overall, ab_first, mcts_first) = 18 dòng
    assert len(loaded) == 18
    assert set(loaded[0].keys()) == set(T7_COLUMNS)

    m0 = [r for r in loaded if int(r["matchup"]) == 0]
    by_split = {r["first_side"]: r for r in m0}
    assert set(by_split.keys()) == {"overall", "ab_first", "mcts_first"}
    assert int(by_split["overall"]["games"]) == 10
    assert int(by_split["ab_first"]["games"]) == 5
    assert int(by_split["mcts_first"]["games"]) == 5
    assert float(by_split["ab_first"]["ab_win_rate"]) == pytest.approx(1.0)
    assert float(by_split["mcts_first"]["ab_win_rate"]) == pytest.approx(0.8)
    assert float(by_split["mcts_first"]["mcts_win_rate"]) == pytest.approx(0.2)

    for r in loaded:
        assert (
            0.0
            <= float(r["ab_win_ci_low"])
            <= float(r["ab_win_rate"])
            <= float(r["ab_win_ci_high"])
            <= 1.0
        )
        assert (
            0.0
            <= float(r["mcts_win_ci_low"])
            <= float(r["mcts_win_rate"])
            <= float(r["mcts_win_ci_high"])
            <= 1.0
        )
        assert float(r["diff_mean"]) == pytest.approx(
            2.0 * float(r["ab_mean_score"]) - 1.0, abs=1e-3
        )


def test_summarize_p33b_rejects_empty_or_wrong_matchup(tmp_path: Path) -> None:
    """File CSV rỗng hoặc cặp đấu không phải alphabeta_gomoku vs mcts phải báo ValueError."""
    empty_csv = tmp_path / "empty.csv"
    write_csv_rows(empty_csv, ["matchup"], [])
    with pytest.raises(ValueError, match="không có dữ liệu"):
        summarize_p33b_table_t7(empty_csv)

    bad_csv = tmp_path / "bad.csv"
    bad_row = _make_synthetic_p33b_rows()[0]
    bad_row["a_params"] = json.dumps({"type": "random"})
    write_csv_rows(bad_csv, list(bad_row.keys()), [bad_row])
    with pytest.raises(ValueError, match="alphabeta_gomoku vs mcts"):
        summarize_p33b_table_t7(bad_csv)
