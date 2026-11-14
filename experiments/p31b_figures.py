"""Generate the P3.1b report figures from per-game runner CSV files."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.stats import wilson_ci

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [100, 500, 1000, 2000, 5000]
COEFFICIENTS = [0.5, 1.0, 1.4, 2.0]


def plot(board, figure, title):
    groups = defaultdict(list)
    with (ROOT / "experiments/results" / f"p31b_{board}.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            params = json.loads(row["a_params"])
            key = params["c"] / 2, params["budget"]["iterations"]
            groups[key].append(row)

    expected = {(c, n) for c in COEFFICIENTS for n in BUDGETS}
    if groups.keys() != expected:
        raise ValueError(f"Incomplete parameter grid for {board}")
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    for c, marker in zip(COEFFICIENTS, ["o", "s", "^", "D"]):
        rates, lows, highs = [], [], []
        for n in BUDGETS:
            rows = groups[c, n]
            if (len(rows) != 200 or {int(r["game_index"]) for r in rows} != set(range(200))
                    or sum(r["first_player"] == "a" for r in rows) != 100):
                raise ValueError(f"Invalid games or color balance at {board}, {c}, {n}")
            wins = sum(row["winner"] == "a" for row in rows)
            low, high = wilson_ci(wins, len(rows))
            rate = wins / len(rows)
            rates.append(rate * 100)
            lows.append(max(0, rate - low) * 100)
            highs.append(max(0, high - rate) * 100)
        ax.errorbar(BUDGETS, rates, yerr=[lows, highs], label=f"c = {c:.1f}", marker=marker,
                    capsize=2, linewidth=1.3, markersize=4)
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS, labels=[str(n) for n in BUDGETS])
    ax.set(title=title, xlabel="Iterations per MCTS move (log scale)", ylabel="MCTS wins (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=9)
    fig.text(0.5, -0.045, "200 games per point · 95% Wilson intervals · c on the [0, 1] scale",
             ha="center", fontsize=9)
    folder = ROOT / "report/figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f"{figure}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot("3x3", "F1", "Tic-Tac-Toe 3×3 · MCTS vs Random")
    plot("9x9", "F2", "Gomoku 9×9 · MCTS vs Random")
