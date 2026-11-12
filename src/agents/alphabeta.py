"""Alpha-Beta (dạng negamax) cho 3×3, với 3 chế độ sắp xếp nước để so số nút (bảng T1)."""

from __future__ import annotations

import random
import time
from math import inf

from src.agents.base import Agent, Budget
from src.agents.minimax import terminal_value
from src.games.base import GameState, Move

# Ba chế độ sắp xếp nước:
#   natural   — giữ thứ tự của legal_moves() (từ trái sang phải, từ trên xuống dưới).
#   random    — xáo ngẫu nhiên tại mỗi nút, tái lập được nhờ `seed`.
#   heuristic — ô nằm trên nhiều đường thắng hơn được thử trước (3×3: giữa, rồi góc, rồi cạnh).
ORDERINGS: tuple[str, ...] = ("natural", "random", "heuristic")


def lines_through(m: Move, size: int) -> int:
    """Số hàng, cột và đường chéo chính đi qua ô `m` trên bàn `size × size`."""
    r, c = m
    return 2 + (r == c) + (r + c == size - 1)


class AlphaBetaAgent(Agent):
    """Alpha-Beta cho ra đúng giá trị của Minimax nhưng bỏ qua các nhánh không thể đổi kết quả.

    Giá trị một thế là 1 / 0 / -1 theo góc nhìn của người sắp đi. Cửa sổ ban đầu là (-inf, +inf).
    Khi nhiều nước cùng giá trị, chọn nước được thử trước theo chế độ sắp xếp đang dùng; với
    `natural` thì trùng với nước MinimaxAgent chọn.

    Ngân sách: chỉ dùng `budget.depth`, cùng quy ước với MinimaxAgent (chạm giới hạn thì coi
    là hòa).
    """

    def __init__(self, ordering: str = "natural", seed: int | None = None) -> None:
        super().__init__()
        if ordering not in ORDERINGS:
            raise ValueError(f"ordering phải thuộc {ORDERINGS}, nhận được {ordering!r}")
        self.ordering = ordering
        self.name = f"alphabeta-{ordering}"
        self.last_value: int | None = None  # giá trị của thế gốc ở lần select_move() gần nhất
        self._rng = random.Random(seed)
        self._nodes = 0
        self._cutoffs = 0
        self._max_depth = 0

    def select_move(self, state: GameState, budget: Budget) -> Move:
        if state.is_terminal():
            raise ValueError("Ván đã kết thúc, không còn nước để chọn")

        start = time.perf_counter()
        self._nodes = 1  # tính cả nút gốc
        self._cutoffs = 0
        self._max_depth = 0
        limit = budget.depth

        best_move: Move | None = None
        alpha = -inf
        for m in self._ordered_moves(state):
            state.play(m)
            value = -self._search(state, 1, limit, -inf, -alpha)
            state.undo()
            if value > alpha:
                alpha, best_move = value, m

        assert best_move is not None
        self.last_value = int(alpha)
        self.last_stats = {
            "nodes": self._nodes,
            "time_s": time.perf_counter() - start,
            "cutoffs": self._cutoffs,
            "depth": self._max_depth,
        }
        return best_move

    def _ordered_moves(self, state: GameState) -> list[Move]:
        moves = state.legal_moves()
        if self.ordering == "random":
            self._rng.shuffle(moves)
        elif self.ordering == "heuristic":
            size = state.board.shape[0]
            # sort() ổn định: các ô ngang điểm giữ thứ tự của legal_moves().
            moves.sort(key=lambda m: -lines_through(m, size))
        return moves

    def _search(
        self, state: GameState, depth: int, limit: int | None, alpha: float, beta: float
    ) -> float:
        self._nodes += 1
        if depth > self._max_depth:
            self._max_depth = depth

        if state.winner() is not None:
            return terminal_value(state)
        if limit is not None and depth >= limit:
            return 0

        best = -inf
        for m in self._ordered_moves(state):
            state.play(m)
            value = -self._search(state, depth + 1, limit, -beta, -alpha)
            state.undo()
            if value > best:
                best = value
                if value > alpha:
                    alpha = value
            if alpha >= beta:
                self._cutoffs += 1
                break
        return best
