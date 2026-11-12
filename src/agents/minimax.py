"""Minimax đầy đủ (dạng negamax), không cắt tỉa. Dùng làm chuẩn đối chiếu trên 3×3."""

from __future__ import annotations

import time

from src.agents.base import Agent, Budget
from src.games.base import GameState, Move


def terminal_value(state: GameState) -> int:
    """Giá trị của một thế đã kết thúc, theo góc nhìn của người sắp đi: 1 thắng, 0 hòa, -1 thua.

    winner() trả 1 / -1 / 0 và người chơi cũng là 1 / -1, nên chỉ cần nhân hai số với nhau.
    """
    result = state.winner()
    if result is None:
        raise ValueError("Thế cờ chưa kết thúc")
    return result * state.current_player()


class MinimaxAgent(Agent):
    """Duyệt toàn bộ cây trò chơi và chọn nước có giá trị cao nhất.

    Giá trị một thế là 1 / 0 / -1 (thắng / hòa / thua) theo góc nhìn của người sắp đi.
    Khi nhiều nước cùng giá trị, chọn nước đứng trước trong legal_moves().

    Ngân sách: chỉ dùng `budget.depth`. Thế chưa kết thúc ở độ sâu giới hạn được coi là hòa (0),
    vì agent này không có hàm lượng giá. Để None thì duyệt tới cuối ván (đúng cho 3×3); nếu đặt
    thì phải >= 1.
    """

    name = "minimax"

    def __init__(self) -> None:
        super().__init__()
        self.last_value: int | None = None  # giá trị của thế gốc ở lần select_move() gần nhất
        self._nodes = 0
        self._max_depth = 0

    def select_move(self, state: GameState, budget: Budget) -> Move:
        if state.is_terminal():
            raise ValueError("Ván đã kết thúc, không còn nước để chọn")
        if budget.depth is not None and budget.depth < 1:
            raise ValueError(f"budget.depth phải >= 1, nhận được {budget.depth}")

        start = time.perf_counter()
        self._nodes = 1  # tính cả nút gốc
        self._max_depth = 0
        limit = budget.depth

        best_move: Move | None = None
        best_value = -2
        for m in state.legal_moves():
            state.play(m)
            value = -self._search(state, 1, limit)
            state.undo()
            if value > best_value:
                best_value, best_move = value, m

        assert best_move is not None
        self.last_value = best_value
        self.last_stats = {
            "nodes": self._nodes,
            "time_s": time.perf_counter() - start,
            "depth": self._max_depth,
        }
        return best_move

    def _search(self, state: GameState, depth: int, limit: int | None) -> int:
        self._nodes += 1
        if depth > self._max_depth:
            self._max_depth = depth

        if state.winner() is not None:
            return terminal_value(state)
        if limit is not None and depth >= limit:
            return 0

        best = -2
        for m in state.legal_moves():
            state.play(m)
            value = -self._search(state, depth + 1, limit)
            state.undo()
            if value > best:
                best = value
        return best
