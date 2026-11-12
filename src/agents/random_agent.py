"""Agent đi ngẫu nhiên đồng nhất: đối thủ chuẩn yếu nhất cho mọi thực nghiệm."""

from __future__ import annotations

import random
import time

from src.agents.base import Agent, Budget
from src.games.base import GameState, Move


class RandomAgent(Agent):
    """Chọn đều ngẫu nhiên một nước trong legal_moves(). Bỏ qua mọi ngân sách.

    Cùng `seed` và cùng chuỗi thế cờ thì cho cùng chuỗi nước đi.
    """

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.seed = seed
        self._rng = random.Random(seed)

    def select_move(self, state: GameState, budget: Budget) -> Move:
        start = time.perf_counter()
        moves = state.legal_moves()
        if not moves:
            raise ValueError("Ván đã kết thúc, không còn nước để chọn")
        move = self._rng.choice(moves)
        self.last_stats = {"nodes": 1, "time_s": time.perf_counter() - start}
        return move
