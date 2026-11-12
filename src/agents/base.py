"""Interface chung cho mọi agent (Random, Minimax, Alpha-Beta, MCTS).

Chủ sở hữu: TV1. Sau mốc M1, mọi thay đổi ở file này phải có đủ 2 người còn lại duyệt.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypedDict

from src.games.base import GameState, Move


@dataclass
class Budget:
    """Ngân sách cho một lần chọn nước. Trường nào là None thì không giới hạn theo trường đó."""

    iterations: int | None = None  # số iteration của MCTS
    depth: int | None = None  # độ sâu tìm kiếm của Minimax / Alpha-Beta
    time_s: float | None = None  # thời gian tối đa cho một nước, tính bằng giây


class RootStat(TypedDict):
    """Thống kê của một nước đi tại gốc cây MCTS."""

    move: Move
    N: int  # số lượt thăm
    W: float  # tổng giá trị, theo góc nhìn của người đi tại gốc
    q: float  # W / N
    uct: float  # giá trị UCT tại thời điểm kết thúc tìm kiếm


class AgentStats(TypedDict, total=False):
    """Thống kê của lần select_move() gần nhất. Agent chỉ điền các trường có ý nghĩa với nó."""

    nodes: int  # số nút đã duyệt (Minimax / Alpha-Beta) hoặc đã tạo (MCTS)
    time_s: float  # thời gian thực tế đã dùng
    cutoffs: int  # số lần cắt tỉa alpha-beta
    depth: int  # độ sâu đã tìm xong
    sims_per_s: float  # số mô phỏng mỗi giây (MCTS)
    root: list[RootStat]  # GUI và bảng T4 đọc trường này


class Agent(ABC):
    """Agent chọn một nước đi cho người đang tới lượt ở thế cờ được đưa vào."""

    name: str = "agent"

    def __init__(self) -> None:
        self.last_stats: AgentStats = {}

    @abstractmethod
    def select_move(self, state: GameState, budget: Budget) -> Move:
        """Trả về một nước hợp lệ cho state.current_player() trong giới hạn `budget`.

        Khi trả về, `state` phải y như lúc nhận vào (mọi play() đã được undo()),
        và `last_stats` đã được ghi mới cho lần gọi này.
        """
