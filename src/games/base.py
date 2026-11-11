"""Interface chung cho trò chơi hai người, tổng bằng không, đi luân phiên.

Chủ sở hữu: TV1. Sau mốc M1, mọi thay đổi ở file này phải có đủ 2 người còn lại duyệt.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

Board = np.ndarray  # shape (n, n), dtype int8: 0 = trống, 1 = X, -1 = O
Move = tuple[int, int]  # (hàng, cột), đánh số từ 0

EMPTY = 0
PLAYER_X = 1  # X luôn đi trước
PLAYER_O = -1
DRAW = 0  # giá trị của winner() khi ván hòa


class GameState(ABC):
    """Một thế cờ có thể thay đổi tại chỗ bằng play() và undo().

    Người chơi được mã hóa bằng 1 (X) và -1 (O), nên đổi lượt là nhân với -1 và
    giá trị theo góc nhìn của đối thủ là giá trị của mình đổi dấu.
    """

    board: Board

    @abstractmethod
    def legal_moves(self) -> list[Move]:
        """Mọi nước hợp lệ ở thế hiện tại; rỗng khi ván đã kết thúc."""

    def candidate_moves(self, radius: int = 2) -> list[Move]:
        """Các nước đáng xét: ô trống cách một quân đã đánh không quá `radius` ô.

        Mặc định trả về legal_moves() (đúng cho 3×3). Caro ghi đè để thu hẹp tập nước.
        """
        return self.legal_moves()

    @abstractmethod
    def play(self, m: Move) -> None:
        """Người đang tới lượt đánh nước `m`, sau đó đổi lượt. `m` phải hợp lệ."""

    @abstractmethod
    def undo(self) -> None:
        """Hoàn tác nước play() gần nhất, khôi phục cả bàn cờ, lượt đi và hash."""

    def is_terminal(self) -> bool:
        """True khi ván đã kết thúc (có người thắng hoặc hòa)."""
        return self.winner() is not None

    @abstractmethod
    def winner(self) -> int | None:
        """1 nếu X thắng, -1 nếu O thắng, 0 nếu hòa, None nếu ván chưa kết thúc."""

    @abstractmethod
    def current_player(self) -> int:
        """Người sắp đi: 1 (X) hoặc -1 (O)."""

    @abstractmethod
    def last_move(self) -> Move | None:
        """Nước vừa đánh gần nhất; None khi chưa có nước nào hoặc đã undo() hết."""

    @abstractmethod
    def move_count(self) -> int:
        """Số nước đã đánh kể từ thế ban đầu; undo() làm giảm đi 1."""

    @abstractmethod
    def hash(self) -> int:
        """Khóa của thế cờ, tính cả lượt đi. Hai thế giống nhau phải cho cùng một khóa."""

    @abstractmethod
    def clone(self) -> GameState:
        """Bản sao độc lập: play()/undo() trên bản sao không ảnh hưởng thế gốc."""
