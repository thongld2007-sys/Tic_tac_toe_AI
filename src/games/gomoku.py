"""Bàn cờ Caro (Gomoku) kích thước tham số n×n (P1.2).

Chủ sở hữu: TV2.  Review: TV3.

Đặc điểm:
  - Kích thước bàn `size × size` tham số hóa (mặc định 15×15, tối thiểu bằng `win_length`).
  - Luật thắng: tạo được dãy từ `win_length` (mặc định 5) quân liên tiếp trở lên theo
    1 trong 4 hướng (ngang, dọc, chéo chính, chéo phụ).
  - Cờ chặn hai đầu (`block_two_ends=True`): dãy >= 5 quân bị quân đối thủ chặn sát ở cả
    hai đầu thì không được tính là thắng (biên bàn cờ không tính là chặn).
  - Băm Zobrist (`hash()`) cập nhật O(1) khi `play()` và khôi phục chính xác khi `undo()`.
  - `candidate_moves(radius=2)`: trả về ô tâm bàn khi bàn trống, hoặc các ô trống nằm trong
    khoảng cách Chebyshev <= `radius` tính từ bất kỳ quân nào đã đánh.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from src.games.base import DRAW, EMPTY, PLAYER_O, PLAYER_X, Board, GameState, Move

DEFAULT_SIZE = 15
DEFAULT_WIN_LENGTH = 5
_ZOBRIST_SEED = 2026_11_12

# 4 hướng kiểm tra thắng: ngang, dọc, chéo chính, chéo phụ.
_DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))


@cache
def _zobrist_keys(size: int) -> tuple[np.ndarray, int]:
    """Sinh bảng khóa Zobrist 64-bit tất định cho bàn `size × size`.

    Trả về `(table, turn_key)`:
      - `table`: mảng shape `(2, size, size)` kiểu uint64 cho PLAYER_X (chỉ số 0)
        và PLAYER_O (chỉ số 1).
      - `turn_key`: khóa uint64 đổi trạng thái mỗi khi chuyển lượt.
    """
    rng = np.random.default_rng(_ZOBRIST_SEED + size)
    info = np.iinfo(np.int64)
    table = rng.integers(1, info.max, size=(2, size, size), dtype=np.int64)
    turn_key = int(rng.integers(1, info.max, dtype=np.int64))
    return table, turn_key


def _player_idx(player: int) -> int:
    return 0 if player == PLAYER_X else 1


class Gomoku(GameState):
    """Trạng thái bàn cờ Caro `size × size`."""

    def __init__(
        self,
        size: int = DEFAULT_SIZE,
        win_length: int = DEFAULT_WIN_LENGTH,
        block_two_ends: bool = False,
    ) -> None:
        if win_length < 3:
            raise ValueError(f"win_length phải >= 3, nhận được {win_length}")
        if size < win_length:
            raise ValueError(f"size ({size}) phải >= win_length ({win_length})")

        self.size = size
        self.win_length = win_length
        self.block_two_ends = block_two_ends
        self.board: Board = np.zeros((size, size), dtype=np.int8)

        self._player = PLAYER_X
        self._history: list[Move] = []
        self._winner: int | None = None

        self._zobrist_table, self._turn_key = _zobrist_keys(size)
        self._hash = 0

    def legal_moves(self) -> list[Move]:
        if self._winner is not None:
            return []
        rows, cols = np.nonzero(self.board == EMPTY)
        return [(int(r), int(c)) for r, c in zip(rows, cols, strict=True)]

    def candidate_moves(self, radius: int = 2) -> list[Move]:
        """Các ô trống trong bán kính Chebyshev `radius` quanh các quân đã đánh.

        Khi bàn trống, trả về duy nhất ô chính giữa bàn `(size // 2, size // 2)`.
        Khi ván đã kết thúc, trả về danh sách rỗng.
        """
        if radius < 1:
            raise ValueError(f"radius phải >= 1, nhận được {radius}")
        if self._winner is not None:
            return []
        if not self._history:
            center = self.size // 2
            return [(center, center)]

        n = self.size
        mask = np.zeros((n, n), dtype=bool)
        for sr, sc in self._history:
            r0 = max(0, sr - radius)
            r1 = min(n, sr + radius + 1)
            c0 = max(0, sc - radius)
            c1 = min(n, sc + radius + 1)
            mask[r0:r1, c0:c1] = True

        mask &= self.board == EMPTY
        rows, cols = np.nonzero(mask)
        return [(int(r), int(c)) for r, c in zip(rows, cols, strict=True)]

    def _checks_win_at(self, r: int, c: int, player: int) -> bool:
        """Kiểm tra nước vừa đánh tại `(r, c)` có giúp `player` thắng hay không."""
        n = self.size
        board = self.board
        opponent = -player

        for dr, dc in _DIRECTIONS:
            back = 0
            while (
                0 <= r - (back + 1) * dr < n
                and 0 <= c - (back + 1) * dc < n
                and board[r - (back + 1) * dr, c - (back + 1) * dc] == player
            ):
                back += 1

            fwd = 0
            while (
                0 <= r + (fwd + 1) * dr < n
                and 0 <= c + (fwd + 1) * dc < n
                and board[r + (fwd + 1) * dr, c + (fwd + 1) * dc] == player
            ):
                fwd += 1

            length = 1 + back + fwd
            if length < self.win_length:
                continue

            if not self.block_two_ends:
                return True

            br, bc = r - (back + 1) * dr, c - (back + 1) * dc
            fr, fc = r + (fwd + 1) * dr, c + (fwd + 1) * dc
            left_blocked = 0 <= br < n and 0 <= bc < n and board[br, bc] == opponent
            right_blocked = 0 <= fr < n and 0 <= fc < n and board[fr, fc] == opponent
            if not (left_blocked and right_blocked):
                return True

        return False

    def play(self, m: Move) -> None:
        r, c = m
        if self._winner is not None:
            raise ValueError("Ván đã kết thúc")
        if not (0 <= r < self.size and 0 <= c < self.size):
            raise ValueError(f"Nước {m} nằm ngoài bàn {self.size}×{self.size}")
        if self.board[r, c] != EMPTY:
            raise ValueError(f"Ô {m} đã có quân")

        player = self._player
        self.board[r, c] = player
        self._history.append((r, c))
        stone_key = int(self._zobrist_table[_player_idx(player), r, c])
        self._hash ^= stone_key ^ self._turn_key
        self._player = -player

        if self._checks_win_at(r, c, player):
            self._winner = player
        elif len(self._history) == self.size * self.size:
            self._winner = DRAW

    def undo(self) -> None:
        if not self._history:
            raise ValueError("Chưa có nước nào để hoàn tác")
        r, c = self._history.pop()
        self._player = -self._player
        stone_key = int(self._zobrist_table[_player_idx(self._player), r, c])
        self._hash ^= stone_key ^ self._turn_key
        self.board[r, c] = EMPTY
        self._winner = None

    def winner(self) -> int | None:
        return self._winner

    def current_player(self) -> int:
        return self._player

    def last_move(self) -> Move | None:
        return self._history[-1] if self._history else None

    def move_count(self) -> int:
        return len(self._history)

    def hash(self) -> int:
        return self._hash

    def clone(self) -> Gomoku:
        other = Gomoku(
            size=self.size,
            win_length=self.win_length,
            block_two_ends=self.block_two_ends,
        )
        other.board = self.board.copy()
        other._player = self._player
        other._history = list(self._history)
        other._winner = self._winner
        other._hash = self._hash
        return other

    def __str__(self) -> str:
        symbols = {EMPTY: ".", PLAYER_X: "X", PLAYER_O: "O"}
        return "\n".join(" ".join(symbols[int(v)] for v in row) for row in self.board)
