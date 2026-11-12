"""Bàn cờ Caro (Gomoku) kích thước tham số n×n (P1.2).

Chủ sở hữu: TV2.  Review: TV3.

Đặc điểm:
  - Kích thước bàn `size × size` tham số hóa (mặc định 15×15, tối thiểu bằng `win_length`).
  - Luật thắng: tạo được dãy từ `win_length` (mặc định 5) quân liên tiếp trở lên theo
    1 trong 4 hướng (ngang, dọc, chéo chính, chéo phụ).
    Lưu ý: `src/eval/patterns.py` được thiết kế cố định cho luật chuẩn `win_length=5`.
  - Cờ chặn hai đầu (`block_two_ends=True`): dãy >= `win_length` quân bị quân đối thủ chặn
    sát ở cả hai đầu thì không được tính là thắng (biên bàn cờ không tính là chặn).
  - Băm Zobrist (`hash()`) cập nhật O(1) khi `play()` và khôi phục chính xác khi `undo()`.
  - `candidate_moves(radius=2)`: trả về ô tâm bàn khi bàn trống, hoặc các ô trống nằm trong
    khoảng cách Chebyshev <= `radius` tính từ bất kỳ quân nào đã đánh. Với `radius=2` (mặc định),
    tập ứng viên được cập nhật tăng dần (incremental) trong O(1) khi `play()` và `undo()`.
"""

from __future__ import annotations

from functools import cache
from itertools import compress

import numpy as np

from src.games.base import DRAW, EMPTY, PLAYER_O, PLAYER_X, Board, GameState, Move

DEFAULT_SIZE = 15
DEFAULT_WIN_LENGTH = 5
_ZOBRIST_SEED = 2026_11_12

# 4 hướng kiểm tra thắng: ngang, dọc, chéo chính, chéo phụ.
_DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))


@cache
def _zobrist_keys(size: int) -> tuple[np.ndarray, int]:
    """Sinh bảng khóa Zobrist int64 dương (63-bit) tất định cho bàn `size × size`.

    Trả về `(table, turn_key)`:
      - `table`: mảng shape `(2, size, size)` kiểu int64 cho PLAYER_X (chỉ số 0)
        và PLAYER_O (chỉ số 1).
      - `turn_key`: khóa int64 đổi trạng thái mỗi khi chuyển lượt.
    """
    rng = np.random.default_rng(_ZOBRIST_SEED + size)
    info = np.iinfo(np.int64)
    table = rng.integers(1, info.max, size=(2, size, size), dtype=np.int64, endpoint=True)
    turn_key = int(rng.integers(1, info.max, dtype=np.int64, endpoint=True))
    return table, turn_key


@cache
def _all_moves(size: int) -> tuple[Move, ...]:
    """Danh sách mọi tọa độ `(r, c)` trên bàn `size × size` theo thứ tự hàng-cột."""
    return tuple((r, c) for r in range(size) for c in range(size))


@cache
def _neighbors(size: int, radius: int) -> tuple[tuple[int, ...], ...]:
    """Với mỗi ô phẳng `r * size + c`, trả về tuple các chỉ số phẳng lân cận trong `radius`."""
    result: list[tuple[int, ...]] = []
    for r in range(size):
        r0 = max(0, r - radius)
        r1 = min(size, r + radius + 1)
        for c in range(size):
            c0 = max(0, c - radius)
            c1 = min(size, c + radius + 1)
            nbrs = tuple(
                nr * size + nc for nr in range(r0, r1) for nc in range(c0, c1) if nr != r or nc != c
            )
            result.append(nbrs)
    return tuple(result)


def _player_idx(player: int) -> int:
    return 0 if player == PLAYER_X else 1


def _normalize_move(m: Move, size: int) -> Move:
    """Kiểm tra và chuẩn hóa tọa độ nước đi về kiểu `(int, int)` chuẩn của Python."""
    try:
        r_raw, c_raw = m
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Nước đi không hợp lệ: {m!r}") from exc

    if (
        isinstance(r_raw, bool)
        or isinstance(c_raw, bool)
        or not isinstance(r_raw, (int, np.integer))
        or not isinstance(c_raw, (int, np.integer))
    ):
        raise ValueError(f"Tọa độ nước đi phải là số nguyên, nhận được {m!r}")

    r, c = int(r_raw), int(c_raw)
    if not (0 <= r < size and 0 <= c < size):
        raise ValueError(f"Nước {(r, c)} nằm ngoài bàn {size}×{size}")
    return r, c


class Gomoku(GameState):
    """Trạng thái bàn cờ Caro `size × size`.

    Lưu ý: Hàm lượng giá mẫu trong `src/eval/patterns.py` cố định 5 quân liên tiếp,
    nên khi dùng chung với module đó cần giữ `win_length=5`.
    """

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

        n_cells = size * size
        self._all_moves = _all_moves(size)
        self._neighbors_r2 = _neighbors(size, 2)
        self._empty_mask: list[int] = [1] * n_cells
        self._nbr_r2: list[int] = [0] * n_cells
        self._cand_r2: list[int] = [0] * n_cells

    def legal_moves(self) -> list[Move]:
        if self._winner is not None:
            return []
        return list(compress(self._all_moves, self._empty_mask))

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
        if radius == 2:
            return list(compress(self._all_moves, self._cand_r2))

        n = self.size
        nbr_table = _neighbors(n, radius)
        empty = self._empty_mask
        flags = [0] * (n * n)
        for sr, sc in self._history:
            for n_idx in nbr_table[sr * n + sc]:
                if empty[n_idx]:
                    flags[n_idx] = 1
        return list(compress(self._all_moves, flags))

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
        if self._winner is not None:
            raise ValueError("Ván đã kết thúc")
        r, c = _normalize_move(m, self.size)
        if self.board[r, c] != EMPTY:
            raise ValueError(f"Ô {(r, c)} đã có quân")

        player = self._player
        self.board[r, c] = player
        self._history.append((r, c))
        stone_key = int(self._zobrist_table[_player_idx(player), r, c])
        self._hash ^= stone_key ^ self._turn_key
        self._player = -player

        idx = r * self.size + c
        self._empty_mask[idx] = 0
        self._cand_r2[idx] = 0
        empty = self._empty_mask
        nbr_r2 = self._nbr_r2
        cand_r2 = self._cand_r2
        for n_idx in self._neighbors_r2[idx]:
            nbr_r2[n_idx] += 1
            if empty[n_idx]:
                cand_r2[n_idx] = 1

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

        idx = r * self.size + c
        self._empty_mask[idx] = 1
        nbr_r2 = self._nbr_r2
        cand_r2 = self._cand_r2
        cand_r2[idx] = 1 if nbr_r2[idx] else 0
        for n_idx in self._neighbors_r2[idx]:
            cnt = nbr_r2[n_idx] - 1
            nbr_r2[n_idx] = cnt
            if not cnt:
                cand_r2[n_idx] = 0

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
        other = Gomoku.__new__(Gomoku)
        other.size = self.size
        other.win_length = self.win_length
        other.block_two_ends = self.block_two_ends
        other.board = self.board.copy()
        other._player = self._player
        other._history = list(self._history)
        other._winner = self._winner
        other._zobrist_table = self._zobrist_table
        other._turn_key = self._turn_key
        other._hash = self._hash
        other._all_moves = self._all_moves
        other._neighbors_r2 = self._neighbors_r2
        other._empty_mask = list(self._empty_mask)
        other._nbr_r2 = list(self._nbr_r2)
        other._cand_r2 = list(self._cand_r2)
        return other

    def __str__(self) -> str:
        symbols = {EMPTY: ".", PLAYER_X: "X", PLAYER_O: "O"}
        return "\n".join(" ".join(symbols[int(v)] for v in row) for row in self.board)
