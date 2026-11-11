"""Tic-Tac-Toe 3×3."""

from __future__ import annotations

import numpy as np

from src.games.base import DRAW, EMPTY, PLAYER_X, GameState, Move

SIZE = 3

# 8 đường thắng: 3 hàng, 3 cột, 2 đường chéo.
LINES: tuple[tuple[Move, Move, Move], ...] = (
    *(tuple((r, c) for c in range(SIZE)) for r in range(SIZE)),
    *(tuple((r, c) for r in range(SIZE)) for c in range(SIZE)),
    tuple((i, i) for i in range(SIZE)),
    tuple((i, SIZE - 1 - i) for i in range(SIZE)),
)

# Với mỗi ô, các đường thắng đi qua ô đó: sau một nước chỉ cần kiểm tra các đường này.
_LINES_THROUGH: dict[Move, tuple[tuple[Move, Move, Move], ...]] = {
    (r, c): tuple(line for line in LINES if (r, c) in line)
    for r in range(SIZE)
    for c in range(SIZE)
}


def _cell_key(m: Move, player: int) -> int:
    """Phần đóng góp của một quân vào hash: mã hóa bàn cờ theo cơ số 3."""
    digit = 1 if player == PLAYER_X else 2
    return digit * 3 ** (m[0] * SIZE + m[1])


class TicTacToe(GameState):
    def __init__(self) -> None:
        self.board = np.zeros((SIZE, SIZE), dtype=np.int8)
        self._player = PLAYER_X
        self._history: list[Move] = []
        self._winner: int | None = None
        self._board_key = 0

    def legal_moves(self) -> list[Move]:
        if self._winner is not None:
            return []
        return [(r, c) for r in range(SIZE) for c in range(SIZE) if self.board[r, c] == EMPTY]

    def play(self, m: Move) -> None:
        r, c = m
        if self._winner is not None:
            raise ValueError("Ván đã kết thúc")
        if not (0 <= r < SIZE and 0 <= c < SIZE):
            raise ValueError(f"Nước {m} nằm ngoài bàn")
        if self.board[r, c] != EMPTY:
            raise ValueError(f"Ô {m} đã có quân")

        player = self._player
        self.board[r, c] = player
        self._history.append((r, c))
        self._board_key += _cell_key((r, c), player)
        self._player = -player

        if any(all(self.board[cell] == player for cell in line) for line in _LINES_THROUGH[r, c]):
            self._winner = player
        elif len(self._history) == SIZE * SIZE:
            self._winner = DRAW

    def undo(self) -> None:
        if not self._history:
            raise ValueError("Chưa có nước nào để hoàn tác")
        m = self._history.pop()
        self._player = -self._player
        self._board_key -= _cell_key(m, self._player)
        self.board[m] = EMPTY
        # play() từ chối đi tiếp khi ván đã xong, nên thế trước nước cuối luôn chưa kết thúc.
        self._winner = None

    def winner(self) -> int | None:
        return self._winner

    def current_player(self) -> int:
        return self._player

    def hash(self) -> int:
        return self._board_key * 2 + (self._player == PLAYER_X)

    def clone(self) -> TicTacToe:
        other = TicTacToe()
        other.board = self.board.copy()
        other._player = self._player
        other._history = list(self._history)
        other._winner = self._winner
        other._board_key = self._board_key
        return other

    def __str__(self) -> str:
        symbols = {EMPTY: ".", PLAYER_X: "X", -PLAYER_X: "O"}
        return "\n".join(" ".join(symbols[int(v)] for v in row) for row in self.board)
