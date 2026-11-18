"""Cờ tướng (Xiangqi): luật, sinh nước hợp lệ và perft.

Chủ sở hữu: TV1 (luật). TV2 viết agent, TV3 viết GUI và thực nghiệm trên module này.

Bàn cờ 10 hàng × 9 cột. Hàng 0 ở phía Đen (trên), hàng 9 ở phía Đỏ (dưới). Đỏ là `PLAYER_X`
(+1) và đi trước, Đen là `PLAYER_O` (-1). Mỗi ô trong `board` chứa `side * loại quân`, với loại
quân: Tướng 1, Sĩ 2, Tượng 3, Mã 4, Xe 5, Pháo 6, Tốt 7.

Một nước đi là cặp `(ô_đi, ô_đến)`, mỗi ô là chỉ số `hàng * 9 + cột` từ 0 đến 89. Cặp này vẫn
đúng kiểu `Move = tuple[int, int]` của interface chung. `move_to_iccs` và `parse_iccs` đổi
qua lại với ký hiệu ICCS (ví dụ "h2e2": cột a–i từ trái sang theo phía Đỏ, hàng 0–9 tính từ
phía Đỏ).

Luật kết thúc ván:
  - Bên tới lượt không còn nước hợp lệ thì thua, kể cả khi không bị chiếu (bị vây hết nước
    cũng thua, khác cờ vua).
  - Hòa khi `no_capture_limit` nước liên tiếp không có quân bị ăn (mặc định 120 nước, tức
    60 nước mỗi bên), khi một thế cờ lặp lại lần thứ ba, hoặc khi đạt `max_plies` nếu có đặt.
    Luật cấm chiếu mãi / đuổi mãi của cờ tướng thi đấu không được cài; lặp ba lần chỉ xử hòa.

Perft (`perft`, `perft_divide`) đếm số nút lá của cây nước hợp lệ đến độ sâu cho trước và bỏ
qua các luật hòa, đúng như cách đếm perft chuẩn. Thế ban đầu cho 44, 1 920, 79 666, 3 290 240
nút ở độ sâu 1–4. Chạy từ dòng lệnh:

    python -m src.games.xiangqi --perft 3
    python -m src.games.xiangqi --perft 2 --fen "<FEN>" --divide
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections.abc import Sequence

import numpy as np

from src.games.base import DRAW, PLAYER_O, PLAYER_X, GameState, Move

ROWS = 10
COLS = 9
SQUARES = ROWS * COLS

RED = PLAYER_X
BLACK = PLAYER_O

KING = 1
ADVISOR = 2
ELEPHANT = 3
HORSE = 4
ROOK = 5
CANNON = 6
PAWN = 7

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
DEFAULT_NO_CAPTURE_LIMIT = 120

# Chữ cái FEN của từng loại quân. Khi đọc FEN chấp nhận thêm E (Tượng) và H (Mã).
_PIECE_LETTERS = {KING: "k", ADVISOR: "a", ELEPHANT: "b", HORSE: "n", ROOK: "r", CANNON: "c"}
_PIECE_LETTERS[PAWN] = "p"
_LETTER_PIECES = {letter: kind for kind, letter in _PIECE_LETTERS.items()}
_LETTER_PIECES.update({"e": ELEPHANT, "h": HORSE})


def _sq(r: int, c: int) -> int:
    return r * COLS + c


def _on_board(r: int, c: int) -> bool:
    return 0 <= r < ROWS and 0 <= c < COLS


def _in_palace(r: int, c: int) -> bool:
    return 3 <= c <= 5 and (r <= 2 or r >= 7)


def _red_half(r: int) -> bool:
    return r >= 5


# ---------------------------------------------------------------------------
# Bảng tra dựng sẵn một lần cho mọi ô
# ---------------------------------------------------------------------------


def _build_rays() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Với mỗi ô: 4 tia (lên, xuống, trái, phải), mỗi tia là các ô theo thứ tự xa dần."""
    rays = []
    for s in range(SQUARES):
        r, c = divmod(s, COLS)
        per_sq = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ray = []
            rr, cc = r + dr, c + dc
            while _on_board(rr, cc):
                ray.append(_sq(rr, cc))
                rr, cc = rr + dr, cc + dc
            per_sq.append(tuple(ray))
        rays.append(tuple(per_sq))
    return tuple(rays)


def _build_horse() -> tuple[tuple[tuple[int, int], ...], ...]:
    """Với mỗi ô: các cặp (ô đến, ô chân mã). Mã bị cản khi ô chân mã có quân."""
    table = []
    for s in range(SQUARES):
        r, c = divmod(s, COLS)
        moves = []
        for dr, dc in ((-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2)):
            rr, cc = r + dr, c + dc
            if not _on_board(rr, cc):
                continue
            leg = _sq(r + dr // 2, c) if abs(dr) == 2 else _sq(r, c + dc // 2)
            moves.append((_sq(rr, cc), leg))
        table.append(tuple(moves))
    return tuple(table)


def _build_horse_attackers(
    horse: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Với mỗi ô t: các cặp (ô con mã, ô chân mã) mà con mã ở đó nhảy được tới t."""
    rev: list[list[tuple[int, int]]] = [[] for _ in range(SQUARES)]
    for h in range(SQUARES):
        for to, leg in horse[h]:
            rev[to].append((h, leg))
    return tuple(tuple(x) for x in rev)


def _build_elephant() -> tuple[tuple[tuple[int, int], ...], ...]:
    """Với mỗi ô: các cặp (ô đến, mắt tượng). Tượng không qua sông nên ô đến cùng nửa bàn."""
    table = []
    for s in range(SQUARES):
        r, c = divmod(s, COLS)
        moves = []
        for dr, dc in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
            rr, cc = r + dr, c + dc
            if _on_board(rr, cc) and _red_half(rr) == _red_half(r):
                moves.append((_sq(rr, cc), _sq(r + dr // 2, c + dc // 2)))
        table.append(tuple(moves))
    return tuple(table)


def _build_palace_steps(deltas: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    """Các bước đi một ô theo `deltas` mà vẫn nằm trong cung (dùng cho Tướng và Sĩ)."""
    table = []
    for s in range(SQUARES):
        r, c = divmod(s, COLS)
        steps = []
        if _in_palace(r, c):
            for dr, dc in deltas:
                rr, cc = r + dr, c + dc
                if _on_board(rr, cc) and _in_palace(rr, cc) and _red_half(rr) == _red_half(r):
                    steps.append(_sq(rr, cc))
        table.append(tuple(steps))
    return tuple(table)


def _build_pawn(side: int) -> tuple[tuple[int, ...], ...]:
    """Tốt đi thẳng một ô về phía đối phương; đã qua sông thì đi ngang thêm được."""
    forward = -1 if side == RED else 1
    table = []
    for s in range(SQUARES):
        r, c = divmod(s, COLS)
        steps = []
        if _on_board(r + forward, c):
            steps.append(_sq(r + forward, c))
        crossed = r <= 4 if side == RED else r >= 5
        if crossed:
            if c > 0:
                steps.append(_sq(r, c - 1))
            if c < COLS - 1:
                steps.append(_sq(r, c + 1))
        table.append(tuple(steps))
    return tuple(table)


RAYS = _build_rays()
HORSE_MOVES = _build_horse()
HORSE_ATTACKERS = _build_horse_attackers(HORSE_MOVES)
ELEPHANT_MOVES = _build_elephant()
ADVISOR_MOVES = _build_palace_steps(((-1, -1), (-1, 1), (1, -1), (1, 1)))
KING_MOVES = _build_palace_steps(((-1, 0), (1, 0), (0, -1), (0, 1)))
PAWN_MOVES = {RED: _build_pawn(RED), BLACK: _build_pawn(BLACK)}


def _build_zobrist() -> tuple[list[list[int]], int]:
    rng = random.Random(20261202)  # hằng cố định: hash giống nhau giữa các lần chạy
    table = [[rng.getrandbits(64) for _ in range(SQUARES)] for _ in range(2 * PAWN + 1)]
    return table, rng.getrandbits(64)


# ZOBRIST[piece + PAWN][ô], piece là mã có dấu trong khoảng -7..7.
ZOBRIST, ZOBRIST_RED_TO_MOVE = _build_zobrist()


# ---------------------------------------------------------------------------
# Ký hiệu ICCS
# ---------------------------------------------------------------------------


def square_name(s: int) -> str:
    """Tên ô theo ICCS: cột a–i, hàng 0–9 tính từ phía Đỏ. Ô 81 (hàng 9, cột 0) là "a0"."""
    if isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < SQUARES:
        raise ValueError(f"Ô phải là số nguyên từ 0 đến {SQUARES - 1}, nhận được {s!r}")
    r, c = divmod(s, COLS)
    return f"{chr(ord('a') + c)}{ROWS - 1 - r}"


def parse_square(name: str) -> int:
    name = name.strip().lower()
    if len(name) != 2 or not ("a" <= name[0] <= "i") or not name[1].isdigit():
        raise ValueError(f"Tên ô không hợp lệ: {name!r}")
    return _sq(ROWS - 1 - int(name[1]), ord(name[0]) - ord("a"))


def move_to_iccs(m: Move) -> str:
    return square_name(m[0]) + square_name(m[1])


def parse_iccs(text: str) -> Move:
    text = text.strip().lower().replace("-", "")
    if len(text) != 4:
        raise ValueError(f"Nước ICCS không hợp lệ: {text!r}")
    return parse_square(text[:2]), parse_square(text[2:])


# ---------------------------------------------------------------------------
# Thế cờ
# ---------------------------------------------------------------------------


class Xiangqi(GameState):
    """Thế cờ tướng, thay đổi tại chỗ bằng play() và undo()."""

    def __init__(
        self,
        fen: str = START_FEN,
        *,
        no_capture_limit: int | None = DEFAULT_NO_CAPTURE_LIMIT,
        max_plies: int | None = None,
    ) -> None:
        for name, value in (("no_capture_limit", no_capture_limit), ("max_plies", max_plies)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} phải là số nguyên >= 1 hoặc None, nhận được {value!r}")
        self.no_capture_limit = no_capture_limit
        self.max_plies = max_plies
        self._load_fen(fen)

    # -- Đọc / ghi FEN -------------------------------------------------------

    def _load_fen(self, fen: str) -> None:
        fields = fen.split()
        if not fields:
            raise ValueError("FEN rỗng")
        rows = fields[0].split("/")
        if len(rows) != ROWS:
            raise ValueError(f"FEN phải có {ROWS} hàng, nhận được {len(rows)}")
        sq = [0] * SQUARES
        kings: dict[int, int] = {}
        for r, row in enumerate(rows):
            c = 0
            for ch in row:
                if ch.isdigit():
                    c += int(ch)
                    continue
                kind = _LETTER_PIECES.get(ch.lower())
                if kind is None:
                    raise ValueError(f"Ký tự quân không hợp lệ trong FEN: {ch!r}")
                if c >= COLS:
                    raise ValueError(f"Hàng {r} của FEN dài quá {COLS} ô")
                side = RED if ch.isupper() else BLACK
                if kind == KING:
                    if side in kings:
                        raise ValueError("FEN có hai Tướng cùng màu")
                    if not _in_palace(r, c) or _red_half(r) != (side == RED):
                        raise ValueError(f"Tướng ở ô {square_name(_sq(r, c))} nằm ngoài cung")
                    kings[side] = _sq(r, c)
                sq[_sq(r, c)] = side * kind
                c += 1
            if c != COLS:
                raise ValueError(f"Hàng {r} của FEN có {c} ô, cần {COLS}")
        if len(kings) != 2:
            raise ValueError("FEN phải có đúng một Tướng mỗi bên")

        turn = fields[1].lower() if len(fields) > 1 else "w"
        if turn in ("w", "r"):
            player = RED
        elif turn == "b":
            player = BLACK
        else:
            raise ValueError(f"Lượt đi trong FEN phải là w/r hoặc b, nhận được {turn!r}")
        no_capture = 0
        if len(fields) > 4:
            if not fields[4].isdigit():
                raise ValueError(f"Số nước không ăn quân trong FEN phải >= 0: {fields[4]!r}")
            no_capture = int(fields[4])

        self._sq = sq
        self._king = kings
        self._player = player
        self._no_capture = no_capture
        self._history: list[tuple[Move, int, int]] = []
        self.board = np.array(sq, dtype=np.int8).reshape(ROWS, COLS)
        self._hash = self._compute_hash()
        self._repetitions: dict[int, int] = {self._hash: 1}
        self._legal_cache: list[Move] | None = None
        self._winner_known = False
        self._winner: int | None = None

    def to_fen(self) -> str:
        rows = []
        for r in range(ROWS):
            out, empty = [], 0
            for c in range(COLS):
                p = self._sq[_sq(r, c)]
                if p == 0:
                    empty += 1
                    continue
                if empty:
                    out.append(str(empty))
                    empty = 0
                letter = _PIECE_LETTERS[abs(p)]
                out.append(letter.upper() if p > 0 else letter)
            if empty:
                out.append(str(empty))
            rows.append("".join(out))
        turn = "w" if self._player == RED else "b"
        full_moves = 1 + len(self._history) // 2
        return f"{'/'.join(rows)} {turn} - - {self._no_capture} {full_moves}"

    def _compute_hash(self) -> int:
        h = ZOBRIST_RED_TO_MOVE if self._player == RED else 0
        for s, p in enumerate(self._sq):
            if p:
                h ^= ZOBRIST[p + PAWN][s]
        return h

    # -- Sinh nước -----------------------------------------------------------

    def _pseudo_moves(self, side: int) -> list[Move]:
        """Mọi nước đi đúng cách đi của quân, chưa xét việc để Tướng mình bị chiếu."""
        sq = self._sq
        out: list[Move] = []
        append = out.append
        for s in range(SQUARES):
            p = sq[s]
            if p * side <= 0:
                continue
            kind = p if p > 0 else -p
            if kind == ROOK:
                for ray in RAYS[s]:
                    for x in ray:
                        q = sq[x]
                        if q == 0:
                            append((s, x))
                        else:
                            if q * side < 0:
                                append((s, x))
                            break
            elif kind == CANNON:
                for ray in RAYS[s]:
                    screen = False
                    for x in ray:
                        q = sq[x]
                        if not screen:
                            if q == 0:
                                append((s, x))
                            else:
                                screen = True
                        elif q != 0:
                            if q * side < 0:
                                append((s, x))
                            break
            elif kind == HORSE:
                for x, leg in HORSE_MOVES[s]:
                    if sq[leg] == 0 and sq[x] * side <= 0:
                        append((s, x))
            elif kind == ELEPHANT:
                for x, eye in ELEPHANT_MOVES[s]:
                    if sq[eye] == 0 and sq[x] * side <= 0:
                        append((s, x))
            elif kind == PAWN:
                for x in PAWN_MOVES[side][s]:
                    if sq[x] * side <= 0:
                        append((s, x))
            else:
                steps = KING_MOVES[s] if kind == KING else ADVISOR_MOVES[s]
                for x in steps:
                    if sq[x] * side <= 0:
                        append((s, x))
        return out

    def _square_attacked(self, t: int, by: int) -> bool:
        """True nếu quân của bên `by` ăn được ô `t` (gồm cả luật hai Tướng đối mặt).

        Chỉ dùng cho ô Tướng: Sĩ và Tượng không rời được nửa bàn của mình nên không bao giờ
        chiếu được Tướng đối phương, và Tướng chỉ "ăn" Tướng kia khi hai Tướng đối mặt.
        """
        sq = self._sq
        rook, cannon, king = by * ROOK, by * CANNON, by * KING
        for i, ray in enumerate(RAYS[t]):
            n = len(ray)
            j = 0
            while j < n and sq[ray[j]] == 0:
                j += 1
            if j == n:
                continue
            q = sq[ray[j]]
            if q == rook or (i < 2 and q == king):
                return True
            j += 1
            while j < n and sq[ray[j]] == 0:
                j += 1
            if j < n and sq[ray[j]] == cannon:
                return True
        horse = by * HORSE
        for h, leg in HORSE_ATTACKERS[t]:
            if sq[h] == horse and sq[leg] == 0:
                return True
        pawn = by * PAWN
        r, c = divmod(t, COLS)
        if by == RED:
            if r < ROWS - 1 and sq[t + COLS] == pawn:
                return True
            crossed = r <= 4
        else:
            if r > 0 and sq[t - COLS] == pawn:
                return True
            crossed = r >= 5
        if crossed:
            if c > 0 and sq[t - 1] == pawn:
                return True
            if c < COLS - 1 and sq[t + 1] == pawn:
                return True
        return False

    def in_check(self, side: int | None = None) -> bool:
        """True nếu Tướng của `side` (mặc định bên tới lượt) đang bị chiếu."""
        side = self._player if side is None else side
        return self._square_attacked(self._king[side], -side)

    def _legal_raw(self) -> list[Move]:
        """Các nước hợp lệ theo luật đi quân, không xét luật hòa hay ván đã kết thúc."""
        side = self._player
        sq = self._sq
        king_sq = self._king[side]
        own_king = side * KING
        legal: list[Move] = []
        for f, t in self._pseudo_moves(side):
            p = sq[f]
            cap = sq[t]
            sq[t] = p
            sq[f] = 0
            if not self._square_attacked(t if p == own_king else king_sq, -side):
                legal.append((f, t))
            sq[f] = p
            sq[t] = cap
        return legal

    def _legal_cached(self) -> list[Move]:
        if self._legal_cache is None:
            self._legal_cache = self._legal_raw()
        return self._legal_cache

    # -- Interface GameState -------------------------------------------------

    def legal_moves(self) -> list[Move]:
        if self.winner() is not None:
            return []
        return list(self._legal_cached())

    def winner(self) -> int | None:
        if not self._winner_known:
            self._winner = self._compute_winner()
            self._winner_known = True
        return self._winner

    def _compute_winner(self) -> int | None:
        if not self._legal_cached():
            return -self._player
        if self.max_plies is not None and len(self._history) >= self.max_plies:
            return DRAW
        if self.no_capture_limit is not None and self._no_capture >= self.no_capture_limit:
            return DRAW
        if self._repetitions.get(self._hash, 0) >= 3:
            return DRAW
        return None

    def play(self, m: Move) -> None:
        if self.winner() is not None:
            raise ValueError("Ván đã kết thúc")
        m = (int(m[0]), int(m[1]))
        if m not in self._legal_cached():
            raise ValueError(f"Nước {m} ({self._safe_iccs(m)}) không hợp lệ")
        f, t = m
        sq = self._sq
        p = sq[f]
        cap = sq[t]
        h = self._hash ^ ZOBRIST[p + PAWN][f] ^ ZOBRIST[p + PAWN][t] ^ ZOBRIST_RED_TO_MOVE
        if cap:
            h ^= ZOBRIST[cap + PAWN][t]
        sq[t] = p
        sq[f] = 0
        board = self.board.reshape(-1)
        board[t] = p
        board[f] = 0
        if p == self._player * KING:
            self._king[self._player] = t
        self._history.append((m, cap, self._no_capture))
        self._no_capture = 0 if cap else self._no_capture + 1
        self._player = -self._player
        self._hash = h
        self._repetitions[h] = self._repetitions.get(h, 0) + 1
        self._legal_cache = None
        self._winner_known = False

    def undo(self) -> None:
        if not self._history:
            raise ValueError("Chưa có nước nào để hoàn tác")
        (f, t), cap, prev_no_capture = self._history.pop()
        count = self._repetitions[self._hash] - 1
        if count:
            self._repetitions[self._hash] = count
        else:
            del self._repetitions[self._hash]
        self._player = -self._player
        sq = self._sq
        p = sq[t]
        sq[f] = p
        sq[t] = cap
        board = self.board.reshape(-1)
        board[f] = p
        board[t] = cap
        if p == self._player * KING:
            self._king[self._player] = f
        self._no_capture = prev_no_capture
        h = self._hash ^ ZOBRIST[p + PAWN][f] ^ ZOBRIST[p + PAWN][t] ^ ZOBRIST_RED_TO_MOVE
        if cap:
            h ^= ZOBRIST[cap + PAWN][t]
        self._hash = h
        self._legal_cache = None
        self._winner_known = False

    def current_player(self) -> int:
        return self._player

    def last_move(self) -> Move | None:
        return self._history[-1][0] if self._history else None

    def move_count(self) -> int:
        return len(self._history)

    def hash(self) -> int:
        return self._hash

    def clone(self) -> Xiangqi:
        other = Xiangqi.__new__(Xiangqi)
        other.no_capture_limit = self.no_capture_limit
        other.max_plies = self.max_plies
        other._sq = list(self._sq)
        other._king = dict(self._king)
        other._player = self._player
        other._no_capture = self._no_capture
        other._history = list(self._history)
        other.board = self.board.copy()
        other._hash = self._hash
        other._repetitions = dict(self._repetitions)
        other._legal_cache = None if self._legal_cache is None else list(self._legal_cache)
        other._winner_known = self._winner_known
        other._winner = self._winner
        return other

    # -- Tiện ích ------------------------------------------------------------

    def piece_at(self, s: int) -> int:
        """Mã quân có dấu tại ô `s` (0 nếu trống)."""
        return self._sq[s]

    def _safe_iccs(self, m: Move) -> str:
        try:
            return move_to_iccs(m)
        except (TypeError, ValueError):
            return "?"

    def __str__(self) -> str:
        lines = []
        for r in range(ROWS):
            cells = []
            for c in range(COLS):
                p = self._sq[_sq(r, c)]
                if p == 0:
                    cells.append(".")
                else:
                    letter = _PIECE_LETTERS[abs(p)]
                    cells.append(letter.upper() if p > 0 else letter)
            lines.append(f"{ROWS - 1 - r} " + " ".join(cells))
        lines.append("  " + " ".join(chr(ord("a") + c) for c in range(COLS)))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Perft
# ---------------------------------------------------------------------------


def _perft_rec(state: Xiangqi, depth: int) -> int:
    moves = state._legal_raw()
    if depth == 1:
        return len(moves)
    sq = state._sq
    kings = state._king
    side = state._player
    own_king = side * KING
    total = 0
    for f, t in moves:
        p = sq[f]
        cap = sq[t]
        sq[t] = p
        sq[f] = 0
        if p == own_king:
            kings[side] = t
        state._player = -side
        total += _perft_rec(state, depth - 1)
        state._player = side
        if p == own_king:
            kings[side] = f
        sq[f] = p
        sq[t] = cap
    return total


def perft(state: Xiangqi, depth: int) -> int:
    """Số nút lá của cây nước hợp lệ sâu `depth` từ `state` (bỏ qua luật hòa).

    Hàm đi và hoàn tác nước trực tiếp trên mảng quân để nhanh hơn play()/undo(); `state`
    được trả về nguyên trạng khi hàm kết thúc.
    """
    if depth < 0:
        raise ValueError(f"depth phải >= 0, nhận được {depth}")
    if depth == 0:
        return 1
    return _perft_rec(state, depth)


def perft_divide(state: Xiangqi, depth: int) -> dict[str, int]:
    """Perft tách theo từng nước đầu, khóa là ký hiệu ICCS (dùng để dò chỗ sai khi đối chiếu)."""
    if depth < 1:
        raise ValueError(f"depth phải >= 1, nhận được {depth}")
    sq = state._sq
    kings = state._king
    side = state._player
    own_king = side * KING
    result: dict[str, int] = {}
    for f, t in state._legal_raw():
        p = sq[f]
        cap = sq[t]
        sq[t] = p
        sq[f] = 0
        if p == own_king:
            kings[side] = t
        state._player = -side
        result[move_to_iccs((f, t))] = perft(state, depth - 1)
        state._player = side
        if p == own_king:
            kings[side] = f
        sq[f] = p
        sq[t] = cap
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Perft cờ tướng")
    parser.add_argument("--perft", type=int, default=3, help="độ sâu (mặc định 3)")
    parser.add_argument("--fen", default=START_FEN, help="thế cờ dạng FEN (mặc định: thế đầu)")
    parser.add_argument("--divide", action="store_true", help="in số nút theo từng nước đầu")
    args = parser.parse_args(argv)

    state = Xiangqi(args.fen)
    print(state)
    if args.divide:
        for name, count in sorted(perft_divide(state, args.perft).items()):
            print(f"{name}: {count}")
    for depth in range(1, args.perft + 1):
        t0 = time.perf_counter()
        nodes = perft(state, depth)
        elapsed = time.perf_counter() - t0
        print(f"perft({depth}) = {nodes}  ({elapsed:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
