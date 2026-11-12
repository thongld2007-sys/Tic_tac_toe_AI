"""Hàm đếm mẫu caro và hàm lượng giá (P1.3a).

Chủ sở hữu: TV3.  Review: TV1.

Mô tả các mẫu cần đếm (theo góc nhìn của `player`):
  - FIVE        : 5 quân liên tiếp → thắng ngay
  - OPEN_FOUR   : _XXXX_ (cả hai đầu trống) → đe doạ kép, thắng gần chắc
  - FOUR        : XXXX_ hoặc _XXXX (một đầu bị chặn) → phải chặn
  - OPEN_THREE  : _XXX_ (cả hai đầu trống, có thể mở thành OPEN_FOUR)
  - THREE       : _XXX (một đầu bị chặn)
  - OPEN_TWO   : _XX_ (cả hai đầu trống)
  - TWO         : XX_ hoặc _XX (một đầu bị chặn)

Hàm chính:
  count_patterns(board, player) -> PatternCount
  evaluate(board, player) -> float          # điểm theo góc nhìn của `player`
"""

from __future__ import annotations

from dataclasses import dataclass

from src.games.base import Board

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

FIVE = "five"
OPEN_FOUR = "open_four"
FOUR = "four"
OPEN_THREE = "open_three"
THREE = "three"
OPEN_TWO = "open_two"
TWO = "two"

# Trọng số cho từng mẫu (theo góc nhìn của người được đánh giá)
_WEIGHTS: dict[str, float] = {
    FIVE: 1_000_000,
    OPEN_FOUR: 50_000,
    FOUR: 10_000,
    OPEN_THREE: 1_000,
    THREE: 200,
    OPEN_TWO: 50,
    TWO: 10,
}

# 4 hướng: (dr, dc)
_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


# ---------------------------------------------------------------------------
# Kiểu dữ liệu kết quả
# ---------------------------------------------------------------------------


@dataclass
class PatternCount:
    """Số lần xuất hiện của mỗi mẫu trên toàn bàn, theo góc nhìn của một người chơi."""

    five: int = 0
    open_four: int = 0
    four: int = 0
    open_three: int = 0
    three: int = 0
    open_two: int = 0
    two: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            FIVE: self.five,
            OPEN_FOUR: self.open_four,
            FOUR: self.four,
            OPEN_THREE: self.open_three,
            THREE: self.three,
            OPEN_TWO: self.open_two,
            TWO: self.two,
        }


# ---------------------------------------------------------------------------
# Hàm nội bộ
# ---------------------------------------------------------------------------


def _extract_line(board: Board, r: int, c: int, dr: int, dc: int) -> list[int]:
    """Trả về dãy giá trị ô trên một đường đi qua (r, c) theo hướng (dr, dc).

    Dãy được cắt tại biên bàn cờ, bao gồm tất cả ô trên đường đó.
    Trả về từ ô đầu tiên đến ô cuối cùng trên đường.
    """
    n = board.shape[0]
    # Tìm điểm bắt đầu (lùi ngược hướng đến biên)
    sr, sc = r, c
    while 0 <= sr - dr < n and 0 <= sc - dc < n:
        sr -= dr
        sc -= dc
    result: list[int] = []
    rr, cc = sr, sc
    while 0 <= rr < n and 0 <= cc < n:
        result.append(int(board[rr, cc]))
        rr += dr
        cc += dc
    return result


def _iter_lines(board: Board) -> list[list[int]]:
    """Trích xuất mọi đường (hàng, cột, chéo chính, chéo phụ) có độ dài >= 2 trong O(n^2)."""
    n = board.shape[0]
    if n < 2:
        return []
    lines: list[list[int]] = []
    for r in range(n):
        lines.append(board[r, :].tolist())
    for c in range(n):
        lines.append(board[:, c].tolist())
    flipped = board[:, ::-1]
    for offset in range(-(n - 2), n - 1):
        lines.append(board.diagonal(offset).tolist())
        lines.append(flipped.diagonal(offset).tolist())
    return lines


def _count_patterns_in_sequence(seq: list[int], player: int) -> PatternCount:
    """Đếm các mẫu cho `player` trong một dãy ô 1-D.

    Mã hoá: `player` → p, đối thủ hoặc biên → b, trống → 0.
    Duyệt mọi đoạn quân `player` liên tiếp có độ dài từ 2 trở lên.
    """
    pc = PatternCount()
    p = player
    n = len(seq)

    i = 0
    while i < n:
        if seq[i] != p:
            i += 1
            continue
        j = i + 1
        while j < n and seq[j] == p:
            j += 1
        length = j - i

        left_open = i > 0 and seq[i - 1] == 0
        right_open = j < n and seq[j] == 0

        if length >= 5:
            pc.five += 1
        elif length == 4:
            if left_open and right_open:
                pc.open_four += 1
            elif left_open or right_open:
                pc.four += 1
        elif length == 3:
            if left_open and right_open:
                pc.open_three += 1
            elif left_open or right_open:
                pc.three += 1
        elif length == 2:
            if left_open and right_open:
                pc.open_two += 1
            elif left_open or right_open:
                pc.two += 1

        i = j

    return pc


# ---------------------------------------------------------------------------
# API công khai
# ---------------------------------------------------------------------------


def count_patterns(board: Board, player: int) -> PatternCount:
    """Đếm các mẫu trên toàn bàn theo góc nhìn của `player` trong thời gian O(n^2).

    Duyệt qua tất cả hàng, cột và hai đường chéo (mỗi đường đúng một lần).
    """
    total = PatternCount()
    for seq in _iter_lines(board):
        if player not in seq:
            continue
        pc = _count_patterns_in_sequence(seq, player)
        total.five += pc.five
        total.open_four += pc.open_four
        total.four += pc.four
        total.open_three += pc.open_three
        total.three += pc.three
        total.open_two += pc.open_two
        total.two += pc.two

    return total


def evaluate(board: Board, player: int) -> float:
    """Điểm heuristic của `player` tại bàn cờ hiện tại.

    Dương → lợi thế cho `player`, âm → bất lợi.
    Trả về ±_WEIGHTS[FIVE] khi một trong hai bên đã có 5 quân liên tiếp.
    """
    my = count_patterns(board, player)
    opp = count_patterns(board, -player)

    # Nếu đã có FIVE thì thắng/thua rõ ràng
    if my.five > 0:
        return float(_WEIGHTS[FIVE])
    if opp.five > 0:
        return float(-_WEIGHTS[FIVE])

    return float(
        _WEIGHTS[OPEN_FOUR] * (my.open_four - opp.open_four)
        + _WEIGHTS[FOUR] * (my.four - opp.four)
        + _WEIGHTS[OPEN_THREE] * (my.open_three - opp.open_three)
        + _WEIGHTS[THREE] * (my.three - opp.three)
        + _WEIGHTS[OPEN_TWO] * (my.open_two - opp.open_two)
        + _WEIGHTS[TWO] * (my.two - opp.two)
    )
