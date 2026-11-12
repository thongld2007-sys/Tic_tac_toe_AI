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

from dataclasses import dataclass, field

import numpy as np

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


def _count_patterns_in_sequence(seq: list[int], player: int) -> PatternCount:
    """Đếm các mẫu cho `player` trong một dãy ô 1-D.

    Mã hoá: `player` → p, đối thủ hoặc biên → b, trống → 0.
    Chúng ta duyệt mọi đoạn con liên tiếp có độ dài từ 2 đến 5.
    """
    pc = PatternCount()
    p = player
    # Thêm sentinel "biên ảo" (giá trị -player) vào hai đầu để xử lý nhất quán
    # Thực ra ta sẽ kiểm tra ô lân cận khi phân tích cửa sổ
    n = len(seq)

    def cell(i: int) -> int:
        """Giá trị ô tại chỉ số i; trả về -player (biên) nếu nằm ngoài."""
        if i < 0 or i >= n:
            return -p  # coi như bị chặn
        return seq[i]

    # Duyệt mọi vị trí bắt đầu của chuỗi quân p liên tiếp
    counted: set[tuple[int, int]] = set()  # tránh đếm trùng

    i = 0
    while i < n:
        if seq[i] != p:
            i += 1
            continue
        # Tìm độ dài đoạn liên tiếp bắt đầu từ i
        j = i
        while j < n and seq[j] == p:
            j += 1
        length = j - i  # số quân liên tiếp từ i đến j-1
        key = (i, j)
        if key in counted:
            i = j
            continue
        counted.add(key)

        left = cell(i - 1)   # ô ngay trước đoạn
        right = cell(j)      # ô ngay sau đoạn

        left_open = left == 0
        right_open = right == 0

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
    """Đếm các mẫu trên toàn bàn theo góc nhìn của `player`.

    Duyệt qua tất cả hàng, cột và hai đường chéo.
    Mỗi đường chỉ được duyệt một lần (ví dụ hướng (0,1) đại diện cho cả
    hướng (0,-1) vì chúng ta lấy toàn bộ dãy từ đầu đến cuối).
    """
    n = board.shape[0]
    total = PatternCount()
    visited_lines: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for r in range(n):
        for c in range(n):
            for dr, dc in _DIRECTIONS:
                # Tính điểm đầu của đường
                sr, sc = r, c
                while 0 <= sr - dr < n and 0 <= sc - dc < n:
                    sr -= dr
                    sc -= dc
                key = ((sr, sc), (dr, dc))
                if key in visited_lines:
                    continue
                visited_lines.add(key)

                seq = _extract_line(board, sr, sc, dr, dc)
                if len(seq) < 2:
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
    Trả về ±∞ khi một trong hai bên đã thắng.
    """
    my = count_patterns(board, player)
    opp = count_patterns(board, -player)

    # Nếu đã có FIVE thì thắng/thua rõ ràng
    if my.five > 0:
        return float(_WEIGHTS[FIVE])
    if opp.five > 0:
        return float(-_WEIGHTS[FIVE])

    score = 0.0
    for pattern, weight in _WEIGHTS.items():
        if pattern == FIVE:
            continue
        score += weight * getattr(my, pattern.replace("-", "_"), 0)
        score -= weight * getattr(opp, pattern.replace("-", "_"), 0)

    return score
