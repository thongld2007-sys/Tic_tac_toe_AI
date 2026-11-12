"""Hàm đếm mẫu caro và hàm lượng giá (P1.3a).

Chủ sở hữu: TV3.  Review: TV1.

Các mẫu, theo góc nhìn của `player` (X là quân của `player`, _ là ô trống):
  - FIVE        : đã có 5 quân liên tiếp trở lên → thắng
  - OPEN_FOUR   : có từ 2 ô trống mà đánh vào là thành FIVE (vd. _XXXX_) → không chặn được
  - FOUR        : có đúng 1 ô như vậy (vd. OXXXX_, XXX_X, XX_XX) → đối thủ phải chặn ngay
  - OPEN_THREE  : đánh thêm 1 quân thì thành OPEN_FOUR (vd. _XXX_ còn chỗ, _X_XX_)
  - THREE       : đánh thêm 1 quân thì chỉ thành FOUR (vd. OXXX__)
  - OPEN_TWO    : đánh thêm 2 quân thì thành OPEN_FOUR (vd. __XX__)
  - TWO         : đánh thêm 2 quân thì chỉ thành FOUR (vd. OXX___)

Mỗi mẫu được xác định bằng cách "đánh thử" thêm quân, nên mẫu đứt quãng (có một ô trống
ở giữa) được xếp đúng mức, còn mẫu nằm trong khoảng không đủ 5 ô (vd. OXXX_O) không được
tính vì không bao giờ thành FIVE.

Luật chặn hai đầu (`block_two_ends=True`): dãy từ 5 quân trở lên bị quân đối thủ chặn ở cả
hai đầu thì không thắng. Biên bàn cờ không tính là chặn. Mặc định là False, tức cứ đủ 5
quân liên tiếp là thắng.

Hàm chính:
  count_patterns(board, player, block_two_ends=False) -> PatternCount
  evaluate(board, player, block_two_ends=False) -> float   # điểm theo góc nhìn của `player`
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, lru_cache

from src.games.base import PLAYER_O, PLAYER_X, Board

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

# Thứ tự từ mạnh đến yếu; chỉ số trong tuple này là "mức" của mẫu.
_LEVELS = (FIVE, OPEN_FOUR, FOUR, OPEN_THREE, THREE, OPEN_TWO, TWO)
_NO_PATTERN = (0,) * len(_LEVELS)

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
FIVE_SCORE: float = float(_WEIGHTS[FIVE])

_WIN_LENGTH = 5


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
#
# Một "đoạn" là dãy ô liên tiếp trên một đường, chỉ gồm quân của người đang xét (1) và
# ô trống (0), kẹp giữa hai quân đối thủ hoặc biên bàn. `closed` là True khi luật chặn
# hai đầu đang bật và cả hai đầu của đoạn đều là quân đối thủ: khi đó dãy quân phủ kín
# cả đoạn không được tính là thắng.


def _has_five(cells: list[int], closed: bool) -> bool:
    """Đoạn đã có dãy từ 5 quân liên tiếp trở lên (và dãy đó hợp lệ theo luật) hay chưa."""
    n = len(cells)
    run = 0
    for i in range(n + 1):
        if i < n and cells[i]:
            run += 1
            continue
        if run >= _WIN_LENGTH and not (closed and run == n):
            return True
        run = 0
    return False


def _winning_cells(cells: list[int], closed: bool) -> int:
    """Số ô trống mà đánh vào thì tạo ngay dãy từ 5 quân trở lên hợp lệ."""
    n = len(cells)
    count = 0
    for i in range(n):
        if cells[i]:
            continue
        start = i
        while start > 0 and cells[start - 1]:
            start -= 1
        end = i
        while end < n - 1 and cells[end + 1]:
            end += 1
        length = end - start + 1
        if length >= _WIN_LENGTH and not (closed and length == n):
            count += 1
    return count


def _level_from_winning_cells(count: int, open_level: int) -> int | None:
    """Từ 2 ô thắng trở lên là mức "mở", đúng 1 ô là mức ngay dưới, 0 ô là không có."""
    if count >= 2:
        return open_level
    if count == 1:
        return open_level + 1
    return None


@cache
def _classify_group(cells: tuple[int, ...], closed: bool) -> int | None:
    """Mức (chỉ số trong _LEVELS) của một cụm quân nằm trong `cells`, None nếu không có mẫu."""
    n = len(cells)
    if n < _WIN_LENGTH:
        return None
    buf = list(cells)
    if _has_five(buf, closed):
        return _LEVELS.index(FIVE)

    level = _level_from_winning_cells(_winning_cells(buf, closed), _LEVELS.index(OPEN_FOUR))
    if level is not None:
        return level

    first = cells.index(1)
    last = n - 1 - cells[::-1].index(1)

    def nearby_empty(lo: int, hi: int) -> list[int]:
        return [i for i in range(max(0, lo - 2), min(n, hi + 3)) if not buf[i]]

    # Đánh thử thêm 1 quân.
    best = 0
    for i in nearby_empty(first, last):
        buf[i] = 1
        best = max(best, _winning_cells(buf, closed))
        buf[i] = 0
    level = _level_from_winning_cells(best, _LEVELS.index(OPEN_THREE))
    if level is not None:
        return level

    # Đánh thử thêm 2 quân.
    best = 0
    for i in nearby_empty(first, last):
        buf[i] = 1
        for j in nearby_empty(min(first, i), max(last, i)):
            buf[j] = 1
            best = max(best, _winning_cells(buf, closed))
            buf[j] = 0
        buf[i] = 0
    return _level_from_winning_cells(best, _LEVELS.index(OPEN_TWO))


@cache
def _classify_segment(
    cells: tuple[int, ...], left_blocked: bool, right_blocked: bool
) -> tuple[int, ...]:
    """Đếm mẫu trong một đoạn. `left_blocked`/`right_blocked`: đầu đó bị đối thủ chặn theo luật."""
    n = len(cells)
    stones = [i for i, value in enumerate(cells) if value]
    if n < _WIN_LENGTH or not stones:
        return _NO_PATTERN

    # Tách thành các cụm: hai quân cách nhau quá 1 ô trống thì thuộc hai cụm khác nhau.
    groups: list[list[int]] = [[stones[0]]]
    for i in stones[1:]:
        if i - groups[-1][-1] > 2:
            groups.append([i])
        else:
            groups[-1].append(i)

    counts = [0] * len(_LEVELS)
    for group in groups:
        # Chỉ giữ cụm đang xét và tối đa 5 ô mỗi bên: đủ cho mọi phép đánh thử ở trên,
        # và làm cho cùng một hình dạng ở các vị trí khác nhau dùng chung kết quả đã nhớ.
        lo = max(0, group[0] - _WIN_LENGTH)
        hi = min(n, group[-1] + _WIN_LENGTH + 1)
        window = [0] * (hi - lo)
        for i in group:
            window[i - lo] = 1
        closed = left_blocked and right_blocked and lo == 0 and hi == n
        level = _classify_group(tuple(window), closed)
        if level is not None:
            counts[level] += 1
    return tuple(counts)


@lru_cache(maxsize=200_000)
def _count_line(
    line: tuple[int, ...], block_two_ends: bool
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Đếm mẫu trên một đường cho cả hai bên: (của PLAYER_X, của PLAYER_O)."""
    n = len(line)
    result = []
    for player in (PLAYER_X, PLAYER_O):
        counts = [0] * len(_LEVELS)
        if player in line:
            start = 0
            for i in range(n + 1):
                if i < n and line[i] != -player:
                    continue
                # line[start:i] là một đoạn; hai đầu là quân đối thủ hoặc biên bàn.
                if i - start >= _WIN_LENGTH:
                    cells = tuple(1 if value == player else 0 for value in line[start:i])
                    found = _classify_segment(
                        cells, block_two_ends and start > 0, block_two_ends and i < n
                    )
                    for level, amount in enumerate(found):
                        counts[level] += amount
                start = i + 1
        result.append(tuple(counts))
    return result[0], result[1]


def _iter_lines(board: Board) -> list[tuple[int, ...]]:
    """Mọi hàng, cột và đường chéo đủ dài để chứa 5 quân, mỗi đường đúng một lần."""
    n = board.shape[0]
    lines = [tuple(row) for row in board.tolist()]
    lines += [tuple(column) for column in board.T.tolist()]
    flipped = board[:, ::-1]
    for offset in range(-(n - _WIN_LENGTH), n - _WIN_LENGTH + 1):
        lines.append(tuple(board.diagonal(offset).tolist()))
        lines.append(tuple(flipped.diagonal(offset).tolist()))
    return lines


def _count_both(board: Board, block_two_ends: bool) -> tuple[list[int], list[int]]:
    """Quét bàn một lần, trả về số mẫu theo mức của (PLAYER_X, PLAYER_O)."""
    totals_x = [0] * len(_LEVELS)
    totals_o = [0] * len(_LEVELS)
    for line in _iter_lines(board):
        if not any(line):
            continue
        counts_x, counts_o = _count_line(line, block_two_ends)
        for level in range(len(_LEVELS)):
            totals_x[level] += counts_x[level]
            totals_o[level] += counts_o[level]
    return totals_x, totals_o


def _check_player(player: int) -> None:
    if player not in (PLAYER_X, PLAYER_O):
        raise ValueError(f"player phải là {PLAYER_X} hoặc {PLAYER_O}, nhận được {player!r}")


# ---------------------------------------------------------------------------
# API công khai
# ---------------------------------------------------------------------------


def count_patterns(board: Board, player: int, block_two_ends: bool = False) -> PatternCount:
    """Đếm các mẫu trên toàn bàn theo góc nhìn của `player`."""
    _check_player(player)
    totals_x, totals_o = _count_both(board, block_two_ends)
    return PatternCount(*(totals_x if player == PLAYER_X else totals_o))


def evaluate(board: Board, player: int, block_two_ends: bool = False) -> float:
    """Điểm heuristic của `player` tại bàn cờ hiện tại.

    Dương → lợi thế cho `player`, âm → bất lợi.
    Trả về ±_WEIGHTS[FIVE] khi một trong hai bên đã có FIVE.
    """
    _check_player(player)
    totals_x, totals_o = _count_both(board, block_two_ends)
    mine, theirs = (totals_x, totals_o) if player == PLAYER_X else (totals_o, totals_x)

    five = _LEVELS.index(FIVE)
    if mine[five] > 0:
        return float(_WEIGHTS[FIVE])
    if theirs[five] > 0:
        return float(-_WEIGHTS[FIVE])

    return float(
        sum(_WEIGHTS[name] * (mine[level] - theirs[level]) for level, name in enumerate(_LEVELS))
    )
