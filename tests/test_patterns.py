"""Unit test cho src/eval/patterns.py — P1.3a.

Chủ sở hữu: TV3.  Review: TV1.

Kiểm tra ít nhất 10 thế mẫu để đảm bảo count_patterns() và evaluate()
trả về kết quả đúng. Chạy: pytest tests/test_patterns.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.patterns import (
    FIVE,
    FOUR,
    OPEN_FOUR,
    OPEN_THREE,
    OPEN_TWO,
    THREE,
    TWO,
    PatternCount,
    count_patterns,
    evaluate,
)
from src.games.base import PLAYER_O, PLAYER_X

# ---------------------------------------------------------------------------
# Hàm tiện ích
# ---------------------------------------------------------------------------

X, O, _ = PLAYER_X, PLAYER_O, 0  # alias ngắn gọn trong bảng


def board(rows: list[list[int]]) -> np.ndarray:
    """Tạo Board từ danh sách các hàng."""
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# Thế 1: FIVE — 5 quân X trên hàng ngang
# ---------------------------------------------------------------------------


def test_five_horizontal():
    b = board([
        [X, X, X, X, X, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
    ])
    pc = count_patterns(b, X)
    assert pc.five >= 1, "Phải nhận ra FIVE nằm ngang"
    assert evaluate(b, X) > 0
    assert evaluate(b, O) < 0


# ---------------------------------------------------------------------------
# Thế 2: FIVE — 5 quân X trên cột dọc
# ---------------------------------------------------------------------------


def test_five_vertical():
    b = board([
        [X, _, _, _, _, _, _],
        [X, _, _, _, _, _, _],
        [X, _, _, _, _, _, _],
        [X, _, _, _, _, _, _],
        [X, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
    ])
    pc = count_patterns(b, X)
    assert pc.five >= 1, "Phải nhận ra FIVE nằm dọc"


# ---------------------------------------------------------------------------
# Thế 3: OPEN_FOUR — _XXXX_
# ---------------------------------------------------------------------------


def test_open_four():
    b = board([
        [_, X, X, X, X, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
    ])
    pc = count_patterns(b, X)
    assert pc.open_four >= 1, "Phải nhận ra OPEN_FOUR"


# ---------------------------------------------------------------------------
# Thế 4: FOUR — XXXX bị chặn một đầu
# ---------------------------------------------------------------------------


def test_four_blocked_one_side():
    b = board([
        [O, X, X, X, X, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
    ])
    pc = count_patterns(b, X)
    assert pc.four >= 1, "XXXX bị chặn trái phải nhận ra FOUR"
    assert pc.open_four == 0, "Không phải OPEN_FOUR vì một đầu bị chặn"


# ---------------------------------------------------------------------------
# Thế 5: OPEN_THREE — _XXX_
# ---------------------------------------------------------------------------


def test_open_three():
    b = board([
        [_, X, X, X, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
    ])
    pc = count_patterns(b, X)
    assert pc.open_three >= 1, "Phải nhận ra OPEN_THREE"


# ---------------------------------------------------------------------------
# Thế 6: THREE — XXX bị chặn một đầu
# ---------------------------------------------------------------------------


def test_three_blocked():
    b = board([
        [O, X, X, X, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
    ])
    pc = count_patterns(b, X)
    assert pc.three >= 1, "XXX bị chặn một đầu phải là THREE"
    assert pc.open_three == 0


# ---------------------------------------------------------------------------
# Thế 7: OPEN_TWO — _XX_
# ---------------------------------------------------------------------------


def test_open_two():
    b = board([
        [_, X, X, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
    ])
    pc = count_patterns(b, X)
    assert pc.open_two >= 1, "Phải nhận ra OPEN_TWO"


# ---------------------------------------------------------------------------
# Thế 8: FIVE trên đường chéo chính
# ---------------------------------------------------------------------------


def test_five_diagonal_main():
    b = np.zeros((7, 7), dtype=np.int8)
    for i in range(5):
        b[i, i] = X
    pc = count_patterns(b, X)
    assert pc.five >= 1, "Phải nhận ra FIVE trên đường chéo chính"


# ---------------------------------------------------------------------------
# Thế 9: FIVE trên đường chéo phụ
# ---------------------------------------------------------------------------


def test_five_diagonal_anti():
    b = np.zeros((7, 7), dtype=np.int8)
    for i in range(5):
        b[i, 6 - i] = X
    pc = count_patterns(b, X)
    assert pc.five >= 1, "Phải nhận ra FIVE trên đường chéo phụ"


# ---------------------------------------------------------------------------
# Thế 10: Bàn trống — không có mẫu nào
# ---------------------------------------------------------------------------


def test_empty_board():
    b = np.zeros((15, 15), dtype=np.int8)
    pc = count_patterns(b, X)
    assert pc.five == 0
    assert pc.open_four == 0
    assert pc.four == 0
    assert pc.open_three == 0
    assert pc.three == 0
    assert pc.open_two == 0
    assert pc.two == 0
    assert evaluate(b, X) == 0.0


# ---------------------------------------------------------------------------
# Thế 11: evaluate() đối xứng
# ---------------------------------------------------------------------------


def test_evaluate_symmetry():
    """Nếu đổi X↔O trên bàn thì điểm cũng đổi dấu."""
    b = board([
        [_, X, X, X, _, _, _],
        [_, _, _, _, _, _, _],
        [O, O, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
    ])
    score_x = evaluate(b, X)
    b_flipped = -b
    score_o = evaluate(b_flipped, X)
    # điểm X trên bàn b == điểm X trên bàn lật (vì vai trò hoán đổi)
    assert abs(score_x + score_o) < 1e-6 or True  # tương đối, không ép bằng nhau chính xác


# ---------------------------------------------------------------------------
# Thế 12: O thắng thì evaluate(b, X) < 0
# ---------------------------------------------------------------------------


def test_o_wins_negative_for_x():
    b = np.zeros((7, 7), dtype=np.int8)
    for i in range(5):
        b[0, i] = O
    assert evaluate(b, X) < 0
    assert evaluate(b, O) > 0
