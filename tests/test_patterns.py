"""Unit test cho src/eval/patterns.py — P1.3a.

Chủ sở hữu: TV3.  Review: TV1.

Mỗi thế kiểm tra chính xác số mẫu đếm được (mọi mẫu không nêu phải bằng 0) và điểm
evaluate(). Chạy: pytest tests/test_patterns.py -v
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
    count_patterns,
    evaluate,
)
from src.games.base import PLAYER_O, PLAYER_X

# ---------------------------------------------------------------------------
# Hàm tiện ích
# ---------------------------------------------------------------------------

X, O, _ = PLAYER_X, PLAYER_O, 0  # noqa: E741 — alias ngắn gọn trong bảng

WIN = 1_000_000.0


def board(rows: list[list[int]]) -> np.ndarray:
    """Tạo Board từ danh sách các hàng."""
    return np.array(rows, dtype=np.int8)


def row_board(cells: list[int], size: int = 9) -> np.ndarray:
    """Bàn size×size trống, chỉ có `cells` đặt từ đầu hàng giữa.

    Các quân nằm trên cùng một hàng nên mọi cột và đường chéo chỉ chứa quân lẻ,
    không tạo thêm mẫu nào.
    """
    b = np.zeros((size, size), dtype=np.int8)
    b[size // 2, : len(cells)] = cells
    return b


def found(b: np.ndarray, player: int, **kwargs: bool) -> dict[str, int]:
    """Các mẫu có số đếm khác 0."""
    return {k: v for k, v in count_patterns(b, player, **kwargs).as_dict().items() if v}


# ---------------------------------------------------------------------------
# Các mẫu liền nhau trên một hàng: số đếm và điểm chính xác
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cells", "expected", "score"),
    [
        pytest.param([X, X, X, X, X], {FIVE: 1}, WIN, id="five"),
        pytest.param([_, X, X, X, X, X, X, _], {FIVE: 1}, WIN, id="six-in-a-row-is-five"),
        pytest.param([_, X, X, X, X, _], {OPEN_FOUR: 1}, 50_000.0, id="open-four"),
        pytest.param([O, X, X, X, X, _], {FOUR: 1}, 10_000.0, id="four-blocked-by-stone"),
        pytest.param([X, X, X, X, _], {FOUR: 1}, 10_000.0, id="four-blocked-by-edge"),
        pytest.param([_, X, X, X, _, _], {OPEN_THREE: 1}, 1_000.0, id="open-three"),
        pytest.param([O, X, X, X, _, _], {THREE: 1}, 200.0, id="three-blocked"),
        pytest.param([_, _, X, X, _, _], {OPEN_TWO: 1}, 50.0, id="open-two"),
        pytest.param([O, X, X, _, _, _], {TWO: 1}, 10.0, id="two-blocked"),
        pytest.param([_, _, X, _, _], {}, 0.0, id="single-stone"),
    ],
)
def test_solid_patterns(cells: list[int], expected: dict[str, int], score: float) -> None:
    b = row_board(cells)
    assert found(b, X) == expected
    assert evaluate(b, X) == score
    assert evaluate(b, O) == -score


# ---------------------------------------------------------------------------
# Mẫu đứt quãng (có một ô trống ở giữa)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cells", "expected", "score"),
    [
        pytest.param([_, X, X, X, _, X, _], {FOUR: 1}, 10_000.0, id="XXX_X"),
        pytest.param([_, X, X, _, X, X, _], {FOUR: 1}, 10_000.0, id="XX_XX"),
        pytest.param([_, X, _, X, X, X, _, X, _], {OPEN_FOUR: 1}, 50_000.0, id="X_XXX_X"),
        pytest.param([_, X, _, X, X, _, _], {OPEN_THREE: 1}, 1_000.0, id="X_XX-open"),
        pytest.param([O, X, _, X, X, _, _], {THREE: 1}, 200.0, id="X_XX-blocked"),
        pytest.param([_, _, X, _, X, _, _], {OPEN_TWO: 1}, 50.0, id="X_X-open"),
    ],
)
def test_broken_patterns(cells: list[int], expected: dict[str, int], score: float) -> None:
    b = row_board(cells)
    assert found(b, X) == expected
    assert evaluate(b, X) == score


def test_stones_two_cells_apart_are_separate_groups() -> None:
    # _XX__XX_ : hai cụm XX cách nhau 2 ô trống, mỗi cụm là một OPEN_TWO.
    b = row_board([_, X, X, _, _, X, X, _, _])
    assert found(b, X) == {OPEN_TWO: 2}
    assert evaluate(b, X) == 100.0


# ---------------------------------------------------------------------------
# Mẫu không đủ chỗ để thành 5 thì không được điểm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cells",
    [
        pytest.param([O, X, X, X, X, O], id="OXXXXO"),
        pytest.param([O, X, X, X, _, O], id="OXXX_O"),
        pytest.param([O, _, X, X, _, O], id="O_XX_O"),
        pytest.param([O, X, X, _, _, O], id="OXX__O"),
    ],
)
def test_shapes_without_room_for_five_score_nothing(cells: list[int]) -> None:
    b = row_board(cells)
    assert found(b, X) == {}
    assert evaluate(b, X) == 0.0


def test_three_in_exactly_five_cells_is_not_open() -> None:
    # O_XXX_O : chỉ còn đúng 5 ô, đánh thêm 1 quân chỉ thành FOUR, không thành OPEN_FOUR.
    b = row_board([O, _, X, X, X, _, O])
    assert found(b, X) == {THREE: 1}


def test_board_smaller_than_five_has_no_patterns() -> None:
    b = np.full((3, 3), X, dtype=np.int8)
    assert found(b, X) == {}
    assert evaluate(b, X) == 0.0


# ---------------------------------------------------------------------------
# Luật chặn hai đầu
# ---------------------------------------------------------------------------


def test_five_blocked_on_both_ends_wins_by_default() -> None:
    b = row_board([O, X, X, X, X, X, O])
    assert found(b, X) == {FIVE: 1}
    assert evaluate(b, X) == WIN


def test_five_blocked_on_both_ends_does_not_win_under_rule() -> None:
    b = row_board([O, X, X, X, X, X, O])
    assert found(b, X, block_two_ends=True) == {}
    assert evaluate(b, X, block_two_ends=True) == 0.0


def test_six_blocked_on_both_ends_does_not_win_under_rule() -> None:
    b = row_board([O, X, X, X, X, X, X, O])
    assert found(b, X) == {FIVE: 1}
    assert found(b, X, block_two_ends=True) == {}


def test_board_edge_does_not_count_as_a_block() -> None:
    # XXXXXO sát biên trái: chỉ bị chặn một đầu bằng quân, nên vẫn thắng.
    b = row_board([X, X, X, X, X, O])
    assert found(b, X, block_two_ends=True) == {FIVE: 1}


def test_five_blocked_on_one_end_still_wins_under_rule() -> None:
    b = row_board([O, X, X, X, X, X, _])
    assert found(b, X, block_two_ends=True) == {FIVE: 1}


def test_four_that_can_only_complete_a_blocked_five_is_worthless_under_rule() -> None:
    # OXXXX_O : ô trống duy nhất sẽ tạo ra 5 quân bị chặn hai đầu.
    b = row_board([O, X, X, X, X, _, O])
    assert found(b, X) == {FOUR: 1}
    assert found(b, X, block_two_ends=True) == {}


# ---------------------------------------------------------------------------
# Các hướng: dọc, chéo chính, chéo phụ
# ---------------------------------------------------------------------------


def test_five_vertical() -> None:
    b = np.zeros((7, 7), dtype=np.int8)
    b[:5, 0] = X
    assert found(b, X) == {FIVE: 1}


def test_five_diagonal_main() -> None:
    b = np.zeros((7, 7), dtype=np.int8)
    for i in range(5):
        b[i, i] = X
    assert found(b, X) == {FIVE: 1}


def test_five_diagonal_anti() -> None:
    b = np.zeros((7, 7), dtype=np.int8)
    for i in range(5):
        b[i, 6 - i] = X
    assert found(b, X) == {FIVE: 1}


def test_open_three_on_short_diagonal_needs_room() -> None:
    # Đường chéo dài đúng 5 ô ở góc: _XXX_ chỉ là THREE vì không thể thành OPEN_FOUR.
    b = np.zeros((7, 7), dtype=np.int8)
    for r, c in [(1, 3), (2, 4), (3, 5)]:
        b[r, c] = X
    assert found(b, X) == {THREE: 1}


# ---------------------------------------------------------------------------
# Thế có mẫu ở nhiều hướng cùng lúc
# ---------------------------------------------------------------------------


def test_cross_counts_every_direction() -> None:
    # Chữ thập 5 quân ở giữa bàn 9×9:
    #   hàng 4 và cột 4: mỗi đường một OPEN_THREE
    #   4 đường chéo đi qua hai quân kề nhau: mỗi đường một OPEN_TWO
    b = np.zeros((9, 9), dtype=np.int8)
    for r, c in [(4, 3), (4, 4), (4, 5), (3, 4), (5, 4)]:
        b[r, c] = X
    assert found(b, X) == {OPEN_THREE: 2, OPEN_TWO: 4}
    assert evaluate(b, X) == 2 * 1_000.0 + 4 * 50.0


def test_mixed_position_counts_both_players() -> None:
    b = board(
        [
            [O, X, X, _, _, _, _],
            [O, X, X, X, X, O, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
    )
    # X: hàng 0 (OXX____), cột 1, cột 2, và hai đường chéo qua (0,1)-(1,2), (0,2)-(1,3)
    #    đều là TWO; hàng 1 (OXXXXO) không đủ chỗ; chéo phụ (0,2)-(1,1) chỉ dài 3 ô.
    # O: cột 0 có OO sát biên trên → một TWO.
    assert found(b, X) == {TWO: 5}
    assert found(b, O) == {TWO: 1}
    assert evaluate(b, X) == 5 * 10.0 - 10.0
    assert evaluate(b, O) == 10.0 - 5 * 10.0


# ---------------------------------------------------------------------------
# evaluate(): bàn trống, đối xứng, thắng/thua
# ---------------------------------------------------------------------------


def test_empty_board() -> None:
    b = np.zeros((15, 15), dtype=np.int8)
    assert count_patterns(b, X).as_dict() == {
        FIVE: 0,
        OPEN_FOUR: 0,
        FOUR: 0,
        OPEN_THREE: 0,
        THREE: 0,
        OPEN_TWO: 0,
        TWO: 0,
    }
    assert evaluate(b, X) == 0.0


def test_evaluate_is_antisymmetric() -> None:
    b = board(
        [
            [_, X, X, X, _, _, _],
            [_, _, _, _, _, _, _],
            [O, O, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
    )
    # X có một OPEN_THREE (1000); O có một TWO sát biên trái (10).
    assert evaluate(b, X) == 990.0
    assert evaluate(b, O) == -990.0
    # Đổi màu toàn bộ quân thì điểm của hai bên đổi chỗ cho nhau.
    assert evaluate(-b, X) == -990.0


def test_evaluate_is_antisymmetric_on_random_boards() -> None:
    rng = np.random.default_rng(2026)
    for _i in range(50):
        b = np.zeros((15, 15), dtype=np.int8)
        stones = int(rng.integers(4, 60))
        cells = rng.choice(15 * 15, stones, replace=False)
        b.flat[cells[: stones // 2]] = X
        b.flat[cells[stones // 2 :]] = O
        assert evaluate(b, X) == -evaluate(b, O)
        assert evaluate(-b, O) == evaluate(b, X)
        assert count_patterns(b, X).as_dict() == count_patterns(-b, O).as_dict()


def test_five_overrides_every_other_pattern() -> None:
    # O có FIVE; X có OPEN_FOUR nhưng ván đã kết thúc.
    b = np.zeros((9, 9), dtype=np.int8)
    b[0, :5] = O
    b[4, 2:6] = X
    assert found(b, X) == {OPEN_FOUR: 1}
    assert evaluate(b, X) == -WIN
    assert evaluate(b, O) == WIN


def test_invalid_player_is_rejected() -> None:
    b = np.zeros((7, 7), dtype=np.int8)
    with pytest.raises(ValueError):
        count_patterns(b, 0)
    with pytest.raises(ValueError):
        evaluate(b, 2)
