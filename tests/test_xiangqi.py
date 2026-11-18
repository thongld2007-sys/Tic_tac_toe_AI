"""Test luật cờ tướng, sinh nước hợp lệ và perft (P4.1b)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.agents.base import Budget
from src.agents.mcts import MCTSAgent
from src.games.base import DRAW, PLAYER_O, PLAYER_X
from src.games.xiangqi import (
    COLS,
    KING,
    START_FEN,
    Xiangqi,
    main,
    move_to_iccs,
    parse_iccs,
    parse_square,
    perft,
    perft_divide,
    square_name,
)


def iccs_moves(state: Xiangqi) -> set[str]:
    return {move_to_iccs(m) for m in state.legal_moves()}


def moves_from(state: Xiangqi, square: str) -> set[str]:
    return {name[2:] for name in iccs_moves(state) if name[:2] == square}


def play_iccs(state: Xiangqi, *names: str) -> None:
    for name in names:
        state.play(parse_iccs(name))


# ---------------------------------------------------------------------------
# Perft
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("depth", "nodes"), [(1, 44), (2, 1920), (3, 79666)])
def test_perft_start_position_matches_reference(depth: int, nodes: int) -> None:
    assert perft(Xiangqi(), depth) == nodes


def test_perft_leaves_state_unchanged() -> None:
    state = Xiangqi()
    play_iccs(state, "h2e2", "h9g7")
    fen, key, board = state.to_fen(), state.hash(), state.board.copy()
    perft(state, 3)
    assert state.to_fen() == fen
    assert state.hash() == key
    assert np.array_equal(state.board, board)


def test_perft_depth_zero_and_negative() -> None:
    assert perft(Xiangqi(), 0) == 1
    with pytest.raises(ValueError):
        perft(Xiangqi(), -1)


def test_perft_divide_sums_to_perft() -> None:
    state = Xiangqi()
    divide = perft_divide(state, 2)
    assert len(divide) == 44
    assert sum(divide.values()) == 1920
    with pytest.raises(ValueError):
        perft_divide(state, 0)


def test_cli_prints_perft(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--perft", "2", "--divide"]) == 0
    out = capsys.readouterr().out
    assert "perft(1) = 44" in out
    assert "perft(2) = 1920" in out
    assert "h2e2: " in out


# ---------------------------------------------------------------------------
# FEN và ký hiệu ICCS
# ---------------------------------------------------------------------------


def test_fen_round_trip_and_start_board() -> None:
    state = Xiangqi()
    assert state.to_fen() == START_FEN
    assert state.current_player() == PLAYER_X
    assert state.board.shape == (10, 9)
    assert state.board[9, 4] == KING
    assert state.board[0, 4] == -KING
    play_iccs(state, "h2e2")
    again = Xiangqi(state.to_fen())
    assert again.to_fen() == state.to_fen()
    assert again.hash() == state.hash()


def test_fen_accepts_alternative_letters_and_black_to_move() -> None:
    state = Xiangqi("rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR b")
    assert state.current_player() == PLAYER_O
    assert state.to_fen().startswith("rnbakabnr/")


@pytest.mark.parametrize(
    "fen",
    [
        "",
        "rnbakabnr/9/9 w",
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNX w",
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABN w",
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNRR w",
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBA1ABNR w",
        "4k4/9/9/9/9/9/9/9/9/K8 w",
        "4k4/9/9/9/9/9/9/9/4K4/4K4 w",
        "4k4/9/9/9/9/9/9/9/9/4K4 x",
    ],
)
def test_invalid_fen_is_rejected(fen: str) -> None:
    with pytest.raises(ValueError):
        Xiangqi(fen)


def test_iccs_names() -> None:
    assert square_name(parse_square("a0")) == "a0"
    assert parse_square("a0") == 9 * COLS
    assert parse_square("i9") == COLS - 1
    assert move_to_iccs(parse_iccs("h2-e2")) == "h2e2"
    for bad in ("j0", "a", "aa"):
        with pytest.raises(ValueError):
            parse_square(bad)
    for bad in ("h2e", "h2e2x", "j0a0"):
        with pytest.raises(ValueError):
            parse_iccs(bad)


# ---------------------------------------------------------------------------
# Cách đi của từng quân
# ---------------------------------------------------------------------------


def test_start_position_has_44_moves_with_known_examples() -> None:
    moves = iccs_moves(Xiangqi())
    assert len(moves) == 44
    assert {"h2e2", "b2e2", "h0g2", "c3c4", "g0e2", "e0e1"} <= moves


def test_horse_leg_blocks() -> None:
    # Mã b0: chân mã c0 bị Tượng chặn nên không nhảy được d1.
    assert moves_from(Xiangqi(), "b0") == {"a2", "c2"}
    state = Xiangqi("3k5/9/9/9/9/9/9/2P6/2N6/4K4 w")
    # Mã c1 bị Tốt ở c2 chặn hướng lên; e0 là Tướng mình.
    assert moves_from(state, "c1") == {"a0", "a2", "e2"}


def test_elephant_eye_and_river() -> None:
    state = Xiangqi("3k5/9/9/9/9/2B6/9/9/9/4K4 w")
    # Tượng c4 đứng sát sông: không qua sông được (a6, e6), chỉ lùi về a2, e2.
    assert moves_from(state, "c4") == {"a2", "e2"}
    blocked = Xiangqi("3k5/9/9/9/9/2B6/3P5/9/9/4K4 w")
    # Mắt tượng d3 có quân nên không đi e2 được.
    assert moves_from(blocked, "c4") == {"a2"}


def test_advisor_and_king_stay_in_palace() -> None:
    state = Xiangqi("3k5/9/9/9/9/9/9/9/9/3AK4 w")
    assert moves_from(state, "d0") == {"e1"}
    assert moves_from(state, "e0") == {"e1", "f0"}
    corner = Xiangqi("4k4/9/9/9/9/9/9/5A3/9/3K5 w")
    assert moves_from(corner, "f2") == {"e1"}
    assert moves_from(corner, "d0") == {"d1"}  # e0 đối mặt Tướng đen e9


def test_rook_moves_and_captures() -> None:
    state = Xiangqi("3k5/9/9/9/9/9/9/9/9/R3K4 w")
    targets = moves_from(state, "a0")
    assert targets == {f"a{r}" for r in range(1, 10)} | {"b0", "c0", "d0"}
    capture = Xiangqi("3k5/9/9/9/9/9/p8/9/9/R3K4 w")
    assert moves_from(capture, "a0") == {"a1", "a2", "a3", "b0", "c0", "d0"}


def test_cannon_needs_exactly_one_screen_to_capture() -> None:
    moves = iccs_moves(Xiangqi())
    assert "b2b9" in moves  # nhảy qua Pháo đen b7 ăn Mã b9
    assert "b2b7" not in moves  # không có ngòi thì không ăn được
    assert "b2b8" not in moves  # sau ngòi chỉ được ăn, không được đi tới ô trống
    assert "b2b3" in moves
    assert "b2b0" not in moves  # không ăn quân mình


def test_pawn_moves_forward_then_sideways_after_river() -> None:
    state = Xiangqi()
    assert moves_from(state, "a3") == {"a4"}
    crossed = Xiangqi("3k5/9/9/9/4P4/9/9/9/9/4K4 w")
    assert moves_from(crossed, "e5") == {"e6", "d5", "f5"}
    black = Xiangqi("3k5/9/9/9/9/4p4/9/9/9/5K3 b")
    assert moves_from(black, "e4") == {"e3", "d4", "f4"}
    last_rank = Xiangqi("P2k5/9/9/9/9/9/9/9/9/4K4 w")
    assert moves_from(last_rank, "a9") == {"b9"}


# ---------------------------------------------------------------------------
# Chiếu, Tướng đối mặt, hết nước
# ---------------------------------------------------------------------------


def test_flying_general_pins_the_only_blocker() -> None:
    # Pháo e1 là quân duy nhất giữa hai Tướng: chỉ được đi dọc cột e, không có ngòi để ăn e9.
    state = Xiangqi("4k4/9/9/9/9/9/9/9/4C4/4K4 w")
    assert moves_from(state, "e1") == {f"e{r}" for r in range(2, 9)}
    pinned_horse = Xiangqi("4k4/9/9/9/9/9/9/9/4N4/4K4 w")
    assert moves_from(pinned_horse, "e1") == set()
    # Tướng đỏ không được bước ra đối mặt Tướng đen trên cột trống.
    open_file = Xiangqi("3k5/9/9/9/9/9/9/9/9/4K4 w")
    assert moves_from(open_file, "e0") == {"e1", "f0"}


def test_moves_must_answer_check() -> None:
    horse_check = Xiangqi("3k5/9/9/9/9/9/9/5n3/9/4K4 w")
    assert horse_check.in_check()
    for m in horse_check.legal_moves():
        horse_check.play(m)
        assert not horse_check.in_check(PLAYER_X)
        horse_check.undo()
    # Mã bị chặn chân thì không chiếu được.
    assert not Xiangqi("3k5/9/9/9/9/9/9/5n3/5P3/4K4 w").in_check()
    # Pháo cần đúng một ngòi.
    assert Xiangqi("4k4/9/9/9/9/4c4/9/4P4/9/4K4 w").in_check()
    assert not Xiangqi("4k4/9/9/9/9/4c4/4P4/4P4/9/4K4 w").in_check()


def test_checkmate_loses_for_side_to_move() -> None:
    state = Xiangqi("3k4R/8R/9/9/9/9/9/9/9/4K4 b")
    assert state.in_check()
    assert state.legal_moves() == []
    assert state.winner() == PLAYER_X
    assert state.is_terminal()


def test_no_moves_without_check_also_loses() -> None:
    state = Xiangqi("3k5/8R/9/9/9/9/9/9/9/4K4 b")
    assert not state.in_check()
    assert state.legal_moves() == []
    assert state.winner() == PLAYER_X


def test_play_mating_move_ends_game() -> None:
    state = Xiangqi("3k5/8R/9/9/9/9/9/9/9/R3K4 w")
    state.play(parse_iccs("a0a9"))
    assert state.winner() == PLAYER_X
    with pytest.raises(ValueError):
        state.play(parse_iccs("d9d8"))
    state.undo()
    assert state.winner() is None


# ---------------------------------------------------------------------------
# Hòa
# ---------------------------------------------------------------------------


def test_threefold_repetition_is_draw() -> None:
    state = Xiangqi()
    cycle = ("b0c2", "b9c7", "c2b0", "c7b9")
    play_iccs(state, *cycle)
    assert state.winner() is None  # thế đầu xuất hiện lần thứ hai
    play_iccs(state, *cycle)
    assert state.winner() == DRAW  # lần thứ ba
    assert state.legal_moves() == []
    state.undo()
    assert state.winner() is None


def test_no_capture_limit_is_draw() -> None:
    state = Xiangqi(no_capture_limit=3)
    play_iccs(state, "h2e2", "h9g7")
    assert state.winner() is None
    play_iccs(state, "h0g2")
    assert state.winner() == DRAW


def test_capture_resets_no_capture_counter() -> None:
    state = Xiangqi(no_capture_limit=3)
    play_iccs(state, "h2e2", "h9g7", "e2e6")  # Pháo ăn Tốt e6
    assert state.winner() is None
    assert state.to_fen().split()[4] == "0"
    state.undo()
    assert state.to_fen().split()[4] == "2"


def test_max_plies_is_draw() -> None:
    state = Xiangqi(max_plies=2)
    play_iccs(state, "h2e2")
    assert state.winner() is None
    play_iccs(state, "h9g7")
    assert state.winner() == DRAW


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValueError):
        Xiangqi(no_capture_limit=0)
    with pytest.raises(ValueError):
        Xiangqi(max_plies=0)


# ---------------------------------------------------------------------------
# play / undo / hash / clone
# ---------------------------------------------------------------------------


def test_illegal_move_and_empty_undo_raise() -> None:
    state = Xiangqi()
    with pytest.raises(ValueError):
        state.play(parse_iccs("b2b7"))
    with pytest.raises(ValueError):
        state.play((0, 1))  # quân đen, chưa tới lượt
    with pytest.raises(ValueError):
        state.undo()


def test_last_move_and_move_count() -> None:
    state = Xiangqi()
    assert state.last_move() is None
    play_iccs(state, "h2e2")
    assert state.last_move() == parse_iccs("h2e2")
    assert state.move_count() == 1
    state.undo()
    assert state.last_move() is None
    assert state.move_count() == 0


def test_transpositions_share_hash() -> None:
    a = Xiangqi()
    play_iccs(a, "h2e2", "h9g7", "b0c2")
    b = Xiangqi()
    play_iccs(b, "b0c2", "h9g7", "h2e2")
    assert a.hash() == b.hash()
    assert a.to_fen().split()[0] == b.to_fen().split()[0]
    c = Xiangqi()
    play_iccs(c, "h2e2", "h9g7")
    assert c.hash() != a.hash()


def test_clone_is_independent() -> None:
    state = Xiangqi()
    play_iccs(state, "h2e2")
    copy = state.clone()
    play_iccs(copy, "h9g7")
    assert state.move_count() == 1
    assert state.to_fen() != copy.to_fen()
    assert state.piece_at(parse_square("g7")) == 0
    copy.undo()
    assert copy.to_fen() == state.to_fen()
    assert copy.hash() == state.hash()


def _independent_legal(state: Xiangqi) -> set[tuple[int, int]]:
    """Nước hợp lệ tính theo cách khác: sau nước đi, đối thủ không có nước giả nào ăn được
    Tướng mình và hai Tướng không đối mặt trên một cột trống."""
    side = state.current_player()
    legal = set()
    for f, t in state._pseudo_moves(side):
        sq = state._sq
        p, cap = sq[f], sq[t]
        sq[t], sq[f] = p, 0
        kings = {s: v for s, v in enumerate(sq) if abs(v) == KING}
        own = next(s for s, v in kings.items() if v == side * KING)
        other = next(s for s, v in kings.items() if v == -side * KING)
        attacked = any(to == own for _, to in state._pseudo_moves(-side))
        if own % COLS == other % COLS:
            lo, hi = sorted((own, other))
            if all(sq[x] == 0 for x in range(lo + COLS, hi, COLS)):
                attacked = True
        sq[f], sq[t] = p, cap
        if not attacked:
            legal.add((f, t))
    return legal


@pytest.mark.parametrize("seed", range(8))
def test_random_games_stay_consistent(seed: int) -> None:
    rng = random.Random(seed)
    state = Xiangqi(max_plies=120)
    hashes = []
    while not state.is_terminal():
        assert set(state.legal_moves()) == _independent_legal(state)
        assert state.hash() == state._compute_hash()
        assert np.array_equal(state.board.reshape(-1), np.array(state._sq, dtype=np.int8))
        hashes.append((state.hash(), state.to_fen()))
        state.play(rng.choice(state.legal_moves()))
    assert state.winner() is not None
    while state.move_count():
        state.undo()
        key, fen = hashes.pop()
        assert state.hash() == key
        assert state.to_fen().split()[:5] == fen.split()[:5]
    assert state.to_fen() == START_FEN


def test_mcts_can_play_xiangqi() -> None:
    state = Xiangqi(max_plies=60)
    agent = MCTSAgent(rollout="l0", seed=1)
    move = agent.select_move(state, Budget(iterations=20))
    assert move in state.legal_moves()
    assert state.to_fen() == START_FEN
