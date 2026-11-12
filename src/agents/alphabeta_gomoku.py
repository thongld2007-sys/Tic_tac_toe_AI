"""Alpha-Beta cho bàn cờ Caro (Gomoku) — P1.3b (R1.3–R1.4).

Chủ sở hữu: TV3.  Review: TV1.

Các kỹ thuật tích hợp:
  1. Giới hạn nước ứng viên (`candidate_moves(radius)` + giới hạn `max_candidates`).
  2. Phát hiện thắng/chặn tức thì (tạo 5 quân thắng ngay hoặc chặn nước thắng đối thủ).
  3. Sắp xếp thứ tự nước đi (Move ordering): TT move -> Killer moves -> Điểm mẫu cục bộ.
  4. Tìm kiếm sâu dần (Iterative Deepening) tuân thủ ngân sách `Budget(depth, time_s)`.
  5. Bảng chuyển vị Zobrist (Transposition Table — TT) với cờ EXACT / LOWERBOUND / UPPERBOUND.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Final

from src.agents.base import Agent, Budget
from src.eval.patterns import _WEIGHTS, FIVE, evaluate
from src.games.base import DRAW, EMPTY, Board, GameState, Move

WIN_SCORE: Final[float] = float(_WEIGHTS[FIVE])

_EXACT: Final[int] = 0
_LOWERBOUND: Final[int] = 1
_UPPERBOUND: Final[int] = 2

_DIRECTIONS: Final[tuple[tuple[int, int], ...]] = ((0, 1), (1, 0), (1, 1), (1, -1))


class _SearchTimeout(Exception):
    """Ngoại lệ nội bộ báo hiệu hết ngân sách thời gian trong Iterative Deepening."""


@dataclass
class TTEntry:
    """Một mục trong bảng chuyển vị (Transposition Table)."""

    depth: int
    value: float
    flag: int
    best_move: Move | None


def _completes_win_at(
    board: Board,
    r: int,
    c: int,
    player: int,
    win_length: int,
    block_two_ends: bool,
) -> bool:
    """Kiểm tra O(1) xem nếu `player` đặt quân vào ô trống `(r, c)` có thắng ngay không."""
    n = board.shape[0]
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

        if 1 + back + fwd < win_length:
            continue

        if not block_two_ends:
            return True

        br, bc = r - (back + 1) * dr, c - (back + 1) * dc
        fr, fc = r + (fwd + 1) * dr, c + (fwd + 1) * dc
        left_blocked = 0 <= br < n and 0 <= bc < n and board[br, bc] == opponent
        right_blocked = 0 <= fr < n and 0 <= fc < n and board[fr, fc] == opponent
        if not (left_blocked and right_blocked):
            return True

    return False


def _line_potential(board: Board, r: int, c: int, dr: int, dc: int, player: int) -> float:
    """Đánh giá nhanh O(1) tiềm năng của một hướng đi qua `(r, c)` khi `player` đánh vào đó."""
    n = board.shape[0]
    opponent = -player

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

    run = 1 + back + fwd
    br, bc = r - (back + 1) * dr, c - (back + 1) * dc
    fr, fc = r + (fwd + 1) * dr, c + (fwd + 1) * dc

    left_open = 0 <= br < n and 0 <= bc < n and board[br, bc] == EMPTY
    right_open = 0 <= fr < n and 0 <= fc < n and board[fr, fc] == EMPTY

    # Kiểm tra khả năng nhảy 1 ô trống (mẫu đứt quãng như XXX_X, _XX_X_)
    gap_left = 0
    if left_open:
        while (
            0 <= br - (gap_left + 1) * dr < n
            and 0 <= bc - (gap_left + 1) * dc < n
            and board[br - (gap_left + 1) * dr, bc - (gap_left + 1) * dc] == player
        ):
            gap_left += 1

    gap_right = 0
    if right_open:
        while (
            0 <= fr + (gap_right + 1) * dr < n
            and 0 <= fc + (gap_right + 1) * dc < n
            and board[fr + (gap_right + 1) * dr, fc + (gap_right + 1) * dc] == player
        ):
            gap_right += 1

    open_ends = int(left_open) + int(right_open)
    effective_run = run + max(gap_left, gap_right)

    if run >= 5:
        left_opp = 0 <= br < n and 0 <= bc < n and board[br, bc] == opponent
        right_opp = 0 <= fr < n and 0 <= fc < n and board[fr, fc] == opponent
        return 500_000.0 if not (left_opp and right_opp) else 50_000.0
    if run == 4:
        if open_ends == 2:
            return 50_000.0
        if open_ends == 1:
            return 10_000.0
        return 0.0
    if effective_run >= 4 and open_ends >= 1:
        return 8_000.0
    if run == 3:
        if open_ends == 2:
            return 2_500.0
        if open_ends == 1:
            return 300.0
        return 0.0
    if effective_run == 3 and open_ends == 2:
        return 1_500.0
    if run == 2:
        if open_ends == 2:
            return 120.0
        if open_ends == 1:
            return 25.0
    return float(open_ends * 4)


def score_move_local(board: Board, move: Move, player: int) -> float:
    """Chấm điểm nhanh O(1) cho `move` kết hợp tấn công (`player`) và phòng thủ (`-player`)."""
    r, c = move
    n = board.shape[0]
    center = (n - 1) * 0.5
    center_bias = max(0.0, n - (abs(r - center) + abs(c - center))) * 0.1

    attack = 0.0
    defense = 0.0
    for dr, dc in _DIRECTIONS:
        attack += _line_potential(board, r, c, dr, dc, player)
        defense += _line_potential(board, r, c, dr, dc, -player)

    return attack + 0.9 * defense + center_bias


def _check_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} phải là số nguyên >= 1, nhận được {value!r}")
    return value


def _check_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} phải là kiểu bool, nhận được {value!r}")
    return value


class AlphaBetaGomoku(Agent):
    """Agent Alpha-Beta cho bàn cờ Caro với ứng viên, thắng/chặn ngay, sắp xếp nước, ID và TT."""

    name: str = "alphabeta-gomoku"

    def __init__(
        self,
        radius: int = 2,
        default_depth: int = 3,
        max_candidates: int | None = 16,
        use_candidates: bool = True,
        use_immediate_tactics: bool = True,
        use_move_ordering: bool = True,
        use_iterative_deepening: bool = True,
        use_tt: bool = True,
    ) -> None:
        super().__init__()
        self.radius = _check_positive_int("radius", radius)
        self.default_depth = _check_positive_int("default_depth", default_depth)
        self.max_candidates = (
            _check_positive_int("max_candidates", max_candidates)
            if max_candidates is not None
            else None
        )
        self.use_candidates = _check_bool("use_candidates", use_candidates)
        self.use_immediate_tactics = _check_bool("use_immediate_tactics", use_immediate_tactics)
        self.use_move_ordering = _check_bool("use_move_ordering", use_move_ordering)
        self.use_iterative_deepening = _check_bool(
            "use_iterative_deepening", use_iterative_deepening
        )
        self.use_tt = _check_bool("use_tt", use_tt)

        self.last_value: float | None = None
        self._tt: dict[int, TTEntry] = {}
        self._tt_game_sig: tuple[int, int, bool] | None = None
        self._killers: dict[int, list[Move]] = {}
        self._nodes = 0
        self._cutoffs = 0
        self._deadline: float | None = None

    def get_params(self) -> dict[str, Any]:
        """Trả về từ điển đầy đủ các tham số cấu hình của agent."""
        return {
            "radius": self.radius,
            "default_depth": self.default_depth,
            "max_candidates": self.max_candidates,
            "use_candidates": self.use_candidates,
            "use_immediate_tactics": self.use_immediate_tactics,
            "use_move_ordering": self.use_move_ordering,
            "use_iterative_deepening": self.use_iterative_deepening,
            "use_tt": self.use_tt,
        }

    def clear_table(self) -> None:
        """Xóa bảng chuyển vị (TT) và killer moves."""
        self._tt.clear()
        self._killers.clear()

    def _get_moves(self, state: GameState) -> list[Move]:
        if self.use_candidates:
            return state.candidate_moves(radius=self.radius)
        return state.legal_moves()

    def _record_killer(self, ply: int, move: Move) -> None:
        killers = self._killers.setdefault(ply, [])
        if move not in killers:
            killers.insert(0, move)
            if len(killers) > 2:
                killers.pop()

    def _prepare_moves(
        self,
        state: GameState,
        ply: int,
        tt_move: Move | None,
        win_length: int,
        block_two_ends: bool,
    ) -> list[Move]:
        """Thu thập ứng viên, kiểm tra thắng/chặn ngay và sắp xếp nước đi."""
        moves = self._get_moves(state)
        if len(moves) <= 1:
            return moves

        board = state.board
        player = state.current_player()
        opponent = -player

        # 1. Phát hiện thắng ngay hoặc bắt buộc chặn nước thắng của đối thủ
        if self.use_immediate_tactics:
            winning_moves = [
                m
                for m in moves
                if _completes_win_at(board, m[0], m[1], player, win_length, block_two_ends)
            ]
            if winning_moves:
                return [winning_moves[0]]

            blocking_moves = [
                m
                for m in moves
                if _completes_win_at(board, m[0], m[1], opponent, win_length, block_two_ends)
            ]
            if blocking_moves:
                moves = blocking_moves

        if not self.use_move_ordering:
            if self.use_candidates and self.max_candidates is not None:
                return moves[: self.max_candidates]
            return moves

        # 2. Chấm điểm cục bộ và sắp xếp ưu tiên: TT move -> Killer moves -> Điểm mẫu cục bộ
        killers = self._killers.get(ply, ())

        def move_priority(m: Move) -> float:
            if m == tt_move:
                return 1e12
            bonus = 5e8 if m in killers else 0.0
            return bonus + score_move_local(board, m, player)

        ordered = sorted(moves, key=move_priority, reverse=True)
        if (
            self.use_candidates
            and self.max_candidates is not None
            and len(ordered) > self.max_candidates
        ):
            return ordered[: self.max_candidates]
        return ordered

    def _negamax(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        win_length: int,
        block_two_ends: bool,
    ) -> float:
        if self._deadline is not None and (self._nodes & 63) == 0:
            if time.perf_counter() >= self._deadline:
                raise _SearchTimeout

        self._nodes += 1

        winner = state.winner()
        if winner is not None:
            if winner == DRAW:
                return 0.0
            # Ưu tiên thắng ở ply nhỏ (nhanh hơn) và thua ở ply lớn (kéo dài hơn)
            win_val = float(WIN_SCORE + max(0, 500 - ply))
            return win_val if winner == state.current_player() else -win_val

        if depth <= 0:
            return evaluate(state.board, state.current_player(), block_two_ends=block_two_ends)

        alpha_orig = alpha
        state_hash = state.hash()
        tt_move: Move | None = None

        if self.use_tt:
            entry = self._tt.get(state_hash)
            if entry is not None:
                tt_move = entry.best_move
                if entry.depth >= depth:
                    if entry.flag == _EXACT:
                        return entry.value
                    if entry.flag == _LOWERBOUND:
                        alpha = max(alpha, entry.value)
                    elif entry.flag == _UPPERBOUND:
                        beta = min(beta, entry.value)
                    if alpha >= beta:
                        self._cutoffs += 1
                        return entry.value

        moves = self._prepare_moves(state, ply, tt_move, win_length, block_two_ends)
        if not moves:
            return 0.0

        best_val = -float("inf")
        best_move: Move | None = moves[0]

        for m in moves:
            state.play(m)
            try:
                val = -self._negamax(
                    state,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    win_length,
                    block_two_ends,
                )
            finally:
                state.undo()

            if val > best_val:
                best_val = val
                best_move = m
            if val > alpha:
                alpha = val
            if alpha >= beta:
                self._cutoffs += 1
                self._record_killer(ply, m)
                break

        if self.use_tt:
            if best_val <= alpha_orig:
                flag = _UPPERBOUND
            elif best_val >= beta:
                flag = _LOWERBOUND
            else:
                flag = _EXACT
            prev = self._tt.get(state_hash)
            if prev is None or prev.depth <= depth:
                self._tt[state_hash] = TTEntry(
                    depth=depth,
                    value=best_val,
                    flag=flag,
                    best_move=best_move,
                )

        return best_val

    def _search_root(
        self,
        state: GameState,
        depth: int,
        root_moves: list[Move],
        win_length: int,
        block_two_ends: bool,
    ) -> tuple[Move, float, list[Move]]:
        """Duyệt tại nút gốc, trả về `(best_move, best_val, new_ordered_root_moves)`."""
        alpha = -float("inf")
        beta = float("inf")
        best_move = root_moves[0]
        best_val = -float("inf")
        scored_moves: list[tuple[float, Move]] = []

        self._nodes += 1

        for m in root_moves:
            if self._deadline is not None and time.perf_counter() >= self._deadline:
                raise _SearchTimeout

            state.play(m)
            try:
                val = -self._negamax(
                    state,
                    depth - 1,
                    -beta,
                    -alpha,
                    1,
                    win_length,
                    block_two_ends,
                )
            finally:
                state.undo()

            scored_moves.append((val, m))
            if val > best_val:
                best_val = val
                best_move = m
            if val > alpha:
                alpha = val
            if val >= WIN_SCORE:
                break

        # Giữ nguyên thứ tự các nước chưa duyệt nếu dừng sớm do tìm thấy nước thắng tuyệt đối
        searched_set = {m for _, m in scored_moves}
        scored_moves.sort(key=lambda item: item[0], reverse=True)
        new_order = [m for _, m in scored_moves] + [m for m in root_moves if m not in searched_set]

        if self.use_tt:
            self._tt[state.hash()] = TTEntry(
                depth=depth,
                value=best_val,
                flag=_EXACT,
                best_move=best_move,
            )

        return best_move, best_val, new_order

    @staticmethod
    def _validate_budget(budget: Budget) -> tuple[int | None, float | None]:
        depth = budget.depth
        time_s = budget.time_s

        if depth is not None:
            depth = _check_positive_int("budget.depth", depth)
        if time_s is not None:
            if isinstance(time_s, bool) or not isinstance(time_s, (int, float)):
                raise ValueError(f"budget.time_s phải là số thực > 0, nhận được {time_s!r}")
            t_val = float(time_s)
            if math.isnan(t_val) or math.isinf(t_val) or t_val <= 0.0:
                raise ValueError(f"budget.time_s phải hữu hạn và > 0, nhận được {time_s!r}")
            time_s = t_val

        return depth, time_s

    def select_move(self, state: GameState, budget: Budget) -> Move:
        """Chọn nước đi tốt nhất cho `state.current_player()` trong giới hạn `budget`."""
        if state.is_terminal():
            raise ValueError("Không thể chọn nước khi ván cờ đã kết thúc")
        budget_depth, budget_time_s = self._validate_budget(budget)

        start_t = time.perf_counter()
        self._nodes = 0
        self._cutoffs = 0
        self._killers.clear()

        default_win = min(5, int(state.board.shape[0]))
        win_length: int = getattr(state, "win_length", default_win)
        block_two_ends: bool = getattr(state, "block_two_ends", False)

        game_sig = (int(state.board.shape[0]), win_length, block_two_ends)
        if self._tt_game_sig != game_sig or len(self._tt) > 250_000:
            self._tt.clear()
            self._tt_game_sig = game_sig

        tt_entry = self._tt.get(state.hash()) if self.use_tt else None
        tt_move = tt_entry.best_move if tt_entry is not None else None
        root_moves = self._prepare_moves(state, 0, tt_move, win_length, block_two_ends)

        if not root_moves:
            legal = state.legal_moves()
            if not legal:
                raise ValueError("Không có nước đi hợp lệ")
            root_moves = [legal[0]]

        # Nếu chỉ có duy nhất 1 nước bắt buộc (bàn trống đánh tâm, thắng ngay, hoặc chặn duy nhất)
        if len(root_moves) == 1:
            self._nodes = 1
            m0 = root_moves[0]
            if _completes_win_at(
                state.board, m0[0], m0[1], state.current_player(), win_length, block_two_ends
            ):
                self.last_value = float(WIN_SCORE + 499)
            else:
                self.last_value = 0.0
            elapsed = time.perf_counter() - start_t
            self.last_stats = {
                "nodes": self._nodes,
                "time_s": elapsed,
                "cutoffs": 0,
                "depth": 1,
            }
            return m0

        max_depth = budget_depth if budget_depth is not None else self.default_depth
        if budget_time_s is not None:
            self._deadline = start_t + budget_time_s * 0.92
            if budget_depth is None:
                max_depth = max(self.default_depth, 6)
        else:
            self._deadline = None

        best_move = root_moves[0]
        best_val = 0.0
        completed_depth = 0

        depths = (
            range(1, max_depth + 1)
            if self.use_iterative_deepening
            else range(max_depth, max_depth + 1)
        )

        for d in depths:
            try:
                move_d, val_d, root_moves = self._search_root(
                    state,
                    d,
                    root_moves,
                    win_length,
                    block_two_ends,
                )
                best_move = move_d
                best_val = val_d
                completed_depth = d
                if abs(val_d) >= WIN_SCORE:
                    break
            except _SearchTimeout:
                break

        if completed_depth == 0:
            completed_depth = 1

        self.last_value = best_val
        elapsed = time.perf_counter() - start_t
        self.last_stats = {
            "nodes": self._nodes,
            "time_s": elapsed,
            "cutoffs": self._cutoffs,
            "depth": completed_depth,
        }
        return best_move


AlphaBetaGomokuAgent = AlphaBetaGomoku
