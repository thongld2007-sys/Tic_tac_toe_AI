"""Agent Cờ tướng (Xiangqi): Alpha-Beta (vật chất + vị trí) và MCTS với Rollout L4 (P4.2b).

Chủ sở hữu: TV2.  Review: TV3.

Thành phần chính:
  1. Hàm lượng giá tĩnh `evaluate_xiangqi(state, player)` kết hợp:
       - Giá trị vật chất chuẩn Cờ tướng (Xe 900, Pháo 450, Mã 400, Sĩ 200, Tượng 200, Tốt 100).
       - Bảng vị trí (Piece-Square Tables — PST 10×9) đối xứng theo chiều dọc giữa Đỏ và Đen,
         phản ánh đặc thù Cờ tướng (Tốt qua sông áp cửu cung, Pháo đầu, Xe chiếm trục/tuần hà,
         Mã trung tâm, Sĩ/Tượng bảo vệ cửu cung).
     Toàn bộ điểm `vật chất + vị trí` được gộp sẵn vào bảng phẳng 15×90 `_SCORE_TABLE` để chấm
     điểm bàn cờ trong một lượt quét mảng 90 ô cực nhanh.
  2. `AlphaBetaXiangqi(Agent)`:
       - Negamax với cắt tỉa Alpha-Beta, ưu tiên chiếu hết sớm (`MATE_SCORE - ply`).
       - Sắp xếp nước đi: TT move -> MVV-LVA cho nước ăn quân -> Killer moves -> chênh lệch PST.
       - Tìm kiếm yên lặng ngắn (Quiescence Search trên các nước ăn quân) tránh hiệu ứng chân trời.
       - Tìm kiếm sâu dần (Iterative Deepening) tuân thủ `Budget(depth, time_s)`.
       - Bảng chuyển vị Zobrist (Transposition Table — TT) dùng `state.hash()`.
  3. `rollout_xiangqi_l4` và `XiangqiMCTSAgent(MCTSAgent)`:
       - Rollout L4 cho Cờ tướng: mô phỏng ngắn `d_cut` nước (mặc định 6 ply) với chính sách ưu
         tiên nước ăn quân MVV-LVA / cải thiện vị trí, sau đó cắt sớm và trả về
         `tanh(evaluate_xiangqi(state, player_at_start) / tau)` trong đoạn `(-1, +1)`.
       - Tự động đăng ký `"xiangqi_l4"` vào `ROLLOUT_POLICIES` để gọi trực tiếp từ `MCTSAgent`
         và cấu hình YAML của `experiments/run.py`.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Final

from src.agents.base import Agent, Budget
from src.agents.mcts import MCTSAgent, register_rollout
from src.games.base import DRAW, GameState, Move
from src.games.xiangqi import (
    ADVISOR,
    BLACK,
    CANNON,
    COLS,
    ELEPHANT,
    HORSE,
    KING,
    PAWN,
    RED,
    ROOK,
    ROWS,
    SQUARES,
    Xiangqi,
)

MATE_SCORE: Final[float] = 100_000.0
DEFAULT_L4_CUTOFF: Final[int] = 6
DEFAULT_L4_TAU: Final[float] = 600.0

_EXACT: Final[int] = 0
_LOWERBOUND: Final[int] = 1
_UPPERBOUND: Final[int] = 2

# Giá trị vật chất cơ bản (centipawns) của từng loại quân Cờ tướng.
PIECE_VALUES: Final[dict[int, int]] = {
    KING: 10_000,
    ROOK: 900,
    CANNON: 450,
    HORSE: 400,
    ADVISOR: 200,
    ELEPHANT: 200,
    PAWN: 100,
}

# Bảng vị trí (PST) 10 hàng × 9 cột theo góc nhìn quân Đỏ (hàng 0 phía Đen, hàng 9 phía Đỏ).
_PST_RED_2D: Final[dict[int, tuple[tuple[int, ...], ...]]] = {
    KING: (
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, -10, -5, -10, 0, 0, 0),
        (0, 0, 0, 0, 4, 0, 0, 0, 0),
        (0, 0, 0, 6, 12, 6, 0, 0, 0),
    ),
    ADVISOR: (
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 10, 0, 10, 0, 0, 0),
        (0, 0, 0, 0, 22, 0, 0, 0, 0),
        (0, 0, 0, 12, 0, 12, 0, 0, 0),
    ),
    ELEPHANT: (
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 10, 0, 0, 0, 10, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (6, 0, 0, 0, 24, 0, 0, 0, 6),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 14, 0, 0, 0, 14, 0, 0),
    ),
    HORSE: (
        (4, 10, 16, 14, 10, 14, 16, 10, 4),
        (8, 18, 24, 26, 22, 26, 24, 18, 8),
        (12, 22, 34, 38, 36, 38, 34, 22, 12),
        (10, 24, 32, 36, 40, 36, 32, 24, 10),
        (8, 20, 28, 32, 34, 32, 28, 20, 8),
        (6, 16, 22, 26, 28, 26, 22, 16, 6),
        (4, 12, 18, 20, 22, 20, 18, 12, 4),
        (2, 10, 16, 18, 16, 18, 16, 10, 2),
        (0, 6, 10, 12, 8, 12, 10, 6, 0),
        (-6, 0, 4, 4, 0, 4, 4, 0, -6),
    ),
    ROOK: (
        (14, 18, 18, 22, 24, 22, 18, 18, 14),
        (16, 24, 24, 28, 30, 28, 24, 24, 16),
        (14, 22, 22, 26, 28, 26, 22, 22, 14),
        (14, 22, 22, 26, 28, 26, 22, 22, 14),
        (12, 20, 20, 24, 26, 24, 20, 20, 12),
        (10, 18, 18, 22, 24, 22, 18, 18, 10),
        (8, 16, 16, 20, 22, 20, 16, 16, 8),
        (8, 14, 14, 18, 20, 18, 14, 14, 8),
        (6, 12, 12, 16, 18, 16, 12, 12, 6),
        (-4, 8, 8, 14, 16, 14, 8, 8, -4),
    ),
    CANNON: (
        (10, 12, 14, 16, 18, 16, 14, 12, 10),
        (10, 14, 16, 20, 24, 20, 16, 14, 10),
        (12, 16, 18, 22, 28, 22, 18, 16, 12),
        (10, 14, 16, 20, 24, 20, 16, 14, 10),
        (8, 12, 14, 18, 22, 18, 14, 12, 8),
        (6, 10, 12, 14, 20, 14, 12, 10, 6),
        (4, 8, 10, 12, 18, 12, 10, 8, 4),
        (8, 14, 12, 16, 26, 16, 12, 14, 8),
        (4, 8, 8, 10, 14, 10, 8, 8, 4),
        (2, 6, 6, 8, 10, 8, 6, 6, 2),
    ),
    PAWN: (
        (20, 26, 36, 46, 50, 46, 36, 26, 20),
        (50, 75, 100, 115, 125, 115, 100, 75, 50),
        (55, 80, 105, 120, 130, 120, 105, 80, 55),
        (45, 70, 95, 110, 120, 110, 95, 70, 45),
        (35, 55, 75, 90, 95, 90, 75, 55, 35),
        (4, 0, 10, 0, 16, 0, 10, 0, 4),
        (0, 0, 6, 0, 12, 0, 6, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
}


def _build_score_tables() -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Dựng bảng phẳng 15×90 gộp sẵn `(vật chất + PST)` theo góc nhìn quân Đỏ.

    Chỉ số hàng bảng là `piece + PAWN` (từ `0` tương ứng `-PAWN` đến `14` tương ứng `+PAWN`).
    Đồng thời trả về bảng `abs_val` kích thước 15 chứa `PIECE_VALUES[abs(piece)]` cho MVV-LVA.
    """
    rows: list[tuple[int, ...]] = [tuple([0] * SQUARES) for _ in range(2 * PAWN + 1)]
    abs_vals = [0] * (2 * PAWN + 1)

    for kind, base_val in PIECE_VALUES.items():
        pst_2d = _PST_RED_2D[kind]
        red_flat: list[int] = []
        black_flat: list[int] = []
        for r in range(ROWS):
            for c in range(COLS):
                red_flat.append(base_val + pst_2d[r][c])
                black_flat.append(-(base_val + pst_2d[ROWS - 1 - r][c]))
        rows[kind + PAWN] = tuple(red_flat)
        rows[-kind + PAWN] = tuple(black_flat)
        abs_vals[kind + PAWN] = base_val
        abs_vals[-kind + PAWN] = base_val

    return tuple(rows), tuple(abs_vals)


_SCORE_TABLE, _ABS_PIECE_VALUE = _build_score_tables()


def evaluate_xiangqi(state: Xiangqi, player: int | None = None) -> float:
    """Lượng giá tĩnh thế cờ tướng (vật chất + vị trí) theo góc nhìn của `player`.

    Nếu `player is None`, lượng giá theo `state.current_player()`.
    Giá trị dương nghĩa là lợi thế cho `player`, giá trị âm là bất lợi.
    """
    if not isinstance(state, Xiangqi):
        raise TypeError(f"evaluate_xiangqi chỉ hỗ trợ Xiangqi, nhận được {type(state)!r}")
    target = state.current_player() if player is None else player
    if target not in (RED, BLACK):
        raise ValueError(f"player phải là RED (+1) hoặc BLACK (-1), nhận được {target!r}")
    score_table = _SCORE_TABLE
    red_score = 0
    for sq_idx in range(SQUARES):
        piece = state.piece_at(sq_idx)
        if piece:
            red_score += score_table[piece + PAWN][sq_idx]
    return float(red_score if target == RED else -red_score)


def score_xiangqi_move(state: Xiangqi, move: Move) -> float:
    """Chấm điểm nhanh O(1) cho nước đi `move` (MVV-LVA cho ăn quân + tiến bộ vị trí PST)."""
    f, t = move
    attacker = state.piece_at(f)
    victim = state.piece_at(t)
    side = 1 if attacker > 0 else -1
    pst_delta = side * (_SCORE_TABLE[attacker + PAWN][t] - _SCORE_TABLE[attacker + PAWN][f])
    if victim:
        # MVV-LVA: ưu tiên ăn quân có giá trị lớn bằng quân có giá trị nhỏ
        mvv = _ABS_PIECE_VALUE[victim + PAWN]
        lva = _ABS_PIECE_VALUE[attacker + PAWN]
        return 10_000.0 + mvv * 16.0 - lva * 0.1 + pst_delta
    return float(pst_delta)


class _SearchTimeout(Exception):
    """Ngoại lệ nội bộ báo hiệu hết ngân sách thời gian trong Iterative Deepening."""


def _check_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} phải là số nguyên >= 1, nhận được {value!r}")
    return value


def _check_non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} phải là số nguyên >= 0, nhận được {value!r}")
    return value


def _check_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} phải là kiểu bool, nhận được {value!r}")
    return value


class AlphaBetaXiangqi(Agent):
    """Agent Alpha-Beta cho Cờ tướng: vật chất + PST, MVV-LVA, Quiescence, ID và Zobrist TT."""

    name: str = "alphabeta-xiangqi"

    def __init__(
        self,
        *,
        default_depth: int = 3,
        quiescence_depth: int = 3,
        use_move_ordering: bool = True,
        use_iterative_deepening: bool = True,
        use_tt: bool = True,
    ) -> None:
        super().__init__()
        self.default_depth = _check_positive_int("default_depth", default_depth)
        self.quiescence_depth = _check_non_negative_int("quiescence_depth", quiescence_depth)
        self.use_move_ordering = _check_bool("use_move_ordering", use_move_ordering)
        self.use_iterative_deepening = _check_bool(
            "use_iterative_deepening", use_iterative_deepening
        )
        self.use_tt = _check_bool("use_tt", use_tt)

        self.last_value: float | None = None
        # Bảng chuyển vị: hash -> (depth, value, flag, best_move)
        self._tt: dict[int, tuple[int, float, int, Move | None]] = {}
        self._killers: dict[int, list[Move]] = {}
        self._game_hashes: set[int] = set()
        self._path_hashes: set[int] = set()
        self._last_move_count = -1
        self._max_check_ply = 0
        self._nodes = 0
        self._cutoffs = 0
        self._deadline: float | None = None

    def get_params(self) -> dict[str, Any]:
        """Trả về từ điển đầy đủ các tham số cấu hình của agent."""
        return {
            "default_depth": self.default_depth,
            "quiescence_depth": self.quiescence_depth,
            "use_move_ordering": self.use_move_ordering,
            "use_iterative_deepening": self.use_iterative_deepening,
            "use_tt": self.use_tt,
        }

    def clear_table(self) -> None:
        """Xóa bảng chuyển vị (TT), killer moves và lịch sử lặp lại."""
        self._tt.clear()
        self._killers.clear()
        self._game_hashes.clear()
        self._path_hashes.clear()
        self._last_move_count = -1

    def _record_killer(self, ply: int, move: Move) -> None:
        killers = self._killers.setdefault(ply, [])
        if move not in killers:
            killers.insert(0, move)
            if len(killers) > 2:
                killers.pop()

    def _order_moves(
        self,
        state: Xiangqi,
        moves: list[Move],
        ply: int,
        tt_move: Move | None,
    ) -> list[Move]:
        if not self.use_move_ordering or len(moves) <= 1:
            return moves
        killers = self._killers.get(ply, ())

        def priority(mv: Move) -> float:
            if mv == tt_move:
                return 1e9
            base = score_xiangqi_move(state, mv)
            if base < 10_000.0 and mv in killers:
                return 5_000.0 + base
            return base

        return sorted(moves, key=priority, reverse=True)

    def _quiescence(
        self,
        state: Xiangqi,
        alpha: float,
        beta: float,
        qdepth: int,
        ply: int,
    ) -> float:
        if self._deadline is not None and (self._nodes & 31) == 0:
            if time.perf_counter() >= self._deadline:
                raise _SearchTimeout

        self._nodes += 1
        win = state.winner()
        if win is not None:
            if win == DRAW:
                return 0.0
            return (MATE_SCORE - ply) if win == state.current_player() else -(MATE_SCORE - ply)

        stand_pat = evaluate_xiangqi(state)
        if stand_pat >= beta:
            self._cutoffs += 1
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        if qdepth <= 0:
            return stand_pat

        # Chỉ mở rộng các nước ăn quân giá trị trong tìm kiếm yên lặng
        captures = [mv for mv in state.legal_moves() if state.piece_at(mv[1]) != 0]
        if not captures:
            return stand_pat
        captures.sort(key=lambda mv: score_xiangqi_move(state, mv), reverse=True)

        for mv in captures:
            state.play(mv)
            val = -self._quiescence(state, -beta, -alpha, qdepth - 1, ply + 1)
            state.undo()
            if val >= beta:
                self._cutoffs += 1
                return beta
            if val > alpha:
                alpha = val
        return alpha

    def _negamax(
        self,
        state: Xiangqi,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
    ) -> float:
        if self._deadline is not None and (self._nodes & 31) == 0:
            if time.perf_counter() >= self._deadline:
                raise _SearchTimeout

        self._nodes += 1
        win = state.winner()
        if win is not None:
            if win == DRAW:
                return 0.0
            return (MATE_SCORE - ply) if win == state.current_player() else -(MATE_SCORE - ply)

        key = state.hash()
        # Tránh lặp lại thế cờ đã xuất hiện khi đang chiếm ưu thế (2-fold repetition detection)
        if ply > 0 and (key in self._game_hashes or key in self._path_hashes):
            return 0.0

        alpha_orig = alpha
        tt_move: Move | None = None

        if self.use_tt:
            entry = self._tt.get(key)
            if entry is not None:
                e_depth, e_val, e_flag, tt_move = entry
                if e_depth >= depth:
                    if e_val > MATE_SCORE - 1000.0:
                        e_val -= ply
                    elif e_val < -MATE_SCORE + 1000.0:
                        e_val += ply
                    if e_flag == _EXACT:
                        return e_val
                    if e_flag == _LOWERBOUND and e_val > alpha:
                        alpha = e_val
                    elif e_flag == _UPPERBOUND and e_val < beta:
                        beta = e_val
                    if alpha >= beta:
                        self._cutoffs += 1
                        return e_val

        if depth <= 0:
            if state.in_check() and ply < self._max_check_ply:
                depth = 1
            elif self.quiescence_depth > 0:
                return self._quiescence(state, alpha, beta, self.quiescence_depth, ply)
            else:
                return evaluate_xiangqi(state)

        moves = self._order_moves(state, state.legal_moves(), ply, tt_move)
        if not moves:
            return -(MATE_SCORE - ply)

        best_val = -math.inf
        best_move: Move | None = moves[0]
        self._path_hashes.add(key)
        try:
            for mv in moves:
                is_capture = state.piece_at(mv[1]) != 0
                state.play(mv)
                val = -self._negamax(state, depth - 1, -beta, -alpha, ply + 1)
                state.undo()

                if val > best_val:
                    best_val = val
                    best_move = mv
                if val > alpha:
                    alpha = val
                if alpha >= beta:
                    self._cutoffs += 1
                    if not is_capture:
                        self._record_killer(ply, mv)
                    break
        finally:
            self._path_hashes.discard(key)

        if self.use_tt:
            if best_val <= alpha_orig:
                flag = _UPPERBOUND
            elif best_val >= beta:
                flag = _LOWERBOUND
            else:
                flag = _EXACT
            store_val = best_val
            if store_val > MATE_SCORE - 1000.0:
                store_val += ply
            elif store_val < -MATE_SCORE + 1000.0:
                store_val -= ply
            if len(self._tt) > 100_000:
                self._tt.clear()
            self._tt[key] = (depth, store_val, flag, best_move)

        return best_val

    def _search_root(
        self,
        state: Xiangqi,
        depth: int,
        pv_move: Move | None,
    ) -> tuple[Move, float]:
        moves = self._order_moves(state, state.legal_moves(), 0, pv_move)
        alpha = -math.inf
        beta = math.inf
        best_move = moves[0]
        best_val = -math.inf

        for mv in moves:
            state.play(mv)
            val = -self._negamax(state, depth - 1, -beta, -alpha, 1)
            state.undo()
            if val > best_val:
                best_val = val
                best_move = mv
            if val > alpha:
                alpha = val

        if self.use_tt:
            self._tt[state.hash()] = (depth, best_val, _EXACT, best_move)
        return best_move, best_val

    def select_move(self, state: GameState, budget: Budget) -> Move:
        if not isinstance(state, Xiangqi):
            raise TypeError(f"AlphaBetaXiangqi chỉ hỗ trợ Xiangqi, nhận được {type(state)!r}")
        if state.is_terminal():
            raise ValueError("Ván đã kết thúc, không còn nước để chọn")
        if budget.depth is not None and (isinstance(budget.depth, bool) or budget.depth < 1):
            raise ValueError(f"budget.depth phải >= 1, nhận được {budget.depth!r}")
        if budget.time_s is not None:
            if (
                isinstance(budget.time_s, bool)
                or not math.isfinite(budget.time_s)
                or budget.time_s <= 0.0
            ):
                raise ValueError(f"budget.time_s phải > 0, nhận được {budget.time_s!r}")

        start = time.perf_counter()
        self._nodes = 0
        self._cutoffs = 0
        self._killers.clear()
        self._deadline = (start + float(budget.time_s)) if budget.time_s is not None else None

        start_count = state.move_count()
        if start_count <= 1 or start_count <= self._last_move_count:
            self._game_hashes.clear()
        self._last_move_count = start_count
        root_hash = state.hash()
        self._game_hashes.add(root_hash)
        self._path_hashes = {root_hash}

        legal = state.legal_moves()
        if len(legal) == 1:
            self.last_value = evaluate_xiangqi(state)
            self.last_stats = {
                "nodes": 1,
                "time_s": time.perf_counter() - start,
                "cutoffs": 0,
                "depth": 1,
            }
            return legal[0]

        max_depth = budget.depth if budget.depth is not None else self.default_depth
        if budget.depth is None and budget.time_s is not None:
            max_depth = max(self.default_depth, 8)

        best_move = max(legal, key=lambda mv: score_xiangqi_move(state, mv))
        best_val = evaluate_xiangqi(state)
        completed_depth = 0

        try:
            depths = range(1, max_depth + 1) if self.use_iterative_deepening else (max_depth,)
            for d in depths:
                self._max_check_ply = d + 2
                mv, val = self._search_root(state, d, best_move)
                best_move, best_val = mv, val
                completed_depth = d
                if abs(best_val) >= MATE_SCORE - 100:
                    break
        except _SearchTimeout:
            pass
        finally:
            while state.move_count() > start_count:
                state.undo()
            self._path_hashes.clear()
            self._deadline = None

        self.last_value = best_val
        self.last_stats = {
            "nodes": self._nodes,
            "time_s": time.perf_counter() - start,
            "cutoffs": self._cutoffs,
            "depth": completed_depth,
        }
        return best_move


# ---------------------------------------------------------------------------
# Rollout L4 và MCTS cho Cờ tướng
# ---------------------------------------------------------------------------


def _pick_xiangqi_rollout_move(state: Xiangqi, rng: random.Random) -> Move:
    """Chọn 1 nước nhanh trong rollout L4: ưu tiên ăn quân giá trị cao hoặc cải thiện vị trí."""
    moves = state.legal_moves()
    best_cap: Move | None = None
    best_cap_score = -1.0

    for mv in moves:
        victim = state.piece_at(mv[1])
        if victim:
            sc = (
                _ABS_PIECE_VALUE[victim + PAWN] * 16.0
                - _ABS_PIECE_VALUE[state.piece_at(mv[0]) + PAWN]
            )
            if sc > best_cap_score:
                best_cap_score = sc
                best_cap = mv

    # 80% ưu tiên nước ăn quân tốt nhất (nếu có), 20% khám phá ngẫu nhiên để giữ đa dạng mẫu
    if best_cap is not None and rng.random() < 0.80:
        return best_cap

    # Nếu không ăn quân: chọn mẫu tốt nhất trong 4 nước ngẫu nhiên (tournament selection O(1))
    sample_k = min(4, len(moves))
    best_mv = moves[rng.randrange(len(moves))]
    best_sc = score_xiangqi_move(state, best_mv)
    for _ in range(sample_k - 1):
        cand = moves[rng.randrange(len(moves))]
        sc = score_xiangqi_move(state, cand)
        if sc > best_sc:
            best_sc = sc
            best_mv = cand
    return best_mv


def rollout_xiangqi_l4(
    state: GameState,
    rng: random.Random,
    max_moves: int | None = None,
    *,
    cutoff_depth: int = DEFAULT_L4_CUTOFF,
    tau: float = DEFAULT_L4_TAU,
) -> float:
    """Rollout L4 cho Cờ tướng: mô phỏng ngắn `d_cut` nước rồi cắt sớm bằng `tanh(eval / tau)`.

    Trả về giá trị trong `[-1.0, +1.0]` theo góc nhìn của `state.current_player()` lúc vào hàm,
    và luôn khôi phục nguyên vẹn `state` trước khi thoát.
    """
    if not isinstance(state, Xiangqi):
        raise TypeError(f"rollout_xiangqi_l4 chỉ hỗ trợ Xiangqi, nhận được {type(state)!r}")
    if isinstance(cutoff_depth, bool) or not isinstance(cutoff_depth, int) or cutoff_depth < 0:
        raise ValueError(f"cutoff_depth phải là số nguyên >= 0, nhận được {cutoff_depth!r}")
    if (
        isinstance(tau, bool)
        or not isinstance(tau, (int, float))
        or not math.isfinite(tau)
        or tau <= 0
    ):
        raise ValueError(f"tau phải là số thực hữu hạn > 0, nhận được {tau!r}")

    player_at_start = state.current_player()
    start_count = state.move_count()
    d_cut = cutoff_depth if max_moves is None else max_moves
    plies = 0

    try:
        while not state.is_terminal():
            if plies >= d_cut:
                raw = evaluate_xiangqi(state, player_at_start)
                return math.tanh(raw / float(tau))
            mv = _pick_xiangqi_rollout_move(state, rng)
            state.play(mv)
            plies += 1

        winner = state.winner()
        assert winner is not None
        return float(winner * player_at_start)
    finally:
        while state.move_count() > start_count:
            state.undo()


class XiangqiMCTSAgent(MCTSAgent):
    """Agent MCTS-UCT cấu hình sẵn Rollout L4 (`xiangqi_l4`) tối ưu cho Cờ tướng."""

    def __init__(
        self,
        *,
        c: float = 1.0,
        selection: str = "robust",
        rollout_depth: int | None = DEFAULT_L4_CUTOFF,
        reuse_tree: bool = True,
        uct_variant: str = "standard",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            c=c,
            selection=selection,
            rollout="xiangqi_l4",
            rollout_depth=rollout_depth,
            use_candidates=False,
            reuse_tree=reuse_tree,
            uct_variant=uct_variant,
            seed=seed,
        )

    def select_move(self, state: GameState, budget: Budget) -> Move:
        if state.is_terminal():
            raise ValueError("Ván đã kết thúc, không còn nước để chọn")
        self._validate_budget(budget)

        # Kiểm tra nhanh nước chiếu bí 1 bước (mate-in-1) tại gốc trước khi chạy MCTS
        if isinstance(state, Xiangqi):
            start = time.perf_counter()
            player = state.current_player()
            legal = state.legal_moves()
            checked = 0
            for mv in legal:
                checked += 1
                state.play(mv)
                win = state.winner()
                state.undo()
                if win == player:
                    elapsed = time.perf_counter() - start
                    self.last_root = None
                    self.last_stats = {
                        "nodes": checked,
                        "time_s": elapsed,
                        "depth": 1,
                        "sims_per_s": (1.0 / elapsed) if elapsed > 0 else 0.0,
                        "root": [{"move": mv, "N": 1, "W": 1.0, "q": 1.0, "uct": 1.0}],
                    }
                    return mv
        return super().select_move(state, budget)


# Đăng ký chính sách rollout L4 của Cờ tướng vào bảng chung của MCTS
register_rollout("xiangqi_l4", rollout_xiangqi_l4)
