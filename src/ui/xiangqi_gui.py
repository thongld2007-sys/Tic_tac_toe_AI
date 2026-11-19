"""Giao diện đồ họa (GUI) cho Cờ tướng (Xiangqi — P4.3b).

Chủ sở hữu: TV3.  Review: TV1.

Tính năng chính:
  1. Vẽ bàn Cờ tướng chuẩn 10 hàng × 9 cột (quân đặt tại giao điểm), có sông Sở Hà – Hán Giới,
     hai Cửu cung đường chéo, tọa độ ICCS (`a`–`i`, `0`–`9`), đánh dấu nước vừa đi, tô sáng quân
     đang chọn, chấm xanh các ô đích hợp lệ và cảnh báo đỏ khi Tướng đang bị chiếu.
  2. Hỗ trợ đủ 4 chế độ chơi:
       - Người vs AI (`human_vs_ai`: Người cầm Đỏ đi trước, AI cầm Đen)
       - AI vs Người (`ai_vs_human`: AI cầm Đỏ, Người cầm Đen)
       - AI vs AI (`ai_vs_ai`: chạy tự động / tạm dừng hoặc đi từng bước)
       - Người vs Người (`human_vs_human`)
  3. Hỗ trợ trực tiếp các agent Cờ tướng từ P4.2b (`alphabeta_xiangqi`, `mcts_xiangqi`, `random`)
     và chỉnh tham số (`depth`, `quiescence_depth`, `iterations`, `time_s`, `c`, `rollout_depth`)
     ngay trên giao diện; hỗ trợ nạp thế cờ mẫu (chiếu bí 1 nước, ăn Xe không bảo vệ, trung cuộc)
     hoặc chuỗi FEN bất kỳ, kèm hoàn tác (`undo`) và bảng lịch sử nước đi theo ký hiệu ICCS.
  4. Tách lõi điều khiển (`XiangqiBoardGeometry`, `XiangqiPlayerConfig`, `XiangqiSession`) hoàn
     toàn khỏi Tkinter để kiểm thử tự động và chạy `--headless-demo` không cần màn hình.

Cách chạy từ thư mục gốc của repo:
    python -m src.ui.xiangqi_gui
    python -m src.ui.xiangqi_gui --mode ai_vs_ai --red alphabeta_xiangqi --black mcts_xiangqi
    python -m src.ui.xiangqi_gui --preset mate_in_1 --headless-demo
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from src.agents.base import Agent, AgentStats, Budget
from src.agents.mcts import FINAL_SELECTIONS, UCT_VARIANTS, MCTSAgent
from src.agents.random_agent import RandomAgent
from src.agents.xiangqi_agents import (
    DEFAULT_L4_CUTOFF,
    AlphaBetaXiangqi,
    XiangqiMCTSAgent,
    evaluate_xiangqi,
)
from src.games.base import DRAW, Move
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
    START_FEN,
    Xiangqi,
    move_to_iccs,
    square_name,
)

XIANGQI_AGENT_TYPES: tuple[str, ...] = (
    "human",
    "alphabeta_xiangqi",
    "mcts_xiangqi",
    "random",
)
XIANGQI_MODES: tuple[str, ...] = (
    "human_vs_ai",
    "ai_vs_human",
    "ai_vs_ai",
    "human_vs_human",
)

# Ký tự Hán tự truyền thống trên quân cờ Đỏ (> 0) và Đen (< 0).
PIECE_GLYPHS: dict[int, str] = {
    KING: "帥",
    ADVISOR: "仕",
    ELEPHANT: "相",
    HORSE: "傌",
    ROOK: "俥",
    CANNON: "炮",
    PAWN: "兵",
    -KING: "將",
    -ADVISOR: "士",
    -ELEPHANT: "象",
    -HORSE: "馬",
    -ROOK: "車",
    -CANNON: "砲",
    -PAWN: "卒",
}

PIECE_NAMES_VI: dict[int, str] = {
    KING: "Tướng",
    ADVISOR: "Sĩ",
    ELEPHANT: "Tượng",
    HORSE: "Mã",
    ROOK: "Xe",
    CANNON: "Pháo",
    PAWN: "Tốt",
}

# Các thế cờ mẫu phục vụ demo và vấn đáp P4.3b.
XIANGQI_PRESETS: dict[str, tuple[str, str]] = {
    "start": ("Thế xuất phát chuẩn", START_FEN),
    "mate_in_1": (
        "Chiếu bí 1 nước (Đỏ đi)",
        "4k4/RR7/9/9/9/9/9/9/9/3K5 w - - 0 1",
    ),
    "free_rook": (
        "Ăn Xe không bảo vệ (Đỏ đi)",
        "3k5/9/4r4/9/9/9/9/4R4/9/5K3 w - - 0 1",
    ),
    "midgame": (
        "Trung cuộc Pháo đầu (Đỏ đi)",
        "r1bakab1r/9/1cn4c1/p1p1p1p1p/9/2P6/P3P1P1P/1C2C2N1/9/RNBAKAB1R w - - 0 1",
    ),
}


def piece_glyph(piece: int) -> str:
    """Trả về ký tự hiển thị trên mặt quân cờ (rỗng nếu ô trống)."""
    return PIECE_GLYPHS.get(int(piece), "")


# ---------------------------------------------------------------------------
# Cấu hình người chơi / thuật toán cho Cờ tướng
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XiangqiPlayerConfig:
    """Cấu hình thuật toán và ngân sách cho một bên Cờ tướng (Đỏ hoặc Đen)."""

    agent_type: str = "human"
    depth: int | None = 2
    quiescence_depth: int = 2
    iterations: int | None = 100
    time_s: float | None = None
    c: float = 1.0
    selection: str = "robust"
    uct_variant: str = "standard"
    rollout_depth: int | None = DEFAULT_L4_CUTOFF
    reuse_tree: bool = True
    use_move_ordering: bool = True
    use_iterative_deepening: bool = True
    use_tt: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        kind = self.agent_type.strip().lower()
        if kind not in XIANGQI_AGENT_TYPES:
            raise ValueError(
                f"agent_type phải thuộc {XIANGQI_AGENT_TYPES}, nhận được {self.agent_type!r}"
            )
        object.__setattr__(self, "agent_type", kind)

        if self.depth is not None and (
            isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 1
        ):
            raise ValueError(f"depth phải là số nguyên >= 1, nhận được {self.depth!r}")
        if (
            isinstance(self.quiescence_depth, bool)
            or not isinstance(self.quiescence_depth, int)
            or self.quiescence_depth < 0
        ):
            raise ValueError(
                f"quiescence_depth phải là số nguyên >= 0, nhận được {self.quiescence_depth!r}"
            )
        if self.iterations is not None and (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations < 1
        ):
            raise ValueError(f"iterations phải là số nguyên >= 1, nhận được {self.iterations!r}")
        if self.time_s is not None:
            if (
                isinstance(self.time_s, bool)
                or not isinstance(self.time_s, (int, float))
                or not math.isfinite(self.time_s)
                or self.time_s <= 0.0
            ):
                raise ValueError(f"time_s phải là số thực > 0, nhận được {self.time_s!r}")
            object.__setattr__(self, "time_s", float(self.time_s))
        if (
            isinstance(self.c, bool)
            or not isinstance(self.c, (int, float))
            or not math.isfinite(self.c)
            or self.c < 0.0
        ):
            raise ValueError(f"c phải là số thực >= 0, nhận được {self.c!r}")
        if self.selection not in FINAL_SELECTIONS:
            raise ValueError(f"selection phải thuộc {FINAL_SELECTIONS}, nhận {self.selection!r}")
        if self.uct_variant not in UCT_VARIANTS:
            raise ValueError(f"uct_variant phải thuộc {UCT_VARIANTS}, nhận {self.uct_variant!r}")
        if self.rollout_depth is not None and (
            isinstance(self.rollout_depth, bool)
            or not isinstance(self.rollout_depth, int)
            or self.rollout_depth < 1
        ):
            raise ValueError(
                f"rollout_depth phải là số nguyên >= 1, nhận được {self.rollout_depth!r}"
            )
        for flag in ("reuse_tree", "use_move_ordering", "use_iterative_deepening", "use_tt"):
            if not isinstance(getattr(self, flag), bool):
                raise ValueError(f"{flag} phải là kiểu bool")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed phải là số nguyên, nhận được {self.seed!r}")

    @property
    def is_human(self) -> bool:
        return self.agent_type == "human"

    def to_budget(self) -> Budget:
        """Tạo `Budget` tương ứng với thuật toán đã cấu hình."""
        if self.agent_type == "human" or self.agent_type == "random":
            return Budget()
        if self.agent_type == "alphabeta_xiangqi":
            if self.time_s is not None:
                return Budget(depth=self.depth, time_s=self.time_s)
            return Budget(depth=self.depth if self.depth is not None else 2)
        if self.agent_type == "mcts_xiangqi":
            if self.time_s is not None:
                return Budget(iterations=self.iterations, time_s=self.time_s)
            return Budget(iterations=self.iterations if self.iterations is not None else 100)
        return Budget()

    @classmethod
    def from_mapping(
        cls,
        data: XiangqiPlayerConfig | Mapping[str, Any] | str,
        **overrides: Any,
    ) -> XiangqiPlayerConfig:
        if isinstance(data, XiangqiPlayerConfig):
            return replace(data, **overrides) if overrides else data
        if isinstance(data, str):
            return cls(agent_type=data, **overrides)
        if not isinstance(data, Mapping):
            raise ValueError(f"Cấu hình người chơi không hợp lệ: {data!r}")
        raw = dict(data)
        raw.update(overrides)
        if "type" in raw and "agent_type" not in raw:
            raw["agent_type"] = raw.pop("type")
        return cls(**raw)


def build_xiangqi_agent(
    config: XiangqiPlayerConfig | Mapping[str, Any] | str,
) -> tuple[Agent | None, Budget]:
    """Khởi tạo `(agent, budget)` cho bàn Cờ tướng từ `XiangqiPlayerConfig`."""
    cfg = XiangqiPlayerConfig.from_mapping(config)
    budget = cfg.to_budget()
    if cfg.agent_type == "human":
        return None, budget
    if cfg.agent_type == "random":
        return RandomAgent(seed=cfg.seed), budget
    if cfg.agent_type == "alphabeta_xiangqi":
        default_d = cfg.depth if cfg.depth is not None else 2
        return (
            AlphaBetaXiangqi(
                default_depth=default_d,
                quiescence_depth=cfg.quiescence_depth,
                use_move_ordering=cfg.use_move_ordering,
                use_iterative_deepening=cfg.use_iterative_deepening,
                use_tt=cfg.use_tt,
            ),
            budget,
        )
    if cfg.agent_type == "mcts_xiangqi":
        return (
            XiangqiMCTSAgent(
                c=cfg.c,
                selection=cfg.selection,
                rollout_depth=cfg.rollout_depth,
                reuse_tree=cfg.reuse_tree,
                uct_variant=cfg.uct_variant,
                seed=cfg.seed,
            ),
            budget,
        )
    raise ValueError(f"Không hỗ trợ agent_type={cfg.agent_type!r}")


# ---------------------------------------------------------------------------
# Hình học bàn Cờ tướng 10×9 (giao điểm, không phụ thuộc Tkinter)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XiangqiBoardGeometry:
    """Ánh xạ tọa độ pixel <-> chỉ số giao điểm `0..89` trên bàn Cờ tướng 10×9."""

    canvas_width: int
    canvas_height: int
    padding: float = 42.0

    def __post_init__(self) -> None:
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            raise ValueError("Kích thước canvas phải > 0")

    def grid_metrics(self) -> tuple[float, float, float]:
        """Trả về `(x0, y0, step)` của lưới giao điểm 9 cột (8 khoảng) × 10 hàng (9 khoảng)."""
        usable_w = max(40.0, float(self.canvas_width) - 2.0 * self.padding)
        usable_h = max(40.0, float(self.canvas_height) - 2.0 * self.padding)
        step = min(usable_w / (COLS - 1), usable_h / (ROWS - 1))
        grid_w = step * (COLS - 1)
        grid_h = step * (ROWS - 1)
        x0 = (self.canvas_width - grid_w) * 0.5
        y0 = (self.canvas_height - grid_h) * 0.5
        return x0, y0, step

    def square_center(self, sq: int) -> tuple[float, float]:
        """Trả về tọa độ pixel `(cx, cy)` của giao điểm `sq` (`0..89`)."""
        if not (0 <= sq < SQUARES):
            raise ValueError(f"Chỉ số ô phải thuộc 0..{SQUARES - 1}, nhận được {sq}")
        r, c = divmod(sq, COLS)
        x0, y0, step = self.grid_metrics()
        return x0 + c * step, y0 + r * step

    def pixel_to_square(self, x: float, y: float) -> int | None:
        """Ánh xạ điểm nhấp chuột `(x, y)` sang chỉ số giao điểm `0..89` gần nhất (hoặc `None`)."""
        x0, y0, step = self.grid_metrics()
        col = round((x - x0) / step)
        row = round((y - y0) / step)
        if not (0 <= row < ROWS and 0 <= col < COLS):
            return None
        cx = x0 + col * step
        cy = y0 + row * step
        if math.hypot(x - cx, y - cy) <= step * 0.48:
            return row * COLS + col
        return None


# ---------------------------------------------------------------------------
# Bộ điều khiển phiên chơi Cờ tướng (XiangqiSession — thuần Python)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XiangqiMoveRecord:
    """Bản ghi một nước đi Cờ tướng kèm ký hiệu ICCS và thống kê AI."""

    ply: int
    player: int
    move: Move
    iccs: str
    agent_name: str
    stats: AgentStats = field(default_factory=dict)


class XiangqiSession:
    """Quản lý trạng thái bàn Cờ tướng, hai người chơi, chọn quân 2 nhịp, undo và nạp FEN."""

    def __init__(
        self,
        *,
        fen: str = START_FEN,
        max_plies: int = 240,
        player_red: XiangqiPlayerConfig | Mapping[str, Any] | str = "human",
        player_black: XiangqiPlayerConfig | Mapping[str, Any] | str = "alphabeta_xiangqi",
    ) -> None:
        self.initial_fen = fen
        self.max_plies = max_plies
        self.players: dict[int, XiangqiPlayerConfig] = {
            RED: XiangqiPlayerConfig.from_mapping(player_red),
            BLACK: XiangqiPlayerConfig.from_mapping(player_black),
        }
        self.state: Xiangqi = Xiangqi(fen, max_plies=max_plies)
        self.history: list[XiangqiMoveRecord] = []
        self.selected_square: int | None = None
        self.legal_targets: set[int] = set()
        self._agents: dict[int, tuple[Agent | None, Budget]] = {}
        self._rebuild_agent(RED)
        self._rebuild_agent(BLACK)

    def _rebuild_agent(self, side: int) -> None:
        self._agents[side] = build_xiangqi_agent(self.players[side])

    def clear_selection(self) -> None:
        self.selected_square = None
        self.legal_targets.clear()

    def new_game(self, *, fen: str | None = None) -> Xiangqi:
        """Khởi tạo ván mới (tùy chọn từ chuỗi FEN mới)."""
        if fen is not None:
            self.initial_fen = fen
        self.state = Xiangqi(self.initial_fen, max_plies=self.max_plies)
        self.history.clear()
        self.clear_selection()
        self._rebuild_agent(RED)
        self._rebuild_agent(BLACK)
        return self.state

    def load_preset(self, name: str) -> Xiangqi:
        """Nạp một thế cờ mẫu đã đăng ký trong `XIANGQI_PRESETS`."""
        key = name.strip().lower()
        if key not in XIANGQI_PRESETS:
            raise ValueError(f"Thế cờ mẫu {name!r} không tồn tại; hỗ trợ {sorted(XIANGQI_PRESETS)}")
        _, fen = XIANGQI_PRESETS[key]
        return self.new_game(fen=fen)

    def configure_player(
        self,
        side: int,
        config: XiangqiPlayerConfig | Mapping[str, Any] | str | None = None,
        **overrides: Any,
    ) -> XiangqiPlayerConfig:
        if side not in (RED, BLACK):
            raise ValueError(f"side phải là RED (+1) hoặc BLACK (-1), nhận được {side!r}")
        base = self.players[side] if config is None else config
        new_cfg = XiangqiPlayerConfig.from_mapping(base, **overrides)
        self.players[side] = new_cfg
        self._rebuild_agent(side)
        return new_cfg

    def set_mode(
        self,
        mode: str,
        *,
        ai_red: XiangqiPlayerConfig | Mapping[str, Any] | str | None = None,
        ai_black: XiangqiPlayerConfig | Mapping[str, Any] | str | None = None,
    ) -> None:
        key = mode.strip().lower()
        if key not in XIANGQI_MODES:
            raise ValueError(f"mode phải thuộc {XIANGQI_MODES}, nhận được {mode!r}")
        default_red = (
            ai_red
            if ai_red is not None
            else (self.players[RED] if not self.players[RED].is_human else "alphabeta_xiangqi")
        )
        default_black = (
            ai_black
            if ai_black is not None
            else (self.players[BLACK] if not self.players[BLACK].is_human else "mcts_xiangqi")
        )
        if key == "human_vs_ai":
            self.configure_player(RED, "human")
            self.configure_player(BLACK, default_black)
        elif key == "ai_vs_human":
            self.configure_player(RED, default_red)
            self.configure_player(BLACK, "human")
        elif key == "ai_vs_ai":
            self.configure_player(RED, default_red)
            self.configure_player(BLACK, default_black)
        elif key == "human_vs_human":
            self.configure_player(RED, "human")
            self.configure_player(BLACK, "human")
        self.clear_selection()

    @property
    def match_mode(self) -> str:
        r_h = self.players[RED].is_human
        b_h = self.players[BLACK].is_human
        if r_h and not b_h:
            return "human_vs_ai"
        if not r_h and b_h:
            return "ai_vs_human"
        if not r_h and not b_h:
            return "ai_vs_ai"
        return "human_vs_human"

    @property
    def last_move(self) -> Move | None:
        return self.history[-1].move if self.history else None

    @property
    def last_stats(self) -> AgentStats:
        for rec in reversed(self.history):
            if rec.stats:
                return copy.deepcopy(rec.stats)
        return {}

    def is_current_player_human(self) -> bool:
        return (not self.state.is_terminal()) and self.players[self.state.current_player()].is_human

    def is_current_player_ai(self) -> bool:
        return (not self.state.is_terminal()) and (
            not self.players[self.state.current_player()].is_human
        )

    def eval_red_score(self) -> float:
        """Điểm lượng giá tĩnh (vật chất + PST) theo góc nhìn bên Đỏ."""
        return evaluate_xiangqi(self.state, RED)

    def status_text(self) -> str:
        win = self.state.winner()
        if win == RED:
            return "Kết thúc: Bên ĐỎ thắng!"
        if win == BLACK:
            return "Kết thúc: Bên ĐEN thắng!"
        if win == DRAW:
            return "Kết thúc: Hòa!"

        cur = self.state.current_player()
        side_str = "Đỏ" if cur == RED else "Đen"
        cfg = self.players[cur]
        role = "Người" if cfg.is_human else cfg.agent_type
        chk = " — ĐANG CHIẾU!" if self.state.in_check() else ""
        return f"Lượt đi: Bên {side_str} ({role}) — Nước #{len(self.history) + 1}{chk}"

    def legal_targets_from(self, sq: int) -> set[int]:
        """Trả về tập chỉ số ô đích hợp lệ xuất phát từ ô `sq`."""
        return {to_sq for from_sq, to_sq in self.state.legal_moves() if from_sq == sq}

    def select_square(self, sq: int, *, auto_reply: bool = False) -> list[XiangqiMoveRecord]:
        """Xử lý nhấp chuột 2 nhịp vào giao điểm `sq` (`0..89`) khi tới lượt Người.

        Trả về danh sách nước đã thực hiện (rỗng nếu chỉ chọn/đổi/bỏ chọn quân).
        """
        if not self.is_current_player_human() or not (0 <= sq < SQUARES):
            return []

        cur = self.state.current_player()
        piece = self.state.piece_at(sq)

        if self.selected_square is None:
            if piece * cur > 0:
                targets = self.legal_targets_from(sq)
                if targets:
                    self.selected_square = sq
                    self.legal_targets = targets
            return []

        if sq == self.selected_square:
            self.clear_selection()
            return []

        if sq in self.legal_targets:
            move = (self.selected_square, sq)
            self.clear_selection()
            return self.play_human_move(move, auto_reply=auto_reply)

        if piece * cur > 0:
            targets = self.legal_targets_from(sq)
            if targets:
                self.selected_square = sq
                self.legal_targets = targets
            else:
                self.clear_selection()
            return []

        self.clear_selection()
        return []

    def play_human_move(
        self,
        move: Move,
        *,
        auto_reply: bool = False,
    ) -> list[XiangqiMoveRecord]:
        if self.state.is_terminal():
            raise ValueError("Ván cờ đã kết thúc")
        mv = (int(move[0]), int(move[1]))
        player = self.state.current_player()
        self.state.play(mv)
        rec = XiangqiMoveRecord(
            ply=len(self.history) + 1,
            player=player,
            move=mv,
            iccs=move_to_iccs(mv),
            agent_name="human",
            stats={},
        )
        self.history.append(rec)
        self.clear_selection()
        played = [rec]
        if auto_reply and self.is_current_player_ai():
            played.append(self.step_ai_move())
        return played

    def step_ai_move(
        self,
        *,
        fallback_agent: XiangqiPlayerConfig | Mapping[str, Any] | str = "alphabeta_xiangqi",
    ) -> XiangqiMoveRecord:
        if self.state.is_terminal():
            raise ValueError("Ván cờ đã kết thúc, AI không thể đi tiếp")
        player = self.state.current_player()
        agent, budget = self._agents[player]
        if agent is None:
            agent, budget = build_xiangqi_agent(fallback_agent)
            assert agent is not None

        before_hash = self.state.hash()
        mv = agent.select_move(self.state, budget)
        if self.state.hash() != before_hash:
            raise RuntimeError(f"Agent {agent.name} làm biến đổi trạng thái sau select_move")

        mv_pair = (int(mv[0]), int(mv[1]))
        self.state.play(mv_pair)
        rec = XiangqiMoveRecord(
            ply=len(self.history) + 1,
            player=player,
            move=mv_pair,
            iccs=move_to_iccs(mv_pair),
            agent_name=agent.name,
            stats=copy.deepcopy(agent.last_stats),
        )
        self.history.append(rec)
        self.clear_selection()
        return rec

    def run_ai_vs_ai(self, *, max_plies: int | None = None) -> list[XiangqiMoveRecord]:
        records: list[XiangqiMoveRecord] = []
        plies = 0
        while not self.state.is_terminal():
            if max_plies is not None and plies >= max_plies:
                break
            records.append(self.step_ai_move())
            plies += 1
        return records

    def undo(self, steps: int | None = None) -> int:
        if not self.history:
            return 0
        if steps is not None:
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
                raise ValueError(f"steps phải là số nguyên >= 1, nhận được {steps!r}")
            target = min(steps, len(self.history))
        else:
            mode = self.match_mode
            if mode in ("human_vs_ai", "ai_vs_human"):
                last_was_ai = not self.players[self.history[-1].player].is_human
                target = 2 if (last_was_ai and len(self.history) >= 2) else 1
            else:
                target = 1

        undone = 0
        for _ in range(target):
            if not self.history:
                break
            self.state.undo()
            self.history.pop()
            undone += 1
        self.clear_selection()

        for side in (RED, BLACK):
            agent, _ = self._agents[side]
            if isinstance(agent, MCTSAgent):
                agent.reset_tree()
        return undone


# ---------------------------------------------------------------------------
# Giao diện đồ họa Tkinter (XiangqiGUI)
# ---------------------------------------------------------------------------


class XiangqiGUI:
    """Cửa sổ đồ họa Tkinter chơi và demo Cờ tướng (P4.3b)."""

    def __init__(
        self,
        session: XiangqiSession | None = None,
        *,
        master: Any | None = None,
        ai_delay_ms: int = 120,
    ) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk

        self._tk = tk
        self._ttk = ttk
        self._messagebox = messagebox
        self.session = session if session is not None else XiangqiSession()
        self.ai_delay_ms = max(20, int(ai_delay_ms))

        self.root = tk.Tk() if master is None else master
        self.root.title("Cờ tướng AI (Xiangqi) — Alpha-Beta (PST) & MCTS-L4 (P4.3b)")
        self.root.geometry("1120x780")
        self.root.minsize(920, 680)

        self._ai_running = False
        self._after_id: str | None = None

        self._build_layout()
        self._sync_controls_from_session()
        self.refresh()

    def _build_layout(self) -> None:
        tk = self._tk
        ttk = self._ttk

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Cột trái: bàn cờ + thanh trạng thái
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.status_var = tk.StringVar(value="")
        ttk.Label(
            left_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor=tk.W, pady=(0, 4))

        self.eval_var = tk.StringVar(value="")
        ttk.Label(left_frame, textvariable=self.eval_var, font=("Segoe UI", 10)).pack(
            anchor=tk.W, pady=(0, 6)
        )

        self.canvas = tk.Canvas(
            left_frame,
            bg="#E9CFA0",
            highlightthickness=1,
            highlightbackground="#78350F",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._draw_board())
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Cột phải: bảng điều khiển + thống kê + lịch sử
        right_frame = ttk.Frame(main_frame, width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        # 1. Chế độ chơi và nút hành động
        ctrl_box = ttk.LabelFrame(right_frame, text="Điều khiển ván đấu", padding=8)
        ctrl_box.pack(fill=tk.X, pady=(0, 8))

        mode_row = ttk.Frame(ctrl_box)
        mode_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(mode_row, text="Chế độ:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value=self.session.match_mode)
        mode_cb = ttk.Combobox(
            mode_row,
            textvariable=self.mode_var,
            values=list(XIANGQI_MODES),
            state="readonly",
            width=18,
        )
        mode_cb.pack(side=tk.RIGHT)
        mode_cb.bind("<<ComboboxSelected>>", lambda _e: self.on_mode_change())

        btn_row1 = ttk.Frame(ctrl_box)
        btn_row1.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(btn_row1, text="Ván mới", command=self.on_new_game).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4)
        )
        ttk.Button(btn_row1, text="Hoàn tác (Undo)", command=self.on_undo).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )

        btn_row2 = ttk.Frame(ctrl_box)
        btn_row2.pack(fill=tk.X)
        ttk.Button(btn_row2, text="AI đi 1 nước", command=self.on_step_ai).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4)
        )
        self.ai_loop_btn_var = tk.StringVar(value="Chạy AI vs AI")
        ttk.Button(
            btn_row2,
            textvariable=self.ai_loop_btn_var,
            command=self.on_toggle_ai_loop,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # 2. Nạp thế cờ mẫu / FEN
        preset_box = ttk.LabelFrame(right_frame, text="Thế cờ mẫu & FEN", padding=8)
        preset_box.pack(fill=tk.X, pady=(0, 8))

        p_row = ttk.Frame(preset_box)
        p_row.pack(fill=tk.X, pady=(0, 4))
        self.preset_var = tk.StringVar(value="start")
        ttk.Combobox(
            p_row,
            textvariable=self.preset_var,
            values=list(XIANGQI_PRESETS.keys()),
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(p_row, text="Nạp mẫu", command=self.on_load_preset).pack(side=tk.RIGHT)

        fen_row = ttk.Frame(preset_box)
        fen_row.pack(fill=tk.X)
        self.fen_var = tk.StringVar(value=self.session.state.to_fen())
        ttk.Entry(fen_row, textvariable=self.fen_var).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4)
        )
        ttk.Button(fen_row, text="Nạp FEN", command=self.on_load_fen).pack(side=tk.RIGHT)

        # 3. Cấu hình bên Đỏ và bên Đen
        self._player_vars: dict[int, dict[str, Any]] = {}
        for side, title in ((RED, "Bên Đỏ (đi trước)"), (BLACK, "Bên Đen (đi sau)")):
            pbox = ttk.LabelFrame(right_frame, text=title, padding=6)
            pbox.pack(fill=tk.X, pady=(0, 6))
            r1 = ttk.Frame(pbox)
            r1.pack(fill=tk.X, pady=(0, 3))
            ttk.Label(r1, text="Agent:").pack(side=tk.LEFT)
            ag_var = tk.StringVar(value=self.session.players[side].agent_type)
            ttk.Combobox(
                r1,
                textvariable=ag_var,
                values=list(XIANGQI_AGENT_TYPES),
                state="readonly",
                width=18,
            ).pack(side=tk.RIGHT)

            r2 = ttk.Frame(pbox)
            r2.pack(fill=tk.X)
            depth_var = tk.StringVar(value=str(self.session.players[side].depth or 2))
            iter_var = tk.StringVar(value=str(self.session.players[side].iterations or 100))
            time_var = tk.StringVar(
                value=""
                if self.session.players[side].time_s is None
                else str(self.session.players[side].time_s)
            )
            ttk.Label(r2, text="depth:").pack(side=tk.LEFT)
            ttk.Entry(r2, textvariable=depth_var, width=4).pack(side=tk.LEFT, padx=(2, 6))
            ttk.Label(r2, text="iters:").pack(side=tk.LEFT)
            ttk.Entry(r2, textvariable=iter_var, width=5).pack(side=tk.LEFT, padx=(2, 6))
            ttk.Label(r2, text="time(s):").pack(side=tk.LEFT)
            ttk.Entry(r2, textvariable=time_var, width=5).pack(side=tk.LEFT, padx=(2, 0))

            self._player_vars[side] = {
                "agent_type": ag_var,
                "depth": depth_var,
                "iterations": iter_var,
                "time_s": time_var,
            }

        ttk.Button(
            right_frame,
            text="Áp dụng cấu hình Đỏ / Đen",
            command=self.on_apply_configs,
        ).pack(fill=tk.X, pady=(0, 8))

        # 4. Thống kê tìm kiếm AI & Lịch sử nước đi ICCS
        stats_box = ttk.LabelFrame(right_frame, text="Thống kê AI gần nhất", padding=6)
        stats_box.pack(fill=tk.X, pady=(0, 8))
        self.stats_var = tk.StringVar(value="Chưa có nước đi AI.")
        ttk.Label(stats_box, textvariable=self.stats_var, wraplength=350, justify=tk.LEFT).pack(
            anchor=tk.W
        )

        hist_box = ttk.LabelFrame(right_frame, text="Lịch sử nước đi (ICCS)", padding=6)
        hist_box.pack(fill=tk.BOTH, expand=True)
        self.history_text = tk.Text(
            hist_box,
            height=10,
            state=tk.DISABLED,
            font=("Consolas", 9),
        )
        self.history_text.pack(fill=tk.BOTH, expand=True)

    def _sync_controls_from_session(self) -> None:
        self.mode_var.set(self.session.match_mode)
        self.fen_var.set(self.session.state.to_fen())
        for side in (RED, BLACK):
            cfg = self.session.players[side]
            pv = self._player_vars[side]
            pv["agent_type"].set(cfg.agent_type)
            pv["depth"].set("" if cfg.depth is None else str(cfg.depth))
            pv["iterations"].set("" if cfg.iterations is None else str(cfg.iterations))
            pv["time_s"].set("" if cfg.time_s is None else str(cfg.time_s))

    def _read_config(self, side: int) -> XiangqiPlayerConfig:
        pv = self._player_vars[side]
        ag_type = pv["agent_type"].get().strip()
        d_raw = pv["depth"].get().strip()
        i_raw = pv["iterations"].get().strip()
        t_raw = pv["time_s"].get().strip()
        return replace(
            self.session.players[side],
            agent_type=ag_type,
            depth=int(d_raw) if d_raw else None,
            iterations=int(i_raw) if i_raw else None,
            time_s=float(t_raw) if t_raw else None,
        )

    def on_apply_configs(self) -> bool:
        try:
            cfg_r = self._read_config(RED)
            cfg_b = self._read_config(BLACK)
        except ValueError as exc:
            self._messagebox.showerror("Lỗi tham số", str(exc), parent=self.root)
            return False
        self.session.configure_player(RED, cfg_r)
        self.session.configure_player(BLACK, cfg_b)
        self.mode_var.set(self.session.match_mode)
        self.refresh()
        if self.session.is_current_player_ai() and self.session.match_mode != "ai_vs_ai":
            self._schedule_ai_step()
        return True

    def on_mode_change(self) -> None:
        self._stop_ai_loop()
        self.session.set_mode(self.mode_var.get().strip())
        self._sync_controls_from_session()
        self.refresh()
        if self.session.is_current_player_ai() and self.session.match_mode != "ai_vs_ai":
            self._schedule_ai_step()

    def on_new_game(self) -> None:
        self._stop_ai_loop()
        if not self.on_apply_configs():
            return
        self.session.new_game(fen=START_FEN)
        self.fen_var.set(self.session.state.to_fen())
        self.refresh()
        if self.session.is_current_player_ai() and self.session.match_mode != "ai_vs_ai":
            self._schedule_ai_step()

    def on_load_preset(self) -> None:
        self._stop_ai_loop()
        try:
            self.session.load_preset(self.preset_var.get().strip())
        except ValueError as exc:
            self._messagebox.showerror("Lỗi thế cờ mẫu", str(exc), parent=self.root)
            return
        self.fen_var.set(self.session.state.to_fen())
        self.refresh()

    def on_load_fen(self) -> None:
        self._stop_ai_loop()
        try:
            self.session.new_game(fen=self.fen_var.get().strip())
        except ValueError as exc:
            self._messagebox.showerror("FEN không hợp lệ", str(exc), parent=self.root)
            return
        self.refresh()

    def on_undo(self) -> None:
        self._stop_ai_loop()
        self.session.undo()
        self.refresh()

    def on_step_ai(self) -> None:
        if self.session.state.is_terminal():
            return
        if not self.on_apply_configs():
            return
        self.session.step_ai_move()
        self.refresh()

    def on_toggle_ai_loop(self) -> None:
        if self._ai_running:
            self._stop_ai_loop()
            return
        if self.session.state.is_terminal():
            self.on_new_game()
        if not self.on_apply_configs():
            return
        if self.session.match_mode != "ai_vs_ai":
            self.session.set_mode("ai_vs_ai")
            self._sync_controls_from_session()
        self._ai_running = True
        self.ai_loop_btn_var.set("Tạm dừng AI vs AI")
        self._schedule_ai_step()

    def _stop_ai_loop(self) -> None:
        self._ai_running = False
        self.ai_loop_btn_var.set("Chạy AI vs AI")
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _schedule_ai_step(self) -> None:
        if self._after_id is not None:
            return
        self._after_id = self.root.after(self.ai_delay_ms, self._run_scheduled_ai_step)

    def _run_scheduled_ai_step(self) -> None:
        self._after_id = None
        if self.session.state.is_terminal():
            self._stop_ai_loop()
            return
        if not self.session.is_current_player_ai() and not self._ai_running:
            return
        self.session.step_ai_move()
        self.refresh()
        if self.session.state.is_terminal():
            self._stop_ai_loop()
            return
        if self._ai_running or self.session.is_current_player_ai():
            self._schedule_ai_step()

    def current_geometry(self) -> XiangqiBoardGeometry:
        w = max(360, int(self.canvas.winfo_width()))
        h = max(400, int(self.canvas.winfo_height()))
        if w <= 1:
            w = 620
        if h <= 1:
            h = 680
        return XiangqiBoardGeometry(canvas_width=w, canvas_height=h)

    def _on_canvas_click(self, event: Any) -> None:
        if self._ai_running or not self.session.is_current_player_human():
            return
        sq = self.current_geometry().pixel_to_square(float(event.x), float(event.y))
        if sq is None:
            return
        played = self.session.select_square(sq, auto_reply=False)
        self.refresh()
        if played and self.session.is_current_player_ai():
            self._schedule_ai_step()

    def refresh(self) -> None:
        self.status_var.set(self.session.status_text())
        ev = self.session.eval_red_score()
        self.eval_var.set(f"Lượng giá tĩnh (góc nhìn Đỏ): {ev:+.0f} điểm")
        self.fen_var.set(self.session.state.to_fen())
        self._update_stats_and_history()
        self._draw_board()

    def _update_stats_and_history(self) -> None:
        tk = self._tk
        stats = self.session.last_stats
        if stats:
            parts: list[str] = []
            if "nodes" in stats:
                parts.append(f"nodes={stats['nodes']}")
            if "depth" in stats:
                parts.append(f"depth={stats['depth']}")
            if "cutoffs" in stats:
                parts.append(f"cutoffs={stats['cutoffs']}")
            if "time_s" in stats:
                parts.append(f"time={stats['time_s'] * 1000.0:.1f}ms")
            if "sims_per_s" in stats:
                parts.append(f"sims/s={stats['sims_per_s']:.0f}")
            line = " | ".join(parts)
            root_list = stats.get("root")
            if isinstance(root_list, list) and root_list:
                top_m = ", ".join(
                    f"{move_to_iccs(it['move'])}:N={it['N']},Q={it['q']:+.2f}"
                    for it in root_list[:3]
                )
                line += f"\nTop MCTS: {top_m}"
            self.stats_var.set(line)
        else:
            self.stats_var.set("Chưa có nước đi AI.")

        self.history_text.configure(state=tk.NORMAL)
        self.history_text.delete("1.0", tk.END)
        for rec in self.session.history:
            side_s = "Đỏ" if rec.player == RED else "Đen"
            extra = ""
            if "nodes" in rec.stats:
                extra = f" [nodes={rec.stats['nodes']}"
                if "time_s" in rec.stats:
                    extra += f", {rec.stats['time_s'] * 1000.0:.0f}ms"
                extra += "]"
            self.history_text.insert(
                tk.END,
                f"#{rec.ply:02d} {side_s} ({rec.agent_name}): {rec.iccs}{extra}\n",
            )
        self.history_text.see(tk.END)
        self.history_text.configure(state=tk.DISABLED)

    def _draw_board(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        geom = self.current_geometry()
        x0, y0, step = geom.grid_metrics()
        grid_w = step * (COLS - 1)
        grid_h = step * (ROWS - 1)

        # Khung nền gỗ bàn cờ
        margin = step * 0.52
        canvas.create_rectangle(
            x0 - margin,
            y0 - margin,
            x0 + grid_w + margin,
            y0 + grid_h + margin,
            fill="#F3D9A4",
            outline="#78350F",
            width=3,
        )

        # 10 đường ngang
        for r in range(ROWS):
            py = y0 + r * step
            canvas.create_line(x0, py, x0 + grid_w, py, fill="#5C3A21", width=1.5)

        # 9 đường dọc (các cột trong 1..7 ngắt qua sông giữa hàng 4 và 5)
        for c in range(COLS):
            px = x0 + c * step
            if c in (0, COLS - 1):
                canvas.create_line(px, y0, px, y0 + grid_h, fill="#5C3A21", width=1.5)
            else:
                canvas.create_line(px, y0, px, y0 + 4 * step, fill="#5C3A21", width=1.5)
                canvas.create_line(px, y0 + 5 * step, px, y0 + grid_h, fill="#5C3A21", width=1.5)

        # Đường chéo Cửu cung (trên: hàng 0..2; dưới: hàng 7..9; cột 3..5)
        for r_top, r_bot in ((0, 2), (7, 9)):
            canvas.create_line(
                x0 + 3 * step,
                y0 + r_top * step,
                x0 + 5 * step,
                y0 + r_bot * step,
                fill="#5C3A21",
                width=1.5,
            )
            canvas.create_line(
                x0 + 5 * step,
                y0 + r_top * step,
                x0 + 3 * step,
                y0 + r_bot * step,
                fill="#5C3A21",
                width=1.5,
            )

        # Dòng sông Sở Hà - Hán Giới (dùng cỡ chữ âm = pixel để khớp tỷ lệ trên màn DPI cao)
        river_y = y0 + 4.5 * step
        river_font = ("Serif", -max(11, int(step * 0.36)), "bold")
        canvas.create_text(x0 + 2.0 * step, river_y, text="楚 河", fill="#78350F", font=river_font)
        canvas.create_text(x0 + 6.0 * step, river_y, text="漢 界", fill="#78350F", font=river_font)

        # Nhãn tọa độ ICCS (cột a..i, hàng 9..0)
        coord_font = ("Segoe UI", -max(9, int(step * 0.22)))
        for c in range(COLS):
            px = x0 + c * step
            canvas.create_text(
                px,
                y0 + grid_h + margin * 0.65,
                text=chr(ord("a") + c),
                fill="#5C3A21",
                font=coord_font,
            )
        for r in range(ROWS):
            py = y0 + r * step
            canvas.create_text(
                x0 - margin * 0.68, py, text=str(ROWS - 1 - r), fill="#5C3A21", font=coord_font
            )

        # Đánh dấu nước vừa đi (ô đi và ô đến)
        last_mv = self.session.last_move
        if last_mv is not None:
            for sq in last_mv:
                cx, cy = geom.square_center(sq)
                rad = step * 0.44
                canvas.create_rectangle(
                    cx - rad, cy - rad, cx + rad, cy + rad, outline="#D97706", width=2.5
                )

        # Vẽ quân cờ trên 90 giao điểm (cỡ chữ âm = pixel để chữ luôn nằm gọn trong vòng tròn)
        in_check = self.session.state.in_check()
        cur_side = self.session.state.current_player()
        piece_rad = step * 0.41
        piece_font = ("Serif", -max(12, int(step * 0.42)), "bold")

        for sq in range(SQUARES):
            piece = self.session.state.piece_at(sq)
            if piece == 0:
                continue
            cx, cy = geom.square_center(sq)
            is_red = piece > 0
            ink = "#B91C1C" if is_red else "#1E293B"
            border = "#2563EB" if sq == self.session.selected_square else ink
            width = 3.2 if sq == self.session.selected_square else 2.0
            if in_check and piece == cur_side * KING:
                border = "#DC2626"
                width = 3.5

            canvas.create_oval(
                cx - piece_rad,
                cy - piece_rad,
                cx + piece_rad,
                cy + piece_rad,
                fill="#FEF9EC",
                outline=border,
                width=width,
            )
            inner_r = piece_rad * 0.84
            canvas.create_oval(
                cx - inner_r,
                cy - inner_r,
                cx + inner_r,
                cy + inner_r,
                fill="",
                outline=ink,
                width=1.0,
            )
            canvas.create_text(cx, cy, text=piece_glyph(piece), fill=ink, font=piece_font)

        # Đánh dấu các ô đích hợp lệ của quân đang chọn
        for target_sq in self.session.legal_targets:
            cx, cy = geom.square_center(target_sq)
            if self.session.state.piece_at(target_sq) != 0:
                rad = step * 0.45
                canvas.create_oval(
                    cx - rad, cy - rad, cx + rad, cy + rad, outline="#16A34A", width=3.0
                )
            else:
                dot = step * 0.14
                canvas.create_oval(
                    cx - dot, cy - dot, cx + dot, cy + dot, fill="#16A34A", outline=""
                )

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Giao diện dòng lệnh & chế độ kiểm thử không cần màn hình (--headless-demo)
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Giao diện đồ họa Cờ tướng (Xiangqi — Alpha-Beta PST & MCTS-L4)."
    )
    parser.add_argument("--mode", choices=XIANGQI_MODES, default="human_vs_ai")
    parser.add_argument("--red", choices=XIANGQI_AGENT_TYPES, default=None)
    parser.add_argument("--black", choices=XIANGQI_AGENT_TYPES, default=None)
    parser.add_argument("--preset", choices=sorted(XIANGQI_PRESETS), default=None)
    parser.add_argument("--fen", type=str, default=None)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--time-s", type=float, default=None)
    parser.add_argument(
        "--headless-demo",
        action="store_true",
        help="Chạy thử một vài nước AI không cần mở cửa sổ (phục vụ kiểm thử tự động).",
    )
    return parser


def create_session_from_args(args: argparse.Namespace) -> XiangqiSession:
    fen = START_FEN
    if args.preset is not None:
        _, fen = XIANGQI_PRESETS[args.preset]
    if args.fen is not None:
        fen = args.fen

    session = XiangqiSession(fen=fen)
    session.set_mode(args.mode)
    if args.red is not None:
        session.configure_player(
            RED,
            args.red,
            depth=args.depth,
            iterations=args.iterations,
            time_s=args.time_s,
        )
    if args.black is not None:
        session.configure_player(
            BLACK,
            args.black,
            depth=args.depth,
            iterations=args.iterations,
            time_s=args.time_s,
        )
    return session


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    session = create_session_from_args(args)

    if args.headless_demo:
        if session.match_mode != "ai_vs_ai":
            session.set_mode("ai_vs_ai", ai_red="alphabeta_xiangqi", ai_black="mcts_xiangqi")
        records = session.run_ai_vs_ai(max_plies=4)
        print(f"Xiangqi headless demo ({len(records)} nước):")
        for rec in records:
            side_s = "Đỏ" if rec.player == RED else "Đen"
            frm, to = square_name(rec.move[0]), square_name(rec.move[1])
            print(f"  #{rec.ply:02d} {side_s} ({rec.agent_name}): {rec.iccs} ({frm}->{to})")
        print(session.status_text())
        return 0

    gui = XiangqiGUI(session)
    gui.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
