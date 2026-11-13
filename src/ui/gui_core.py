"""Giao diện đồ họa (GUI) phần lõi cho Tic-Tac-Toe 3×3 và Caro n×n (P6.2).

Chủ sở hữu: TV2.  Review: TV3.

Tính năng chính:
  1. Vẽ bàn cờ trực quan cho cả Tic-Tac-Toe 3×3 và Caro (Gomoku) kích thước tùy chọn
     (9×9, 11×11, 15×15...), hiển thị tọa độ hàng/cột, đánh dấu nước vừa đi, tô sáng
     đường thắng và hỗ trợ lớp phủ màu/nhãn trên từng ô (`cell_overlays`) để bảng phân tích
     (`src/ui/gui_panel.py` — P6.1b) có thể vẽ heatmap số lượt thăm `N`.
  2. Hỗ trợ đầy đủ các chế độ:
       - Người vs AI (`human_vs_ai`: Người cầm X, AI cầm O)
       - AI vs Người (`ai_vs_human`: AI cầm X, Người cầm O)
       - AI vs AI (`ai_vs_ai`: tự động chạy/tạm dừng hoặc đi từng bước)
       - Người vs Người (`human_vs_human`)
  3. Chọn và đổi thuật toán (`human`, `random`, `minimax`, `alphabeta`, `alphabeta_gomoku`,
     `mcts`) cùng mọi tham số (`iterations`, `depth`, `time_s`, `c`, `selection`, `rollout`,
     `rollout_depth`, `use_candidates`, `candidate_radius`, `reuse_tree`, `ordering`,
     `max_candidates`, `use_immediate_tactics`, `use_move_ordering`,
     `use_iterative_deepening`, `use_tt`, `seed`) trực tiếp trên giao diện hoặc qua cấu hình
     mà không phải sửa code và không cần khởi động lại chương trình.
  4. Hoàn tác (`undo`) thông minh: hỗ trợ hoàn tác từng nước đơn lẻ (`steps=1`) hoặc hoàn tác
     cả lượt Người + AI (`steps=None` trong chế độ Người vs AI), khôi phục chính xác trạng thái
     bàn cờ, lượt đi, lịch sử nước đi và thống kê `last_stats`.

Cách chạy từ thư mục gốc của repo:
    python -m src.ui.gui_core
    python -m src.ui.gui_core --game gomoku --size 9 --x-agent human --o-agent mcts
"""

from __future__ import annotations

import argparse
import copy
import importlib
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from src.agents.alphabeta import ORDERINGS, AlphaBetaAgent
from src.agents.alphabeta_gomoku import AlphaBetaGomoku
from src.agents.base import Agent, AgentStats, Budget
from src.agents.mcts import FINAL_SELECTIONS, ROLLOUT_POLICIES, MCTSAgent
from src.agents.minimax import MinimaxAgent
from src.agents.random_agent import RandomAgent
from src.games.base import DRAW, EMPTY, PLAYER_O, PLAYER_X, GameState, Move
from src.games.gomoku import DEFAULT_SIZE, DEFAULT_WIN_LENGTH, Gomoku
from src.games.tictactoe import LINES as TICTACTOE_LINES
from src.games.tictactoe import TicTacToe

GAME_TYPES: tuple[str, ...] = ("tictactoe", "gomoku")
MATCH_MODES: tuple[str, ...] = (
    "human_vs_ai",
    "ai_vs_human",
    "ai_vs_ai",
    "human_vs_human",
)
AGENT_TYPES: tuple[str, ...] = (
    "human",
    "mcts",
    "alphabeta",
    "alphabeta_gomoku",
    "minimax",
    "random",
)

# Các thế cờ mẫu phục vụ kiểm thử và vấn đáp (ví dụ thế bẫy E1 trên bàn 3×3).
PRESET_POSITIONS: dict[str, tuple[GameConfig, tuple[Move, ...]]] = {}

_GOMOKU_DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))


def available_rollout_policies() -> list[str]:
    """Trả về danh sách các chính sách rollout đã đăng ký (tự động nạp `src.agents.rollouts`)."""
    try:
        importlib.import_module("src.agents.rollouts")
    except ModuleNotFoundError:
        pass
    return sorted(ROLLOUT_POLICIES.keys())


# ---------------------------------------------------------------------------
# Cấu hình trò chơi và người chơi / thuật toán
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameConfig:
    """Cấu hình khởi tạo bàn cờ (`TicTacToe` 3×3 hoặc `Gomoku` n×n)."""

    game_type: str = "tictactoe"
    size: int = DEFAULT_SIZE
    win_length: int = DEFAULT_WIN_LENGTH
    block_two_ends: bool = False

    def __post_init__(self) -> None:
        kind = self.game_type.strip().lower()
        if kind not in GAME_TYPES:
            raise ValueError(f"game_type phải thuộc {GAME_TYPES}, nhận được {self.game_type!r}")
        object.__setattr__(self, "game_type", kind)

        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 3:
            raise ValueError(f"size phải là số nguyên >= 3, nhận được {self.size!r}")
        if (
            isinstance(self.win_length, bool)
            or not isinstance(self.win_length, int)
            or self.win_length < 3
        ):
            raise ValueError(f"win_length phải là số nguyên >= 3, nhận được {self.win_length!r}")
        if not isinstance(self.block_two_ends, bool):
            raise ValueError(f"block_two_ends phải là kiểu bool, nhận được {self.block_two_ends!r}")
        if kind == "gomoku" and self.size < self.win_length:
            raise ValueError(
                f"size ({self.size}) phải >= win_length ({self.win_length}) cho bàn Caro"
            )

    @property
    def board_size(self) -> int:
        """Kích thước cạnh bàn cờ (3 cho Tic-Tac-Toe, `size` cho Caro)."""
        return 3 if self.game_type == "tictactoe" else self.size

    @classmethod
    def from_mapping(cls, data: GameConfig | Mapping[str, Any]) -> GameConfig:
        """Tạo `GameConfig` từ từ điển (hỗ trợ cả khóa `type` như trong file YAML)."""
        if isinstance(data, GameConfig):
            return data
        if not isinstance(data, Mapping):
            raise ValueError(f"Cấu hình trò chơi phải là Mapping, nhận được {data!r}")
        raw = dict(data)
        game_type = str(raw.pop("game_type", raw.pop("type", "tictactoe")))
        size = int(raw.pop("size", DEFAULT_SIZE if game_type.lower() == "gomoku" else 3))
        win_length = int(raw.pop("win_length", DEFAULT_WIN_LENGTH))
        block_two_ends = bool(raw.pop("block_two_ends", False))
        if raw:
            unknown = sorted(raw.keys())
            raise ValueError(f"Tham số trò chơi không hợp lệ: {unknown}")
        return cls(
            game_type=game_type,
            size=size,
            win_length=win_length,
            block_two_ends=block_two_ends,
        )


def build_game(config: GameConfig | Mapping[str, Any] | None = None) -> GameState:
    """Khởi tạo đối tượng `GameState` mới từ `GameConfig`."""
    cfg = GameConfig() if config is None else GameConfig.from_mapping(config)
    if cfg.game_type == "tictactoe":
        return TicTacToe()
    return Gomoku(
        size=cfg.size,
        win_length=cfg.win_length,
        block_two_ends=cfg.block_two_ends,
    )


@dataclass(frozen=True)
class PlayerConfig:
    """Cấu hình thuật toán và tham số cho một bên người chơi (X hoặc O)."""

    agent_type: str = "human"
    # Ngân sách (Budget)
    iterations: int | None = 1000
    depth: int | None = None
    time_s: float | None = None
    # Tham số MCTS
    c: float = math.sqrt(2.0)
    selection: str = "robust"
    rollout: str = "l2"
    rollout_depth: int | None = None
    use_candidates: bool = True
    candidate_radius: int = 2
    reuse_tree: bool = True
    # Tham số Alpha-Beta 3×3 và Caro
    ordering: str = "heuristic"
    max_candidates: int | None = 16
    use_immediate_tactics: bool = True
    use_move_ordering: bool = True
    use_iterative_deepening: bool = True
    use_tt: bool = True
    # Seed ngẫu nhiên
    seed: int = 42

    def __post_init__(self) -> None:
        kind = self.agent_type.strip().lower()
        if kind not in AGENT_TYPES:
            raise ValueError(f"agent_type phải thuộc {AGENT_TYPES}, nhận được {self.agent_type!r}")
        object.__setattr__(self, "agent_type", kind)

        if self.iterations is not None:
            if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
                raise ValueError(
                    f"iterations phải là số nguyên >= 1, nhận được {self.iterations!r}"
                )
            if self.iterations < 1:
                raise ValueError(f"iterations phải >= 1, nhận được {self.iterations}")

        if self.depth is not None:
            if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 1:
                raise ValueError(f"depth phải là số nguyên >= 1, nhận được {self.depth!r}")

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
            raise ValueError(f"c phải là số thực hữu hạn >= 0, nhận được {self.c!r}")
        object.__setattr__(self, "c", float(self.c))

        sel = self.selection.strip().lower()
        if sel not in FINAL_SELECTIONS:
            raise ValueError(
                f"selection phải thuộc {FINAL_SELECTIONS}, nhận được {self.selection!r}"
            )
        object.__setattr__(self, "selection", sel)

        rollout_key = self.rollout.strip().lower()
        allowed_rollouts = available_rollout_policies()
        if rollout_key not in allowed_rollouts:
            raise ValueError(f"rollout phải thuộc {allowed_rollouts}, nhận được {self.rollout!r}")
        object.__setattr__(self, "rollout", rollout_key)

        if self.rollout_depth is not None:
            if (
                isinstance(self.rollout_depth, bool)
                or not isinstance(self.rollout_depth, int)
                or self.rollout_depth < 1
            ):
                raise ValueError(
                    f"rollout_depth phải là số nguyên >= 1, nhận được {self.rollout_depth!r}"
                )

        if (
            isinstance(self.candidate_radius, bool)
            or not isinstance(self.candidate_radius, int)
            or self.candidate_radius < 1
        ):
            raise ValueError(
                f"candidate_radius phải là số nguyên >= 1, nhận được {self.candidate_radius!r}"
            )

        ord_key = self.ordering.strip().lower()
        if ord_key not in ORDERINGS:
            raise ValueError(f"ordering phải thuộc {ORDERINGS}, nhận được {self.ordering!r}")
        object.__setattr__(self, "ordering", ord_key)

        if self.max_candidates is not None:
            if (
                isinstance(self.max_candidates, bool)
                or not isinstance(self.max_candidates, int)
                or self.max_candidates < 1
            ):
                raise ValueError(
                    f"max_candidates phải là số nguyên >= 1, nhận được {self.max_candidates!r}"
                )

        for flag_name in (
            "use_candidates",
            "reuse_tree",
            "use_immediate_tactics",
            "use_move_ordering",
            "use_iterative_deepening",
            "use_tt",
        ):
            val = getattr(self, flag_name)
            if not isinstance(val, bool):
                raise ValueError(f"{flag_name} phải là kiểu bool, nhận được {val!r}")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed phải là số nguyên, nhận được {self.seed!r}")

    @property
    def is_human(self) -> bool:
        """True nếu bên này do người trực tiếp đánh."""
        return self.agent_type == "human"

    def to_budget(self, game_type: str = "tictactoe") -> Budget:
        """Sinh `Budget` tương ứng với loại thuật toán và loại bàn cờ."""
        if self.agent_type == "human":
            return Budget()
        if self.agent_type == "mcts":
            iters = self.iterations
            if iters is None and self.time_s is None:
                iters = 1000
            return Budget(iterations=iters, time_s=self.time_s)
        if self.agent_type in ("minimax", "alphabeta", "alphabeta_gomoku"):
            depth = self.depth
            if game_type == "gomoku" and depth is None and self.time_s is None:
                # Giới hạn độ sâu mặc định trên bàn Caro để không treo giao diện
                depth = 2 if self.agent_type == "minimax" else 3
            return Budget(depth=depth, time_s=self.time_s)
        return Budget(iterations=self.iterations, depth=self.depth, time_s=self.time_s)

    @classmethod
    def from_mapping(
        cls,
        data: PlayerConfig | Mapping[str, Any] | str,
        **overrides: Any,
    ) -> PlayerConfig:
        """Tạo `PlayerConfig` linh hoạt từ tên thuật toán, từ điển hoặc `PlayerConfig`."""
        if isinstance(data, PlayerConfig):
            return replace(data, **overrides) if overrides else data
        if isinstance(data, str):
            return cls(agent_type=data, **overrides)
        if not isinstance(data, Mapping):
            raise ValueError(
                f"Cấu hình người chơi phải là PlayerConfig, str hoặc Mapping, nhận {data!r}"
            )
        raw = dict(data)
        raw.update(overrides)
        if "type" in raw and "agent_type" not in raw:
            raw["agent_type"] = raw.pop("type")
        if "radius" in raw and "candidate_radius" not in raw:
            raw["candidate_radius"] = raw.pop("radius")
        if "default_depth" in raw and "depth" not in raw:
            raw["depth"] = raw.pop("default_depth")
        budget = raw.pop("budget", None)
        if isinstance(budget, Mapping):
            for k in ("iterations", "depth", "time_s"):
                if k in budget and k not in raw:
                    raw[k] = budget[k]
        return cls(**raw)


def build_agent_for_state(
    config: PlayerConfig | Mapping[str, Any] | str,
    state: GameState,
) -> tuple[Agent | None, Budget]:
    """Khởi tạo `(agent, budget)` cho `state` theo `PlayerConfig`.

    - Với `agent_type == "human"`, trả về `(None, Budget())`.
    - Với `agent_type == "alphabeta"`, tự động dùng `AlphaBetaAgent` trên bàn 3×3 và
      `AlphaBetaGomoku` trên bàn Caro (`Gomoku`) để người dùng đổi bàn cờ hoặc đổi
      thuật toán mà không cần sửa code.
    """
    cfg = PlayerConfig.from_mapping(config)
    game_type = "gomoku" if isinstance(state, Gomoku) else "tictactoe"
    budget = cfg.to_budget(game_type)

    if cfg.agent_type == "human":
        return None, budget
    if cfg.agent_type == "random":
        return RandomAgent(seed=cfg.seed), budget
    if cfg.agent_type == "minimax":
        return MinimaxAgent(), budget
    if cfg.agent_type == "alphabeta":
        if isinstance(state, Gomoku):
            default_d = cfg.depth if cfg.depth is not None else 3
            return (
                AlphaBetaGomoku(
                    radius=cfg.candidate_radius,
                    default_depth=default_d,
                    max_candidates=cfg.max_candidates,
                    use_candidates=cfg.use_candidates,
                    use_immediate_tactics=cfg.use_immediate_tactics,
                    use_move_ordering=cfg.use_move_ordering,
                    use_iterative_deepening=cfg.use_iterative_deepening,
                    use_tt=cfg.use_tt,
                ),
                budget,
            )
        return AlphaBetaAgent(ordering=cfg.ordering, seed=cfg.seed), budget
    if cfg.agent_type == "alphabeta_gomoku":
        default_d = cfg.depth if cfg.depth is not None else 3
        return (
            AlphaBetaGomoku(
                radius=cfg.candidate_radius,
                default_depth=default_d,
                max_candidates=cfg.max_candidates,
                use_candidates=cfg.use_candidates,
                use_immediate_tactics=cfg.use_immediate_tactics,
                use_move_ordering=cfg.use_move_ordering,
                use_iterative_deepening=cfg.use_iterative_deepening,
                use_tt=cfg.use_tt,
            ),
            budget,
        )
    if cfg.agent_type == "mcts":
        return (
            MCTSAgent(
                c=cfg.c,
                selection=cfg.selection,
                rollout=cfg.rollout,
                rollout_depth=cfg.rollout_depth,
                use_candidates=cfg.use_candidates,
                candidate_radius=cfg.candidate_radius,
                reuse_tree=cfg.reuse_tree,
                seed=cfg.seed,
            ),
            budget,
        )
    raise ValueError(f"Không hỗ trợ agent_type={cfg.agent_type!r}")


# Đăng ký sẵn thế cờ bẫy E1 trên bàn 3×3 (O(0,0), X(0,1), O(1,1), X(2,0) — tới lượt X).
PRESET_POSITIONS["E1"] = (
    GameConfig(game_type="tictactoe"),
    ((0, 1), (0, 0), (2, 0), (1, 1)),
)


# ---------------------------------------------------------------------------
# Hình học bàn cờ và phát hiện đường thắng (thuần Python, không phụ thuộc X11)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardGeometry:
    """Tính toán tọa độ pixel <-> ô bàn cờ `(row, col)` trên khung vẽ."""

    board_size: int
    canvas_width: int
    canvas_height: int
    padding: float = 36.0

    def __post_init__(self) -> None:
        if self.board_size < 1:
            raise ValueError(f"board_size phải >= 1, nhận được {self.board_size}")
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            raise ValueError("Kích thước canvas phải > 0")

    def grid_box(self) -> tuple[float, float, float, float]:
        """Trả về `(x0, y0, cell_size, grid_span)` của vùng lưới hình vuông."""
        usable_w = max(20.0, float(self.canvas_width) - 2.0 * self.padding)
        usable_h = max(20.0, float(self.canvas_height) - 2.0 * self.padding)
        span = min(usable_w, usable_h)
        cell_size = span / self.board_size
        x0 = (self.canvas_width - span) * 0.5
        y0 = (self.canvas_height - span) * 0.5
        return x0, y0, cell_size, span

    def cell_bbox(self, row: int, col: int) -> tuple[float, float, float, float]:
        """Trả về khung chữ nhật `(x1, y1, x2, y2)` của ô `(row, col)`."""
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError(f"Ô ({row}, {col}) nằm ngoài bàn {self.board_size}×{self.board_size}")
        x0, y0, cell_size, _ = self.grid_box()
        x1 = x0 + col * cell_size
        y1 = y0 + row * cell_size
        return x1, y1, x1 + cell_size, y1 + cell_size

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        """Trả về tâm `(cx, cy)` của ô `(row, col)`."""
        x1, y1, x2, y2 = self.cell_bbox(row, col)
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    def pixel_to_cell(self, x: float, y: float) -> Move | None:
        """Ánh xạ tọa độ điểm nhấp chuột `(x, y)` sang ô `(row, col)`, hoặc `None` nếu ngoài bàn."""
        x0, y0, cell_size, span = self.grid_box()
        if not (x0 <= x < x0 + span and y0 <= y < y0 + span):
            return None
        col = int((x - x0) // cell_size)
        row = int((y - y0) // cell_size)
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return row, col
        return None


def find_winning_cells(state: GameState) -> list[Move]:
    """Trả về danh sách các ô tạo thành đường thắng trên `state` (rỗng nếu chưa có người thắng)."""
    winner = state.winner()
    if winner not in (PLAYER_X, PLAYER_O):
        return []

    board = state.board
    n = int(board.shape[0])

    if isinstance(state, TicTacToe):
        for line in TICTACTOE_LINES:
            if all(int(board[r, c]) == winner for r, c in line):
                return list(line)
        return []

    win_length = getattr(state, "win_length", DEFAULT_WIN_LENGTH)
    block_two_ends = bool(getattr(state, "block_two_ends", False))
    opponent = -winner

    for r in range(n):
        for c in range(n):
            if int(board[r, c]) != winner:
                continue
            for dr, dc in _GOMOKU_DIRECTIONS:
                pr, pc = r - dr, c - dc
                if 0 <= pr < n and 0 <= pc < n and int(board[pr, pc]) == winner:
                    continue
                cells: list[Move] = []
                cr, cc = r, c
                while 0 <= cr < n and 0 <= cc < n and int(board[cr, cc]) == winner:
                    cells.append((cr, cc))
                    cr += dr
                    cc += dc
                if len(cells) < win_length:
                    continue
                if block_two_ends:
                    left_blocked = 0 <= pr < n and 0 <= pc < n and int(board[pr, pc]) == opponent
                    right_blocked = 0 <= cr < n and 0 <= cc < n and int(board[cr, cc]) == opponent
                    if left_blocked and right_blocked:
                        continue
                return cells
    return []


# ---------------------------------------------------------------------------
# Bộ điều khiển phiên chơi (GameSession — không phụ thuộc Tkinter)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveRecord:
    """Bản ghi một nước đi trong lịch sử ván đấu."""

    ply: int
    player: int
    move: Move
    agent_name: str
    stats: AgentStats = field(default_factory=dict)


class GameSession:
    """Quản lý trạng thái bàn cờ, hai người chơi, lịch sử nước đi và hoàn tác (`undo`).

    Tách biệt hoàn toàn khỏi thư viện hiển thị để mọi luồng nghiệp vụ (Người vs AI,
    AI vs AI, đổi thuật toán/tham số động, undo, nạp thế cờ) đều kiểm thử tự động được.
    """

    def __init__(
        self,
        game_config: GameConfig | Mapping[str, Any] | None = None,
        player_x: PlayerConfig | Mapping[str, Any] | str = "human",
        player_o: PlayerConfig | Mapping[str, Any] | str = "mcts",
    ) -> None:
        self.game_config = (
            GameConfig() if game_config is None else GameConfig.from_mapping(game_config)
        )
        self.players: dict[int, PlayerConfig] = {
            PLAYER_X: PlayerConfig.from_mapping(player_x),
            PLAYER_O: PlayerConfig.from_mapping(player_o),
        }
        self.state: GameState = build_game(self.game_config)
        self.history: list[MoveRecord] = []
        self._agents: dict[int, tuple[Agent | None, Budget]] = {}
        self._rebuild_agent(PLAYER_X)
        self._rebuild_agent(PLAYER_O)

    def _rebuild_agent(self, player: int) -> None:
        self._agents[player] = build_agent_for_state(self.players[player], self.state)

    def new_game(
        self,
        game_config: GameConfig | Mapping[str, Any] | None = None,
    ) -> GameState:
        """Bắt đầu ván cờ mới (tùy chọn đổi cấu hình bàn cờ `game_config`)."""
        if game_config is not None:
            self.game_config = GameConfig.from_mapping(game_config)
        self.state = build_game(self.game_config)
        self.history.clear()
        self._rebuild_agent(PLAYER_X)
        self._rebuild_agent(PLAYER_O)
        return self.state

    def configure_player(
        self,
        player: int,
        config: PlayerConfig | Mapping[str, Any] | str | None = None,
        **overrides: Any,
    ) -> PlayerConfig:
        """Đổi thuật toán hoặc tham số của `player` (`PLAYER_X` hoặc `PLAYER_O`) ngay lập tức.

        Không cần sửa code và không làm mất trạng thái bàn cờ đang chơi.
        """
        if player not in (PLAYER_X, PLAYER_O):
            raise ValueError(f"player phải là PLAYER_X (1) hoặc PLAYER_O (-1), nhận {player!r}")
        base = self.players[player] if config is None else config
        new_cfg = PlayerConfig.from_mapping(base, **overrides)
        self.players[player] = new_cfg
        self._rebuild_agent(player)
        return new_cfg

    def set_mode(
        self,
        mode: str,
        *,
        ai_x: PlayerConfig | Mapping[str, Any] | str | None = None,
        ai_o: PlayerConfig | Mapping[str, Any] | str | None = None,
    ) -> None:
        """Thiết lập nhanh một trong 4 chế độ chơi (`MATCH_MODES`)."""
        key = mode.strip().lower()
        if key not in MATCH_MODES:
            raise ValueError(f"mode phải thuộc {MATCH_MODES}, nhận được {mode!r}")

        default_ai_x = (
            ai_x
            if ai_x is not None
            else (self.players[PLAYER_X] if not self.players[PLAYER_X].is_human else "mcts")
        )
        default_ai_o = (
            ai_o
            if ai_o is not None
            else (self.players[PLAYER_O] if not self.players[PLAYER_O].is_human else "mcts")
        )

        if key == "human_vs_ai":
            self.configure_player(PLAYER_X, "human")
            self.configure_player(PLAYER_O, default_ai_o)
        elif key == "ai_vs_human":
            self.configure_player(PLAYER_X, default_ai_x)
            self.configure_player(PLAYER_O, "human")
        elif key == "ai_vs_ai":
            self.configure_player(PLAYER_X, default_ai_x)
            self.configure_player(PLAYER_O, default_ai_o)
        elif key == "human_vs_human":
            self.configure_player(PLAYER_X, "human")
            self.configure_player(PLAYER_O, "human")

    @property
    def match_mode(self) -> str:
        """Trả về tên chế độ hiện tại dựa trên cấu hình của X và O."""
        x_human = self.players[PLAYER_X].is_human
        o_human = self.players[PLAYER_O].is_human
        if x_human and not o_human:
            return "human_vs_ai"
        if not x_human and o_human:
            return "ai_vs_human"
        if not x_human and not o_human:
            return "ai_vs_ai"
        return "human_vs_human"

    @property
    def move_history(self) -> list[Move]:
        """Danh sách các nước `(row, col)` đã đánh theo thứ tự từ đầu ván."""
        return [rec.move for rec in self.history]

    @property
    def last_record(self) -> MoveRecord | None:
        """Bản ghi nước đi gần nhất, hoặc `None` nếu bàn đang trống."""
        return self.history[-1] if self.history else None

    @property
    def last_move(self) -> Move | None:
        """Tọa độ `(row, col)` của nước đi gần nhất, hoặc `None`."""
        return self.history[-1].move if self.history else None

    @property
    def last_stats(self) -> AgentStats:
        """Thống kê `last_stats` của nước AI gần nhất (hoặc `{}` nếu chưa có)."""
        for rec in reversed(self.history):
            if rec.stats:
                return copy.deepcopy(rec.stats)
        return {}

    @property
    def winning_cells(self) -> list[Move]:
        """Các ô thuộc đường thắng nếu ván cờ đã có người thắng."""
        return find_winning_cells(self.state)

    def is_current_player_human(self) -> bool:
        """True nếu ván chưa kết thúc và người sắp đi là `"human"`."""
        if self.state.is_terminal():
            return False
        return self.players[self.state.current_player()].is_human

    def is_current_player_ai(self) -> bool:
        """True nếu ván chưa kết thúc và người sắp đi là AI."""
        if self.state.is_terminal():
            return False
        return not self.players[self.state.current_player()].is_human

    def get_agent(self, player: int) -> tuple[Agent | None, Budget]:
        """Trả về `(agent, budget)` đang gắn với `player`."""
        return self._agents[player]

    def status_text(self) -> str:
        """Chuỗi mô tả trạng thái ván cờ hiện tại (dùng cho thanh trạng thái GUI)."""
        winner = self.state.winner()
        if winner == PLAYER_X:
            return "Kết thúc: Quân X thắng!"
        if winner == PLAYER_O:
            return "Kết thúc: Quân O thắng!"
        if winner == DRAW:
            return "Kết thúc: Hòa!"

        cur = self.state.current_player()
        side = "X" if cur == PLAYER_X else "O"
        cfg = self.players[cur]
        role = "Người" if cfg.is_human else cfg.agent_type
        return f"Lượt đi: Quân {side} ({role}) — Nước #{len(self.history) + 1}"

    def play_human_move(self, move: Move, *, auto_reply: bool = False) -> list[MoveRecord]:
        """Thực hiện nước `move` của người chơi (tự động cho AI đáp trả nếu `auto_reply=True`)."""
        if self.state.is_terminal():
            raise ValueError("Ván cờ đã kết thúc")
        r, c = int(move[0]), int(move[1])
        player = self.state.current_player()
        self.state.play((r, c))
        rec = MoveRecord(
            ply=len(self.history) + 1,
            player=player,
            move=(r, c),
            agent_name="human",
            stats={},
        )
        self.history.append(rec)
        played = [rec]

        if auto_reply and self.is_current_player_ai():
            played.append(self.step_ai_move())
        return played

    def step_ai_move(
        self,
        *,
        fallback_agent: PlayerConfig | Mapping[str, Any] | str = "mcts",
    ) -> MoveRecord:
        """Cho AI thực hiện 1 nước đi tại thế cờ hiện tại.

        Nếu bên đang tới lượt được cấu hình là `"human"` (ví dụ người dùng bấm nút
        "AI đi 1 nước / Gợi ý"), hàm dùng `fallback_agent` (mặc định MCTS) để chọn nước.
        """
        if self.state.is_terminal():
            raise ValueError("Ván cờ đã kết thúc, AI không thể đi tiếp")

        player = self.state.current_player()
        agent, budget = self._agents[player]
        if agent is None:
            agent, budget = build_agent_for_state(fallback_agent, self.state)
            assert agent is not None

        before_hash = self.state.hash()
        before_board = self.state.board.copy()
        move = agent.select_move(self.state, budget)
        if self.state.hash() != before_hash or not (self.state.board == before_board).all():
            raise RuntimeError(f"Agent {agent.name} làm biến đổi trạng thái bàn cờ sau select_move")

        r, c = int(move[0]), int(move[1])
        self.state.play((r, c))
        stats_copy: AgentStats = copy.deepcopy(agent.last_stats)
        rec = MoveRecord(
            ply=len(self.history) + 1,
            player=player,
            move=(r, c),
            agent_name=agent.name,
            stats=stats_copy,
        )
        self.history.append(rec)
        return rec

    def run_ai_vs_ai(self, *, max_plies: int | None = None) -> list[MoveRecord]:
        """Chạy liên tiếp các nước AI cho tới khi kết thúc ván hoặc đạt `max_plies`."""
        records: list[MoveRecord] = []
        plies = 0
        while not self.state.is_terminal():
            if max_plies is not None and plies >= max_plies:
                break
            records.append(self.step_ai_move())
            plies += 1
        return records

    def undo(self, steps: int | None = None) -> int:
        """Hoàn tác nước đi và khôi phục lịch sử, trạng thái bàn cờ cùng bộ nhớ cây MCTS.

        - Nếu `steps` được truyền cụ thể (`steps >= 1`): hoàn tác tối đa `steps` nước (ply).
        - Nếu `steps is None` (mặc định cho nút Undo trên GUI):
            + Ở chế độ Người vs AI (`human_vs_ai` hoặc `ai_vs_human`): nếu đang tới lượt
              Người (hoặc ván cờ vừa kết thúc ngay sau nước của AI) và đã có từ 2 nước trở lên,
              hoàn tác 2 nước (nước của AI + nước của Người) để quay về đúng lượt Người.
            + Các trường hợp còn lại: hoàn tác 1 nước.
        Trả về số nước (ply) thực tế đã hoàn tác.
        """
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
                if last_was_ai and len(self.history) >= 2:
                    target = 2
                else:
                    target = 1
            else:
                target = 1

        undone = 0
        for _ in range(target):
            if not self.history:
                break
            self.state.undo()
            self.history.pop()
            undone += 1

        # Đặt lại gốc cây lưu tạm của MCTS (nếu có) để tránh dùng cây của tương lai đã hoàn tác
        for p in (PLAYER_X, PLAYER_O):
            agent, _ = self._agents[p]
            if isinstance(agent, MCTSAgent):
                agent.reset_tree()

        return undone

    def load_position(
        self,
        moves: Sequence[Move],
        game_config: GameConfig | Mapping[str, Any] | None = None,
    ) -> GameState:
        """Khởi tạo lại bàn cờ và đánh sẵn chuỗi nước `moves` (phục vụ nạp thế cờ / thế E1)."""
        self.new_game(game_config)
        for mv in moves:
            r, c = int(mv[0]), int(mv[1])
            player = self.state.current_player()
            self.state.play((r, c))
            self.history.append(
                MoveRecord(
                    ply=len(self.history) + 1,
                    player=player,
                    move=(r, c),
                    agent_name="preset",
                    stats={},
                )
            )
        return self.state

    def load_preset(self, name: str = "E1") -> GameState:
        """Nạp nhanh một thế cờ mẫu đã đăng ký trong `PRESET_POSITIONS` (mặc định `'E1'`)."""
        key = name.strip().upper()
        if key not in PRESET_POSITIONS:
            allowed = sorted(PRESET_POSITIONS.keys())
            raise ValueError(f"Thế cờ mẫu {name!r} không tồn tại; hỗ trợ {allowed}")
        cfg, moves = PRESET_POSITIONS[key]
        return self.load_position(moves, game_config=cfg)


# ---------------------------------------------------------------------------
# Giao diện đồ họa Tkinter (BoardGUI)
# ---------------------------------------------------------------------------


@dataclass
class CellOverlay:
    """Thông tin lớp phủ hiển thị trên một ô bàn cờ (dùng cho heatmap N của P6.1b)."""

    bg_color: str | None = None
    text: str | None = None
    text_color: str = "#1E293B"


class BoardGUI:
    """Cửa sổ chính Tkinter cho Tic-Tac-Toe 3×3 và Caro n×n.

    Hỗ trợ:
      - Click chuột trực tiếp trên bàn cờ khi tới lượt Người.
      - Chuyển đổi trò chơi (3×3 / Caro), kích thước bàn, luật chặn 2 đầu.
      - Chuyển đổi chế độ chơi (Người vs AI, AI vs Người, AI vs AI, Người vs Người).
      - Chỉnh thuật toán và mọi tham số cho từng bên X / O ngay trên giao diện.
      - Undo (hoàn tác lượt hoặc hoàn tác 1 nước), nạp nhanh thế E1, chạy/dừng AI vs AI.
      - Cung cấp `extension_frame`, `register_observer(fn)` và `set_cell_overlays(overlays)`
        để tích hợp bảng phân tích (`src/ui/gui_panel.py` — P6.1b).
    """

    def __init__(
        self,
        session: GameSession | None = None,
        *,
        master: Any | None = None,
        ai_delay_ms: int = 120,
    ) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk

        self._tk = tk
        self._ttk = ttk
        self._messagebox = messagebox
        self.session = session if session is not None else GameSession()
        self.ai_delay_ms = max(10, int(ai_delay_ms))

        self.root = tk.Tk() if master is None else master
        self.root.title("Tic-Tac-Toe 3×3 & Caro AI — Minimax, Alpha-Beta & MCTS-UCT")
        self.root.geometry("1180x760")
        self.root.minsize(980, 660)

        self._hover_cell: Move | None = None
        self._cell_overlays: dict[Move, CellOverlay] = {}
        self._observers: list[Callable[[GameSession], None]] = []
        self._ai_running: bool = False
        self._after_id: str | None = None

        self._build_ui()
        self._sync_controls_from_session()
        self.refresh()

        if self.session.is_current_player_ai() and self.session.match_mode != "ai_vs_ai":
            self._schedule_ai_step()

    # --- Điểm nối mở rộng cho P6.1b (gui_panel.py) ---

    def register_observer(self, callback: Callable[[GameSession], None]) -> None:
        """Đăng ký hàm callback được gọi mỗi khi bàn cờ hoặc thống kê thay đổi."""
        self._observers.append(callback)

    def set_cell_overlays(self, overlays: Mapping[Move, CellOverlay | Mapping[str, Any]]) -> None:
        """Cập nhật lớp phủ màu/nhãn (ví dụ heatmap N) trên các ô bàn cờ và vẽ lại."""
        parsed: dict[Move, CellOverlay] = {}
        for mv, item in overlays.items():
            key = (int(mv[0]), int(mv[1]))
            if isinstance(item, CellOverlay):
                parsed[key] = item
            elif isinstance(item, Mapping):
                parsed[key] = CellOverlay(
                    bg_color=item.get("bg_color"),
                    text=item.get("text"),
                    text_color=str(item.get("text_color", "#1E293B")),
                )
        self._cell_overlays = parsed
        self._draw_board()

    def clear_cell_overlays(self) -> None:
        """Xóa toàn bộ lớp phủ trên bàn cờ."""
        self._cell_overlays.clear()
        self._draw_board()

    # --- Xây dựng bố cục giao diện ---

    def _build_ui(self) -> None:
        tk = self._tk
        ttk = self._ttk

        main_pane = ttk.Frame(self.root, padding=8)
        main_pane.pack(fill=tk.BOTH, expand=True)
        main_pane.columnconfigure(0, weight=3)
        main_pane.columnconfigure(1, weight=2)
        main_pane.rowconfigure(0, weight=1)

        # Cột trái: Thanh trạng thái + Canvas bàn cờ + Thanh nút điều khiển nhanh
        left_frame = ttk.Frame(main_pane)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value=self.session.status_text())
        status_label = ttk.Label(
            left_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 12, "bold"),
            padding=(4, 4),
        )
        status_label.grid(row=0, column=0, sticky="ew")

        self.canvas = tk.Canvas(
            left_frame,
            width=620,
            height=620,
            bg="#F8FAFC",
            highlightthickness=1,
            highlightbackground="#CBD5E1",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _e: self._draw_board())
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

        action_bar = ttk.Frame(left_frame, padding=(0, 6, 0, 0))
        action_bar.grid(row=2, column=0, sticky="ew")

        ttk.Button(action_bar, text="Ván mới", command=self.on_new_game).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(action_bar, text="Undo (Hoàn tác)", command=self.on_undo).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(action_bar, text="Undo 1 nước", command=lambda: self.on_undo(steps=1)).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(action_bar, text="AI đi 1 nước", command=self.on_step_ai).pack(
            side=tk.LEFT, padx=4
        )
        self.ai_loop_btn_var = tk.StringVar(value="Chạy AI vs AI")
        ttk.Button(
            action_bar,
            textvariable=self.ai_loop_btn_var,
            command=self.on_toggle_ai_loop,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_bar, text="Nạp thế E1 (3×3)", command=self.on_load_e1).pack(
            side=tk.LEFT, padx=4
        )

        # Cột phải: Cấu hình trò chơi, thuật toán/tham số và thống kê
        right_frame = ttk.Frame(main_pane)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(2, weight=1)

        self._build_game_box(right_frame)
        self._build_players_notebook(right_frame)
        self._build_stats_and_history_box(right_frame)

    def _build_game_box(self, parent: Any) -> None:
        tk = self._tk
        ttk = self._ttk

        box = ttk.LabelFrame(parent, text="1. Bàn cờ & Chế độ chơi", padding=8)
        box.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        for col in range(4):
            box.columnconfigure(col, weight=1)

        ttk.Label(box, text="Trò chơi:").grid(row=0, column=0, sticky="w", pady=2)
        self.game_type_var = tk.StringVar(value=self.session.game_config.game_type)
        game_cb = ttk.Combobox(
            box,
            textvariable=self.game_type_var,
            values=list(GAME_TYPES),
            state="readonly",
            width=12,
        )
        game_cb.grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        game_cb.bind("<<ComboboxSelected>>", lambda _e: self.on_new_game())

        ttk.Label(box, text="Cỡ bàn Caro:").grid(row=0, column=2, sticky="w", pady=2)
        self.board_size_var = tk.StringVar(value=str(self.session.game_config.size))
        size_cb = ttk.Combobox(
            box,
            textvariable=self.board_size_var,
            values=["9", "11", "13", "15", "19"],
            width=6,
        )
        size_cb.grid(row=0, column=3, sticky="ew", padx=4, pady=2)
        size_cb.bind("<<ComboboxSelected>>", lambda _e: self.on_new_game())

        self.block_two_var = tk.BooleanVar(value=self.session.game_config.block_two_ends)
        ttk.Checkbutton(
            box,
            text="Luật chặn hai đầu (Caro)",
            variable=self.block_two_var,
            command=self.on_new_game,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(box, text="Chế độ nhanh:").grid(row=1, column=2, sticky="w", pady=2)
        self.mode_var = tk.StringVar(value=self.session.match_mode)
        mode_cb = ttk.Combobox(
            box,
            textvariable=self.mode_var,
            values=list(MATCH_MODES),
            state="readonly",
            width=14,
        )
        mode_cb.grid(row=1, column=3, sticky="ew", padx=4, pady=2)
        mode_cb.bind("<<ComboboxSelected>>", lambda _e: self.on_quick_mode_change())

    def _build_players_notebook(self, parent: Any) -> None:
        ttk = self._ttk

        outer = ttk.LabelFrame(
            parent,
            text="2. Thuật toán & Tham số (đổi trực tiếp, không cần sửa code)",
            padding=6,
        )
        outer.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        outer.columnconfigure(0, weight=1)

        self.player_notebook = ttk.Notebook(outer)
        self.player_notebook.grid(row=0, column=0, sticky="ew")

        self._player_vars: dict[int, dict[str, Any]] = {}
        for player, title in ((PLAYER_X, "Quân X (Đi trước)"), (PLAYER_O, "Quân O (Đi sau)")):
            tab = ttk.Frame(self.player_notebook, padding=6)
            self.player_notebook.add(tab, text=title)
            self._player_vars[player] = self._build_single_player_tab(tab, player)

        ttk.Button(
            outer,
            text="Áp dụng thuật toán & tham số",
            command=self.on_apply_player_configs,
        ).grid(row=1, column=0, sticky="ew", pady=(6, 0))

    def _build_single_player_tab(self, tab: Any, player: int) -> dict[str, Any]:
        tk = self._tk
        ttk = self._ttk
        cfg = self.session.players[player]
        vars_map: dict[str, Any] = {}

        for col in range(4):
            tab.columnconfigure(col, weight=1)

        ttk.Label(tab, text="Thuật toán:").grid(row=0, column=0, sticky="w", pady=2)
        vars_map["agent_type"] = tk.StringVar(value=cfg.agent_type)
        agent_cb = ttk.Combobox(
            tab,
            textvariable=vars_map["agent_type"],
            values=list(AGENT_TYPES),
            state="readonly",
            width=15,
        )
        agent_cb.grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        agent_cb.bind("<<ComboboxSelected>>", lambda _e: self.on_apply_player_configs())

        ttk.Label(tab, text="Seed:").grid(row=0, column=2, sticky="w", pady=2)
        vars_map["seed"] = tk.StringVar(value=str(cfg.seed))
        ttk.Entry(tab, textvariable=vars_map["seed"], width=8).grid(
            row=0, column=3, sticky="ew", padx=4, pady=2
        )

        # Dòng ngân sách: Iterations, Depth, Time (s)
        ttk.Label(tab, text="Iterations (MCTS):").grid(row=1, column=0, sticky="w", pady=2)
        vars_map["iterations"] = tk.StringVar(
            value="" if cfg.iterations is None else str(cfg.iterations)
        )
        ttk.Entry(tab, textvariable=vars_map["iterations"], width=8).grid(
            row=1, column=1, sticky="ew", padx=4, pady=2
        )

        ttk.Label(tab, text="Depth (AB/MM):").grid(row=1, column=2, sticky="w", pady=2)
        vars_map["depth"] = tk.StringVar(value="" if cfg.depth is None else str(cfg.depth))
        ttk.Entry(tab, textvariable=vars_map["depth"], width=8).grid(
            row=1, column=3, sticky="ew", padx=4, pady=2
        )

        ttk.Label(tab, text="Time limit (s):").grid(row=2, column=0, sticky="w", pady=2)
        vars_map["time_s"] = tk.StringVar(value="" if cfg.time_s is None else str(cfg.time_s))
        ttk.Entry(tab, textvariable=vars_map["time_s"], width=8).grid(
            row=2, column=1, sticky="ew", padx=4, pady=2
        )

        ttk.Label(tab, text="Hằng số UCT c:").grid(row=2, column=2, sticky="w", pady=2)
        vars_map["c"] = tk.StringVar(value=f"{cfg.c:.4g}")
        c_cb = ttk.Combobox(
            tab,
            textvariable=vars_map["c"],
            values=["0.5", "1.0", "1.414", "2.0"],
            width=8,
        )
        c_cb.grid(row=2, column=3, sticky="ew", padx=4, pady=2)

        # Dòng MCTS: Selection, Rollout, Rollout depth, Radius
        ttk.Label(tab, text="Selection (MCTS):").grid(row=3, column=0, sticky="w", pady=2)
        vars_map["selection"] = tk.StringVar(value=cfg.selection)
        sel_cb = ttk.Combobox(
            tab,
            textvariable=vars_map["selection"],
            values=list(FINAL_SELECTIONS),
            state="readonly",
            width=12,
        )
        sel_cb.grid(row=3, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(tab, text="Rollout (MCTS):").grid(row=3, column=2, sticky="w", pady=2)
        vars_map["rollout"] = tk.StringVar(value=cfg.rollout)
        roll_cb = ttk.Combobox(
            tab,
            textvariable=vars_map["rollout"],
            values=available_rollout_policies(),
            state="readonly",
            width=10,
        )
        roll_cb.grid(row=3, column=3, sticky="ew", padx=4, pady=2)

        ttk.Label(tab, text="Ordering (AB 3×3):").grid(row=4, column=0, sticky="w", pady=2)
        vars_map["ordering"] = tk.StringVar(value=cfg.ordering)
        ord_cb = ttk.Combobox(
            tab,
            textvariable=vars_map["ordering"],
            values=list(ORDERINGS),
            state="readonly",
            width=12,
        )
        ord_cb.grid(row=4, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(tab, text="Bán kính ứng viên:").grid(row=4, column=2, sticky="w", pady=2)
        vars_map["candidate_radius"] = tk.StringVar(value=str(cfg.candidate_radius))
        ttk.Spinbox(
            tab,
            from_=1,
            to=5,
            textvariable=vars_map["candidate_radius"],
            width=6,
        ).grid(row=4, column=3, sticky="ew", padx=4, pady=2)

        ttk.Label(tab, text="Rollout depth:").grid(row=5, column=0, sticky="w", pady=2)
        vars_map["rollout_depth"] = tk.StringVar(
            value="" if cfg.rollout_depth is None else str(cfg.rollout_depth)
        )
        ttk.Entry(tab, textvariable=vars_map["rollout_depth"], width=8).grid(
            row=5, column=1, sticky="ew", padx=4, pady=2
        )

        ttk.Label(tab, text="Max candidates:").grid(row=5, column=2, sticky="w", pady=2)
        vars_map["max_candidates"] = tk.StringVar(
            value="" if cfg.max_candidates is None else str(cfg.max_candidates)
        )
        ttk.Entry(tab, textvariable=vars_map["max_candidates"], width=8).grid(
            row=5, column=3, sticky="ew", padx=4, pady=2
        )

        # Các cờ bật/tắt kỹ thuật
        flags_frame = ttk.Frame(tab)
        flags_frame.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(4, 0))

        vars_map["use_candidates"] = tk.BooleanVar(value=cfg.use_candidates)
        ttk.Checkbutton(
            flags_frame,
            text="Ứng viên",
            variable=vars_map["use_candidates"],
        ).pack(side=tk.LEFT, padx=(0, 8))

        vars_map["reuse_tree"] = tk.BooleanVar(value=cfg.reuse_tree)
        ttk.Checkbutton(
            flags_frame,
            text="Tái dùng cây MCTS",
            variable=vars_map["reuse_tree"],
        ).pack(side=tk.LEFT, padx=(0, 8))

        vars_map["use_tt"] = tk.BooleanVar(value=cfg.use_tt)
        ttk.Checkbutton(
            flags_frame,
            text="Bảng băm TT",
            variable=vars_map["use_tt"],
        ).pack(side=tk.LEFT, padx=(0, 8))

        vars_map["use_iterative_deepening"] = tk.BooleanVar(value=cfg.use_iterative_deepening)
        ttk.Checkbutton(
            flags_frame,
            text="Sâu dần (ID)",
            variable=vars_map["use_iterative_deepening"],
        ).pack(side=tk.LEFT)

        return vars_map

    def _build_stats_and_history_box(self, parent: Any) -> None:
        tk = self._tk
        ttk = self._ttk

        bottom_notebook = ttk.Notebook(parent)
        bottom_notebook.grid(row=2, column=0, sticky="nsew")

        # Tab 1: Thống kê nước đi & Lịch sử
        info_tab = ttk.Frame(bottom_notebook, padding=6)
        bottom_notebook.add(info_tab, text="Thống kê & Lịch sử")
        info_tab.columnconfigure(0, weight=1)
        info_tab.rowconfigure(1, weight=1)

        self.stats_summary_var = tk.StringVar(value="Chưa có thống kê nước đi AI.")
        ttk.Label(
            info_tab,
            textvariable=self.stats_summary_var,
            justify=tk.LEFT,
            font=("Consolas", 9),
        ).grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.history_text = tk.Text(
            info_tab,
            height=12,
            width=42,
            font=("Consolas", 9),
            state=tk.DISABLED,
            bg="#F8FAFC",
        )
        self.history_text.grid(row=1, column=0, sticky="nsew")

        # Tab 2: Khung mở rộng dành sẵn cho P6.1b (gui_panel.py của TV1)
        self.extension_frame = ttk.Frame(bottom_notebook, padding=6)
        bottom_notebook.add(self.extension_frame, text="Bảng phân tích (P6.1b)")

    # --- Đồng bộ dữ liệu cấu hình giữa UI và GameSession ---

    def _sync_controls_from_session(self) -> None:
        self.game_type_var.set(self.session.game_config.game_type)
        self.board_size_var.set(str(self.session.game_config.size))
        self.block_two_var.set(self.session.game_config.block_two_ends)
        self.mode_var.set(self.session.match_mode)

        for player in (PLAYER_X, PLAYER_O):
            cfg = self.session.players[player]
            v = self._player_vars[player]
            v["agent_type"].set(cfg.agent_type)
            v["seed"].set(str(cfg.seed))
            v["iterations"].set("" if cfg.iterations is None else str(cfg.iterations))
            v["depth"].set("" if cfg.depth is None else str(cfg.depth))
            v["time_s"].set("" if cfg.time_s is None else str(cfg.time_s))
            v["c"].set(f"{cfg.c:.4g}")
            v["selection"].set(cfg.selection)
            v["rollout"].set(cfg.rollout)
            v["ordering"].set(cfg.ordering)
            v["candidate_radius"].set(str(cfg.candidate_radius))
            v["rollout_depth"].set("" if cfg.rollout_depth is None else str(cfg.rollout_depth))
            v["max_candidates"].set("" if cfg.max_candidates is None else str(cfg.max_candidates))
            v["use_candidates"].set(cfg.use_candidates)
            v["reuse_tree"].set(cfg.reuse_tree)
            v["use_tt"].set(cfg.use_tt)
            v["use_iterative_deepening"].set(cfg.use_iterative_deepening)

    def _read_player_config_from_ui(self, player: int) -> PlayerConfig:
        base = self.session.players[player]
        v = self._player_vars[player]
        iters_str = v["iterations"].get().strip()
        depth_str = v["depth"].get().strip()
        time_str = v["time_s"].get().strip()
        seed_str = v["seed"].get().strip()
        c_str = v["c"].get().strip()
        rad_str = v["candidate_radius"].get().strip()
        rdepth_str = v["rollout_depth"].get().strip()
        maxc_str = v["max_candidates"].get().strip()

        return replace(
            base,
            agent_type=v["agent_type"].get().strip(),
            iterations=int(iters_str) if iters_str else None,
            depth=int(depth_str) if depth_str else None,
            time_s=float(time_str) if time_str else None,
            c=float(c_str) if c_str else math.sqrt(2.0),
            selection=v["selection"].get().strip(),
            rollout=v["rollout"].get().strip(),
            rollout_depth=int(rdepth_str) if rdepth_str else None,
            ordering=v["ordering"].get().strip(),
            candidate_radius=int(rad_str) if rad_str else 2,
            max_candidates=int(maxc_str) if maxc_str else None,
            use_candidates=bool(v["use_candidates"].get()),
            reuse_tree=bool(v["reuse_tree"].get()),
            use_tt=bool(v["use_tt"].get()),
            use_iterative_deepening=bool(v["use_iterative_deepening"].get()),
            seed=int(seed_str) if seed_str else 42,
        )

    # --- Xử lý sự kiện từ người dùng ---

    def on_apply_player_configs(self) -> bool:
        """Đọc cấu hình X và O từ giao diện và áp dụng ngay vào `session`."""
        try:
            cfg_x = self._read_player_config_from_ui(PLAYER_X)
            cfg_o = self._read_player_config_from_ui(PLAYER_O)
        except ValueError as exc:
            self._messagebox.showerror("Lỗi tham số", str(exc), parent=self.root)
            return False

        self.session.configure_player(PLAYER_X, cfg_x)
        self.session.configure_player(PLAYER_O, cfg_o)
        self.mode_var.set(self.session.match_mode)
        self.refresh()

        if self.session.is_current_player_ai() and self.session.match_mode != "ai_vs_ai":
            self._schedule_ai_step()
        return True

    def on_quick_mode_change(self) -> None:
        """Chuyển nhanh chế độ chơi từ combobox `Chế độ nhanh`."""
        self._stop_ai_loop()
        mode = self.mode_var.get().strip()
        self.session.set_mode(mode)
        self._sync_controls_from_session()
        self.refresh()
        if self.session.is_current_player_ai() and self.session.match_mode != "ai_vs_ai":
            self._schedule_ai_step()

    def on_new_game(self) -> None:
        """Khởi tạo ván mới theo loại bàn cờ và cấu hình hiện tại."""
        self._stop_ai_loop()
        if not self.on_apply_player_configs():
            return
        try:
            gtype = self.game_type_var.get().strip()
            size_val = int(self.board_size_var.get().strip() or str(DEFAULT_SIZE))
            gcfg = GameConfig(
                game_type=gtype,
                size=3 if gtype == "tictactoe" else size_val,
                win_length=3 if gtype == "tictactoe" else DEFAULT_WIN_LENGTH,
                block_two_ends=bool(self.block_two_var.get()),
            )
        except ValueError as exc:
            self._messagebox.showerror("Lỗi cấu hình bàn cờ", str(exc), parent=self.root)
            return

        self.session.new_game(gcfg)
        self._cell_overlays.clear()
        self.refresh()
        if self.session.is_current_player_ai() and self.session.match_mode != "ai_vs_ai":
            self._schedule_ai_step()

    def on_undo(self, steps: int | None = None) -> int:
        """Hoàn tác nước đi trên GUI."""
        self._stop_ai_loop()
        undone = self.session.undo(steps=steps)
        self.refresh()
        return undone

    def on_step_ai(self) -> MoveRecord | None:
        """Cho AI đánh đúng 1 nước tại thế cờ hiện tại."""
        if self.session.state.is_terminal():
            return None
        self.on_apply_player_configs()
        rec = self.session.step_ai_move()
        self.refresh()
        return rec

    def on_toggle_ai_loop(self) -> None:
        """Bật / tắt chế độ tự động đánh liên tiếp (AI vs AI)."""
        if self._ai_running:
            self._stop_ai_loop()
            return
        if self.session.state.is_terminal():
            self.on_new_game()
        if not self.on_apply_player_configs():
            return
        if self.session.match_mode != "ai_vs_ai":
            self.session.set_mode("ai_vs_ai")
            self._sync_controls_from_session()
        self._ai_running = True
        self.ai_loop_btn_var.set("Tạm dừng AI vs AI")
        self._schedule_ai_step()

    def on_load_e1(self) -> None:
        """Nạp nhanh thế cờ bẫy E1 trên bàn 3×3."""
        self._stop_ai_loop()
        self.session.load_preset("E1")
        self._sync_controls_from_session()
        self.refresh()

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

    # --- Tương tác chuột trên bàn cờ ---

    def current_geometry(self) -> BoardGeometry:
        """Trả về đối tượng `BoardGeometry` khớp với kích thước hiện tại của Canvas."""
        w = max(240, int(self.canvas.winfo_width()))
        h = max(240, int(self.canvas.winfo_height()))
        if w <= 1:
            w = 620
        if h <= 1:
            h = 620
        return BoardGeometry(
            board_size=self.session.game_config.board_size,
            canvas_width=w,
            canvas_height=h,
        )

    def handle_cell_click(self, row: int, col: int, *, run_ai_sync: bool = False) -> bool:
        """Xử lý khi người chơi nhấp vào ô `(row, col)`.

        Nếu `run_ai_sync=True`, nước trả lời của AI (nếu có) được chạy đồng bộ ngay lập tức
        (phục vụ kiểm thử tự động); ngược lại hẹn giờ qua `root.after` để giao diện mượt.
        """
        if self._ai_running or not self.session.is_current_player_human():
            return False
        move = (int(row), int(col))
        if move not in self.session.state.legal_moves():
            return False

        self.session.play_human_move(move, auto_reply=run_ai_sync)
        self.refresh()
        if not run_ai_sync and self.session.is_current_player_ai():
            self._schedule_ai_step()
        return True

    def _on_canvas_click(self, event: Any) -> None:
        cell = self.current_geometry().pixel_to_cell(float(event.x), float(event.y))
        if cell is not None:
            self.handle_cell_click(cell[0], cell[1], run_ai_sync=False)

    def _on_canvas_motion(self, event: Any) -> None:
        cell = self.current_geometry().pixel_to_cell(float(event.x), float(event.y))
        if cell != self._hover_cell:
            self._hover_cell = cell
            self._draw_board()

    def _on_canvas_leave(self, _event: Any) -> None:
        if self._hover_cell is not None:
            self._hover_cell = None
            self._draw_board()

    # --- Vẽ bàn cờ và cập nhật thống kê ---

    def refresh(self) -> None:
        """Cập nhật toàn bộ thanh trạng thái, hình vẽ bàn cờ, lịch sử và thông báo observer."""
        self.status_var.set(self.session.status_text())
        self._update_stats_and_history()
        self._draw_board()
        for obs in self._observers:
            obs(self.session)

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
            summary_line = " | ".join(parts)
            root_list = stats.get("root")
            if isinstance(root_list, list) and root_list:
                top_moves = ", ".join(
                    f"{item['move']}:N={item['N']},Q={item['q']:+.2f}" for item in root_list[:3]
                )
                summary_line += f"\nTop MCTS: {top_moves}"
            self.stats_summary_var.set(summary_line)
        else:
            self.stats_summary_var.set("Chưa có thống kê nước đi AI.")

        self.history_text.configure(state=tk.NORMAL)
        self.history_text.delete("1.0", tk.END)
        for rec in self.session.history:
            side = "X" if rec.player == PLAYER_X else "O"
            extra = ""
            if "nodes" in rec.stats:
                extra += f" [nodes={rec.stats['nodes']}"
                if "time_s" in rec.stats:
                    extra += f", {rec.stats['time_s'] * 1000.0:.0f}ms"
                extra += "]"
            self.history_text.insert(
                tk.END,
                f"#{rec.ply:02d} {side} ({rec.agent_name}): {rec.move}{extra}\n",
            )
        self.history_text.see(tk.END)
        self.history_text.configure(state=tk.DISABLED)

    def _draw_board(self) -> None:
        canvas = self.canvas
        canvas.delete("all")

        geom = self.current_geometry()
        n = geom.board_size
        x0, y0, cell_size, span = geom.grid_box()
        is_gomoku = self.session.game_config.game_type == "gomoku"

        board_bg = "#F5E6C8" if is_gomoku else "#FFFFFF"
        canvas.create_rectangle(
            x0,
            y0,
            x0 + span,
            y0 + span,
            fill=board_bg,
            outline="#475569",
            width=2,
        )

        win_cells = set(self.session.winning_cells)
        last_mv = self.session.last_move
        board = self.session.state.board

        # 1. Tô nền từng ô (đường thắng, lớp phủ heatmap, ô đang rê chuột)
        for r in range(n):
            for c in range(n):
                x1, y1, x2, y2 = geom.cell_bbox(r, c)
                mv = (r, c)
                fill_color: str | None = None
                overlay = self._cell_overlays.get(mv)
                if mv in win_cells:
                    fill_color = "#FEF08A"
                elif overlay is not None and overlay.bg_color:
                    fill_color = overlay.bg_color
                elif (
                    mv == self._hover_cell
                    and int(board[r, c]) == EMPTY
                    and self.session.is_current_player_human()
                ):
                    fill_color = "#E0F2FE"

                if fill_color is not None:
                    canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="")

                if overlay is not None and overlay.text and int(board[r, c]) == EMPTY:
                    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                    font_sz = max(7, min(11, int(cell_size * 0.22)))
                    canvas.create_text(
                        cx,
                        cy,
                        text=overlay.text,
                        fill=overlay.text_color,
                        font=("Segoe UI", font_sz, "bold"),
                    )

        # 2. Vẽ đường lưới và nhãn tọa độ hàng / cột
        line_w = 2 if n == 3 else 1
        for i in range(n + 1):
            px = x0 + i * cell_size
            py = y0 + i * cell_size
            canvas.create_line(px, y0, px, y0 + span, fill="#64748B", width=line_w)
            canvas.create_line(x0, py, x0 + span, py, fill="#64748B", width=line_w)

        coord_font = ("Segoe UI", max(8, min(11, int(cell_size * 0.25))))
        for i in range(n):
            cx, _ = geom.cell_center(0, i)
            _, cy = geom.cell_center(i, 0)
            canvas.create_text(cx, y0 - 14, text=str(i), fill="#475569", font=coord_font)
            canvas.create_text(x0 - 14, cy, text=str(i), fill="#475569", font=coord_font)

        # 3. Vẽ quân X, O và đánh dấu nước vừa đi
        for r in range(n):
            for c in range(n):
                val = int(board[r, c])
                if val == EMPTY:
                    continue
                x1, y1, x2, y2 = geom.cell_bbox(r, c)
                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                pad = cell_size * (0.22 if n == 3 else 0.16)

                if is_gomoku:
                    # Vẽ quân cờ tròn trên bàn Caro
                    stone_fill = "#1E293B" if val == PLAYER_X else "#F8FAFC"
                    stone_outline = "#0F172A" if val == PLAYER_X else "#334155"
                    canvas.create_oval(
                        x1 + pad,
                        y1 + pad,
                        x2 - pad,
                        y2 - pad,
                        fill=stone_fill,
                        outline=stone_outline,
                        width=2,
                    )
                    label_color = "#F8FAFC" if val == PLAYER_X else "#0F172A"
                    canvas.create_text(
                        cx,
                        cy,
                        text="X" if val == PLAYER_X else "O",
                        fill=label_color,
                        font=("Segoe UI", max(8, int(cell_size * 0.34)), "bold"),
                    )
                else:
                    # Vẽ X và O truyền thống trên bàn 3×3
                    stroke = max(3, int(cell_size * 0.08))
                    if val == PLAYER_X:
                        canvas.create_line(
                            x1 + pad,
                            y1 + pad,
                            x2 - pad,
                            y2 - pad,
                            fill="#1D4ED8",
                            width=stroke,
                            capstyle=self._tk.ROUND,
                        )
                        canvas.create_line(
                            x1 + pad,
                            y2 - pad,
                            x2 - pad,
                            y1 + pad,
                            fill="#1D4ED8",
                            width=stroke,
                            capstyle=self._tk.ROUND,
                        )
                    else:
                        canvas.create_oval(
                            x1 + pad,
                            y1 + pad,
                            x2 - pad,
                            y2 - pad,
                            outline="#DC2626",
                            width=stroke,
                        )

                if last_mv == (r, c):
                    canvas.create_rectangle(
                        x1 + 3,
                        y1 + 3,
                        x2 - 3,
                        y2 - 3,
                        outline="#F97316",
                        width=2,
                    )

        # 4. Vẽ đường nối qua các ô thắng cuộc
        win_list = self.session.winning_cells
        if len(win_list) >= 2:
            sx, sy = geom.cell_center(*win_list[0])
            ex, ey = geom.cell_center(*win_list[-1])
            canvas.create_line(
                sx,
                sy,
                ex,
                ey,
                fill="#16A34A",
                width=max(3, int(cell_size * 0.09)),
                capstyle=self._tk.ROUND,
            )

    def run(self) -> None:
        """Bắt đầu vòng lặp sự kiện chính của cửa sổ Tkinter."""
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Giao diện dòng lệnh (CLI)
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh cho `python -m src.ui.gui_core`."""
    parser = argparse.ArgumentParser(
        description="GUI phần lõi cho Tic-Tac-Toe 3×3 và Caro n×n (P6.2)."
    )
    parser.add_argument(
        "--game",
        choices=GAME_TYPES,
        default="tictactoe",
        help="Loại trò chơi mở ban đầu (mặc định: tictactoe).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=9,
        help="Kích thước bàn Caro n×n khi --game=gomoku (mặc định: 9).",
    )
    parser.add_argument(
        "--block-two-ends",
        action="store_true",
        help="Bật luật chặn hai đầu trên bàn Caro.",
    )
    parser.add_argument(
        "--x-agent",
        choices=AGENT_TYPES,
        default="human",
        help="Thuật toán cầm quân X (mặc định: human).",
    )
    parser.add_argument(
        "--o-agent",
        choices=AGENT_TYPES,
        default="mcts",
        help="Thuật toán cầm quân O (mặc định: mcts).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Ngân sách iterations mặc định cho MCTS (mặc định: 1000).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Giới hạn độ sâu cho Minimax / Alpha-Beta.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=math.sqrt(2.0),
        help="Hằng số khám phá UCT c (mặc định: sqrt(2)).",
    )
    parser.add_argument(
        "--rollout",
        type=str,
        default="l2",
        help="Chính sách rollout cho MCTS: l0, l1, l2, random (mặc định: l2).",
    )
    parser.add_argument(
        "--selection",
        choices=FINAL_SELECTIONS,
        default="robust",
        help="Tiêu chuẩn chọn nước cuối của MCTS (mặc định: robust).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Nạp sẵn một thế cờ mẫu (ví dụ: E1).",
    )
    parser.add_argument(
        "--headless-demo",
        action="store_true",
        help="Chạy thử một ván tự động không mở cửa sổ đồ họa (dùng cho CI/kiểm thử).",
    )
    return parser


def create_session_from_args(args: argparse.Namespace) -> GameSession:
    """Khởi tạo `GameSession` từ các tham số dòng lệnh."""
    game_cfg = GameConfig(
        game_type=args.game,
        size=3 if args.game == "tictactoe" else args.size,
        win_length=3 if args.game == "tictactoe" else DEFAULT_WIN_LENGTH,
        block_two_ends=bool(args.block_two_ends),
    )
    common_kwargs = {
        "iterations": args.iterations,
        "depth": args.depth,
        "c": args.c,
        "rollout": args.rollout,
        "selection": args.selection,
    }
    px = PlayerConfig(agent_type=args.x_agent, **common_kwargs)
    po = PlayerConfig(agent_type=args.o_agent, **common_kwargs)
    session = GameSession(game_config=game_cfg, player_x=px, player_o=po)
    if args.preset:
        session.load_preset(args.preset)
    return session


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào chính khi chạy `python -m src.ui.gui_core`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        session = create_session_from_args(args)
    except ValueError as exc:
        print(f"Lỗi cấu hình: {exc}", file=sys.stderr)
        return 2

    if args.headless_demo:
        if session.players[PLAYER_X].is_human and session.players[PLAYER_O].is_human:
            session.set_mode("ai_vs_ai")
        records = session.run_ai_vs_ai()
        print(f"{session.status_text()} (số nước đã đi: {len(records)})")
        return 0

    app = BoardGUI(session=session)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
