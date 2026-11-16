"""Bảng phân tích cho GUI: heatmap N, top 5 nước W/N/UCT và nạp thế cờ (P6.1b).

Chủ sở hữu: TV1.  Review: TV2.

Bảng phân tích gắn vào khung `extension_frame` mà GUI lõi (`src/ui/gui_core.py`, P6.2) để sẵn,
và dùng ba điểm nối của `BoardGUI`: `register_observer`, `set_cell_overlays`,
`clear_cell_overlays`. Mọi phần tính toán nằm trong các hàm thuần (không cần màn hình) để kiểm
thử tự động được; lớp `AnalysisPanel` chỉ lo việc hiển thị.

Tính năng:
  1. Heatmap N: tô màu từng ô theo số lượt thăm `N` của nút con tương ứng tại gốc cây MCTS,
     ô càng đậm thì càng được thăm nhiều; số ghi trong ô là `N`.
  2. Bảng top 5 nước tại gốc: `N`, tỷ lệ `N` trên tổng, `W`, `W/N` và điểm UCT.
     Lưu ý thang đo: MCTS trong repo tính thưởng thắng = +1, hòa = 0, thua = -1, nên
     `W/N` nằm trong [-1, +1] chứ không phải [0, 1] như ví dụ của đề bài.
  3. Phân tích thế hiện tại mà không đánh nước nào (chạy MCTS trên bản sao của bàn cờ).
  4. Nạp thế cờ: thế mẫu (ví dụ E1) bằng một nút bấm, từ ô nhập hoặc từ file văn bản.

Hai cách viết một thế cờ (xem `parse_position`):
    a) Danh sách nước theo thứ tự đã đánh, X đi trước, mỗi nước là `hàng,cột` tính từ 0:
           game: gomoku
           size: 9
           moves: 4,4 4,5 3,3
    b) Sơ đồ bàn cờ, mỗi dòng một hàng, dùng `X`, `O` và `.` cho ô trống:
           O X .
           . O .
           X . .

Cách chạy từ thư mục gốc của repo (nhận mọi tham số của `python -m src.ui.gui_core`):
    python -m src.ui.gui_panel
    python -m src.ui.gui_panel --preset E1
    python -m src.ui.gui_panel --position the_co.txt
    python -m src.ui.gui_panel --preset E1 --headless-demo
"""

from __future__ import annotations

import copy
import math
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.games.base import PLAYER_X, Move
from src.games.gomoku import DEFAULT_WIN_LENGTH
from src.ui.gui_core import (
    PRESET_POSITIONS,
    BoardGUI,
    CellOverlay,
    GameConfig,
    GameSession,
    PlayerConfig,
    build_agent_for_state,
    build_arg_parser,
    build_game,
    create_session_from_args,
)

TOP_K = 5
DEFAULT_ANALYSIS_ITERATIONS = 2000

# Dải màu của heatmap, từ ô ít được thăm (nhạt) tới ô được thăm nhiều nhất (đậm).
HEAT_LOW = (255, 247, 237)
HEAT_HIGH = (194, 65, 12)
HEAT_TEXT_DARK = "#1E293B"
HEAT_TEXT_LIGHT = "#FFFFFF"

_EMPTY_SYMBOLS = "._"
_SHUFFLE_ATTEMPTS = 500


# ---------------------------------------------------------------------------
# Thống kê tại gốc cây MCTS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RootRow:
    """Thống kê của một nước đi (một nút con của gốc cây MCTS)."""

    move: Move
    n: int
    w: float
    q: float
    uct: float
    share: float


@dataclass(frozen=True)
class Analysis:
    """Kết quả phân tích một thế cờ: các nước tại gốc, xếp theo `N` giảm dần."""

    state_hash: int
    move_count: int
    player: int
    rows: tuple[RootRow, ...]
    source: str
    best_move: Move | None = None


def root_rows(stats: Mapping[str, Any] | None) -> list[RootRow]:
    """Đổi `last_stats["root"]` của MCTS thành danh sách `RootRow`, xếp theo `N` giảm dần.

    Trả về danh sách rỗng nếu `stats` không có thống kê gốc (ví dụ agent là Alpha-Beta).
    """
    if not stats:
        return []
    raw = stats.get("root")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []

    items = [i for i in raw if isinstance(i, Mapping) and "move" in i and "N" in i]
    total = sum(int(item["N"]) for item in items)
    rows: list[RootRow] = []
    for item in items:
        n = int(item["N"])
        w = float(item.get("W", 0.0))
        rows.append(
            RootRow(
                move=(int(item["move"][0]), int(item["move"][1])),
                n=n,
                w=w,
                q=w / n if n > 0 else 0.0,
                uct=float(item.get("uct", math.inf if n == 0 else math.nan)),
                share=n / total if total > 0 else 0.0,
            )
        )
    rows.sort(key=lambda row: (-row.n, -row.q, row.move))
    return rows


def top_moves(rows: Sequence[RootRow], k: int = TOP_K) -> list[RootRow]:
    """`k` nước được thăm nhiều nhất (`rows` đã xếp theo `N` giảm dần)."""
    if k < 1:
        raise ValueError(f"k phải >= 1, nhận được {k}")
    return list(rows[:k])


def format_number(value: float, digits: int = 2) -> str:
    """Viết một số thực cho bảng; vô cùng thành `∞`, không xác định thành `-`."""
    if math.isnan(value):
        return "-"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    return f"{value:+.{digits}f}"


def row_cells(rank: int, row: RootRow) -> tuple[str, ...]:
    """Các ô chữ của một dòng trong bảng top nước: hạng, nước, N, %N, W, W/N, UCT."""
    return (
        str(rank),
        f"({row.move[0]}, {row.move[1]})",
        str(row.n),
        f"{row.share * 100.0:.1f}%",
        format_number(row.w, 1),
        format_number(row.q, 3),
        format_number(row.uct, 3),
    )


TABLE_HEADERS: tuple[str, ...] = ("#", "Nước", "N", "%N", "W", "W/N", "UCT")


def format_table(rows: Sequence[RootRow], k: int = TOP_K) -> str:
    """Bảng top `k` nước dạng văn bản (dùng cho chế độ không mở cửa sổ)."""
    lines = [list(TABLE_HEADERS)]
    lines += [list(row_cells(i, row)) for i, row in enumerate(top_moves(rows, k), start=1)]
    widths = [max(len(line[col]) for line in lines) for col in range(len(TABLE_HEADERS))]
    return "\n".join(
        "  ".join(cell.rjust(width) for cell, width in zip(line, widths, strict=True))
        for line in lines
    )


# ---------------------------------------------------------------------------
# Heatmap N
# ---------------------------------------------------------------------------


def heat_color(fraction: float) -> str:
    """Màu `#RRGGBB` cho mức `fraction` trong [0, 1] (0 = nhạt nhất, 1 = đậm nhất)."""
    f = min(1.0, max(0.0, float(fraction)))
    channels = (round(lo + (hi - lo) * f) for lo, hi in zip(HEAT_LOW, HEAT_HIGH, strict=True))
    return "#" + "".join(f"{ch:02X}" for ch in channels)


def heatmap_overlays(rows: Sequence[RootRow]) -> dict[Move, CellOverlay]:
    """Lớp phủ heatmap cho `BoardGUI.set_cell_overlays`: màu theo `N / N_max`, chữ là `N`."""
    n_max = max((row.n for row in rows), default=0)
    overlays: dict[Move, CellOverlay] = {}
    for row in rows:
        fraction = row.n / n_max if n_max > 0 else 0.0
        overlays[row.move] = CellOverlay(
            bg_color=heat_color(fraction),
            text=str(row.n),
            text_color=HEAT_TEXT_LIGHT if fraction >= 0.6 else HEAT_TEXT_DARK,
        )
    return overlays


# ---------------------------------------------------------------------------
# Đọc và ghi thế cờ
# ---------------------------------------------------------------------------


def _replay(config: GameConfig, moves: Sequence[Move]) -> None:
    """Đánh lại `moves` trên bàn mới; báo `ValueError` nếu có nước không hợp lệ."""
    state = build_game(config)
    for index, move in enumerate(moves, start=1):
        try:
            state.play(move)
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Nước thứ {index} {move} không hợp lệ: {exc}") from exc


def _parse_moves(text: str) -> list[Move]:
    moves: list[Move] = []
    for token in text.replace(";", " ").split():
        parts = token.strip("()").split(",")
        if len(parts) != 2:
            raise ValueError(f"Nước {token!r} phải có dạng hàng,cột (ví dụ 1,2)")
        try:
            moves.append((int(parts[0]), int(parts[1])))
        except ValueError as exc:
            raise ValueError(f"Nước {token!r} phải gồm hai số nguyên") from exc
    return moves


def _parse_bool(value: str) -> bool:
    key = value.strip().lower()
    if key in ("true", "yes", "1", "on", "có"):
        return True
    if key in ("false", "no", "0", "off", "không"):
        return False
    raise ValueError(f"Giá trị {value!r} phải là true hoặc false")


def _order_stones(config: GameConfig, xs: list[Move], os_: list[Move]) -> tuple[Move, ...]:
    """Tìm một thứ tự đánh hợp lệ (X trước, xen kẽ) dẫn tới các quân trong sơ đồ.

    Sơ đồ không cho biết thứ tự đánh. Thứ tự đọc theo hàng có thể làm ván kết thúc sớm
    (một bên đủ hàng trước khi đặt hết quân), khi đó thử xáo trộn có seed cố định.
    """
    if len(xs) - len(os_) not in (0, 1):
        raise ValueError(
            f"Số quân không hợp lệ: X = {len(xs)}, O = {len(os_)} "
            "(X đi trước nên X phải bằng O hoặc hơn O đúng 1 quân)"
        )
    rng = random.Random(0)
    xs, os_ = list(xs), list(os_)
    for _ in range(_SHUFFLE_ATTEMPTS):
        moves: list[Move] = []
        for i, x_move in enumerate(xs):
            moves.append(x_move)
            if i < len(os_):
                moves.append(os_[i])
        try:
            _replay(config, moves)
        except ValueError:
            rng.shuffle(xs)
            rng.shuffle(os_)
            continue
        return tuple(moves)
    raise ValueError("Sơ đồ không ứng với thế cờ nào của một ván hợp lệ (ván đã kết thúc sớm)")


def _parse_diagram(rows: list[str], options: Mapping[str, str]) -> tuple[GameConfig, list[Move]]:
    grid: list[list[str]] = []
    for line in rows:
        cells = [ch for ch in line.replace("|", " ") if not ch.isspace()]
        grid.append(cells)
    n = len(grid)
    for r, cells in enumerate(grid):
        if len(cells) != n:
            raise ValueError(f"Sơ đồ phải vuông: có {n} hàng nhưng hàng {r} có {len(cells)} ô")
    config = _config_from_options(options, diagram_size=n)
    size = config.board_size
    if size != n:
        raise ValueError(f"Sơ đồ {n}×{n} không khớp với bàn {size}×{size}")
    xs = [(r, c) for r in range(n) for c in range(n) if grid[r][c] in "Xx"]
    os_ = [(r, c) for r in range(n) for c in range(n) if grid[r][c] in "Oo"]
    return config, list(_order_stones(config, xs, os_))


def _config_from_options(
    options: Mapping[str, str],
    *,
    diagram_size: int | None = None,
    default: GameConfig | None = None,
) -> GameConfig:
    base = default if default is not None else GameConfig()
    if diagram_size is not None:
        base = GameConfig(
            game_type="tictactoe" if diagram_size == 3 else "gomoku",
            size=max(diagram_size, DEFAULT_WIN_LENGTH),
        )
    game_type = options.get("game", base.game_type).strip().lower()
    if game_type in ("3x3", "ttt", "tic-tac-toe"):
        game_type = "tictactoe"
    if game_type == "caro":
        game_type = "gomoku"
    try:
        size = int(options["size"]) if "size" in options else base.size
        win_length = int(options["win_length"]) if "win_length" in options else base.win_length
    except ValueError as exc:
        raise ValueError("size và win_length phải là số nguyên") from exc
    block = base.block_two_ends
    if "block_two_ends" in options:
        block = _parse_bool(options["block_two_ends"])
    if game_type == "tictactoe":
        return GameConfig(game_type="tictactoe")
    return GameConfig(game_type=game_type, size=size, win_length=win_length, block_two_ends=block)


_OPTION_KEYS = ("game", "size", "win_length", "block_two_ends", "moves")


def parse_position(
    text: str,
    default_config: GameConfig | None = None,
) -> tuple[GameConfig, tuple[Move, ...]]:
    """Đọc một thế cờ từ văn bản, trả về `(cấu hình bàn, chuỗi nước đi)`.

    Chấp nhận danh sách nước (`moves: 0,1 0,0 ...` hoặc chỉ `0,1 0,0 ...`) hoặc sơ đồ bàn cờ.
    Các dòng `game:`, `size:`, `win_length:`, `block_two_ends:` là tùy chọn; nếu thiếu thì
    danh sách nước dùng `default_config`, còn sơ đồ suy ra bàn từ kích thước của nó.
    Dòng trống và phần sau dấu `#` bị bỏ qua. Báo `ValueError` kèm lý do nếu thế cờ sai.
    """
    options: dict[str, str] = {}
    diagram: list[str] = []
    move_text: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or set(line) <= set("-+ "):
            continue
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() in _OPTION_KEYS:
            name = key.strip().lower()
            if name == "moves":
                move_text.append(value)
            else:
                options[name] = value.strip()
        elif set(line) <= set("XxOo|" + _EMPTY_SYMBOLS + " \t"):
            diagram.append(line)
        else:
            move_text.append(line)

    if diagram and move_text:
        raise ValueError("Chỉ dùng một trong hai cách: danh sách nước hoặc sơ đồ bàn cờ")
    if diagram:
        config, moves = _parse_diagram(diagram, options)
    else:
        config = _config_from_options(options, default=default_config)
        moves = _parse_moves(" ".join(move_text))
    _replay(config, moves)
    return config, tuple(moves)


def format_position(session: GameSession) -> str:
    """Viết thế cờ hiện tại của `session` thành văn bản mà `parse_position` đọc lại được."""
    cfg = session.game_config
    lines = [f"game: {cfg.game_type}"]
    if cfg.game_type == "gomoku":
        lines.append(f"size: {cfg.size}")
        lines.append(f"win_length: {cfg.win_length}")
        lines.append(f"block_two_ends: {str(cfg.block_two_ends).lower()}")
    lines.append("moves: " + " ".join(f"{r},{c}" for r, c in session.move_history))
    return "\n".join(lines) + "\n"


def load_position_text(session: GameSession, text: str) -> None:
    """Đọc `text` rồi nạp thế cờ vào `session` (bàn cũ giữ nguyên nếu `text` sai)."""
    config, moves = parse_position(text, default_config=session.game_config)
    session.load_position(moves, game_config=config)


def load_position_file(session: GameSession, path: str | Path) -> None:
    """Nạp thế cờ từ file văn bản UTF-8."""
    load_position_text(session, Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Phân tích một thế cờ
# ---------------------------------------------------------------------------


def analysis_config(session: GameSession, iterations: int | None = None) -> PlayerConfig:
    """Cấu hình MCTS dùng để phân tích thế hiện tại.

    Nếu bên đang tới lượt là MCTS thì dùng đúng tham số của bên đó (c, rollout, ...), để bảng
    phân tích cho thấy chính agent ấy đang nghĩ gì; nếu không thì dùng MCTS mặc định.
    """
    player = session.state.current_player()
    current = session.players[player]
    if current.agent_type == "mcts":
        base = current
    else:
        rollout = "l0" if session.game_config.game_type == "tictactoe" else "l2"
        base = PlayerConfig(agent_type="mcts", rollout=rollout)
    if iterations is not None:
        base = replace(base, iterations=iterations, time_s=None)
    return replace(base, reuse_tree=False)


def analyze_position(session: GameSession, iterations: int | None = None) -> Analysis:
    """Chạy MCTS trên bản sao của thế hiện tại; không đánh nước nào, không đổi `session`."""
    if session.state.is_terminal():
        raise ValueError("Ván đã kết thúc, không còn nước nào để phân tích")
    config = analysis_config(session, iterations)
    state = session.state.clone()
    agent, budget = build_agent_for_state(config, state)
    assert agent is not None
    best = agent.select_move(state, budget)
    stats = copy.deepcopy(agent.last_stats)
    total = budget.iterations if budget.iterations is not None else f"{budget.time_s} s"
    return Analysis(
        state_hash=session.state.hash(),
        move_count=session.state.move_count(),
        player=session.state.current_player(),
        rows=tuple(root_rows(stats)),
        source=f"phân tích thế hiện tại, {agent.name}, ngân sách {total}",
        best_move=(int(best[0]), int(best[1])),
    )


def analysis_from_last_move(session: GameSession) -> Analysis | None:
    """Thống kê gốc của nước vừa đánh, nếu nước đó do MCTS chọn; ngược lại `None`.

    Các số liệu thuộc về thế cờ ngay trước nước đó, tức là cây mà agent đã dựng để chọn nước.
    """
    record = session.last_record
    if record is None:
        return None
    rows = root_rows(record.stats)
    if not rows:
        return None
    return Analysis(
        state_hash=session.state.hash(),
        move_count=session.state.move_count(),
        player=record.player,
        rows=tuple(rows),
        source=f"nước #{record.ply} của {record.agent_name}",
        best_move=record.move,
    )


def player_symbol(player: int) -> str:
    """Ký hiệu của người chơi: `X` hoặc `O`."""
    return "X" if player == PLAYER_X else "O"


# ---------------------------------------------------------------------------
# Giao diện Tkinter
# ---------------------------------------------------------------------------


class AnalysisPanel:
    """Bảng phân tích gắn vào `BoardGUI.extension_frame`.

    Bảng hiển thị phân tích của thế đang thấy trên bàn nếu người dùng vừa bấm "Phân tích";
    nếu không thì hiển thị thống kê gốc của nước MCTS vừa đánh. Khi bàn cờ đổi sang thế khác
    mà không có số liệu nào khớp, bảng và heatmap được xóa để tránh hiển thị số liệu cũ.
    """

    def __init__(
        self,
        gui: BoardGUI,
        *,
        top_k: int = TOP_K,
        iterations: int = DEFAULT_ANALYSIS_ITERATIONS,
    ) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self._tk = tk
        self._ttk = ttk
        self._filedialog = filedialog
        self._messagebox = messagebox
        self.gui = gui
        self.top_k = top_k
        self.analysis: Analysis | None = None
        self.shown: Analysis | None = None

        self.heatmap_var = tk.BooleanVar(value=True)
        self.iterations_var = tk.StringVar(value=str(iterations))
        self.preset_var = tk.StringVar(value="E1" if "E1" in PRESET_POSITIONS else "")
        self.position_var = tk.StringVar(value="")
        self.source_var = tk.StringVar(value="")

        self._build_ui(gui.extension_frame)
        gui.register_observer(self.on_session_change)
        self.on_session_change(gui.session)

    # --- Bố cục ---

    def _build_ui(self, parent: Any) -> None:
        tk, ttk = self._tk, self._ttk
        parent.columnconfigure(0, weight=1)

        ttk.Label(parent, textvariable=self.source_var, wraplength=330, justify=tk.LEFT).grid(
            row=0, column=0, sticky="ew"
        )

        columns = ("rank", "move", "n", "share", "w", "q", "uct")
        widths = (24, 54, 50, 50, 56, 56, 56)
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", height=self.top_k)
        for column, header, width in zip(columns, TABLE_HEADERS, widths, strict=True):
            self.tree.heading(column, text=header)
            self.tree.column(column, width=width, minwidth=width, anchor=tk.E, stretch=True)
        self.tree.grid(row=1, column=0, sticky="ew", pady=(4, 2))

        ttk.Label(
            parent,
            text="W/N trong [-1, +1]: thắng +1, hòa 0, thua -1 (tính cho bên đi nước đó).",
            wraplength=330,
            justify=tk.LEFT,
            foreground="#475569",
        ).grid(row=2, column=0, sticky="ew")

        analyze_bar = ttk.Frame(parent)
        analyze_bar.grid(row=3, column=0, sticky="ew", pady=(6, 2))
        ttk.Checkbutton(
            analyze_bar,
            text="Heatmap N",
            variable=self.heatmap_var,
            command=self.on_toggle_heatmap,
        ).pack(side=tk.LEFT)
        ttk.Label(analyze_bar, text="Iterations:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(analyze_bar, textvariable=self.iterations_var, width=7).pack(side=tk.LEFT)
        ttk.Button(analyze_bar, text="Phân tích", command=self.on_analyze).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        preset_bar = ttk.Frame(parent)
        preset_bar.grid(row=4, column=0, sticky="ew", pady=2)
        ttk.Button(
            preset_bar, text="Mở thế E1 và phân tích", command=self.on_open_e1
        ).pack(side=tk.LEFT)
        ttk.Combobox(
            preset_bar,
            textvariable=self.preset_var,
            values=sorted(PRESET_POSITIONS),
            state="readonly",
            width=6,
        ).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(preset_bar, text="Nạp thế mẫu", command=self.on_load_preset).pack(side=tk.LEFT)

        position_bar = ttk.Frame(parent)
        position_bar.grid(row=5, column=0, sticky="ew", pady=2)
        position_bar.columnconfigure(1, weight=1)
        ttk.Label(position_bar, text="Nước:").grid(row=0, column=0, padx=(0, 2))
        entry = ttk.Entry(position_bar, textvariable=self.position_var)
        entry.grid(row=0, column=1, sticky="ew")
        entry.bind("<Return>", lambda _event: self.on_load_text())
        ttk.Button(position_bar, text="Nạp", width=5, command=self.on_load_text).grid(
            row=0, column=2, padx=(2, 0)
        )

        file_bar = ttk.Frame(parent)
        file_bar.grid(row=6, column=0, sticky="ew", pady=2)
        ttk.Button(file_bar, text="Mở file thế cờ…", command=self.on_open_file).pack(side=tk.LEFT)
        ttk.Button(file_bar, text="Lưu thế cờ…", command=self.on_save_file).pack(
            side=tk.LEFT, padx=(6, 0)
        )

    # --- Hiển thị ---

    def on_session_change(self, session: GameSession) -> None:
        """Observer của `BoardGUI`: chọn số liệu khớp với bàn cờ rồi vẽ bảng và heatmap."""
        analysis = self.analysis
        if analysis is not None and (
            analysis.state_hash != session.state.hash()
            or analysis.move_count != session.state.move_count()
        ):
            analysis = self.analysis = None
        if analysis is None:
            analysis = analysis_from_last_move(session)
        self.show(analysis)

    def show(self, analysis: Analysis | None) -> None:
        """Vẽ bảng top nước và heatmap cho `analysis` (hoặc xóa nếu `None`)."""
        self.shown = analysis
        self.tree.delete(*self.tree.get_children())
        if analysis is None:
            self.source_var.set(
                "Chưa có số liệu MCTS cho thế này. Bấm “Phân tích” để chạy MCTS trên thế "
                "hiện tại."
            )
        else:
            total = sum(row.n for row in analysis.rows)
            self.source_var.set(
                f"Bên đi: {player_symbol(analysis.player)}. Tổng N = {total}. "
                f"Nguồn: {analysis.source}. Nước được chọn: {analysis.best_move}."
            )
            for rank, row in enumerate(top_moves(analysis.rows, self.top_k), start=1):
                self.tree.insert("", self._tk.END, values=row_cells(rank, row))
        self._apply_heatmap()

    def _apply_heatmap(self) -> None:
        if self.shown is not None and self.heatmap_var.get():
            self.gui.set_cell_overlays(heatmap_overlays(self.shown.rows))
        else:
            self.gui.clear_cell_overlays()

    def table_values(self) -> list[tuple[str, ...]]:
        """Nội dung các dòng đang hiển thị trong bảng (phục vụ kiểm thử)."""
        return [tuple(self.tree.item(item, "values")) for item in self.tree.get_children()]

    # --- Xử lý nút bấm ---

    def _show_error(self, title: str, exc: Exception) -> None:
        self._messagebox.showerror(title, str(exc), parent=self.gui.root)

    def _read_iterations(self) -> int:
        try:
            iterations = int(self.iterations_var.get().strip())
        except ValueError as exc:
            raise ValueError("Iterations phải là số nguyên >= 1") from exc
        if iterations < 1:
            raise ValueError("Iterations phải là số nguyên >= 1")
        return iterations

    def on_toggle_heatmap(self) -> None:
        """Bật hoặc tắt lớp phủ heatmap trên bàn cờ."""
        self._apply_heatmap()

    def on_analyze(self) -> bool:
        """Chạy MCTS trên thế hiện tại và hiển thị kết quả; trả về False nếu không chạy được."""
        try:
            self.analysis = analyze_position(self.gui.session, self._read_iterations())
        except ValueError as exc:
            self._show_error("Không phân tích được", exc)
            return False
        self.show(self.analysis)
        return True

    def _after_load(self) -> None:
        """Đồng bộ lại GUI lõi sau khi `session` được nạp thế cờ mới."""
        self.analysis = None
        self.gui._stop_ai_loop()
        self.gui._sync_controls_from_session()
        self.gui.refresh()

    def on_open_e1(self) -> bool:
        """Một nút bấm: nạp thế bẫy E1 trên bàn 3×3 rồi phân tích ngay."""
        self.gui.on_load_e1()
        return self.on_analyze()

    def on_load_preset(self) -> bool:
        """Nạp thế mẫu đang chọn trong danh sách."""
        try:
            self.gui.session.load_preset(self.preset_var.get())
        except ValueError as exc:
            self._show_error("Không nạp được thế mẫu", exc)
            return False
        self._after_load()
        return True

    def load_text(self, text: str) -> bool:
        """Nạp thế cờ từ văn bản; báo lỗi và giữ nguyên bàn cờ nếu văn bản sai."""
        try:
            load_position_text(self.gui.session, text)
        except ValueError as exc:
            self._show_error("Thế cờ không hợp lệ", exc)
            return False
        self._after_load()
        return True

    def on_load_text(self) -> bool:
        """Nạp thế cờ từ ô nhập danh sách nước."""
        return self.load_text(self.position_var.get())

    def on_open_file(self) -> bool:
        """Chọn một file văn bản rồi nạp thế cờ trong đó."""
        path = self._filedialog.askopenfilename(
            parent=self.gui.root,
            title="Mở file thế cờ",
            filetypes=[("Văn bản", "*.txt"), ("Tất cả", "*.*")],
        )
        if not path:
            return False
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self._show_error("Không đọc được file", exc)
            return False
        return self.load_text(text)

    def on_save_file(self) -> bool:
        """Lưu thế cờ hiện tại thành file văn bản đọc lại được."""
        path = self._filedialog.asksaveasfilename(
            parent=self.gui.root,
            title="Lưu thế cờ",
            defaultextension=".txt",
            filetypes=[("Văn bản", "*.txt")],
        )
        if not path:
            return False
        try:
            Path(path).write_text(format_position(self.gui.session), encoding="utf-8")
        except OSError as exc:
            self._show_error("Không ghi được file", exc)
            return False
        return True


def attach_panel(gui: BoardGUI, **kwargs: Any) -> AnalysisPanel:
    """Gắn bảng phân tích vào một `BoardGUI` đã tạo."""
    return AnalysisPanel(gui, **kwargs)


# ---------------------------------------------------------------------------
# Giao diện dòng lệnh
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào khi chạy `python -m src.ui.gui_panel`: GUI lõi kèm bảng phân tích."""
    parser = build_arg_parser()
    parser.description = "GUI Tic-Tac-Toe 3×3 và Caro kèm bảng phân tích MCTS (P6.1b)."
    parser.add_argument(
        "--position",
        type=str,
        default=None,
        help="Nạp thế cờ từ file văn bản (danh sách nước hoặc sơ đồ bàn cờ).",
    )
    parser.add_argument(
        "--analysis-iterations",
        type=int,
        default=DEFAULT_ANALYSIS_ITERATIONS,
        help=f"Số iteration khi bấm Phân tích (mặc định: {DEFAULT_ANALYSIS_ITERATIONS}).",
    )
    args = parser.parse_args(argv)
    try:
        if args.analysis_iterations < 1:
            raise ValueError("--analysis-iterations phải >= 1")
        session = create_session_from_args(args)
        if args.position:
            load_position_file(session, args.position)
    except (ValueError, OSError, UnicodeDecodeError) as exc:
        print(f"Lỗi cấu hình: {exc}", file=sys.stderr)
        return 2

    if args.headless_demo:
        # Không mở cửa sổ: in bảng top nước của thế hiện tại (dùng cho CI và kiểm thử).
        try:
            analysis = analyze_position(session, args.analysis_iterations)
        except ValueError as exc:
            print(f"Lỗi: {exc}", file=sys.stderr)
            return 2
        print(f"Bên đi: {player_symbol(analysis.player)}. Nguồn: {analysis.source}.")
        print(format_table(analysis.rows))
        print(f"Nước được chọn: {analysis.best_move}")
        return 0

    app = BoardGUI(session=session)
    attach_panel(app, iterations=args.analysis_iterations)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
