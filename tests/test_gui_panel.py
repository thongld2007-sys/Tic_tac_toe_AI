"""Kiểm thử bảng phân tích của GUI (P6.1b): thống kê gốc, heatmap N, nạp thế cờ."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.games.base import PLAYER_O, PLAYER_X
from src.games.gomoku import Gomoku
from src.games.tictactoe import TicTacToe
from src.ui.gui_core import PRESET_POSITIONS, BoardGUI, GameConfig, GameSession
from src.ui.gui_panel import (
    HEAT_HIGH,
    HEAT_LOW,
    TABLE_HEADERS,
    AnalysisPanel,
    RootRow,
    analysis_config,
    analysis_from_last_move,
    analyze_position,
    format_number,
    format_position,
    format_table,
    heat_color,
    heatmap_overlays,
    load_position_file,
    load_position_text,
    main,
    parse_position,
    root_rows,
    row_cells,
    top_moves,
)

E1_MOVES = ((0, 1), (0, 0), (2, 0), (1, 1))
E1_DIAGRAM = "O | X | .\n-----------\n . | O | .\n-----------\n X | . | .\n"


def _stats() -> dict:
    return {
        "nodes": 21,
        "root": [
            {"move": (1, 0), "N": 8, "W": -4.0, "q": -0.5, "uct": 0.4},
            {"move": (2, 1), "N": 10, "W": 4.0, "q": 0.4, "uct": 1.2},
            {"move": (2, 2), "N": 2, "W": -2.0, "q": -1.0, "uct": 0.7},
            {"move": (1, 2), "N": 0, "W": 0.0, "q": 0.0, "uct": math.inf},
        ],
    }


# --- Thống kê tại gốc ---


def test_root_rows_sorted_by_visits_with_share_and_ratio() -> None:
    rows = root_rows(_stats())
    assert [row.move for row in rows] == [(2, 1), (1, 0), (2, 2), (1, 2)]
    assert [row.n for row in rows] == [10, 8, 2, 0]
    assert rows[0].q == pytest.approx(0.4)
    assert rows[1].q == pytest.approx(-0.5)
    assert sum(row.share for row in rows) == pytest.approx(1.0)
    assert rows[0].share == pytest.approx(0.5)
    assert rows[3].q == 0.0 and math.isinf(rows[3].uct)


@pytest.mark.parametrize("stats", [None, {}, {"nodes": 5}, {"root": []}, {"root": "abc"}])
def test_root_rows_empty_when_agent_has_no_root_statistics(stats: dict | None) -> None:
    assert root_rows(stats) == []


def test_top_moves_keeps_k_most_visited() -> None:
    rows = root_rows(_stats())
    assert [row.move for row in top_moves(rows, 2)] == [(2, 1), (1, 0)]
    assert len(top_moves(rows)) == 4
    with pytest.raises(ValueError):
        top_moves(rows, 0)


def test_table_cells_and_text_table() -> None:
    rows = root_rows(_stats())
    assert row_cells(1, rows[0]) == ("1", "(2, 1)", "10", "50.0%", "+4.0", "+0.400", "+1.200")
    assert row_cells(4, rows[3])[-1] == "∞"
    assert format_number(math.nan) == "-"
    lines = format_table(rows, 3).splitlines()
    assert lines[0].split() == list(TABLE_HEADERS)
    assert len(lines) == 4
    assert lines[1].split()[:4] == ["1", "(2,", "1)", "10"]


# --- Heatmap ---


def test_heat_color_endpoints_and_clamping() -> None:
    assert heat_color(0.0) == "#" + "".join(f"{ch:02X}" for ch in HEAT_LOW)
    assert heat_color(1.0) == "#" + "".join(f"{ch:02X}" for ch in HEAT_HIGH)
    assert heat_color(-3.0) == heat_color(0.0)
    assert heat_color(7.0) == heat_color(1.0)
    middle = heat_color(0.5)
    assert middle not in (heat_color(0.0), heat_color(1.0))


def test_heatmap_overlays_scale_with_visit_count() -> None:
    rows = root_rows(_stats())
    overlays = heatmap_overlays(rows)
    assert set(overlays) == {(2, 1), (1, 0), (2, 2), (1, 2)}
    assert overlays[(2, 1)].bg_color == heat_color(1.0)
    assert overlays[(1, 2)].bg_color == heat_color(0.0)
    assert overlays[(1, 0)].bg_color == heat_color(0.8)
    assert [overlays[m].text for m in ((2, 1), (1, 0), (2, 2), (1, 2))] == ["10", "8", "2", "0"]
    assert overlays[(2, 1)].text_color != overlays[(2, 2)].text_color
    assert heatmap_overlays([]) == {}
    unvisited = [RootRow(move=(0, 0), n=0, w=0.0, q=0.0, uct=math.inf, share=0.0)]
    assert heatmap_overlays(unvisited)[(0, 0)].bg_color == heat_color(0.0)


# --- Đọc và ghi thế cờ ---


def test_parse_move_list_uses_default_game_or_header() -> None:
    config, moves = parse_position("0,1 0,0 2,0 1,1")
    assert config.game_type == "tictactoe"
    assert moves == E1_MOVES

    config, moves = parse_position("# thế thử\ngame: gomoku\nsize: 9\nmoves: 4,4 4,5; (3,3)\n")
    assert (config.game_type, config.size) == ("gomoku", 9)
    assert moves == ((4, 4), (4, 5), (3, 3))

    default = GameConfig(game_type="gomoku", size=11, block_two_ends=True)
    config, moves = parse_position("5,5 5,6", default_config=default)
    assert config == default and moves == ((5, 5), (5, 6))

    config, _ = parse_position("game: caro\nsize: 7\nblock_two_ends: true\nmoves: 3,3")
    assert (config.game_type, config.size, config.block_two_ends) == ("gomoku", 7, True)


def test_parse_diagram_from_the_assignment_gives_e1() -> None:
    config, moves = parse_position(E1_DIAGRAM)
    assert config.game_type == "tictactoe"
    state = TicTacToe()
    for move in moves:
        state.play(move)
    preset = TicTacToe()
    for move in PRESET_POSITIONS["E1"][1]:
        preset.play(move)
    assert (state.board == preset.board).all()
    assert state.current_player() == PLAYER_X


def test_parse_diagram_reorders_stones_when_reading_order_ends_the_game() -> None:
    # Đọc theo hàng thì X đủ ba quân ở hàng 0 trước khi đặt hết quân; phải tìm thứ tự khác.
    config, moves = parse_position("X X X\nO O X\nO X O")
    state = TicTacToe()
    for move in moves:
        state.play(move)
    assert state.move_count() == 9 and state.is_terminal()
    assert moves[-1] in ((0, 0), (0, 1), (0, 2))


def test_parse_diagram_for_gomoku_board() -> None:
    rows = [". " * 9 for _ in range(9)]
    rows[4] = ". . . X O . . . ."
    rows[3] = ". . . X . . . . ."
    config, moves = parse_position("\n".join(rows))
    assert (config.game_type, config.size) == ("gomoku", 9)
    state = Gomoku(9)
    for move in moves:
        state.play(move)
    assert state.move_count() == 3 and state.current_player() == PLAYER_O


@pytest.mark.parametrize(
    "text",
    [
        "0,0 0,0",
        "0,0 3,3",
        "1 2",
        "a,b",
        "X X .\n. . .\n. . .",
        "X O\n. .",
        "X O .\n. .",
        "X X X\nO O O\n. . .",
        "X . .\n. . .\n. . .\nmoves: 0,0",
        "game: chess\nmoves: 0,0",
        "game: gomoku\nsize: abc\nmoves: 0,0",
        "game: gomoku\nsize: 9\nblock_two_ends: maybe\nmoves: 0,0",
        "0,0 0,1 1,0 1,1 2,0 2,1",
    ],
)
def test_parse_position_rejects_invalid_text(text: str) -> None:
    with pytest.raises(ValueError):
        parse_position(text)


def test_format_position_round_trip_and_file(tmp_path: Path) -> None:
    session = GameSession(
        game_config={"type": "gomoku", "size": 9, "block_two_ends": True},
        player_x="human",
        player_o="human",
    )
    session.load_position([(4, 4), (4, 5), (3, 3)])
    text = format_position(session)
    config, moves = parse_position(text)
    assert config == session.game_config
    assert moves == tuple(session.move_history)

    path = tmp_path / "the_co.txt"
    path.write_text(text, encoding="utf-8")
    other = GameSession()
    load_position_file(other, path)
    assert isinstance(other.state, Gomoku)
    assert other.state.hash() == session.state.hash()


def test_load_position_text_keeps_board_when_text_is_invalid() -> None:
    session = GameSession(player_x="human", player_o="human")
    session.play_human_move((1, 1))
    before = session.state.hash()
    with pytest.raises(ValueError):
        load_position_text(session, "0,0 0,0")
    assert session.state.hash() == before
    assert session.move_history == [(1, 1)]


# --- Phân tích thế cờ ---


def test_analyze_e1_finds_the_blocking_move_without_changing_session() -> None:
    session = GameSession(player_x="human", player_o="human")
    session.load_preset("E1")
    before = session.state.hash()
    analysis = analyze_position(session, iterations=600)
    assert session.state.hash() == before and session.state.move_count() == 4
    assert len(session.history) == 4
    assert analysis.player == PLAYER_X
    assert analysis.best_move == (2, 2)
    assert analysis.rows[0].move == (2, 2)
    assert sum(row.n for row in analysis.rows) == 600
    assert {row.move for row in analysis.rows} == set(session.state.legal_moves())
    assert len(top_moves(analysis.rows)) == 5


def test_analysis_config_follows_the_mcts_player_to_move() -> None:
    session = GameSession(
        player_x={"type": "mcts", "c": 0.7, "rollout": "l0", "iterations": 50, "seed": 3},
        player_o="human",
    )
    config = analysis_config(session)
    assert (config.c, config.iterations, config.seed) == (0.7, 50, 3)
    assert config.reuse_tree is False
    assert analysis_config(session, 80).iterations == 80

    # Sau nước của X thì tới lượt O (người): dùng MCTS mặc định.
    session.step_ai_move()
    fallback = analysis_config(session, 40)
    assert fallback.agent_type == "mcts" and fallback.iterations == 40
    assert fallback.c == pytest.approx(math.sqrt(2.0))


def test_analyze_rejects_finished_game() -> None:
    session = GameSession(player_x="human", player_o="human")
    session.load_position([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
    with pytest.raises(ValueError):
        analyze_position(session, iterations=10)


def test_analysis_from_last_move_only_for_mcts_moves() -> None:
    session = GameSession(
        player_x={"type": "mcts", "iterations": 60, "rollout": "l0", "seed": 1},
        player_o={"type": "alphabeta"},
    )
    assert analysis_from_last_move(session) is None
    record = session.step_ai_move()
    analysis = analysis_from_last_move(session)
    assert analysis is not None
    assert analysis.player == PLAYER_X and analysis.best_move == record.move
    assert sum(row.n for row in analysis.rows) == 60

    session.step_ai_move()
    assert analysis_from_last_move(session) is None


# --- Dòng lệnh ---


def test_cli_headless_prints_top_moves_for_e1(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--preset", "E1", "--headless-demo", "--analysis-iterations", "400"])
    assert code == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[1].split() == list(TABLE_HEADERS)
    assert len(lines) == 8
    assert lines[2].split()[:3] == ["1", "(2,", "2)"]
    assert lines[-1].endswith("(2, 2)")


def test_cli_position_file_and_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "e1.txt"
    path.write_text(E1_DIAGRAM, encoding="utf-8")
    assert main(["--position", str(path), "--headless-demo", "--analysis-iterations", "50"]) == 0
    assert "Bên đi: X" in capsys.readouterr().out

    bad = tmp_path / "bad.txt"
    bad.write_text("0,0 0,0", encoding="utf-8")
    assert main(["--position", str(bad), "--headless-demo"]) == 2
    assert main(["--position", str(tmp_path / "missing.txt"), "--headless-demo"]) == 2
    assert main(["--headless-demo", "--analysis-iterations", "0"]) == 2
    assert "Lỗi" in capsys.readouterr().err


# --- Cửa sổ Tkinter ---


def test_tkinter_analysis_panel_when_display_available() -> None:
    """Khi có màn hình, kiểm tra bảng phân tích trên cửa sổ thật: E1 bằng một nút bấm."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Môi trường CI không có DISPLAY cho Tkinter")

    try:
        session = GameSession(player_x="human", player_o="human")
        gui = BoardGUI(session=session, master=root)
        panel = AnalysisPanel(gui, iterations=500)
        errors: list[str] = []
        panel._show_error = lambda title, exc: errors.append(f"{title}: {exc}")

        # Bàn trống, chưa có số liệu: bảng rỗng, không có lớp phủ.
        assert panel.table_values() == []
        assert gui._cell_overlays == {}

        # Một nút bấm mở thế E1 và phân tích.
        assert panel.on_open_e1() is True
        assert gui.session.state.move_count() == 4
        values = panel.table_values()
        assert len(values) == 5
        assert values[0][1] == "(2, 2)"
        assert set(gui._cell_overlays) == set(gui.session.state.legal_moves())
        assert gui._cell_overlays[(2, 2)].bg_color == heat_color(1.0)
        assert gui._cell_overlays[(2, 2)].text == values[0][2]

        # Tắt rồi bật heatmap.
        panel.heatmap_var.set(False)
        panel.on_toggle_heatmap()
        assert gui._cell_overlays == {}
        panel.heatmap_var.set(True)
        panel.on_toggle_heatmap()
        assert len(gui._cell_overlays) == 5

        # Bàn cờ đổi thì số liệu cũ phải biến mất.
        assert gui.handle_cell_click(2, 2, run_ai_sync=True) is True
        assert panel.table_values() == []
        assert gui._cell_overlays == {}

        # Nước do MCTS đánh thì bảng tự hiện thống kê gốc của nước đó.
        gui.session.configure_player(
            PLAYER_O, {"type": "mcts", "iterations": 80, "rollout": "l0", "seed": 2}
        )
        record = gui.on_step_ai()
        assert record is not None
        assert panel.shown is not None and panel.shown.best_move == record.move
        assert panel.table_values()

        # Nạp thế cờ từ ô nhập: Caro 9×9.
        panel.position_var.set("game: gomoku")
        assert panel.load_text("game: gomoku\nsize: 9\nmoves: 4,4 4,5 3,3") is True
        assert isinstance(gui.session.state, Gomoku)
        assert gui.game_type_var.get() == "gomoku"
        assert panel.table_values() == []

        # Văn bản sai: báo lỗi, giữ nguyên bàn cờ.
        assert panel.load_text("4,4 4,4") is False
        assert gui.session.state.move_count() == 3
        panel.iterations_var.set("abc")
        assert panel.on_analyze() is False
        assert len(errors) == 2

        assert panel.on_load_preset() is True
        assert gui.session.state.move_count() == 4
    finally:
        root.destroy()
