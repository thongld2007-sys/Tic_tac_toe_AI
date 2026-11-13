"""Unit test cho GUI phần lõi `src/ui/gui_core.py` (P6.2)."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.alphabeta import AlphaBetaAgent
from src.agents.alphabeta_gomoku import AlphaBetaGomoku
from src.agents.mcts import MCTSAgent
from src.agents.minimax import MinimaxAgent
from src.agents.random_agent import RandomAgent
from src.games.base import EMPTY, PLAYER_O, PLAYER_X
from src.games.gomoku import Gomoku
from src.games.tictactoe import TicTacToe
from src.ui.gui_core import (
    AGENT_TYPES,
    BoardGeometry,
    BoardGUI,
    CellOverlay,
    GameConfig,
    GameSession,
    PlayerConfig,
    available_rollout_policies,
    build_agent_for_state,
    build_game,
    find_winning_cells,
    main,
)


def test_game_config_and_build_game_for_3x3_and_gomoku() -> None:
    """Khởi tạo được cả bàn 3×3 và bàn Caro với kích thước/luật tùy chọn."""
    cfg_ttt = GameConfig.from_mapping({"type": "tictactoe"})
    assert cfg_ttt.game_type == "tictactoe"
    assert cfg_ttt.board_size == 3
    state_ttt = build_game(cfg_ttt)
    assert isinstance(state_ttt, TicTacToe)
    assert state_ttt.board.shape == (3, 3)

    cfg_gomoku = GameConfig.from_mapping(
        {"type": "gomoku", "size": 9, "win_length": 5, "block_two_ends": True}
    )
    assert cfg_gomoku.game_type == "gomoku"
    assert cfg_gomoku.board_size == 9
    state_gomoku = build_game(cfg_gomoku)
    assert isinstance(state_gomoku, Gomoku)
    assert state_gomoku.board.shape == (9, 9)
    assert state_gomoku.block_two_ends is True

    with pytest.raises(ValueError, match="game_type"):
        GameConfig(game_type="chess")
    with pytest.raises(ValueError, match="size"):
        GameConfig(game_type="gomoku", size=4, win_length=5)
    with pytest.raises(ValueError, match="Tham số trò chơi"):
        GameConfig.from_mapping({"type": "tictactoe", "unknown": 1})


def test_player_config_and_dynamic_agent_builder_across_games() -> None:
    """Mọi thuật toán và tham số đều khởi tạo được cho cả 3×3 và Caro mà không sửa code."""
    ttt = TicTacToe()
    gomoku = Gomoku(size=9)

    assert "l0" in available_rollout_policies()
    assert "l1" in available_rollout_policies()
    assert "l2" in available_rollout_policies()

    for agent_name in AGENT_TYPES:
        agent_3x3, budget_3x3 = build_agent_for_state(agent_name, ttt)
        agent_gmk, budget_gmk = build_agent_for_state(agent_name, gomoku)
        if agent_name == "human":
            assert agent_3x3 is None
            assert agent_gmk is None
        else:
            assert agent_3x3 is not None
            assert agent_gmk is not None
            mv_3x3 = agent_3x3.select_move(
                ttt,
                budget_3x3,
            )
            assert mv_3x3 in ttt.legal_moves()

    # Kiểm tra 'alphabeta' tự động chọn đúng lớp cho 3×3 và Caro
    ab_3x3, _ = build_agent_for_state({"type": "alphabeta", "ordering": "heuristic"}, ttt)
    ab_gmk, b_gmk = build_agent_for_state(
        {
            "type": "alphabeta",
            "depth": 2,
            "radius": 2,
            "max_candidates": 10,
            "use_tt": False,
            "use_iterative_deepening": False,
        },
        gomoku,
    )
    assert isinstance(ab_3x3, AlphaBetaAgent)
    assert ab_3x3.ordering == "heuristic"
    assert isinstance(ab_gmk, AlphaBetaGomoku)
    assert ab_gmk.default_depth == 2
    assert ab_gmk.max_candidates == 10
    assert ab_gmk.use_tt is False
    assert ab_gmk.use_iterative_deepening is False
    assert b_gmk.depth == 2

    # Kiểm tra tham số hóa MCTS qua từ điển (kèm nested budget giống YAML)
    mcts_agent, mcts_budget = build_agent_for_state(
        {
            "type": "mcts",
            "c": 0.75,
            "selection": "max_robust",
            "rollout": "l1",
            "rollout_depth": 12,
            "use_candidates": True,
            "candidate_radius": 1,
            "reuse_tree": False,
            "seed": 99,
            "budget": {"iterations": 120, "time_s": 0.5},
        },
        gomoku,
    )
    assert isinstance(mcts_agent, MCTSAgent)
    assert mcts_agent.c == pytest.approx(0.75)
    assert mcts_agent.selection == "max_robust"
    assert mcts_agent.rollout == "l1"
    assert mcts_agent.rollout_depth == 12
    assert mcts_agent.use_candidates is True
    assert mcts_agent.candidate_radius == 1
    assert mcts_agent.reuse_tree is False
    assert mcts_budget.iterations == 120
    assert mcts_budget.time_s == pytest.approx(0.5)

    # Kiểm tra Minimax và Random
    mm_agent, mm_budget = build_agent_for_state("minimax", gomoku)
    assert isinstance(mm_agent, MinimaxAgent)
    assert mm_budget.depth == 2  # tự động giới hạn độ sâu an toàn trên Caro
    rnd_agent, _ = build_agent_for_state({"type": "random", "seed": 7}, ttt)
    assert isinstance(rnd_agent, RandomAgent)


def test_player_config_validation_rejects_invalid_values() -> None:
    """PlayerConfig kiểm tra chặt các giá trị tham số sai."""
    with pytest.raises(ValueError, match="agent_type"):
        PlayerConfig(agent_type="unknown")
    with pytest.raises(ValueError, match="iterations"):
        PlayerConfig(iterations=0)
    with pytest.raises(ValueError, match="depth"):
        PlayerConfig(depth=0)
    with pytest.raises(ValueError, match="time_s"):
        PlayerConfig(time_s=-1.0)
    with pytest.raises(ValueError, match="c "):
        PlayerConfig(c=-0.1)
    with pytest.raises(ValueError, match="selection"):
        PlayerConfig(selection="invalid")
    with pytest.raises(ValueError, match="rollout"):
        PlayerConfig(rollout="l99")
    with pytest.raises(ValueError, match="ordering"):
        PlayerConfig(ordering="backwards")


def test_board_geometry_pixel_and_cell_mapping() -> None:
    """Ánh xạ hai chiều tâm ô <-> tọa độ chuột chính xác trên cả bàn 3×3 và 15×15."""
    for size in (3, 9, 15):
        geom = BoardGeometry(board_size=size, canvas_width=640, canvas_height=600, padding=30.0)
        for r in range(size):
            for c in range(size):
                cx, cy = geom.cell_center(r, c)
                assert geom.pixel_to_cell(cx, cy) == (r, c)

        assert geom.pixel_to_cell(5.0, 5.0) is None
        assert geom.pixel_to_cell(635.0, 595.0) is None
        with pytest.raises(ValueError, match="nằm ngoài bàn"):
            geom.cell_bbox(size, 0)


def test_find_winning_cells_on_3x3_and_gomoku_with_block_two_ends() -> None:
    """Phát hiện đúng danh sách các ô tạo nên đường thắng trên 3×3 và Caro."""
    ttt = TicTacToe()
    for mv in [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]:
        ttt.play(mv)
    assert find_winning_cells(ttt) == [(0, 0), (0, 1), (0, 2)]

    gmk = Gomoku(size=9, win_length=5, block_two_ends=True)
    # Tạo 1 dây 5 quân X ở hàng 2 bị chặn cả 2 đầu bởi O(2,1) và O(2,7) -> chưa thắng
    for mv in [
        (2, 2),
        (2, 1),
        (2, 3),
        (2, 7),
        (2, 4),
        (0, 0),
        (2, 5),
        (0, 1),
        (2, 6),
        (0, 2),
    ]:
        gmk.play(mv)
    assert gmk.winner() is None
    assert find_winning_cells(gmk) == []

    # Sau đó X tạo dây 5 quân không bị chặn 2 đầu ở hàng 4 -> thắng ở hàng 4
    for col in range(5):
        gmk.play((4, col))
        if col < 4:
            gmk.play((1, col))
    assert find_winning_cells(gmk) == [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]


def test_session_human_vs_ai_and_smart_undo_on_3x3_and_gomoku() -> None:
    """Chế độ Người vs AI: người đi, AI tự trả lời, và Undo hoàn tác đúng cặp nước đi."""
    session = GameSession(
        game_config={"type": "tictactoe"},
        player_x="human",
        player_o={"type": "alphabeta", "ordering": "heuristic"},
    )
    assert session.match_mode == "human_vs_ai"
    assert session.is_current_player_human()

    records = session.play_human_move((1, 1), auto_reply=True)
    assert len(records) == 2
    assert records[0].player == PLAYER_X
    assert records[0].move == (1, 1)
    assert records[1].player == PLAYER_O
    assert session.state.move_count() == 2
    assert session.is_current_player_human()
    assert session.last_stats.get("nodes", 0) > 0

    # Undo mặc định trong chế độ Người vs AI hoàn tác cả nước AI + nước Người (2 ply)
    undone = session.undo()
    assert undone == 2
    assert session.state.move_count() == 0
    assert session.history == []
    assert (session.state.board == EMPTY).all()

    # Chuyển sang bàn Caro 9×9 ngay trên cùng session mà không cần tạo lại
    session.new_game({"type": "gomoku", "size": 9})
    session.configure_player(
        PLAYER_O,
        {"type": "mcts", "rollout": "l2", "iterations": 60, "seed": 11},
    )
    records_gmk = session.play_human_move((4, 4), auto_reply=True)
    assert len(records_gmk) == 2
    assert session.state.move_count() == 2
    assert "sims_per_s" in session.last_stats
    assert "root" in session.last_stats

    # Undo 1 nước đơn lẻ (chỉ hoàn tác nước của AI, tới lượt O)
    assert session.undo(steps=1) == 1
    assert session.state.move_count() == 1
    assert session.state.current_player() == PLAYER_O


def test_session_ai_vs_ai_and_hot_swap_algorithm_mid_game() -> None:
    """AI vs AI chơi trọn ván và cho phép đổi thuật toán/tham số ngay giữa ván cờ."""
    session = GameSession(
        game_config={"type": "gomoku", "size": 9},
        player_x={"type": "mcts", "rollout": "l1", "iterations": 40, "seed": 1},
        player_o={"type": "random", "seed": 2},
    )
    assert session.match_mode == "ai_vs_ai"

    # Chạy 2 ply đầu
    first_two = session.run_ai_vs_ai(max_plies=2)
    assert len(first_two) == 2
    board_snapshot = session.state.board.copy()

    # Đổi thuật toán của cả X và O ngay giữa ván cờ mà không làm mất bàn cờ hiện tại
    session.configure_player(
        PLAYER_X,
        {"type": "alphabeta_gomoku", "depth": 2, "max_candidates": 8},
    )
    session.configure_player(
        PLAYER_O,
        {"type": "mcts", "rollout": "l2", "c": 0.8, "selection": "max", "iterations": 40},
    )
    assert np.array_equal(session.state.board, board_snapshot)

    next_two = session.run_ai_vs_ai(max_plies=2)
    assert len(next_two) == 2
    assert next_two[0].agent_name == "alphabeta-gomoku"
    assert next_two[1].agent_name == "mcts-c0.80-max"
    assert session.state.move_count() == 4


def test_session_preset_e1_and_mcts_blocks_trap() -> None:
    """Nạp thế cờ mẫu E1 và kiểm tra MCTS chọn đúng ô chặn (2, 2)."""
    session = GameSession()
    session.load_preset("E1")
    assert session.state.move_count() == 4
    assert session.state.current_player() == PLAYER_X

    session.configure_player(
        PLAYER_X,
        {"type": "mcts", "iterations": 600, "rollout": "l0", "seed": 42},
    )
    rec = session.step_ai_move()
    assert rec.move == (2, 2)
    assert rec.stats.get("root")


def test_cli_headless_demo_and_invalid_args(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI `--headless-demo` chạy trọn ván trả mã 0; tham số sai trả mã 2."""
    code = main(
        [
            "--game",
            "tictactoe",
            "--x-agent",
            "alphabeta",
            "--o-agent",
            "random",
            "--headless-demo",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Kết thúc:" in out

    bad_code = main(["--game", "gomoku", "--size", "3", "--headless-demo"])
    assert bad_code == 2


def test_tkinter_board_gui_widgets_when_display_available() -> None:
    """Khi môi trường có màn hình (Windows/macOS), kiểm tra trực tiếp cửa sổ BoardGUI."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Môi trường CI không có DISPLAY cho Tkinter")

    try:
        session = GameSession(
            game_config={"type": "tictactoe"},
            player_x="human",
            player_o={"type": "random", "seed": 10},
        )
        gui = BoardGUI(session=session, master=root)

        # Kiểm tra click ô (0, 0) đồng bộ với nước trả lời của AI
        clicked = gui.handle_cell_click(0, 0, run_ai_sync=True)
        assert clicked is True
        assert gui.session.state.move_count() == 2

        # Kiểm tra lớp phủ heatmap (điểm nối cho P6.1b)
        gui.set_cell_overlays({(1, 1): CellOverlay(bg_color="#BBF7D0", text="N=42")})
        gui.clear_cell_overlays()

        # Kiểm tra Undo trên GUI
        assert gui.on_undo() == 2
        assert gui.session.state.move_count() == 0

        # Kiểm tra chuyển sang bàn Caro 9×9 và đổi thuật toán từ các biến UI
        gui.game_type_var.set("gomoku")
        gui.board_size_var.set("9")
        gui._player_vars[PLAYER_O]["agent_type"].set("mcts")
        gui._player_vars[PLAYER_O]["iterations"].set("30")
        gui._player_vars[PLAYER_O]["rollout"].set("l2")
        gui.on_new_game()

        assert isinstance(gui.session.state, Gomoku)
        assert gui.session.state.size == 9
        assert gui.session.players[PLAYER_O].agent_type == "mcts"
        assert gui.session.players[PLAYER_O].iterations == 30
        assert gui.handle_cell_click(4, 4, run_ai_sync=True) is True
        assert gui.session.state.move_count() == 2
    finally:
        root.destroy()
