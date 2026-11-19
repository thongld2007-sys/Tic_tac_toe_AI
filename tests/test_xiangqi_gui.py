"""Unit test cho giao diện đồ họa Cờ tướng (`src/ui/xiangqi_gui.py`, P4.3b)."""

from __future__ import annotations

import pytest

from src.games.xiangqi import BLACK, RED, ROOK, SQUARES, parse_iccs
from src.ui.xiangqi_gui import (
    XIANGQI_PRESETS,
    XiangqiBoardGeometry,
    XiangqiPlayerConfig,
    XiangqiSession,
    build_xiangqi_agent,
    main,
    piece_glyph,
)


def test_xiangqi_board_geometry_roundtrip() -> None:
    """Ánh xạ pixel <-> giao điểm 0..89 chính xác trên toàn bộ 90 giao điểm."""
    geom = XiangqiBoardGeometry(canvas_width=640, canvas_height=700)
    for sq in range(SQUARES):
        cx, cy = geom.square_center(sq)
        assert geom.pixel_to_square(cx, cy) == sq
        assert geom.pixel_to_square(cx + 2.0, cy - 2.0) == sq
    assert geom.pixel_to_square(0.0, 0.0) is None
    with pytest.raises(ValueError):
        geom.square_center(-1)
    with pytest.raises(ValueError):
        geom.square_center(SQUARES)


def test_xiangqi_player_config_and_agent_builder() -> None:
    """Kiểm tra xây dựng agent và kiểm tra tham số không hợp lệ."""
    ag_h, _ = build_xiangqi_agent("human")
    assert ag_h is None

    ag_ab, bud_ab = build_xiangqi_agent(
        XiangqiPlayerConfig(agent_type="alphabeta_xiangqi", depth=2, quiescence_depth=1)
    )
    assert ag_ab is not None
    assert bud_ab.depth == 2

    ag_mcts, bud_mcts = build_xiangqi_agent(
        XiangqiPlayerConfig(agent_type="mcts_xiangqi", iterations=50, rollout_depth=4)
    )
    assert ag_mcts is not None
    assert bud_mcts.iterations == 50

    with pytest.raises(ValueError):
        XiangqiPlayerConfig(agent_type="invalid")
    with pytest.raises(ValueError):
        XiangqiPlayerConfig(agent_type="alphabeta_xiangqi", depth=0)
    with pytest.raises(ValueError):
        XiangqiPlayerConfig(agent_type="mcts_xiangqi", time_s=0.0)


def test_xiangqi_session_two_click_human_vs_ai_and_undo() -> None:
    """Kiểm tra luồng chọn quân 2 nhịp, AI tự đáp trả, ký hiệu ICCS và hoàn tác (undo)."""
    session = XiangqiSession(
        player_red="human",
        player_black=XiangqiPlayerConfig(
            agent_type="alphabeta_xiangqi", depth=1, quiescence_depth=0
        ),
    )
    assert session.match_mode == "human_vs_ai"
    assert piece_glyph(ROOK) == "俥"
    assert piece_glyph(-ROOK) == "車"

    # Nhấp Pháo Đỏ h2 -> chọn quân và hiện các ô đích hợp lệ (bao gồm e2: Pháo đầu)
    from_sq, to_sq = parse_iccs("h2e2")
    assert session.select_square(from_sq) == []
    assert session.selected_square == from_sq
    assert to_sq in session.legal_targets

    # Nhấp lại chính ô đang chọn -> bỏ chọn
    assert session.select_square(from_sq) == []
    assert session.selected_square is None

    # Chọn lại rồi đánh h2e2 với auto_reply=True -> cả Người (Đỏ) và AI (Đen) cùng đi
    session.select_square(from_sq)
    played = session.select_square(to_sq, auto_reply=True)
    assert len(played) == 2
    assert played[0].iccs == "h2e2"
    assert played[0].player == RED
    assert played[1].player == BLACK
    assert len(session.history) == 2

    # Undo mặc định trong Người vs AI hoàn tác cả 2 nước (AI + Người) về lại lượt Đỏ
    undone = session.undo()
    assert undone == 2
    assert len(session.history) == 0
    assert session.state.current_player() == RED


def test_xiangqi_presets_and_headless_demo() -> None:
    """Kiểm tra nạp các thế cờ mẫu (mate_in_1, free_rook) và chế độ --headless-demo."""
    session = XiangqiSession(
        player_red=XiangqiPlayerConfig(agent_type="alphabeta_xiangqi", depth=2, quiescence_depth=0),
        player_black="random",
    )
    for preset_name in XIANGQI_PRESETS:
        session.load_preset(preset_name)
        assert session.state.move_count() == 0

    # Thế chiếu bí 1 nước: Alpha-Beta Đỏ thắng ngay trong 1 nước
    session.load_preset("mate_in_1")
    rec = session.step_ai_move()
    assert rec.player == RED
    assert session.state.is_terminal()
    assert session.state.winner() == RED
    assert "ĐỎ thắng" in session.status_text()

    # Kiểm tra CLI --headless-demo chạy sạch không cần cửa sổ X11
    rc = main(
        [
            "--preset",
            "midgame",
            "--mode",
            "ai_vs_ai",
            "--depth",
            "1",
            "--iterations",
            "15",
            "--headless-demo",
        ]
    )
    assert rc == 0


def test_select_square_edge_cases_switch_and_capture() -> None:
    """Kiểm tra chọn ô trống, ô đối thủ, ô ngoài bàn, đổi quân chọn và ăn quân trên free_rook."""
    session = XiangqiSession(player_red="human", player_black="human")
    session.load_preset("free_rook")

    # Nhấp ô ngoài bàn hoặc ô trống khi chưa chọn quân -> không thay đổi
    assert session.select_square(-1) == []
    assert session.select_square(SQUARES) == []
    assert session.select_square(0) == []
    assert session.selected_square is None

    # Nhấp quân Đen (Xe Đen e7) khi tới lượt Đỏ -> không chọn được
    _, black_rook_sq = parse_iccs("e2e7")
    assert session.select_square(black_rook_sq) == []
    assert session.selected_square is None

    # Chọn Tướng Đỏ f0 rồi đổi sang chọn Xe Đỏ e2
    red_king_sq, _ = parse_iccs("f0f1")
    red_rook_sq, _ = parse_iccs("e2e7")
    session.select_square(red_king_sq)
    assert session.selected_square == red_king_sq
    session.select_square(red_rook_sq)
    assert session.selected_square == red_rook_sq
    assert black_rook_sq in session.legal_targets

    # Nhấp vào một ô trống không hợp lệ -> hủy chọn
    assert session.select_square(0) == []
    assert session.selected_square is None

    # Chọn lại Xe Đỏ e2 và ăn Xe Đen e7
    session.select_square(red_rook_sq)
    played = session.select_square(black_rook_sq)
    assert len(played) == 1
    assert played[0].iccs == "e2e7"
    assert session.state.piece_at(black_rook_sq) == ROOK
    assert session.state.current_player() == BLACK


def test_undo_multiple_steps_and_validation() -> None:
    """Kiểm tra hoàn tác nhiều bước (`steps=1`, `steps=3`, vượt quá lịch sử, bàn trống, lỗi)."""
    session = XiangqiSession(player_red="human", player_black="human")
    assert session.undo() == 0

    moves_iccs = ("h2e2", "h9g7", "h0g2", "i9h9")
    for iccs in moves_iccs:
        session.play_human_move(parse_iccs(iccs))
    assert len(session.history) == 4
    assert session.state.move_count() == 4

    # Hoàn tác 1 bước
    assert session.undo(steps=1) == 1
    assert len(session.history) == 3
    assert session.state.current_player() == BLACK

    # Hoàn tác 2 bước nữa -> còn 1 nước (lượt Đen)
    assert session.undo(steps=2) == 2
    assert len(session.history) == 1
    assert session.history[0].iccs == "h2e2"

    # Truyền steps > len(history) -> hoàn tác hết phần còn lại
    assert session.undo(steps=10) == 1
    assert len(session.history) == 0
    assert session.state.move_count() == 0
    assert session.state.current_player() == RED

    with pytest.raises(ValueError, match="steps"):
        session.play_human_move(parse_iccs("h2e2"))
        session.undo(steps=0)
    with pytest.raises(ValueError, match="steps"):
        session.undo(steps=True)


def test_load_invalid_fen_and_preset_raises() -> None:
    """Kiểm tra nạp chuỗi FEN sai định dạng, thiếu Tướng và tên thế cờ mẫu không tồn tại."""
    session = XiangqiSession()
    for bad_fen in (
        "",
        "not-a-valid-fen",
        "9/9/9/9/9/9/9/9/9/9 w - - 0 1",  # thiếu cả hai Tướng
        "4k4/9/9/9/9 w - - 0 1",  # thiếu hàng
    ):
        with pytest.raises(ValueError):
            session.new_game(fen=bad_fen)

    with pytest.raises(ValueError, match="Thế cờ mẫu"):
        session.load_preset("khong_ton_tai")


def test_run_ai_vs_ai_to_completion_and_mode_switching() -> None:
    """Kiểm tra chạy AI vs AI tới hết ván (terminal) và chuyển đổi đủ 4 chế độ chơi."""
    # Thế tàn cuộc hai Xe Đỏ đấu Tướng Đen đơn độc: Alpha-Beta Đỏ phải chiếu bí nhanh tới hết ván
    endgame_fen = "4k4/9/9/9/9/9/9/1R7/9/R3K4 w - - 0 1"
    session = XiangqiSession(
        fen=endgame_fen,
        max_plies=40,
        player_red=XiangqiPlayerConfig(agent_type="alphabeta_xiangqi", depth=2, quiescence_depth=0),
        player_black="random",
    )
    assert session.match_mode == "ai_vs_ai"
    records = session.run_ai_vs_ai()
    assert len(records) >= 1
    assert session.state.is_terminal()
    assert session.state.winner() == RED
    assert session.last_move == records[-1].move
    assert "nodes" in session.last_stats

    with pytest.raises(ValueError, match="kết thúc"):
        session.step_ai_move()

    # Kiểm tra chuyển đủ 4 chế độ chơi và đổi thuật toán giữa chừng
    session.new_game()
    for mode_name in ("human_vs_ai", "ai_vs_human", "ai_vs_ai", "human_vs_human"):
        session.set_mode(mode_name)
        assert session.match_mode == mode_name
    with pytest.raises(ValueError, match="mode"):
        session.set_mode("che_do_la")
