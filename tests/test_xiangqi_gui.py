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
