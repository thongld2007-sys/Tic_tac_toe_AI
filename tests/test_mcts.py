"""Unit test cho MCTS-UCT trên bàn 3x3 (P2.2a)."""

from __future__ import annotations

import math
import random

import pytest
import yaml

from experiments.run import build_agent, normalize_config, run_experiment
from src.agents.base import Budget
from src.agents.mcts import (
    DEFAULT_C,
    MCTSAgent,
    MCTSNode,
    default_rollout,
    register_rollout,
    select_final_child,
)
from src.agents.minimax import MinimaxAgent
from src.games.base import DRAW, PLAYER_O, PLAYER_X
from src.games.gomoku import Gomoku
from src.games.tictactoe import TicTacToe


def make_position_e1() -> TicTacToe:
    """Dựng thế bẫy E1 trên bàn 3x3 (tới lượt X):

        O | X | .
        ---------
        . | O | .
        ---------
        X | . | .

    O đang dọa thắng chéo chính (0, 0)-(1, 1)-(2, 2), nên X bắt buộc phải
    chặn tại ô dưới phải (2, 2).
    """
    state = TicTacToe()
    for move in [(0, 1), (0, 0), (2, 0), (1, 1)]:
        state.play(move)
    assert state.current_player() == PLAYER_X
    return state


def test_mcts_node_uct_score_and_selection() -> None:
    """Công thức UCT trả về inf khi N=0 và khớp W/N + c*sqrt(ln(N_p)/N_i) khi N>0."""
    root = MCTSNode(move=None, parent=None, player_just_moved=PLAYER_O, untried_moves=[])
    root.N = 20

    child_unvisited = MCTSNode(
        move=(0, 0), parent=root, player_just_moved=PLAYER_X, untried_moves=[], depth=1
    )
    child_a = MCTSNode(
        move=(0, 1), parent=root, player_just_moved=PLAYER_X, untried_moves=[], depth=1
    )
    child_a.N = 10
    child_a.W = 6.0

    child_b = MCTSNode(
        move=(0, 2), parent=root, player_just_moved=PLAYER_X, untried_moves=[], depth=1
    )
    child_b.N = 10
    child_b.W = 2.0

    root.children = [child_a, child_b, child_unvisited]

    assert child_unvisited.q == 0.0
    assert math.isinf(child_unvisited.uct_score(DEFAULT_C))

    expected_a = 0.6 + DEFAULT_C * math.sqrt(math.log(20) / 10)
    assert child_a.q == pytest.approx(0.6)
    assert child_a.uct_score(DEFAULT_C) == pytest.approx(expected_a)

    # Còn nút chưa thăm thì select_child_uct ưu tiên nút N=0 (uct = inf).
    assert root.select_child_uct(DEFAULT_C) is child_unvisited

    # Khi mọi nút con đều đã thăm và c=0 (thuần khai thác), chọn nút có q lớn nhất.
    root.children = [child_a, child_b]
    assert root.select_child_uct(0.0) is child_a


def test_backpropagation_flips_perspective_alternating_levels() -> None:
    """Kiểm tra R2.4: lan truyền ngược đổi dấu giá trị qua từng tầng của cây."""
    agent = MCTSAgent(seed=1)

    # Giả sử tại gốc tới lượt X (nên player_just_moved ở root là O).
    # Tầng 1: X vừa đi (player_just_moved = X)
    # Tầng 2: O vừa đi (player_just_moved = O)
    # Tầng 3: X vừa đi (player_just_moved = X) -> tại nút lá này tới lượt O.
    root = MCTSNode(move=None, parent=None, player_just_moved=PLAYER_O, untried_moves=[])
    depth1_x = MCTSNode(
        move=(0, 0), parent=root, player_just_moved=PLAYER_X, untried_moves=[], depth=1
    )
    depth2_o = MCTSNode(
        move=(1, 1), parent=depth1_x, player_just_moved=PLAYER_O, untried_moves=[], depth=2
    )
    depth3_x = MCTSNode(
        move=(2, 2), parent=depth2_o, player_just_moved=PLAYER_X, untried_moves=[], depth=3
    )

    # Trường hợp 1: Kết quả mô phỏng từ nút depth3_x là X thắng.
    # Vì tại depth3_x người vừa đi là X (player_just_moved = X), giá trị theo góc nhìn
    # của X là +1.0.
    agent._backpropagate(depth3_x, +1.0)

    assert depth3_x.N == 1 and depth3_x.W == pytest.approx(+1.0)  # tốt cho X
    assert depth2_o.N == 1 and depth2_o.W == pytest.approx(-1.0)  # xấu cho O
    assert depth1_x.N == 1 and depth1_x.W == pytest.approx(+1.0)  # tốt cho X
    assert root.N == 1 and root.W == pytest.approx(-1.0)  # góc nhìn player_just_moved của root (O)

    # Trường hợp 2: Kết quả mô phỏng tiếp theo là O thắng -> xấu cho X (-1.0 tại depth3_x).
    agent._backpropagate(depth3_x, -1.0)

    assert depth3_x.N == 2 and depth3_x.W == pytest.approx(0.0)
    assert depth2_o.N == 2 and depth2_o.W == pytest.approx(0.0)
    assert depth1_x.N == 2 and depth1_x.W == pytest.approx(0.0)
    assert root.N == 2 and root.W == pytest.approx(0.0)

    # Trường hợp 3: Hòa (0.0) -> mọi tầng giữ nguyên W, chỉ tăng N.
    agent._backpropagate(depth3_x, 0.0)
    for node in (depth3_x, depth2_o, depth1_x, root):
        assert node.N == 3
        assert node.W == pytest.approx(0.0)


def test_backpropagation_matches_winning_leaf_for_both_players() -> None:
    """Khi một nước đi tại gốc thắng ngay, nút con đó phải nhận W > 0 (q = +1.0)."""
    # Thế cờ X thắng ngay tại (0, 2)
    state = TicTacToe()
    for mv in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        state.play(mv)

    agent = MCTSAgent(c=0.5, seed=7)
    chosen = agent.select_move(state, Budget(iterations=200))
    assert chosen == (0, 2)

    win_stat = next(item for item in agent.last_stats["root"] if item["move"] == (0, 2))
    assert win_stat["q"] > 0.9
    assert win_stat["W"] > 0


def test_default_rollout_restores_state_and_returns_correct_sign() -> None:
    """default_rollout phải trả về đúng dấu theo người sắp đi và khôi phục nguyên state."""
    # Thế cờ chỉ còn đúng 1 ô trống (2, 2) và X đi vào đó sẽ thắng đường chéo.
    state = TicTacToe()
    for mv in [(0, 0), (0, 1), (1, 1), (0, 2), (2, 0), (1, 0), (1, 2), (2, 1)]:
        state.play(mv)
    # Bàn hiện tại:
    # X: (0,0), (1,1), (2,0), (1,2)
    # O: (0,1), (0,2), (1,0), (2,1)
    # Tới lượt X, chỉ còn ô (2, 2); X đánh (2, 2) sẽ thắng chéo (0,0)-(1,1)-(2,2).
    before_board = state.board.copy()
    before_hash = state.hash()
    before_count = state.move_count()

    val = default_rollout(state, random.Random(42))
    assert val == pytest.approx(1.0)
    assert state.move_count() == before_count
    assert state.hash() == before_hash
    assert (state.board == before_board).all()


@pytest.mark.parametrize("seed", [0, 7, 42, 123, 2026])
def test_mcts_selects_bottom_right_on_position_e1(seed: int) -> None:
    """Tiêu chuẩn hoàn thành P2.2a: Chọn đúng ô dưới phải (2, 2) ở thế E1 với >= 1000 iteration."""
    state = make_position_e1()
    before_board = state.board.copy()
    before_hash = state.hash()

    agent = MCTSAgent(c=DEFAULT_C, selection="robust", seed=seed)
    move = agent.select_move(state, Budget(iterations=1000))

    assert move == (2, 2)
    # State không bị biến đổi sau khi select_move kết thúc
    assert (state.board == before_board).all()
    assert state.hash() == before_hash

    # Trong bảng thống kê tại gốc, nước (2, 2) phải đứng đầu về lượt thăm N và có q cao nhất
    root_stats = agent.last_stats["root"]
    assert len(root_stats) == 5  # Thế E1 có 5 ô trống
    assert root_stats[0]["move"] == (2, 2)
    best_q = max(item["q"] for item in root_stats)
    assert root_stats[0]["q"] == pytest.approx(best_q)
    # Các nước không chặn (2, 2) đều để O thắng ngay ở nước kế nên q âm sâu
    for item in root_stats[1:]:
        assert item["q"] < 0.0


def test_mcts_blocks_immediate_loss_as_player_o() -> None:
    """Khi tới lượt O và X đang dọa thắng ngay ở (0, 2), MCTS của O phải chặn (0, 2)."""
    state = TicTacToe()
    for mv in [(0, 0), (1, 1), (0, 1)]:
        state.play(mv)
    assert state.current_player() == PLAYER_O

    agent = MCTSAgent(seed=99)
    move = agent.select_move(state, Budget(iterations=600))
    assert move == (0, 2)


@pytest.mark.parametrize("selection", ["robust", "max", "max_robust"])
def test_final_move_selection_modes_on_e1(selection: str) -> None:
    """Cả 3 chế độ chọn nước cuối ('robust', 'max', 'max_robust') đều chọn đúng (2, 2) ở thế E1."""
    state = make_position_e1()
    agent = MCTSAgent(selection=selection, seed=42)
    assert agent.select_move(state, Budget(iterations=1000)) == (2, 2)


def test_select_final_child_tie_break_between_max_and_robust() -> None:
    """Kiểm tra trực tiếp _select_final_child khi nút max-N và nút max-q khác nhau."""
    root = MCTSNode(move=None, parent=None, player_just_moved=PLAYER_O, untried_moves=[])

    # Nút A: rất nhiều lượt thăm (N=100), tỷ lệ thắng tốt và ổn định (q=0.70)
    child_high_n = MCTSNode(
        move=(0, 0), parent=root, player_just_moved=PLAYER_X, untried_moves=[], depth=1
    )
    child_high_n.N = 100
    child_high_n.W = 70.0

    # Nút B: mới thăm 1 lần nhưng tình cờ thắng (N=1, q=1.0)
    child_lucky = MCTSNode(
        move=(0, 1), parent=root, player_just_moved=PLAYER_X, untried_moves=[], depth=1
    )
    child_lucky.N = 1
    child_lucky.W = 1.0

    root.children = [child_high_n, child_lucky]

    robust_agent = MCTSAgent(c=DEFAULT_C, selection="robust")
    max_agent = MCTSAgent(c=DEFAULT_C, selection="max")
    max_robust_agent = MCTSAgent(c=DEFAULT_C, selection="max_robust")

    assert robust_agent._select_final_child(root) is child_high_n
    assert max_agent._select_final_child(root) is child_lucky
    # max_robust dùng cận dưới tin cậy q - c/sqrt(N) nên tránh bị đánh lừa bởi N=1
    assert max_robust_agent._select_final_child(root) is child_high_n


def test_mcts_reproducibility_and_stats_schema() -> None:
    """Cùng seed và cùng số iteration phải cho kết quả và cây tìm kiếm hệt nhau."""
    state = make_position_e1()

    a1 = MCTSAgent(seed=12345)
    m1 = a1.select_move(state, Budget(iterations=300))

    a2 = MCTSAgent(seed=12345)
    m2 = a2.select_move(state, Budget(iterations=300))

    assert m1 == m2
    assert a1.last_stats["nodes"] == a2.last_stats["nodes"]
    assert a1.last_stats["depth"] == a2.last_stats["depth"]
    assert [(r["move"], r["N"], r["W"]) for r in a1.last_stats["root"]] == [
        (r["move"], r["N"], r["W"]) for r in a2.last_stats["root"]
    ]
    assert sum(r["N"] for r in a1.last_stats["root"]) == 300
    assert a1.last_stats["time_s"] >= 0.0
    assert a1.last_stats["sims_per_s"] > 0.0


def test_mcts_time_budget_and_default_budget() -> None:
    """MCTS chạy được với ngân sách thời gian (time_s) và ngân sách mặc định."""
    state = TicTacToe()
    agent = MCTSAgent(seed=11)

    move_timed = agent.select_move(state, Budget(time_s=0.03))
    assert move_timed in state.legal_moves()
    assert sum(r["N"] for r in agent.last_stats["root"]) >= 1

    # Budget rỗng sẽ dùng DEFAULT_ITERATIONS = 1000
    move_default = agent.select_move(state, Budget())
    assert move_default in state.legal_moves()
    assert sum(r["N"] for r in agent.last_stats["root"]) == 1000


def test_mcts_use_candidates_on_gomoku_3x3() -> None:
    """Chế độ use_candidates=True hoạt động đúng trên bàn Gomoku."""
    state = Gomoku(size=3, win_length=3)
    for mv in [(0, 1), (0, 0), (2, 0), (1, 1)]:
        state.play(mv)

    agent = MCTSAgent(use_candidates=True, candidate_radius=2, seed=42)
    assert agent.seed == 42
    move = agent.select_move(state, Budget(iterations=600))
    assert move == (2, 2)


def test_mcts_expansion_avoids_top_left_order_bias() -> None:
    """Các nút con mở rộng không bị cố định ở góc trên-trái khi số iteration nhỏ."""
    state = Gomoku(size=9, win_length=5)
    agent = MCTSAgent(use_candidates=False, seed=42)
    agent.select_move(state, Budget(iterations=10))
    assert agent.last_root is not None
    expanded_moves = [ch.move for ch in agent.last_root.children]
    first_ten_top_left = [(0, c) for c in range(9)] + [(1, 0)]
    assert expanded_moves != first_ten_top_left


def test_mcts_unbeaten_against_random_on_3x3() -> None:
    """MCTS (600 iteration) đấu 20 ván với đối thủ ngẫu nhiên trên 3x3 không thua ván nào."""
    for game_idx in range(20):
        state = TicTacToe()
        mcts_player = PLAYER_X if game_idx < 10 else PLAYER_O
        mcts = MCTSAgent(seed=100 + game_idx)
        rng = random.Random(500 + game_idx)

        while not state.is_terminal():
            if state.current_player() == mcts_player:
                mv = mcts.select_move(state, Budget(iterations=600))
            else:
                mv = rng.choice(state.legal_moves())
            state.play(mv)

        winner = state.winner()
        assert winner in (mcts_player, DRAW)


def test_mcts_draws_against_minimax_on_3x3() -> None:
    """MCTS (1500 iteration) đấu với Minimax tối ưu trên 3x3 giữ hòa cả khi đi trước và đi sau."""
    minimax = MinimaxAgent()
    for mcts_player in (PLAYER_X, PLAYER_O):
        state = TicTacToe()
        mcts = MCTSAgent(seed=2026)
        while not state.is_terminal():
            if state.current_player() == mcts_player:
                mv = mcts.select_move(state, Budget(iterations=1500))
            else:
                mv = minimax.select_move(state, Budget())
            state.play(mv)
        assert state.winner() == DRAW


def test_mcts_invalid_parameters_and_terminal_state() -> None:
    """Kiểm tra bắt lỗi tham số không hợp lệ và khi gọi trên thế cờ đã kết thúc."""
    with pytest.raises(ValueError, match="c"):
        MCTSAgent(c=-0.1)
    with pytest.raises(ValueError, match="c"):
        MCTSAgent(c=True)
    with pytest.raises(ValueError, match="c"):
        MCTSAgent(c=float("nan"))

    with pytest.raises(ValueError, match="selection"):
        MCTSAgent(selection="invalid")

    with pytest.raises(ValueError, match="use_candidates"):
        MCTSAgent(use_candidates=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="candidate_radius"):
        MCTSAgent(candidate_radius=0)
    with pytest.raises(ValueError, match="candidate_radius"):
        MCTSAgent(candidate_radius=True)

    with pytest.raises(ValueError, match="rollout"):
        MCTSAgent(rollout="unknown_policy")

    with pytest.raises(ValueError, match="rollout_depth"):
        MCTSAgent(rollout_depth=0)
    with pytest.raises(ValueError, match="rollout_depth"):
        MCTSAgent(rollout_depth=True)

    with pytest.raises(ValueError, match="reuse_tree"):
        MCTSAgent(reuse_tree=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="seed"):
        MCTSAgent(seed=True)

    agent = MCTSAgent(seed=1)
    state = TicTacToe()

    with pytest.raises(ValueError, match="iterations"):
        agent.select_move(state, Budget(iterations=0))
    with pytest.raises(ValueError, match="iterations"):
        agent.select_move(state, Budget(iterations=True))

    with pytest.raises(ValueError, match="time_s"):
        agent.select_move(state, Budget(time_s=0.0))
    with pytest.raises(ValueError, match="time_s"):
        agent.select_move(state, Budget(time_s=True))

    # Đánh tới khi X thắng
    for mv in [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]:
        state.play(mv)
    assert state.is_terminal()

    with pytest.raises(ValueError, match="kết thúc"):
        agent.select_move(state, Budget(iterations=10))


def test_mcts_all_r26_parameters_configurable_via_yaml() -> None:
    """Tiêu chí P2.2b: Mọi tham số R2.6 của MCTS đều chỉnh và bung tổ hợp được qua YAML."""
    yaml_text = """
name: test_mcts_yaml_r26
seed: 2026
games: 2
swap_first: true
game:
  type: gomoku
  size: 5
matchups:
  - a:
      type: mcts
      c: [0.5, 1.4]
      selection: [robust, max, max_robust]
      rollout: l0
      rollout_depth: 10
      use_candidates: true
      candidate_radius: 2
      reuse_tree: true
      budget:
        iterations: 30
        time_s: 0.5
    b:
      type: random
"""
    cfg = normalize_config(yaml.safe_load(yaml_text))
    assert len(cfg["matchups"]) == 2 * 3

    first_spec = cfg["matchups"][0]["a"]
    agent, budget = build_agent(first_spec, seed=99)
    assert isinstance(agent, MCTSAgent)
    assert agent.get_params() == {
        "c": 0.5,
        "selection": "robust",
        "rollout": "l0",
        "rollout_depth": 10,
        "use_candidates": True,
        "candidate_radius": 2,
        "reuse_tree": True,
        "seed": 99,
    }
    assert budget == Budget(iterations=30, depth=None, time_s=0.5)

    rows = run_experiment(cfg, workers=1)
    assert len(rows) == 6 * 2


def test_mcts_tree_reuse_across_moves_and_reset() -> None:
    """Khi bật reuse_tree=True, nhánh cây con khớp state.hash() được tái sử dụng ở lượt sau."""
    state = TicTacToe()
    agent = MCTSAgent(reuse_tree=True, seed=42)

    m1 = agent.select_move(state, Budget(iterations=100))
    assert agent.last_root is not None
    child_after_m1 = next(ch for ch in agent.last_root.children if ch.move == m1)
    visits_on_m1 = child_after_m1.N
    assert visits_on_m1 > 0

    # Đánh nước m1 của X, tới lượt O (cùng agent hoặc nhánh con trực tiếp)
    state.play(m1)
    agent.select_move(state, Budget(iterations=50))
    assert agent.last_root is not None
    assert agent.last_root.parent is None
    assert agent.last_root.move is None
    # Tổng lượt thăm tại gốc mới phải bằng visits_on_m1 cũ + 50 vòng lặp mới
    assert agent.last_root.N == visits_on_m1 + 50

    # Khi gọi reset_tree() hoặc sang ván mới không khớp hash, cây được tạo mới
    agent.reset_tree()
    assert agent.last_root is None


def test_mcts_custom_rollout_registration_and_rollout_depth() -> None:
    """Kiểm tra đăng ký chính sách rollout tùy chỉnh và truyền max_moves vào rollout."""
    seen_max_moves: list[int | None] = []

    def custom_with_depth(
        state: TicTacToe,
        rng: random.Random,
        max_moves: int | None = None,
    ) -> float:
        seen_max_moves.append(max_moves)
        return 0.0

    register_rollout("custom_depth", custom_with_depth)  # type: ignore[arg-type]
    agent_depth = MCTSAgent(rollout="custom_depth", rollout_depth=4, seed=7)
    assert agent_depth.rollout == "custom_depth"

    state = make_position_e1()
    move = agent_depth.select_move(state, Budget(iterations=20))
    assert move in state.legal_moves()
    assert seen_max_moves and all(m == 4 for m in seen_max_moves)

    # Hàm rollout 2 tham số (không có max_moves) vẫn hoạt động bình thường làm fallback
    register_rollout("zero_eval", lambda s, rng: 0.0)
    agent_two_arg = MCTSAgent(rollout="zero_eval", rollout_depth=2, seed=7)
    assert agent_two_arg.select_move(state, Budget(iterations=20)) in state.legal_moves()

    # select_final_child hàm độc lập kiểm tra lỗi đầu vào
    assert agent_two_arg.last_root is not None
    assert select_final_child(agent_two_arg.last_root, "robust").move in state.legal_moves()
    with pytest.raises(ValueError, match="selection"):
        select_final_child(agent_two_arg.last_root, "unknown")
