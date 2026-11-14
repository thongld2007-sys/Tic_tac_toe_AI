"""Unit test cho chính sách rollout L0–L4 và MCTS trên bàn Caro (P2.3a & P2.3b)."""

from __future__ import annotations

import math
import random
import subprocess
import sys

import pytest
import yaml

from experiments.run import normalize_config, run_experiment, summarize
from src.agents.base import Budget
from src.agents.mcts import MCTSAgent, resolve_rollout
from src.agents.rollouts import (
    _find_initial_wins_1d,
    _gomoku_tables,
    _load_gomoku_grid,
    evaluate_leaf_tanh,
    find_winning_moves,
    measure_sims_per_second,
    rollout_l0,
    rollout_l1,
    rollout_l2,
    rollout_l3,
    rollout_l4,
    score_move_pattern,
    select_softmax_move,
    select_tactical_move,
)
from src.eval.patterns import evaluate as evaluate_patterns
from src.games.base import PLAYER_O, PLAYER_X
from src.games.gomoku import Gomoku
from src.games.tictactoe import TicTacToe


def _play_moves(state: Gomoku | TicTacToe, moves: list[tuple[int, int]]) -> None:
    for mv in moves:
        state.play(mv)


def test_rollout_policies_registered_and_resolvable() -> None:
    """Các chính sách 'l0'–'l4' được đăng ký và giải được qua resolve_rollout & MCTSAgent."""
    for name, expected_fn in (
        ("l0", rollout_l0),
        ("l1", rollout_l1),
        ("l2", rollout_l2),
        ("l3", rollout_l3),
        ("l4", rollout_l4),
    ):
        resolved_name, resolved_fn = resolve_rollout(name.upper())
        assert resolved_name == name
        assert resolved_fn is expected_fn

        agent = MCTSAgent(rollout=name, use_candidates=True, reuse_tree=True, seed=7)
        assert agent.rollout == name
        assert agent.rollout_fn is expected_fn


def test_l0_resolves_to_rollout_l0_in_fresh_process_and_worker_determinism() -> None:
    """Trong tiến trình mới chưa import rollouts, MCTSAgent(rollout='l0') vẫn trỏ tới rollout_l0."""
    code = (
        "import sys; "
        "from src.agents.mcts import MCTSAgent; "
        "assert 'src.agents.rollouts' not in sys.modules; "
        "agent = MCTSAgent(rollout='l0'); "
        "from src.agents.rollouts import rollout_l0; "
        "assert agent.rollout_fn is rollout_l0"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    # Kết quả chạy l0, l1, l2 qua runner không phụ thuộc số tiến trình (workers=1 vs workers=2)
    yaml_text = """
name: test_l0_worker_determinism
seed: 2026
games: 4
swap_first: true
opening_plies: 2
game:
  type: gomoku
  size: 9
matchups:
  - a:
      type: mcts
      rollout: [l0, l1, l2]
      use_candidates: true
      candidate_radius: 2
      reuse_tree: true
      budget:
        iterations: 50
    b:
      type: random
"""
    cfg = normalize_config(yaml.safe_load(yaml_text))
    rows_w1 = run_experiment(cfg, workers=1)
    rows_w2 = run_experiment(cfg, workers=2)
    keys = ("matchup", "game_index", "winner", "score_a", "plies", "a_moves", "a_nodes")
    assert [{k: r[k] for k in keys} for r in rows_w1] == [{k: r[k] for k in keys} for r in rows_w2]


@pytest.mark.parametrize("rollout_fn", [rollout_l0, rollout_l1, rollout_l2, rollout_l3, rollout_l4])
def test_rollouts_preserve_state_on_gomoku_and_tictactoe(rollout_fn) -> None:
    """Mọi hàm rollout L0–L4 phải giữ nguyên bàn cờ, hash và số nước đi sau khi chạy."""
    for state in (Gomoku(size=9), TicTacToe()):
        if isinstance(state, Gomoku):
            _play_moves(state, [(4, 4), (4, 5), (3, 4), (3, 5), (5, 4)])
        else:
            _play_moves(state, [(0, 0), (1, 1), (0, 1)])

        before_board = state.board.copy()
        before_hash = state.hash()
        before_count = state.move_count()
        before_player = state.current_player()

        val_full = rollout_fn(state, random.Random(42))
        assert -1.0 <= val_full <= 1.0
        assert state.move_count() == before_count
        assert state.hash() == before_hash
        assert state.current_player() == before_player
        assert (state.board == before_board).all()

        # Khi cắt sớm với max_moves nhỏ, giá trị vẫn nằm trong [-1, 1] và giữ nguyên state
        val_trunc = rollout_fn(state, random.Random(42), max_moves=1)
        assert -1.0 <= val_trunc <= 1.0
        assert state.move_count() == before_count
        assert state.hash() == before_hash
        assert (state.board == before_board).all()


def test_l1_and_l2_take_immediate_winning_move_on_gomoku() -> None:
    """Khi người sắp đi có nước thắng ngay (4 quân liên tiếp), L1 và L2 thắng ngay ở ply đầu."""
    state = Gomoku(size=9)
    # X: (4,2), (4,3), (4,4), (4,5) -> thắng ngay tại (4,1) hoặc (4,6)
    # O: (2,2), (2,3), (2,4), (2,5) -> O cũng đang có 4 quân, nhưng hiện tới lượt X!
    _play_moves(
        state,
        [(4, 2), (2, 2), (4, 3), (2, 3), (4, 4), (2, 4), (4, 5), (2, 5)],
    )
    assert state.current_player() == PLAYER_X

    win_moves = set(find_winning_moves(state, PLAYER_X))
    assert win_moves == {(4, 1), (4, 6)}

    for use_cand in (False, True):
        mv = select_tactical_move(state, random.Random(1), use_candidates=use_cand)
        assert mv in win_moves

    # Ngay cả khi đối thủ cũng có đường 4 và max_moves=1, L1/L2 vẫn ưu tiên thắng ngay cho X (+1.0)
    assert rollout_l1(state, random.Random(1), max_moves=1) == pytest.approx(1.0)
    assert rollout_l2(state, random.Random(1), max_moves=1) == pytest.approx(1.0)


def test_l1_and_l2_block_immediate_opponent_threat_on_gomoku() -> None:
    """Khi chưa có nước thắng ngay nhưng đối thủ có 4 quân dọa thắng, L1 và L2 bắt buộc chặn."""
    state = Gomoku(size=9)
    # X có 4 quân bị chặn một đầu bởi O tại (4, 1): O(4,1) - X(4,2), X(4,3), X(4,4), X(4,5)
    # Chỉ còn duy nhất ô (4, 6) là điểm thắng ngay của X. Tới lượt O phải chặn (4, 6).
    _play_moves(
        state,
        [
            (4, 2),
            (4, 1),
            (4, 3),
            (0, 0),
            (4, 4),
            (0, 1),
            (4, 5),
        ],
    )
    assert state.current_player() == PLAYER_O
    assert find_winning_moves(state, PLAYER_O) == []
    assert find_winning_moves(state, PLAYER_X) == [(4, 6)]

    for use_cand in (False, True):
        chosen = select_tactical_move(state, random.Random(99), use_candidates=use_cand)
        assert chosen == (4, 6)

    # Nếu chỉ chạy 2 nước (O chặn (4,6), sau đó X đi), X không thể thắng ở nước thứ 2
    assert rollout_l1(state, random.Random(99), max_moves=2) == pytest.approx(0.0)
    assert rollout_l2(state, random.Random(99), max_moves=2) == pytest.approx(0.0)


def test_block_two_ends_rule_respected_in_winning_moves_and_rollouts() -> None:
    """Luật chặn hai đầu: dãy 5 quân bị chặn cả 2 đầu không được tính là nước thắng ngay."""
    # Xây dựng thế: O tại (4, 1) và O tại (4, 7); X có (4, 2), (4, 3), (4, 4), (4, 5).
    # Ô trống (4, 6) nằm giữa X(4,5) và O(4,7): nếu X đánh (4, 6) sẽ tạo 5 quân X (4,2)..(4,6)
    # nhưng bị O chặn ở cả (4, 1) lẫn (4, 7).
    moves = [
        (4, 2),
        (4, 1),
        (4, 3),
        (4, 7),
        (4, 4),
        (0, 0),
        (4, 5),
        (0, 1),
    ]

    state_free = Gomoku(size=9, win_length=5, block_two_ends=False)
    _play_moves(state_free, moves)
    assert find_winning_moves(state_free, PLAYER_X) == [(4, 6)]
    assert rollout_l1(state_free, random.Random(5), max_moves=1) == pytest.approx(1.0)
    assert rollout_l2(state_free, random.Random(5), max_moves=1) == pytest.approx(1.0)

    state_blocked = Gomoku(size=9, win_length=5, block_two_ends=True)
    _play_moves(state_blocked, moves)
    assert find_winning_moves(state_blocked, PLAYER_X) == []
    assert rollout_l1(state_blocked, random.Random(5), max_moves=1) == pytest.approx(0.0)
    assert rollout_l2(state_blocked, random.Random(5), max_moves=1) == pytest.approx(0.0)


def test_l2_restricts_random_fallback_to_candidate_moves() -> None:
    """Khi không có thắng/chặn ngay, L2 chỉ chọn các nước trong bán kính candidate_moves."""
    state = Gomoku(size=15)
    state.play((7, 7))
    candidates = set(state.candidate_moves(radius=2))

    rng = random.Random(123)
    for _ in range(50):
        mv = select_tactical_move(state, rng, use_candidates=True, radius=2)
        assert mv in candidates

    with pytest.raises(ValueError, match="radius"):
        rollout_l2(state, rng, radius=0)


def test_l1_and_l2_on_tictactoe_trap_position_e1() -> None:
    """Trên bàn TicTacToe 3×3 ở thế bẫy E1, L1 và L2 luôn chặn ngay ô (2, 2)."""
    state = TicTacToe()
    _play_moves(state, [(0, 1), (0, 0), (2, 0), (1, 1)])
    assert state.current_player() == PLAYER_X

    assert find_winning_moves(state, PLAYER_X) == []
    assert find_winning_moves(state, PLAYER_O) == [(2, 2)]

    for use_cand in (False, True):
        assert select_tactical_move(state, random.Random(7), use_candidates=use_cand) == (2, 2)


def test_incremental_initial_wins_matches_bruteforce_on_random_games() -> None:
    """Hàm quét nhanh _find_initial_wins_1d khớp tuyệt đối với quét vét cạn trên mọi thế cờ."""
    rng = random.Random(2026)
    for block_two in (False, True):
        for _ in range(30):
            state = Gomoku(size=9, win_length=5, block_two_ends=block_two)
            stride, base_grid, cell_indices, _, dirs, _ = _gomoku_tables(9, 2)
            while not state.is_terminal():
                grid, stones_x, stones_o = _load_gomoku_grid(state, stride, base_grid, cell_indices)
                wx_set, _ = _find_initial_wins_1d(grid, stones_x, PLAYER_X, 5, block_two, dirs)
                wo_set, _ = _find_initial_wins_1d(grid, stones_o, PLAYER_O, 5, block_two, dirs)
                bf_x = {(r + 1) * stride + (c + 1) for r, c in find_winning_moves(state, PLAYER_X)}
                bf_o = {(r + 1) * stride + (c + 1) for r, c in find_winning_moves(state, PLAYER_O)}
                assert wx_set == bf_x
                assert wo_set == bf_o
                state.play(rng.choice(state.candidate_moves(2)))


def test_gomoku_mcts_tree_reuse_and_sims_per_second_measurement() -> None:
    """MCTS trên Caro tái sử dụng cây con khớp hash và đo được tốc độ mô phỏng/giây (L0–L2)."""
    state = Gomoku(size=9)
    state.play((4, 4))
    state.play((4, 5))

    for policy in ("l0", "l1", "l2"):
        metrics = measure_sims_per_second(
            state,
            rollout=policy,
            iterations=200,
            use_candidates=True,
            candidate_radius=2,
            reuse_tree=False,
            seed=42,
        )
        assert metrics["rollout_sims_per_s"] > 100.0
        assert metrics["mcts_sims_per_s"] > 100.0
        assert metrics["mcts_time_s"] > 0.0
        assert metrics["nodes"] > 1.0
        assert metrics["depth"] >= 1.0

    with pytest.raises(ValueError, match="iterations"):
        measure_sims_per_second(state, iterations=0)

    # Kiểm tra tái sử dụng cây qua 2 ply (nước của MCTS + nước trả lời của đối thủ nằm trong cây)
    agent = MCTSAgent(rollout="l2", use_candidates=True, reuse_tree=True, seed=42)
    m1 = agent.select_move(state, Budget(iterations=150))
    assert agent.last_root is not None
    child_m1 = next(ch for ch in agent.last_root.children if ch.move == m1)
    assert child_m1.children, "Nhánh được chọn nhiều nhất phải đã mở rộng tầng 2"
    opp_reply = child_m1.children[0].move
    assert opp_reply is not None
    prior_visits = child_m1.children[0].N
    assert prior_visits > 0

    state.play(m1)
    state.play(opp_reply)
    agent.select_move(state, Budget(iterations=100))
    assert agent.last_root is not None
    assert agent.last_root.N == prior_visits + 100


def test_mcts_l2_beats_random_at_1000_iterations_on_gomoku() -> None:
    """Tiêu chí P2.3a: MCTS (1000 iteration, L2, ứng viên, tái dùng cây) thắng Random >= 95%."""
    yaml_text = """
name: test_p23a_mcts_l2_vs_random
seed: 2026
games: 20
swap_first: true
opening_plies: 2
game:
  type: gomoku
  size: 9
matchups:
  - a:
      type: mcts
      rollout: l2
      use_candidates: true
      candidate_radius: 2
      reuse_tree: true
      budget:
        iterations: 1000
    b:
      type: random
"""
    cfg = normalize_config(yaml.safe_load(yaml_text))
    rows = run_experiment(cfg)
    summary = summarize(rows)[0]
    win_rate = summary["a_wins"] / summary["games"]
    assert win_rate >= 0.95, f"Tỷ lệ thắng Random chỉ đạt {win_rate:.1%} (< 95%)"


def test_l3_local_pattern_scoring_and_softmax_sampling() -> None:
    """P2.3b: L3 chấm điểm mẫu cục bộ cao cho nước tạo OPEN_FOUR và ưu tiên chọn theo softmax."""
    state = Gomoku(size=9)
    # X có 3 quân mở 2 đầu tại (4,3), (4,4), (4,5); O có các quân rời ở góc trên
    _play_moves(
        state,
        [(4, 3), (1, 1), (4, 4), (1, 3), (4, 5), (2, 1)],
    )
    assert state.current_player() == PLAYER_X

    # Hai đầu (4, 2) và (4, 6) tạo OPEN_FOUR cho X nên phải có điểm mẫu cao hơn hẳn ô xa (8, 8)
    s_open4_left = score_move_pattern(state, (4, 2), PLAYER_X)
    s_open4_right = score_move_pattern(state, (4, 6), PLAYER_X)
    s_far = score_move_pattern(state, (8, 8), PLAYER_X)
    assert s_open4_left >= 4.0
    assert s_open4_right >= 4.0
    assert s_far == pytest.approx(0.0)

    # Với nhiệt độ thấp (temperature=0.1), select_softmax_move gần như luôn chọn (4,2) hoặc (4,6)
    rng = random.Random(2026)
    hits = sum(
        select_softmax_move(state, rng, temperature=0.1, radius=2) in {(4, 2), (4, 6)}
        for _ in range(50)
    )
    assert hits >= 45

    # Với nhiệt độ cực nhỏ subnormal (temperature=1e-310), không tràn 1/T thành inf gây NaN
    assert select_softmax_move(state, random.Random(7), temperature=1e-310, radius=2) in {
        (4, 2),
        (4, 6),
    }
    assert rollout_l3(state, random.Random(7), temperature=1e-310) == pytest.approx(1.0)

    # Với nhiệt độ mặc định (temperature=1.0), tần suất chọn (4,2)/(4,6) vẫn vượt trội so với L2
    rng_default = random.Random(2026)
    hits_default = sum(
        select_softmax_move(state, rng_default, temperature=1.0, radius=2) in {(4, 2), (4, 6)}
        for _ in range(200)
    )
    assert hits_default >= 80

    with pytest.raises(ValueError, match="temperature"):
        rollout_l3(state, rng, temperature=0.0)
    with pytest.raises(ValueError, match="radius"):
        rollout_l3(state, rng, radius=0)
    with pytest.raises(ValueError, match="ngoài bàn"):
        score_move_pattern(state, (9, 9))


def test_l4_truncation_and_tanh_pattern_evaluation() -> None:
    """P2.3b: L4 cắt sớm tại d_cut và trả về tanh(evaluate(s, player_at_start) / tau)."""
    state = Gomoku(size=9)
    # Tạo thế X áp đảo hoàn toàn (có OPEN_THREE tại hàng 4), tới lượt X hoặc O
    _play_moves(
        state,
        [(4, 3), (1, 1), (4, 4), (1, 8), (4, 5), (8, 1)],
    )
    assert state.current_player() == PLAYER_X

    # Khi cutoff_depth=0 (cắt ngay lập tức không đi thêm nước nào), L4 trả về đúng tanh(eval / tau)
    tau = 2000.0
    expected_x = math.tanh(evaluate_patterns(state.board, PLAYER_X, state.block_two_ends) / tau)
    assert expected_x > 0.3
    val_cut0 = rollout_l4(state, random.Random(7), cutoff_depth=0, tau=tau)
    assert val_cut0 == pytest.approx(expected_x)
    assert evaluate_leaf_tanh(state, PLAYER_X, tau=tau) == pytest.approx(expected_x)
    assert evaluate_leaf_tanh(state, PLAYER_O, tau=tau) == pytest.approx(-expected_x)

    # Khi chạy với cutoff_depth=2 (chưa đủ số nước để tạo 5 quân), giá trị thuộc (0, 1)
    val_cut2 = rollout_l4(state, random.Random(7), cutoff_depth=2, tau=tau)
    assert 0.0 < val_cut2 < 1.0

    # Khi có nước thắng ngay trong giới hạn d_cut, L4 trả về chính xác +1.0
    state_win1 = Gomoku(size=9)
    _play_moves(
        state_win1,
        [(4, 2), (1, 1), (4, 3), (1, 2), (4, 4), (1, 3), (4, 5), (1, 5)],
    )
    assert rollout_l4(state_win1, random.Random(7), cutoff_depth=2) == pytest.approx(1.0)

    # Kiểm tra tham số tau thực sự có tác dụng trên TicTacToe 3×3 và Gomoku có win_length != 5
    state_ttt = TicTacToe()
    state_ttt.play((1, 1))  # X đi giữa (nằm trên 4 đường thắng), tới lượt O -> raw = -40.0
    for tau_test in (1.0, 10000.0):
        expected_ttt = math.tanh(-40.0 / tau_test)
        assert rollout_l4(state_ttt, random.Random(7), cutoff_depth=0, tau=tau_test) == (
            pytest.approx(expected_ttt)
        )
        assert evaluate_leaf_tanh(state_ttt, tau=tau_test) == pytest.approx(expected_ttt)

    state_g4 = Gomoku(size=5, win_length=4)
    state_g4.play((2, 2))  # X đi tâm 5×5 (nằm trên 8 đoạn dài 4), tới lượt O -> raw = -80.0
    for tau_test in (1.0, 10000.0):
        expected_g4 = math.tanh(-80.0 / tau_test)
        assert rollout_l4(state_g4, random.Random(7), cutoff_depth=0, tau=tau_test) == (
            pytest.approx(expected_g4)
        )
        assert evaluate_leaf_tanh(state_g4, tau=tau_test) == pytest.approx(expected_g4)

    with pytest.raises(ValueError, match="cutoff_depth"):
        rollout_l4(state, random.Random(7), cutoff_depth=-1)
    with pytest.raises(ValueError, match="tau"):
        rollout_l4(state, random.Random(7), tau=0.0)


def test_l3_and_l4_run_in_experiment_runner_and_beat_random() -> None:
    """Tiêu chí P2.3b: L3 và L4 chạy được trong runner (kể cả đa tiến trình) và thắng Random."""
    yaml_text = """
name: test_p23b_l3_l4_runner
seed: 2026
games: 4
swap_first: true
opening_plies: 2
game:
  type: gomoku
  size: 9
matchups:
  - a:
      type: mcts
      rollout: [l3, l4]
      rollout_depth: 6
      use_candidates: true
      candidate_radius: 2
      reuse_tree: true
      budget:
        iterations: 80
    b:
      type: random
"""
    cfg = normalize_config(yaml.safe_load(yaml_text))
    rows_w1 = run_experiment(cfg, workers=1)
    rows_w2 = run_experiment(cfg, workers=2)
    keys = ("matchup", "game_index", "winner", "score_a", "plies", "a_moves", "a_nodes")
    assert [{k: r[k] for k in keys} for r in rows_w1] == [{k: r[k] for k in keys} for r in rows_w2]

    summaries = summarize(rows_w2)
    assert len(summaries) == 2
    for entry in summaries:
        assert entry["score_a"] >= 0.8
