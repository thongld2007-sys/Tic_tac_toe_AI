"""Tìm kiếm Cây Monte Carlo dùng công thức UCT (MCTS-UCT) — P2.2a.

Chủ sở hữu: TV2.  Review: TV3.

Bốn pha của mỗi vòng lặp MCTS:
  1. Selection (Lựa chọn): từ nút gốc, khi nút hiện tại đã mở rộng hết các nước đi
     (`not node.untried_moves`) và chưa kết thúc, chọn nút con cực đại hóa điểm UCT:
         UCT(i) = W_i / N_i + c * sqrt(ln(N_p) / N_i)
  2. Expansion (Mở rộng): nếu thế cờ tại nút hiện tại chưa kết thúc và còn nước chưa thử,
     mở rộng thêm một nút con mới từ `untried_moves`.
  3. Simulation / Rollout (Mô phỏng): từ nút vừa chọn/mở rộng, ước lượng giá trị thế cờ
     trong đoạn [-1.0, +1.0] theo góc nhìn của người sắp đi (`state.current_player()`).
     Mặc định là `default_rollout` (L0: đánh ngẫu nhiên đồng nhất tới cuối ván).
  4. Backpropagation (Cập nhật ngược): lan truyền kết quả ngược từ nút vừa mô phỏng
     lên nút gốc, đổi dấu (`value = -value`) qua mỗi tầng để `node.W` tại mỗi nút luôn
     biểu diễn tổng giá trị theo góc nhìn của `node.player_just_moved` (người vừa thực hiện
     nước đi dẫn tới nút đó).
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from math import inf, log, sqrt

from src.agents.base import Agent, Budget, RootStat
from src.agents.minimax import terminal_value
from src.games.base import GameState, Move

DEFAULT_C = math.sqrt(2.0)
DEFAULT_ITERATIONS = 1000

FINAL_SELECTIONS: tuple[str, ...] = ("robust", "max", "max_robust")

# Hàm rollout nhận (state, rng) và trả về giá trị trong [-1.0, +1.0]
# theo góc nhìn của state.current_player() tại thời điểm bắt đầu rollout.
RolloutFn = Callable[[GameState, random.Random], float]


def default_rollout(state: GameState, rng: random.Random) -> float:
    """Rollout L0: hai bên chọn nước ngẫu nhiên đồng nhất tới khi ván kết thúc.

    Trả về +1.0 nếu người sắp đi ở `state` ban đầu thắng, -1.0 nếu thua, 0.0 nếu hòa.
    Khi hàm kết thúc, `state` được khôi phục nguyên vẹn như lúc truyền vào.
    """
    player_at_start = state.current_player()
    start_count = state.move_count()
    try:
        while not state.is_terminal():
            moves = state.legal_moves()
            state.play(rng.choice(moves))
        winner = state.winner()
        assert winner is not None
        return float(winner * player_at_start)
    finally:
        while state.move_count() > start_count:
            state.undo()


class MCTSNode:
    """Một nút trong cây tìm kiếm MCTS.

    Thuộc tính `W` luôn được tích lũy theo góc nhìn của `player_just_moved` (người chơi
    ở nút cha đã chọn nước `move` để đi xuống nút này). Nhờ vậy khi nút cha dùng UCT để
    so sánh các nút con của nó, `child.W / child.N` chính là kỳ vọng thắng của người chơi
    đang tới lượt tại nút cha.
    """

    __slots__ = (
        "move",
        "parent",
        "player_just_moved",
        "children",
        "untried_moves",
        "N",
        "W",
        "depth",
    )

    def __init__(
        self,
        move: Move | None,
        parent: MCTSNode | None,
        player_just_moved: int,
        untried_moves: list[Move],
        depth: int = 0,
    ) -> None:
        self.move = move
        self.parent = parent
        self.player_just_moved = player_just_moved
        self.children: list[MCTSNode] = []
        self.untried_moves: list[Move] = list(untried_moves)
        self.N: int = 0
        self.W: float = 0.0
        self.depth: int = depth

    @property
    def q(self) -> float:
        """Giá trị trung bình W / N theo góc nhìn của `player_just_moved`."""
        return self.W / self.N if self.N > 0 else 0.0

    def uct_score(self, c: float) -> float:
        """Điểm UCT(i) = W_i / N_i + c * sqrt(ln(N_p) / N_i)."""
        if self.N == 0:
            return inf
        parent_visits = self.parent.N if self.parent is not None and self.parent.N > 0 else 1
        return (self.W / self.N) + c * sqrt(log(parent_visits) / self.N)

    def select_child_uct(self, c: float) -> MCTSNode:
        """Chọn nút con có điểm UCT cao nhất."""
        return max(self.children, key=lambda child: child.uct_score(c))


class MCTSAgent(Agent):
    """Agent Tìm kiếm Cây Monte Carlo với công thức UCT (MCTS-UCT).

    Tham số:
      - `c`: hằng số khám phá trong công thức UCT (mặc định sqrt(2)).
      - `selection`: quy tắc chọn nước đi cuối tại gốc:
          * `"robust"`: chọn nút con có số lượt thăm `N` lớn nhất (mặc định).
          * `"max"`: chọn nút con có tỷ lệ thắng `q = W / N` lớn nhất.
          * `"max_robust"`: chọn nút con đạt cả `max N` và `max q`; nếu không có nút nào
            đồng thời đạt cả hai thì chọn nút có điểm biên dưới an toàn `q - c / sqrt(N)`
            cao nhất trong các nút đã thăm.
      - `rollout_fn`: hàm mô phỏng `rollout(state, rng) -> float` (mặc định `default_rollout`).
      - `use_candidates`: nếu True, thu hẹp các nước mở rộng bằng `state.candidate_moves(radius)`
        thay vì `state.legal_moves()` (hữu ích cho bàn Caro lớn).
      - `candidate_radius`: bán kính dùng khi `use_candidates=True` (mặc định 2).
      - `seed`: hạt giống ngẫu nhiên để tái lập kết quả.
    """

    def __init__(
        self,
        c: float = DEFAULT_C,
        selection: str = "robust",
        rollout_fn: RolloutFn = default_rollout,
        use_candidates: bool = False,
        candidate_radius: int = 2,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if c < 0:
            raise ValueError(f"Hằng số khám phá c phải >= 0, nhận được {c}")
        if selection not in FINAL_SELECTIONS:
            raise ValueError(f"selection phải thuộc {FINAL_SELECTIONS}, nhận được {selection!r}")
        if candidate_radius < 1:
            raise ValueError(f"candidate_radius phải >= 1, nhận được {candidate_radius}")

        self.c = float(c)
        self.selection = selection
        self.rollout_fn = rollout_fn
        self.use_candidates = use_candidates
        self.candidate_radius = candidate_radius
        self.seed = seed
        self.name = f"mcts-c{self.c:.2f}-{selection}"
        self._rng = random.Random(seed)
        self.last_root: MCTSNode | None = None

    def _moves_for(self, state: GameState) -> list[Move]:
        if self.use_candidates:
            return state.candidate_moves(self.candidate_radius)
        return state.legal_moves()

    def _backpropagate(self, node: MCTSNode, value_for_node_player: float) -> None:
        """Lan truyền kết quả ngược từ `node` lên gốc, đổi dấu qua mỗi tầng."""
        value = value_for_node_player
        curr: MCTSNode | None = node
        while curr is not None:
            curr.N += 1
            curr.W += value
            value = -value
            curr = curr.parent

    def _select_final_child(self, root: MCTSNode) -> MCTSNode:
        """Chọn nút con tại gốc khi kết thúc tìm kiếm theo chế độ `self.selection`."""
        visited = [child for child in root.children if child.N > 0]
        pool = visited if visited else root.children
        if not pool:
            raise ValueError("Nút gốc không có nước đi con nào")

        if self.selection == "robust":
            return max(pool, key=lambda ch: (ch.N, ch.q))
        if self.selection == "max":
            return max(pool, key=lambda ch: (ch.q, ch.N))

        # max_robust: nếu nút có N lớn nhất cũng đạt q lớn nhất thì chọn ngay;
        # ngược lại ưu tiên nút có cận dưới tin cậy (Lower Confidence Bound) cao nhất.
        by_n = max(pool, key=lambda ch: (ch.N, ch.q))
        by_q = max(pool, key=lambda ch: (ch.q, ch.N))
        if by_n is by_q:
            return by_n
        lcb_factor = self.c if self.c > 0 else 1.0
        return max(pool, key=lambda ch: (ch.q - lcb_factor / sqrt(ch.N), ch.N))

    def select_move(self, state: GameState, budget: Budget) -> Move:
        if state.is_terminal():
            raise ValueError("Ván đã kết thúc, không còn nước để chọn")
        if budget.iterations is not None and budget.iterations < 1:
            raise ValueError(f"budget.iterations phải >= 1, nhận được {budget.iterations}")
        if budget.time_s is not None and budget.time_s <= 0:
            raise ValueError(f"budget.time_s phải > 0, nhận được {budget.time_s}")

        max_iterations = budget.iterations
        max_time_s = budget.time_s
        if max_iterations is None and max_time_s is None:
            max_iterations = DEFAULT_ITERATIONS

        start = time.perf_counter()
        deadline = (start + max_time_s) if max_time_s is not None else None
        root_move_count = state.move_count()

        root = MCTSNode(
            move=None,
            parent=None,
            player_just_moved=-state.current_player(),
            untried_moves=self._moves_for(state),
            depth=0,
        )
        nodes_created = 1
        max_depth = 0
        iterations_done = 0

        try:
            while True:
                if max_iterations is not None and iterations_done >= max_iterations:
                    break
                if deadline is not None and iterations_done > 0 and time.perf_counter() >= deadline:
                    break

                node = root

                # 1. Selection: đi xuống theo UCT khi nút đã mở hết các nhánh con.
                while not node.untried_moves and node.children and not state.is_terminal():
                    node = node.select_child_uct(self.c)
                    assert node.move is not None
                    state.play(node.move)

                # 2. Expansion: chọn ngẫu nhiên 1 nhánh con chưa thử trong O(1)
                # (hoán đổi với phần tử cuối rồi pop) để tránh thiên lệch vị trí trên bàn lớn.
                if node.untried_moves and not state.is_terminal():
                    idx = self._rng.randrange(len(node.untried_moves))
                    node.untried_moves[idx], node.untried_moves[-1] = (
                        node.untried_moves[-1],
                        node.untried_moves[idx],
                    )
                    move = node.untried_moves.pop()
                    player = state.current_player()
                    state.play(move)
                    child = MCTSNode(
                        move=move,
                        parent=node,
                        player_just_moved=player,
                        untried_moves=self._moves_for(state),
                        depth=node.depth + 1,
                    )
                    node.children.append(child)
                    node = child
                    nodes_created += 1

                if node.depth > max_depth:
                    max_depth = node.depth

                # 3. Simulation (Rollout): định giá theo góc nhìn của state.current_player().
                if state.is_terminal():
                    rollout_val = float(terminal_value(state))
                else:
                    rollout_val = float(self.rollout_fn(state, self._rng))

                # Khôi phục state về lại thế gốc trước khi sang vòng lặp tiếp theo.
                while state.move_count() > root_move_count:
                    state.undo()

                # 4. Backpropagation: đổi sang góc nhìn của node.player_just_moved (-rollout_val).
                self._backpropagate(node, -rollout_val)
                iterations_done += 1
        finally:
            while state.move_count() > root_move_count:
                state.undo()

        elapsed = time.perf_counter() - start
        best_child = self._select_final_child(root)
        assert best_child.move is not None

        root_stats: list[RootStat] = [
            {
                "move": ch.move,
                "N": ch.N,
                "W": ch.W,
                "q": ch.q,
                "uct": ch.uct_score(self.c),
            }
            for ch in sorted(root.children, key=lambda c_node: (-c_node.N, -c_node.q))
            if ch.move is not None
        ]

        self.last_root = root
        self.last_stats = {
            "nodes": nodes_created,
            "time_s": elapsed,
            "depth": max_depth,
            "sims_per_s": (iterations_done / elapsed) if elapsed > 0 else 0.0,
            "root": root_stats,
        }
        return best_child.move
