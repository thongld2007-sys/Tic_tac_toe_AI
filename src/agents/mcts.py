"""Tìm kiếm Cây Monte Carlo dùng công thức UCT (MCTS-UCT) — P2.2a & P2.2b.

Chủ sở hữu: TV2.  Review: TV3.

Bốn pha của mỗi vòng lặp MCTS:
  1. Selection (Lựa chọn): từ nút gốc, khi nút hiện tại đã mở rộng hết các nước đi
     (`not node.untried_moves`) và chưa kết thúc, chọn nút con cực đại hóa điểm UCT:
         UCT(i) = W_i / N_i + c * sqrt(ln(N_p) / N_i)
  2. Expansion (Mở rộng): nếu thế cờ tại nút hiện tại chưa kết thúc và còn nước chưa thử,
     chọn ngẫu nhiên 1 nước từ `untried_moves` trong O(1) để tạo nút con mới.
  3. Simulation / Rollout (Mô phỏng): từ nút vừa chọn/mở rộng, ước lượng giá trị thế cờ
     trong đoạn [-1.0, +1.0] theo góc nhìn của người sắp đi (`state.current_player()`).
     Mặc định là `default_rollout` (L0: đánh ngẫu nhiên đồng nhất tới cuối ván).
  4. Backpropagation (Cập nhật ngược): lan truyền kết quả ngược từ nút vừa mô phỏng
     lên nút gốc, đổi dấu (`value = -value`) qua mỗi tầng để `node.W` tại mỗi nút luôn
     biểu diễn tổng giá trị theo góc nhìn của `node.player_just_moved` (người vừa thực hiện
     nước đi dẫn tới nút đó).

Tham số hóa đầy đủ (R2.6 — chỉnh được qua YAML trong `experiments/run.py`):
  - `c`: hằng số khám phá UCT (`c >= 0`, mặc định `sqrt(2)`).
  - `selection`: quy tắc chọn nước cuối tại gốc (`"robust"`, `"max"`, `"max_robust"`).
  - `rollout`: tên chính sách rollout (`"random"` / `"l0"`, hoặc các chính sách đăng ký qua
    `register_rollout`) hoặc hàm `Callable[[GameState, random.Random], float]`.
  - `rollout_depth`: giới hạn số nước mô phỏng tối đa trong mỗi rollout (`None` = tới cuối ván).
  - `use_candidates`: thu hẹp nước mở rộng bằng `state.candidate_moves(candidate_radius)`.
  - `candidate_radius`: bán kính lân cận Chebyshev khi `use_candidates=True` (mặc định `2`).
  - `reuse_tree`: tái sử dụng nhánh cây con khớp `state.hash()` ở các nước đi tiếp theo.
  - Ngân sách `Budget(iterations=..., time_s=...)`: hỗ trợ giới hạn theo số vòng lặp, theo
    thời gian thực (`time_s`), hoặc kết hợp cả hai (dừng khi chạm ngưỡng nào đến trước).
"""

from __future__ import annotations

import importlib
import inspect
import math
import random
import time
from collections.abc import Callable
from math import inf, log, sqrt
from typing import Any

from src.agents.base import Agent, Budget, RootStat
from src.agents.minimax import terminal_value
from src.games.base import GameState, Move

DEFAULT_C = math.sqrt(2.0)
DEFAULT_ITERATIONS = 1000

FINAL_SELECTIONS: tuple[str, ...] = ("robust", "max", "max_robust")

# Hàm rollout nhận (state, rng) và trả về giá trị trong [-1.0, +1.0]
# theo góc nhìn của state.current_player() tại thời điểm bắt đầu rollout.
RolloutFn = Callable[[GameState, random.Random], float]


def default_rollout(
    state: GameState,
    rng: random.Random,
    max_moves: int | None = None,
) -> float:
    """Rollout L0: hai bên chọn nước ngẫu nhiên đồng nhất tới khi ván kết thúc.

    Trả về +1.0 nếu người sắp đi ở `state` ban đầu thắng, -1.0 nếu thua, 0.0 nếu hòa
    (hoặc khi chạm ngưỡng `max_moves` mà chưa bên nào thắng).
    Khi hàm kết thúc, `state` được khôi phục nguyên vẹn như lúc truyền vào.
    """
    player_at_start = state.current_player()
    start_count = state.move_count()
    plies = 0
    try:
        while not state.is_terminal():
            if max_moves is not None and plies >= max_moves:
                return 0.0
            moves = state.legal_moves()
            state.play(rng.choice(moves))
            plies += 1
        winner = state.winner()
        assert winner is not None
        return float(winner * player_at_start)
    finally:
        while state.move_count() > start_count:
            state.undo()


ROLLOUT_POLICIES: dict[str, RolloutFn] = {
    "random": default_rollout,
    "l0": default_rollout,
}


def register_rollout(name: str, fn: RolloutFn) -> None:
    """Đăng ký một chính sách rollout theo tên (không phân biệt hoa thường) để gọi từ YAML."""
    key = name.strip().lower()
    if not key:
        raise ValueError("Tên chính sách rollout không được rỗng")
    if not callable(fn):
        raise ValueError(f"Hàm rollout cho {name!r} phải là callable")
    ROLLOUT_POLICIES[key] = fn


def resolve_rollout(rollout: str | RolloutFn) -> tuple[str, RolloutFn]:
    """Chuẩn hóa tham số `rollout` (tên chuỗi hoặc hàm) thành `(rollout_name, rollout_fn)`."""
    if callable(rollout):
        name = getattr(rollout, "__name__", "custom")
        if rollout is default_rollout:
            name = "random"
        return name, rollout
    if isinstance(rollout, str):
        key = rollout.strip().lower()
        if key not in ROLLOUT_POLICIES:
            # Thử nạp module src.agents.rollouts (P2.3) nếu đã được cài đặt
            try:
                importlib.import_module("src.agents.rollouts")
            except ModuleNotFoundError:
                pass
        if key in ROLLOUT_POLICIES:
            return key, ROLLOUT_POLICIES[key]
        allowed = sorted(ROLLOUT_POLICIES)
        raise ValueError(f"rollout phải thuộc {allowed} hoặc là một hàm, nhận được {rollout!r}")
    raise ValueError(f"rollout phải là tên chuỗi hoặc hàm callable, nhận được {rollout!r}")


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
        "state_hash",
    )

    def __init__(
        self,
        move: Move | None,
        parent: MCTSNode | None,
        player_just_moved: int,
        untried_moves: list[Move],
        depth: int = 0,
        state_hash: int | None = None,
    ) -> None:
        self.move = move
        self.parent = parent
        self.player_just_moved = player_just_moved
        self.children: list[MCTSNode] = []
        self.untried_moves: list[Move] = list(untried_moves)
        self.N: int = 0
        self.W: float = 0.0
        self.depth: int = depth
        self.state_hash: int | None = state_hash

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


def select_final_child(
    root: MCTSNode,
    selection: str = "robust",
    c: float = DEFAULT_C,
) -> MCTSNode:
    """Chọn nút con tốt nhất tại `root` theo quy tắc `selection` ('robust', 'max', 'max_robust')."""
    if selection not in FINAL_SELECTIONS:
        raise ValueError(f"selection phải thuộc {FINAL_SELECTIONS}, nhận được {selection!r}")

    visited = [child for child in root.children if child.N > 0]
    pool = visited if visited else root.children
    if not pool:
        raise ValueError("Nút gốc không có nước đi con nào")

    if selection == "robust":
        return max(pool, key=lambda ch: (ch.N, ch.q))
    if selection == "max":
        return max(pool, key=lambda ch: (ch.q, ch.N))

    # max_robust: nếu nút có N lớn nhất cũng đạt q lớn nhất thì chọn ngay;
    # ngược lại ưu tiên nút có cận dưới tin cậy (Lower Confidence Bound) cao nhất.
    by_n = max(pool, key=lambda ch: (ch.N, ch.q))
    by_q = max(pool, key=lambda ch: (ch.q, ch.N))
    if by_n is by_q:
        return by_n
    lcb_factor = c if c > 0 else 1.0
    return max(pool, key=lambda ch: (ch.q - lcb_factor / sqrt(ch.N), ch.N))


def _rebase_depth(node: MCTSNode, depth: int = 0) -> tuple[int, int]:
    """Cập nhật lại `depth` của toàn bộ cây con gốc `node`; trả về `(node_count, max_depth)`."""
    node.depth = depth
    total_nodes = 1
    max_d = depth
    stack = list(node.children)
    while stack:
        curr = stack.pop()
        assert curr.parent is not None
        curr.depth = curr.parent.depth + 1
        total_nodes += 1
        if curr.depth > max_d:
            max_d = curr.depth
        stack.extend(curr.children)
    return total_nodes, max_d


class MCTSAgent(Agent):
    """Agent Tìm kiếm Cây Monte Carlo với công thức UCT (MCTS-UCT).

    Tham số (R2.6):
      - `c`: hằng số khám phá trong công thức UCT (mặc định `sqrt(2)`).
      - `selection`: quy tắc chọn nước đi cuối tại gốc:
          * `"robust"`: chọn nút con có số lượt thăm `N` lớn nhất (mặc định).
          * `"max"`: chọn nút con có tỷ lệ thắng `q = W / N` lớn nhất.
          * `"max_robust"`: chọn nút con đạt cả `max N` và `max q`; nếu không có nút nào
            đồng thời đạt cả hai thì chọn nút có điểm biên dưới an toàn `q - c / sqrt(N)`
            cao nhất trong các nút đã thăm.
      - `rollout`: tên chính sách (`"random"` / `"l0"`) hoặc hàm `rollout(state, rng) -> float`.
      - `rollout_fn`: bí danh tương thích ngược của `rollout` khi truyền trực tiếp hàm.
      - `rollout_depth`: giới hạn số nước tối đa của mỗi lần rollout (`None` = tới hết ván).
      - `use_candidates`: nếu True, thu hẹp các nước mở rộng bằng `state.candidate_moves(radius)`
        thay vì `state.legal_moves()` (hữu ích cho bàn Caro lớn).
      - `candidate_radius`: bán kính dùng khi `use_candidates=True` (mặc định 2).
      - `reuse_tree`: nếu True, giữ lại nhánh cây con khớp với thế cờ ở lượt đi kế tiếp.
      - `seed`: hạt giống ngẫu nhiên để tái lập kết quả.
    """

    def __init__(
        self,
        c: float = DEFAULT_C,
        selection: str = "robust",
        rollout_fn: RolloutFn | None = None,
        use_candidates: bool = False,
        candidate_radius: int = 2,
        seed: int | None = None,
        *,
        rollout: str | RolloutFn = "random",
        rollout_depth: int | None = None,
        reuse_tree: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            raise ValueError(f"Hằng số khám phá c phải là số thực >= 0, nhận được {c!r}")
        c_val = float(c)
        if math.isnan(c_val) or math.isinf(c_val) or c_val < 0.0:
            raise ValueError(f"Hằng số khám phá c phải hữu hạn và >= 0, nhận được {c!r}")

        if selection not in FINAL_SELECTIONS:
            raise ValueError(f"selection phải thuộc {FINAL_SELECTIONS}, nhận được {selection!r}")

        if not isinstance(use_candidates, bool):
            raise ValueError(f"use_candidates phải là bool, nhận được {use_candidates!r}")

        if isinstance(candidate_radius, bool) or not isinstance(candidate_radius, int):
            raise ValueError(
                f"candidate_radius phải là số nguyên >= 1, nhận được {candidate_radius!r}"
            )
        if candidate_radius < 1:
            raise ValueError(f"candidate_radius phải >= 1, nhận được {candidate_radius}")

        if rollout_depth is not None:
            if (
                isinstance(rollout_depth, bool)
                or not isinstance(rollout_depth, int)
                or rollout_depth < 1
            ):
                raise ValueError(
                    f"rollout_depth phải là số nguyên >= 1, nhận được {rollout_depth!r}"
                )

        if not isinstance(reuse_tree, bool):
            raise ValueError(f"reuse_tree phải là bool, nhận được {reuse_tree!r}")

        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError(f"seed phải là số nguyên hoặc None, nhận được {seed!r}")

        chosen_rollout: str | RolloutFn = rollout_fn if rollout_fn is not None else rollout
        rollout_name, resolved_fn = resolve_rollout(chosen_rollout)
        params = inspect.signature(resolved_fn).parameters
        accepts_max_moves = "max_moves" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        self.c = c_val
        self.selection = selection
        self.rollout = rollout_name
        self.rollout_fn = resolved_fn
        self._rollout_accepts_max_moves = accepts_max_moves
        self.rollout_depth = rollout_depth
        self.use_candidates = use_candidates
        self.candidate_radius = candidate_radius
        self.reuse_tree = reuse_tree
        self.seed = seed
        self.name = f"mcts-c{self.c:.2f}-{selection}"
        self._rng = random.Random(seed)
        self.last_root: MCTSNode | None = None

    def get_params(self) -> dict[str, Any]:
        """Trả về từ điển đầy đủ các tham số cấu hình (R2.6) của agent."""
        return {
            "c": self.c,
            "selection": self.selection,
            "rollout": self.rollout,
            "rollout_depth": self.rollout_depth,
            "use_candidates": self.use_candidates,
            "candidate_radius": self.candidate_radius,
            "reuse_tree": self.reuse_tree,
            "seed": self.seed,
        }

    def reset_tree(self) -> None:
        """Xóa cây tìm kiếm đã lưu (dùng khi bắt đầu ván mới nếu bật `reuse_tree=True`)."""
        self.last_root = None

    def _moves_for(self, state: GameState) -> list[Move]:
        if self.use_candidates:
            return state.candidate_moves(self.candidate_radius)
        return state.legal_moves()

    def _find_reusable_root(self, state: GameState) -> MCTSNode | None:
        """Tìm nút trong `self.last_root` (tới độ sâu 2) khớp với `state.hash()` hiện tại."""
        if not self.reuse_tree or self.last_root is None:
            return None
        target_hash = state.hash()
        target_just_moved = -state.current_player()

        candidates: list[MCTSNode] = [self.last_root]
        for ch in self.last_root.children:
            candidates.append(ch)
            candidates.extend(ch.children)

        legal_set = set(self._moves_for(state))
        for cand in candidates:
            if cand.state_hash == target_hash and cand.player_just_moved == target_just_moved:
                existing_moves = {ch.move for ch in cand.children if ch.move is not None} | set(
                    cand.untried_moves
                )
                if existing_moves <= legal_set:
                    cand.parent = None
                    cand.move = None
                    missing = [mv for mv in self._moves_for(state) if mv not in existing_moves]
                    if missing:
                        cand.untried_moves.extend(missing)
                    return cand
        return None

    def _run_rollout(self, state: GameState) -> float:
        if self.rollout_depth is not None and self._rollout_accepts_max_moves:
            return float(
                self.rollout_fn(  # type: ignore[call-arg]
                    state, self._rng, max_moves=self.rollout_depth
                )
            )
        return float(self.rollout_fn(state, self._rng))

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
        return select_final_child(root, selection=self.selection, c=self.c)

    @staticmethod
    def _validate_budget(budget: Budget) -> tuple[int | None, float | None]:
        max_iterations = budget.iterations
        max_time_s = budget.time_s

        if max_iterations is not None:
            if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
                raise ValueError(
                    f"budget.iterations phải là số nguyên >= 1, nhận được {max_iterations!r}"
                )
            if max_iterations < 1:
                raise ValueError(f"budget.iterations phải >= 1, nhận được {max_iterations}")

        if max_time_s is not None:
            if isinstance(max_time_s, bool) or not isinstance(max_time_s, (int, float)):
                raise ValueError(f"budget.time_s phải là số thực > 0, nhận được {max_time_s!r}")
            t_val = float(max_time_s)
            if math.isnan(t_val) or math.isinf(t_val) or t_val <= 0.0:
                raise ValueError(f"budget.time_s phải hữu hạn và > 0, nhận được {max_time_s!r}")
            max_time_s = t_val

        if max_iterations is None and max_time_s is None:
            max_iterations = DEFAULT_ITERATIONS

        return max_iterations, max_time_s

    def select_move(self, state: GameState, budget: Budget) -> Move:
        if state.is_terminal():
            raise ValueError("Ván đã kết thúc, không còn nước để chọn")
        max_iterations, max_time_s = self._validate_budget(budget)

        start = time.perf_counter()
        deadline = (start + max_time_s) if max_time_s is not None else None
        root_move_count = state.move_count()

        reused = self._find_reusable_root(state)
        if reused is not None:
            root = reused
            nodes_created, max_depth = _rebase_depth(root, 0)
        else:
            root = MCTSNode(
                move=None,
                parent=None,
                player_just_moved=-state.current_player(),
                untried_moves=self._moves_for(state),
                depth=0,
                state_hash=state.hash(),
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
                        state_hash=state.hash(),
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
                    rollout_val = self._run_rollout(state)

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
