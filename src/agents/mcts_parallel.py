"""Agent MCTS song song hóa theo gốc (Root Parallelization — P4.3a / Bonus B4).

Chủ sở hữu: TV3.  Review: TV1.

Ý tưởng thuật toán (Chaslot et al., 2008 — Parallel Monte-Carlo Tree Search):
  - Khởi chạy $K = \\text{num\\_workers}$ cây MCTS độc lập từ cùng trạng thái gốc $s_0$
    với các hạt giống ngẫu nhiên phân tách tất định bằng SHA-256.
  - Khi hết ngân sách (`budget.iterations` hoặc `budget.time_s`), thu thập các nút con
    tại gốc của cả $K$ cây và cộng dồn thống kê:
    $$N(a) = \\sum_{k=1}^K N_k(a), \\quad W(a) = \\sum_{k=1}^K W_k(a)$$
  - Chọn nước đi cuối cùng trên nút gốc đã hợp nhất theo quy tắc `selection` (`robust`,
    `max`, hoặc `max_robust`).
"""

from __future__ import annotations

import atexit
import hashlib
import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

from src.agents.base import Agent, Budget, RootStat
from src.agents.mcts import (
    DEFAULT_C,
    MCTSAgent,
    MCTSNode,
    RolloutFn,
    resolve_rollout,
    select_final_child,
)
from src.games.base import GameState, Move

_ROOT_PROCESS_POOLS: dict[int, ProcessPoolExecutor] = {}


def _get_root_process_pool(num_workers: int) -> ProcessPoolExecutor:
    """Lấy hoặc khởi tạo `ProcessPoolExecutor` tái sử dụng cho `num_workers` tiến trình."""
    pool = _ROOT_PROCESS_POOLS.get(num_workers)
    if pool is None:
        pool = ProcessPoolExecutor(max_workers=num_workers)
        _ROOT_PROCESS_POOLS[num_workers] = pool
    return pool


def shutdown_root_process_pools() -> None:
    """Đóng toàn bộ các `ProcessPoolExecutor` đã lưu trong bộ nhớ đệm toàn cục."""
    for pool in list(_ROOT_PROCESS_POOLS.values()):
        try:
            pool.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass
    _ROOT_PROCESS_POOLS.clear()


atexit.register(shutdown_root_process_pools)


def _derive_worker_seed(base_seed: int, call_idx: int, state_hash: int, worker_id: int) -> int:
    """Sinh hạt giống 63-bit độc lập và tất định cho từng worker tại mỗi lượt tìm kiếm."""
    payload = f"rootpar:{base_seed}:{call_idx}:{state_hash}:{worker_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _run_root_worker_task(
    args: tuple[GameState, dict[str, Any], Budget, int],
) -> dict[str, Any]:
    """Hàm thực thi tìm kiếm MCTS của 1 worker (ở cấp module để `ProcessPoolExecutor` pickle)."""
    state, agent_kwargs, worker_budget, worker_seed = args
    agent = MCTSAgent(seed=worker_seed, **agent_kwargs)
    agent.select_move(state, worker_budget)
    root = agent.last_root
    children_stats: list[tuple[Move, int, float, float, float]] = []
    if root is not None:
        for ch in root.children:
            if ch.move is not None and ch.N > 0:
                children_stats.append((ch.move, ch.N, ch.W, ch.W2, ch.prior))
    return {
        "children": children_stats,
        "nodes": int(agent.last_stats.get("nodes", 0)),
        "depth": int(agent.last_stats.get("depth", 0)),
        "iterations": int(root.N) if root is not None else 0,
    }


def _run_rollout_batch_worker(
    args: tuple[GameState, str, int, int | None, int],
) -> int:
    """Chạy `count` lần rollout độc lập trong tiến trình worker để đo thông lượng song song thực."""
    state, rollout_name, count, depth, seed = args
    _, fn = resolve_rollout(rollout_name)
    rng = random.Random(seed)
    if depth is not None:
        for _ in range(count):
            fn(state, rng, max_moves=depth)  # type: ignore[call-arg]
    else:
        for _ in range(count):
            fn(state, rng)
    return count


def measure_parallel_rollouts_per_second(
    state: GameState,
    rollout: str = "l2_b4",
    *,
    num_workers: int = 4,
    num_rollouts_per_worker: int = 250,
    split_total: bool = False,
    seed: int = 2026,
) -> float:
    """Đo thông lượng `rollout_sims_per_s` thực tế khi chạy song song trên `num_workers` worker."""
    if num_workers < 1:
        raise ValueError(f"num_workers phải >= 1, nhận được {num_workers}")
    depth = 8 if rollout in ("l4", "l4_b4") else None
    if num_workers == 1:
        t0 = time.perf_counter()
        _run_rollout_batch_worker((state.clone(), rollout, num_rollouts_per_worker, depth, seed))
        dt = time.perf_counter() - t0
        return (num_rollouts_per_worker / dt) if dt > 0.0 else 0.0

    pool = _get_root_process_pool(num_workers)
    # Warm-up pool
    warm_tasks = [(state.clone(), rollout, 5, depth, seed + w) for w in range(num_workers)]
    list(pool.map(_run_rollout_batch_worker, warm_tasks))

    tasks: list[tuple[GameState, str, int, int | None, int]] = []
    for w_id in range(num_workers):
        if split_total:
            cnt = max(1, (num_rollouts_per_worker + w_id) // num_workers)
        else:
            cnt = num_rollouts_per_worker
        tasks.append((state.clone(), rollout, cnt, depth, seed + 1000 + w_id))

    t0 = time.perf_counter()
    done_counts = list(pool.map(_run_rollout_batch_worker, tasks))
    dt = time.perf_counter() - t0
    total_done = sum(done_counts)
    return (total_done / dt) if dt > 0.0 else 0.0


class RootParallelMCTSAgent(Agent):
    """Agent MCTS song song hóa theo gốc (Root Parallelization — P4.3a / Bonus B4).

    Tham số:
      - `num_workers`: số cây MCTS độc lập chạy song song tại gốc (mặc định `4`).
      - `c`: hằng số khám phá UCT (mặc định `DEFAULT_C`).
      - `selection`: quy tắc chọn nước đi cuối tại gốc (`"robust"`, `"max"`, `"max_robust"`).
      - `uct_variant`: biến thể công thức UCT (`"standard"`, `"no_sqrt"`, `"no_log"`,
        `"sqrt_n"`, `"ucb1_tuned"`, `"progressive_bias"`).
      - `rollout`: tên chính sách rollout (`"l2_b4"`, `"l4_b4"`, ...) hoặc hàm callable.
      - `rollout_depth`: độ sâu cắt sớm truyền vào rollout (`None` = mặc định của chính sách).
      - `use_candidates`: thu hẹp nước mở rộng tại mỗi nút bằng `candidate_moves(radius)`.
      - `candidate_radius`: bán kính lân cận khi `use_candidates=True` (mặc định `2`).
      - `split_iterations`: khi `budget.iterations` được truyền vào:
          * Nếu `False` (mặc định): mỗi worker chạy đủ `budget.iterations` vòng lặp song song
            (tăng tổng mẫu mô phỏng lên `num_workers` lần trong xấp xỉ cùng thời gian thực).
          * Nếu `True`: chia tổng `budget.iterations` cho `num_workers` worker (giữ tổng số
            vòng lặp không đổi nhưng giảm thời gian thực trên mỗi nước đi).
      - `executor`: `"process"` (mặc định, chạy đa tiến trình vượt qua GIL; tự động chuyển
        sang `"thread"` nếu đang ở trong tiến trình con daemon hoặc khi `rollout` là hàm
        cục bộ không pickle được) hoặc `"thread"`.
      - `seed`: hạt giống ngẫu nhiên gốc.
    """

    def __init__(
        self,
        num_workers: int = 4,
        c: float = DEFAULT_C,
        selection: str = "robust",
        *,
        uct_variant: str = "standard",
        rollout: str | RolloutFn = "l2_b4",
        rollout_depth: int | None = None,
        use_candidates: bool = True,
        candidate_radius: int = 2,
        split_iterations: bool = False,
        executor: str = "process",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError(f"num_workers phải là số nguyên >= 1, nhận được {num_workers!r}")
        if not isinstance(split_iterations, bool):
            raise ValueError(f"split_iterations phải là kiểu bool, nhận được {split_iterations!r}")
        if executor not in ("process", "thread"):
            raise ValueError(f"executor phải là 'process' hoặc 'thread', nhận được {executor!r}")

        # Kiểm tra hợp lệ các tham số MCTS chuẩn thông qua một MCTSAgent mẫu
        probe = MCTSAgent(
            c=c,
            selection=selection,
            uct_variant=uct_variant,
            rollout=rollout,
            rollout_depth=rollout_depth,
            use_candidates=use_candidates,
            candidate_radius=candidate_radius,
            reuse_tree=False,
            seed=seed,
        )
        self.num_workers = num_workers
        self.c = probe.c
        self.selection = probe.selection
        self.uct_variant = probe.uct_variant
        self.rollout = probe.rollout
        self.rollout_fn = probe.rollout_fn
        self._raw_rollout: str | RolloutFn = rollout
        self.rollout_depth = probe.rollout_depth
        self.use_candidates = probe.use_candidates
        self.candidate_radius = probe.candidate_radius
        self.split_iterations = split_iterations
        self.executor = executor
        self.seed = seed
        self._call_count = 0
        self.last_root: MCTSNode | None = None
        self.name = f"mcts-rootpar{num_workers}-{self.rollout}"

    def get_params(self) -> dict[str, Any]:
        """Trả về từ điển tham số cấu hình của agent song song hóa theo gốc."""
        return {
            "num_workers": self.num_workers,
            "c": self.c,
            "selection": self.selection,
            "uct_variant": self.uct_variant,
            "rollout": self.rollout,
            "rollout_depth": self.rollout_depth,
            "use_candidates": self.use_candidates,
            "candidate_radius": self.candidate_radius,
            "split_iterations": self.split_iterations,
            "executor": self.executor,
            "seed": self.seed,
        }

    def close(self) -> None:
        """Giải phóng các tiến trình worker trong pool toàn cục (nếu có)."""
        shutdown_root_process_pools()

    def select_move(self, state: GameState, budget: Budget) -> Move:
        """Chạy `num_workers` cây MCTS song song tại gốc, gộp `(N, W)` và chọn nước đi tốt nhất."""
        if state.is_terminal():
            raise ValueError("Ván đã kết thúc, không còn nước để chọn")
        max_iterations, max_time_s = MCTSAgent._validate_budget(budget)
        self._call_count += 1
        base_seed = 0 if self.seed is None else self.seed

        rollout_arg: str | RolloutFn = (
            self._raw_rollout if callable(self._raw_rollout) else self.rollout
        )
        agent_kwargs: dict[str, Any] = {
            "c": self.c,
            "selection": self.selection,
            "uct_variant": self.uct_variant,
            "rollout": rollout_arg,
            "rollout_depth": self.rollout_depth,
            "use_candidates": self.use_candidates,
            "candidate_radius": self.candidate_radius,
            "reuse_tree": False,
        }

        if self.num_workers == 1:
            single_seed = _derive_worker_seed(base_seed, self._call_count, state.hash(), 0)
            single_agent = MCTSAgent(seed=single_seed, **agent_kwargs)
            move = single_agent.select_move(state, budget)
            self.last_root = single_agent.last_root
            self.last_stats = dict(single_agent.last_stats)
            self.last_stats["workers"] = 1
            self.last_stats["iterations"] = (
                int(self.last_root.N) if self.last_root is not None else 0
            )
            return move

        start_t = time.perf_counter()
        tasks: list[tuple[GameState, dict[str, Any], Budget, int]] = []
        for w_id in range(self.num_workers):
            if max_iterations is not None and self.split_iterations:
                w_iters = max(1, (max_iterations + w_id) // self.num_workers)
            else:
                w_iters = max_iterations
            w_budget = Budget(iterations=w_iters, time_s=max_time_s)
            w_seed = _derive_worker_seed(base_seed, self._call_count, state.hash(), w_id)
            tasks.append((state.clone(), agent_kwargs, w_budget, w_seed))

        use_process_pool = (
            self.executor == "process"
            and not mp.current_process().daemon
            and not callable(self._raw_rollout)
        )
        if use_process_pool:
            pool = _get_root_process_pool(self.num_workers)
            results = list(pool.map(_run_root_worker_task, tasks))
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as tpool:
                results = list(tpool.map(_run_root_worker_task, tasks))

        agg_n: dict[Move, int] = {}
        agg_w: dict[Move, float] = {}
        agg_w2: dict[Move, float] = {}
        agg_prior: dict[Move, float] = {}
        total_new_nodes = 0
        total_iters = 0
        max_depth = 0

        for res in results:
            total_new_nodes += max(0, int(res["nodes"]) - 1)
            total_iters += int(res["iterations"])
            if int(res["depth"]) > max_depth:
                max_depth = int(res["depth"])
            for mv, n_val, w_val, w2_val, prior_val in res["children"]:
                agg_n[mv] = agg_n.get(mv, 0) + n_val
                agg_w[mv] = agg_w.get(mv, 0.0) + w_val
                agg_w2[mv] = agg_w2.get(mv, 0.0) + w2_val
                agg_prior[mv] = prior_val

        player = state.current_player()
        merged_root = MCTSNode(
            move=None,
            parent=None,
            player_just_moved=-player,
            untried_moves=[],
            depth=0,
            state_hash=state.hash(),
        )
        merged_root.N = total_iters

        for mv, n_val in agg_n.items():
            ch = MCTSNode(
                move=mv,
                parent=merged_root,
                player_just_moved=player,
                untried_moves=[],
                depth=1,
                prior=agg_prior.get(mv, 0.0),
            )
            ch.N = n_val
            ch.W = agg_w[mv]
            ch.W2 = agg_w2[mv]
            merged_root.children.append(ch)

        if not merged_root.children:
            fallback_moves = (
                state.candidate_moves(self.candidate_radius)
                if self.use_candidates
                else state.legal_moves()
            )
            if not fallback_moves:
                fallback_moves = state.legal_moves()
            ch = MCTSNode(
                move=fallback_moves[0],
                parent=merged_root,
                player_just_moved=player,
                untried_moves=[],
                depth=1,
            )
            ch.N = 1
            merged_root.children.append(ch)

        best_child = select_final_child(merged_root, selection=self.selection, c=self.c)
        assert best_child.move is not None
        elapsed = time.perf_counter() - start_t

        root_stats: list[RootStat] = [
            {
                "move": ch.move,
                "N": ch.N,
                "W": ch.W,
                "q": ch.q,
                "uct": ch.uct_score(self.c, variant=self.uct_variant),
            }
            for ch in sorted(merged_root.children, key=lambda c_node: (-c_node.N, -c_node.q))
            if ch.move is not None
        ]

        self.last_root = merged_root
        self.last_stats = {
            "nodes": 1 + total_new_nodes,
            "iterations": total_iters,
            "time_s": elapsed,
            "depth": max_depth,
            "sims_per_s": (total_iters / elapsed) if elapsed > 0.0 else 0.0,
            "workers": self.num_workers,
            "root": root_stats,
        }
        return best_child.move
