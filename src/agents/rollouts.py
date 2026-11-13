"""Các chính sách mô phỏng (Rollout Policies L0–L2) và tiện ích MCTS trên Caro (P2.3a).

Chủ sở hữu: TV3.  Review: TV1.

Các mức chính sách rollout cho MCTS-UCT (R2.7, R5.1):
  - **L0 (`rollout_l0`)**: Chọn nước ngẫu nhiên đồng nhất trên toàn bộ các nước hợp lệ
    (`legal_moves`) cho tới khi ván cờ kết thúc (hoặc chạm ngưỡng `max_moves`).
  - **L1 (`rollout_l1`)**: Ưu tiên chiến thuật tức thì 1 bước:
      1. Nếu người sắp đi có nước thắng ngay ở lượt này, đánh ngay nước đó (kết thúc rollout).
      2. Nếu không, nhưng đối thủ có nước dọa thắng ngay ở lượt kế tiếp, bắt buộc đánh vào
         ô chặn (chọn ngẫu nhiên nếu có nhiều ô chặn).
      3. Nếu không có thắng/chặn ngay, chọn ngẫu nhiên đồng nhất từ toàn bộ `legal_moves`.
  - **L2 (`rollout_l2`)**: Giống L1 về ưu tiên thắng ngay / chặn thua ngay, nhưng thu hẹp
    tập nước đi ngẫu nhiên vào vùng ứng viên quanh các quân đã đánh (`candidate_moves(radius)`,
    mặc định `radius=2`).

Tối ưu tốc độ mô phỏng (`sims_per_s`) trên bàn Caro (`Gomoku`):
  - Biểu diễn bàn cờ bằng mảng phẳng 1 chiều có viền bảo vệ (`BORDER = 2`) giúp loại bỏ hoàn
    toàn kiểm tra biên 2 chiều và chi phí truy cập phần tử `numpy.ndarray` trong vòng lặp.
  - Không làm biến đổi trạng thái `Gomoku` gốc trong quá trình mô phỏng (tránh chi phí cập
    nhật/hoàn tác Zobrist hash và mặt nạ lân cận qua hàng chục nước đi thử).
  - Theo dõi tăng dần (incremental) các đầu mút dọa thắng dọc theo 4 hướng của nước vừa đánh:
    mỗi nước đi mới chỉ kiểm tra tối đa 8 đầu mút thay vì quét lại toàn bộ bàn cờ.
  - Chỉ sử dụng các thuộc tính và phương thức công khai của `Gomoku` (`board`, `size`,
    `win_length`, `block_two_ends`, `candidate_moves`, `legal_moves`) và `MCTSAgent`
    (`rollout_fn`, `select_move`, `last_stats`), không phụ thuộc thành phần nội bộ (`_...`).
  - Lưu ý thực nghiệm: khi chạy nhiều tiến trình (`workers`) song song, tranh chấp CPU có thể
    làm thời gian mỗi nước tăng lên; khi chạy thực nghiệm ngân sách thời gian cố định ở P3.3
    cần giữ cấu hình `workers` nhất quán giữa các phương pháp.
"""

from __future__ import annotations

import inspect
import random
import time
from functools import cache

from src.agents.base import Budget
from src.agents.mcts import MCTSAgent, RolloutFn, default_rollout, register_rollout
from src.games.base import PLAYER_O, PLAYER_X, GameState, Move
from src.games.gomoku import Gomoku

_BORDER = 2


@cache
def _gomoku_tables(
    n: int,
    radius: int,
) -> tuple[
    int,
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[int, int, int, int],
    int,
]:
    """Bảng tra cứu 1D có viền (`stride = n + 2`) được lưu sẵn cho bàn Caro `n × n`.

    Trả về `(stride, base_grid, cell_indices, nbrs, dirs, center_idx)`:
      - `stride`: bước nhảy hàng `n + 2`.
      - `base_grid`: mảng 1D kích thước `(n + 2)^2` với viền `_BORDER = 2` và lòng bàn `0`.
      - `cell_indices`: danh sách `n^2` chỉ số 1D của các ô trong bàn cờ.
      - `nbrs`: tại mỗi chỉ số 1D `idx`, tuple các chỉ số 1D lân cận trong khoảng cách
        Chebyshev `<= radius`.
      - `dirs`: 4 bước nhảy hướng `(1, stride, stride + 1, stride - 1)`.
      - `center_idx`: chỉ số 1D của ô tâm bàn `(n // 2, n // 2)`.
    """
    stride = n + 2
    total = stride * stride
    base_grid = [_BORDER] * total
    cell_indices: list[int] = []
    for r in range(n):
        row_off = (r + 1) * stride + 1
        for c in range(n):
            idx = row_off + c
            base_grid[idx] = 0
            cell_indices.append(idx)

    nbrs: list[tuple[int, ...]] = [() for _ in range(total)]
    for r in range(n):
        r0 = max(0, r - radius)
        r1 = min(n, r + radius + 1)
        for c in range(n):
            c0 = max(0, c - radius)
            c1 = min(n, c + radius + 1)
            idx = (r + 1) * stride + (c + 1)
            nbrs[idx] = tuple(
                (nr + 1) * stride + (nc + 1)
                for nr in range(r0, r1)
                for nc in range(c0, c1)
                if nr != r or nc != c
            )

    dirs = (1, stride, stride + 1, stride - 1)
    center_idx = (n // 2 + 1) * stride + (n // 2 + 1)
    return stride, tuple(base_grid), tuple(cell_indices), tuple(nbrs), dirs, center_idx


def _load_gomoku_grid(
    state: Gomoku,
    stride: int,
    base_grid: tuple[int, ...],
    cell_indices: tuple[int, ...],
) -> tuple[list[int], list[int], list[int]]:
    """Nạp trạng thái `Gomoku` vào mảng phẳng 1D `grid` và tách danh sách quân X, O."""
    grid = list(base_grid)
    stones_x: list[int] = []
    stones_o: list[int] = []

    flat = state.board.ravel().tolist()
    for k, val in enumerate(flat):
        if val:
            idx = cell_indices[k]
            grid[idx] = val
            if val == PLAYER_X:
                stones_x.append(idx)
            elif val == PLAYER_O:
                stones_o.append(idx)

    return grid, stones_x, stones_o


def _check_win_1d(
    grid: list[int],
    idx0: int,
    player: int,
    win_len: int,
    block_two: bool,
    dirs: tuple[int, int, int, int],
) -> bool:
    """Kiểm tra đặt quân `player` vào ô trống `idx0` trên `grid` có thắng ngay không."""
    opp = -player
    for d in dirs:
        bp = idx0 - d
        b = 0
        while grid[bp] == player:
            b += 1
            bp -= d
        fp = idx0 + d
        f = 0
        while grid[fp] == player:
            f += 1
            fp += d
        if 1 + b + f >= win_len:
            if not block_two or not (grid[bp] == opp and grid[fp] == opp):
                return True
    return False


def _find_initial_wins_1d(
    grid: list[int],
    stones: list[int],
    player: int,
    win_len: int,
    block_two: bool,
    dirs: tuple[int, int, int, int],
) -> tuple[set[int], list[int]]:
    """Tìm mọi ô trống trên `grid` giúp `player` thắng ngay bằng cách quét đầu mút chuỗi quân."""
    wins_set: set[int] = set()
    wins_list: list[int] = []
    if len(stones) < win_len - 1:
        return wins_set, wins_list

    opp = -player
    need_pair = win_len >= 5
    for s in stones:
        for d in dirs:
            if grid[s - d] == player:
                continue
            if need_pair and grid[s + d] != player:
                continue
            fp = s + d
            seg = 1
            while grid[fp] == player:
                seg += 1
                fp += d
            bp = s - d

            if grid[bp] == 0:
                ebp = bp - d
                ext_b = 0
                while grid[ebp] == player:
                    ext_b += 1
                    ebp -= d
                if seg + 1 + ext_b >= win_len:
                    if not block_two or not (grid[ebp] == opp and grid[fp] == opp):
                        if bp not in wins_set:
                            wins_set.add(bp)
                            wins_list.append(bp)

            if grid[fp] == 0:
                efp = fp + d
                ext_f = 0
                while grid[efp] == player:
                    ext_f += 1
                    efp += d
                if seg + 1 + ext_f >= win_len:
                    if not block_two or not (grid[bp] == opp and grid[efp] == opp):
                        if fp not in wins_set:
                            wins_set.add(fp)
                            wins_list.append(fp)

    return wins_set, wins_list


def _rollout_l0_gomoku(
    state: Gomoku,
    rng: random.Random,
    max_moves: int | None = None,
) -> float:
    """Rollout L0 tối ưu trên mảng 1D cho bàn Caro `Gomoku`."""
    if state.is_terminal():
        winner = state.winner()
        assert winner is not None
        return float(winner * state.current_player())

    n = state.size
    win_len = state.win_length
    block_two = state.block_two_ends
    stride, base_grid, cell_indices, _, dirs, _ = _gomoku_tables(n, 2)
    grid, _, _ = _load_gomoku_grid(state, stride, base_grid, cell_indices)

    pool = [idx for idx in cell_indices if grid[idx] == 0]
    remaining = len(pool)
    limit = remaining if max_moves is None else min(remaining, max_moves)

    player_at_start = state.current_player()
    player = player_at_start

    for _ in range(limit):
        pick = rng.randrange(remaining)
        idx0 = pool[pick]
        remaining -= 1
        pool[pick] = pool[remaining]
        grid[idx0] = player
        opp = -player

        for d in dirs:
            bp = idx0 - d
            b = 0
            while grid[bp] == player:
                b += 1
                bp -= d
            fp = idx0 + d
            f = 0
            while grid[fp] == player:
                f += 1
                fp += d
            if 1 + b + f >= win_len:
                if not block_two or not (grid[bp] == opp and grid[fp] == opp):
                    return float(player * player_at_start)

        player = opp

    return 0.0


def _rollout_tactical_gomoku(
    state: Gomoku,
    rng: random.Random,
    *,
    use_candidates: bool,
    radius: int = 2,
    max_moves: int | None = None,
) -> float:
    """Rollout chiến thuật L1 (`use_candidates=False`) và L2 (`use_candidates=True`) trên Caro."""
    if state.is_terminal():
        winner = state.winner()
        assert winner is not None
        return float(winner * state.current_player())

    n = state.size
    win_len = state.win_length
    block_two = state.block_two_ends
    stride, base_grid, cell_indices, nbrs, dirs, center_idx = _gomoku_tables(n, radius)
    grid, stones_x, stones_o = _load_gomoku_grid(state, stride, base_grid, cell_indices)

    start_count = len(stones_x) + len(stones_o)
    max_possible = n * n - start_count
    limit = max_possible if max_moves is None else min(max_possible, max_moves)

    wins_x_set, wins_x_list = _find_initial_wins_1d(
        grid, stones_x, PLAYER_X, win_len, block_two, dirs
    )
    wins_o_set, wins_o_list = _find_initial_wins_1d(
        grid, stones_o, PLAYER_O, win_len, block_two, dirs
    )

    if use_candidates:
        in_pool = [0] * (stride * stride)
        for s in stones_x:
            in_pool[s] = 1
        for s in stones_o:
            in_pool[s] = 1
        pool: list[int] = []
        if start_count == 0:
            in_pool[center_idx] = 1
            pool.append(center_idx)
        else:
            for s in stones_x:
                for n_idx in nbrs[s]:
                    if not in_pool[n_idx]:
                        in_pool[n_idx] = 1
                        pool.append(n_idx)
            for s in stones_o:
                for n_idx in nbrs[s]:
                    if not in_pool[n_idx]:
                        in_pool[n_idx] = 1
                        pool.append(n_idx)
    else:
        in_pool = []
        pool = [idx for idx in cell_indices if grid[idx] == 0]
    pool_len = len(pool)

    player_at_start = state.current_player()
    player = player_at_start

    for _ in range(limit):
        if player == PLAYER_X:
            my_list, my_set = wins_x_list, wins_x_set
            opp_list, opp_set = wins_o_list, wins_o_set
        else:
            my_list, my_set = wins_o_list, wins_o_set
            opp_list, opp_set = wins_x_list, wins_x_set

        # 1. Thắng ngay ở lượt hiện tại
        for idx in my_list:
            if grid[idx] == 0 and (
                not block_two or _check_win_1d(grid, idx, player, win_len, block_two, dirs)
            ):
                return float(player * player_at_start)
        if my_list:
            my_list.clear()
            my_set.clear()

        # 2. Chặn nước thắng ngay của đối thủ
        opp = -player
        valid_blocks: list[int] = []
        for idx in opp_list:
            if grid[idx] == 0 and (
                not block_two or _check_win_1d(grid, idx, opp, win_len, block_two, dirs)
            ):
                valid_blocks.append(idx)
        if len(valid_blocks) != len(opp_list):
            opp_list[:] = valid_blocks
            opp_set.clear()
            opp_set.update(valid_blocks)

        idx0 = 0
        if valid_blocks:
            idx0 = valid_blocks[0] if len(valid_blocks) == 1 else rng.choice(valid_blocks)
        else:
            # 3. Chọn ngẫu nhiên từ tập ứng viên (L2) hoặc toàn bộ ô trống (L1)
            while pool_len > 0:
                pick = rng.randrange(pool_len)
                cand = pool[pick]
                pool_len -= 1
                pool[pick] = pool[pool_len]
                if grid[cand] == 0:
                    idx0 = cand
                    break
            if idx0 == 0:
                for cand in cell_indices:
                    if grid[cand] == 0:
                        idx0 = cand
                        break
                if idx0 == 0:
                    return 0.0

        grid[idx0] = player

        # Kiểm tra thắng tại idx0 và cập nhật tăng dần các đầu mút dọa thắng mới của `player`
        for d in dirs:
            bp = idx0 - d
            b = 0
            while grid[bp] == player:
                b += 1
                bp -= d
            fp = idx0 + d
            f = 0
            while grid[fp] == player:
                f += 1
                fp += d
            seg = 1 + b + f
            if seg >= win_len:
                if not block_two or not (grid[bp] == opp and grid[fp] == opp):
                    return float(player * player_at_start)

            if grid[bp] == 0:
                ebp = bp - d
                ext_b = 0
                while grid[ebp] == player:
                    ext_b += 1
                    ebp -= d
                if seg + 1 + ext_b >= win_len:
                    if not block_two or not (grid[ebp] == opp and grid[fp] == opp):
                        if bp not in my_set:
                            my_set.add(bp)
                            my_list.append(bp)

            if grid[fp] == 0:
                efp = fp + d
                ext_f = 0
                while grid[efp] == player:
                    ext_f += 1
                    efp += d
                if seg + 1 + ext_f >= win_len:
                    if not block_two or not (grid[bp] == opp and grid[efp] == opp):
                        if fp not in my_set:
                            my_set.add(fp)
                            my_list.append(fp)

        if use_candidates:
            in_pool[idx0] = 1
            for n_idx in nbrs[idx0]:
                if not in_pool[n_idx]:
                    in_pool[n_idx] = 1
                    if pool_len < len(pool):
                        pool[pool_len] = n_idx
                    else:
                        pool.append(n_idx)
                    pool_len += 1

        player = opp

    return 0.0


def find_winning_moves(
    state: GameState,
    player: int | None = None,
    *,
    use_candidates: bool = False,
    radius: int = 2,
) -> list[Move]:
    """Trả về danh sách các nước đi hợp lệ giúp `player` thắng ngay trong 1 nước.

    Nếu `player` là `None`, mặc định kiểm tra cho `state.current_player()`.
    Nếu `use_candidates=True`, chỉ xét các nước trong `state.candidate_moves(radius)`.
    Hàm giữ nguyên trạng thái `state` sau khi kết thúc.
    """
    if state.is_terminal():
        return []
    target = state.current_player() if player is None else player
    moves = state.candidate_moves(radius) if use_candidates else state.legal_moves()
    if not moves and use_candidates:
        moves = state.legal_moves()

    if isinstance(state, Gomoku):
        stride, base_grid, cell_indices, _, dirs, _ = _gomoku_tables(state.size, 2)
        grid, _, _ = _load_gomoku_grid(state, stride, base_grid, cell_indices)
        win_len = state.win_length
        block_two = state.block_two_ends
        return [
            mv
            for mv in moves
            if _check_win_1d(
                grid,
                (mv[0] + 1) * stride + (mv[1] + 1),
                target,
                win_len,
                block_two,
                dirs,
            )
        ]

    # Đối với GameState tổng quát (ví dụ TicTacToe 3×3)
    winning: list[Move] = []
    if target == state.current_player():
        for mv in moves:
            state.play(mv)
            if state.winner() == target:
                winning.append(mv)
            state.undo()
        return winning

    # Khi cần kiểm tra nước thắng ngay của đối thủ trên GameState tổng quát:
    # thử đóng vai đối thủ đánh vào từng ô trống thông qua một nước đệm hoặc kiểm tra dòng.
    if len(moves) == 1:
        return []
    for i, mv in enumerate(moves):
        # Chọn một nước đệm dummy khác mv mà không kết thúc ván ngay cho current_player
        dummy: Move | None = None
        for j, other in enumerate(moves):
            if i == j:
                continue
            state.play(other)
            if not state.is_terminal():
                dummy = other
                break
            state.undo()
        if dummy is None:
            continue
        state.play(mv)
        if state.winner() == target:
            winning.append(mv)
        state.undo()
        state.undo()
    return winning


def select_tactical_move(
    state: GameState,
    rng: random.Random,
    *,
    use_candidates: bool = False,
    radius: int = 2,
) -> Move:
    """Chọn 1 nước đi theo quy tắc L1 (`use_candidates=False`) hoặc L2 (`use_candidates=True`).

    Thứ tự ưu tiên:
      1. Nước thắng ngay của `state.current_player()`.
      2. Nước chặn thắng ngay của đối thủ (`-state.current_player()`).
      3. Nước ngẫu nhiên đồng nhất từ `candidate_moves(radius)` (L2) hoặc `legal_moves()` (L1).
    """
    if state.is_terminal():
        raise ValueError("Ván đã kết thúc, không còn nước để chọn")

    player = state.current_player()
    win_moves = find_winning_moves(state, player, use_candidates=use_candidates, radius=radius)
    if win_moves:
        return win_moves[0] if len(win_moves) == 1 else rng.choice(win_moves)

    block_moves = find_winning_moves(state, -player, use_candidates=use_candidates, radius=radius)
    if block_moves:
        return block_moves[0] if len(block_moves) == 1 else rng.choice(block_moves)

    pool = state.candidate_moves(radius) if use_candidates else state.legal_moves()
    if not pool:
        pool = state.legal_moves()
    return rng.choice(pool)


def _rollout_tactical_generic(
    state: GameState,
    rng: random.Random,
    *,
    use_candidates: bool,
    radius: int = 2,
    max_moves: int | None = None,
) -> float:
    """Rollout L1/L2 cho `GameState` tổng quát (ví dụ `TicTacToe`), khôi phục nguyên `state`."""
    player_at_start = state.current_player()
    start_count = state.move_count()
    plies = 0
    try:
        while not state.is_terminal():
            if max_moves is not None and plies >= max_moves:
                return 0.0
            move = select_tactical_move(state, rng, use_candidates=use_candidates, radius=radius)
            state.play(move)
            plies += 1
        winner = state.winner()
        assert winner is not None
        return float(winner * player_at_start)
    finally:
        while state.move_count() > start_count:
            state.undo()


def rollout_l0(
    state: GameState,
    rng: random.Random,
    max_moves: int | None = None,
) -> float:
    """Rollout L0: chọn nước ngẫu nhiên đồng nhất tới cuối ván (hoặc `max_moves`)."""
    if isinstance(state, Gomoku):
        return _rollout_l0_gomoku(state, rng, max_moves=max_moves)
    return default_rollout(state, rng, max_moves=max_moves)


def rollout_l1(
    state: GameState,
    rng: random.Random,
    max_moves: int | None = None,
) -> float:
    """Rollout L1: ưu tiên thắng ngay / chặn thua ngay, còn lại chọn ngẫu nhiên trên toàn bàn."""
    if isinstance(state, Gomoku):
        return _rollout_tactical_gomoku(state, rng, use_candidates=False, max_moves=max_moves)
    return _rollout_tactical_generic(state, rng, use_candidates=False, max_moves=max_moves)


def rollout_l2(
    state: GameState,
    rng: random.Random,
    max_moves: int | None = None,
    *,
    radius: int = 2,
) -> float:
    """Rollout L2: ưu tiên thắng ngay / chặn thua ngay, còn lại chọn ngẫu nhiên trong ứng viên."""
    if radius < 1:
        raise ValueError(f"radius phải >= 1, nhận được {radius}")
    if isinstance(state, Gomoku):
        return _rollout_tactical_gomoku(
            state, rng, use_candidates=True, radius=radius, max_moves=max_moves
        )
    return _rollout_tactical_generic(
        state, rng, use_candidates=True, radius=radius, max_moves=max_moves
    )


def measure_sims_per_second(
    state: GameState,
    rollout: str | RolloutFn = "l2",
    *,
    iterations: int = 1000,
    use_candidates: bool = True,
    candidate_radius: int = 2,
    reuse_tree: bool = False,
    rollout_depth: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Đo tốc độ mô phỏng (`sims_per_s`) của chính sách rollout độc lập và trong MCTS.

    Trả về từ điển gồm:
      - `rollout_sims_per_s`: số lần chạy hàm rollout độc lập trên mỗi giây.
      - `mcts_sims_per_s`: tốc độ `sims_per_s` thực tế khi chạy `MCTSAgent.select_move`.
      - `mcts_time_s`: tổng thời gian (giây) của `MCTSAgent.select_move`.
      - `nodes`: tổng số nút cây MCTS đã tạo.
      - `depth`: độ sâu lớn nhất của cây MCTS.
    """
    if iterations < 1:
        raise ValueError(f"iterations phải >= 1, nhận được {iterations}")

    agent = MCTSAgent(
        rollout=rollout,
        rollout_depth=rollout_depth,
        use_candidates=use_candidates,
        candidate_radius=candidate_radius,
        reuse_tree=reuse_tree,
        seed=seed,
    )
    rollout_fn = agent.rollout_fn
    params = inspect.signature(rollout_fn).parameters
    accepts_max_moves = "max_moves" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    rng = random.Random(seed)
    t0 = time.perf_counter()
    if rollout_depth is not None and accepts_max_moves:
        for _ in range(iterations):
            rollout_fn(state, rng, max_moves=rollout_depth)  # type: ignore[call-arg]
    else:
        for _ in range(iterations):
            rollout_fn(state, rng)
    rollout_elapsed = time.perf_counter() - t0
    rollout_sps = (iterations / rollout_elapsed) if rollout_elapsed > 0 else 0.0

    agent.select_move(state, Budget(iterations=iterations))
    stats = agent.last_stats
    return {
        "rollout_sims_per_s": float(rollout_sps),
        "mcts_sims_per_s": float(stats["sims_per_s"]),
        "mcts_time_s": float(stats["time_s"]),
        "nodes": float(stats["nodes"]),
        "depth": float(stats["depth"]),
    }


# Đăng ký các chính sách L0–L2 vào bảng chính sách chung của MCTS
register_rollout("l0", rollout_l0)
register_rollout("l1", rollout_l1)
register_rollout("l2", rollout_l2)
