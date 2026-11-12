"""Module thống kê thực nghiệm: khoảng tin cậy Wilson, kiểm định hai tỷ lệ và bootstrap (P3.2a).

Chủ sở hữu: TV2.  Review: TV3.

Các hàm chính:
  - `wilson_ci(successes, n, confidence=0.95)`: khoảng tin cậy Wilson cho tỷ lệ nhị thức,
    khớp chính xác với `statsmodels.stats.proportion.proportion_confint(..., method="wilson")`.
  - `two_proportion_ztest(k1, n1, k2, n2, alternative="two-sided", pooled=True)`: kiểm định z
    cho hai tỷ lệ độc lập, khớp chính xác với `statsmodels.stats.proportion.proportions_ztest`.
  - `bootstrap_ci(data, ...)`: khoảng tin cậy bootstrap phi tham số (percentile / basic) cho
    kỳ vọng (hoặc hàm thống kê tùy chọn) của mẫu, dùng hạt giống tường minh (`seed`).
  - `bootstrap_diff_ci(data_a, data_b, ...)`: khoảng tin cậy bootstrap cho hiệu hai kỳ vọng
    (độc lập hoặc ghép cặp `paired=True`).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from statistics import NormalDist
from typing import NamedTuple

import numpy as np

_STD_NORMAL = NormalDist(0.0, 1.0)

ALTERNATIVES: tuple[str, ...] = (
    "two-sided",
    "two_sided",
    "larger",
    "greater",
    "smaller",
    "less",
)
BOOTSTRAP_METHODS: tuple[str, ...] = ("percentile", "basic")


class ConfidenceInterval(NamedTuple):
    """Khoảng tin cậy `[low, high]`, hỗ trợ giải nén 2 phần tử `low, high = ...`."""

    low: float
    high: float

    @property
    def center(self) -> float:
        """Điểm giữa của khoảng tin cậy."""
        return 0.5 * (self.low + self.high)

    @property
    def half_width(self) -> float:
        """Bán kính (nửa độ rộng) của khoảng tin cậy."""
        return 0.5 * (self.high - self.low)

    @property
    def width(self) -> float:
        """Độ rộng `high - low` của khoảng tin cậy."""
        return self.high - self.low


class ZTestResult(NamedTuple):
    """Kết quả kiểm định z cho hai tỷ lệ `(z_stat, p_value)`."""

    z_stat: float
    p_value: float

    @property
    def statistic(self) -> float:
        """Bí danh tương thích quy ước của scipy/statsmodels."""
        return self.z_stat

    @property
    def pvalue(self) -> float:
        """Bí danh tương thích quy ước của scipy/statsmodels."""
        return self.p_value


def _check_int(name: str, value: object, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} phải là số nguyên, nhận được {value!r}")
    iv = int(value)
    if iv < minimum:
        raise ValueError(f"{name} phải >= {minimum}, nhận được {iv}")
    return iv


def _check_count(name: str, count: int | float, n: int) -> float:
    if isinstance(count, bool) or not isinstance(count, (int, float, np.number)):
        raise ValueError(f"{name} phải là số thực hoặc số nguyên, nhận được {count!r}")
    val = float(count)
    if math.isnan(val) or val < 0.0 or val > float(n):
        raise ValueError(f"{name} phải nằm trong đoạn [0, {n}], nhận được {count!r}")
    return val


def _check_confidence(confidence: float) -> float:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float, np.number)):
        raise ValueError(f"confidence phải là số thực trong (0, 1), nhận được {confidence!r}")
    conf = float(confidence)
    if math.isnan(conf) or not (0.0 < conf < 1.0):
        raise ValueError(f"confidence phải nằm trong khoảng mở (0, 1), nhận được {confidence!r}")
    return conf


def normal_z(confidence: float = 0.95) -> float:
    """Giá trị tới hạn chuẩn tắc hai phía `z_{1 - alpha/2}` với `alpha = 1 - confidence`."""
    conf = _check_confidence(confidence)
    alpha = 1.0 - conf
    return float(_STD_NORMAL.inv_cdf(1.0 - 0.5 * alpha))


def wilson_ci(
    successes: int | float,
    n: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Khoảng tin cậy điểm số Wilson cho tỷ lệ nhị thức `p = successes / n`.

    Công thức khớp chính xác với:
        `statsmodels.stats.proportion.proportion_confint(..., method="wilson")`

    Tham số:
      - `successes`: số lần thành công `k` (trong đoạn `[0, n]`).
      - `n`: tổng số phép thử (`n >= 1`).
      - `confidence`: mức tin cậy trong khoảng `(0, 1)`, mặc định `0.95`.
    """
    n_int = _check_int("n", n, minimum=1)
    k = _check_count("successes", successes, n_int)
    z = normal_z(confidence)

    p_hat = k / n_int
    z2 = z * z
    denom = 1.0 + z2 / n_int
    center = (p_hat + z2 / (2.0 * n_int)) / denom
    radicand = (p_hat * (1.0 - p_hat)) / n_int + z2 / (4.0 * n_int * n_int)
    half = (z / denom) * math.sqrt(max(0.0, radicand))

    low = max(0.0, min(1.0, center - half))
    high = max(0.0, min(1.0, center + half))
    return ConfidenceInterval(low=low, high=high)


def two_proportion_ztest(
    k1: int | float,
    n1: int,
    k2: int | float,
    n2: int,
    *,
    alternative: str = "two-sided",
    pooled: bool = True,
) -> ZTestResult:
    """Kiểm định z cho hai tỷ lệ độc lập `p1 = k1 / n1` và `p2 = k2 / n2`.

    Công thức khớp chính xác với:
        `statsmodels.stats.proportion.proportions_ztest([k1, k2], [n1, n2], ...)`

    Tham số:
      - `k1`, `n1`: số thành công và cỡ mẫu của nhóm 1.
      - `k2`, `n2`: số thành công và cỡ mẫu của nhóm 2.
      - `alternative`:
          * `"two-sided"` (hoặc `"two_sided"`): H1: p1 != p2
          * `"larger"` (hoặc `"greater"`): H1: p1 > p2
          * `"smaller"` (hoặc `"less"`): H1: p1 < p2
      - `pooled`: nếu True (mặc định), dùng phương sai gộp dưới H0: p1 = p2;
        nếu False, dùng phương sai riêng của từng mẫu.
    """
    n1_int = _check_int("n1", n1, minimum=1)
    n2_int = _check_int("n2", n2, minimum=1)
    c1 = _check_count("k1", k1, n1_int)
    c2 = _check_count("k2", k2, n2_int)
    if not isinstance(pooled, bool):
        raise ValueError(f"pooled phải là kiểu bool, nhận được {pooled!r}")
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative phải thuộc {ALTERNATIVES}, nhận được {alternative!r}")

    p1 = c1 / n1_int
    p2 = c2 / n2_int
    diff = p1 - p2

    if pooled:
        p_pool = (c1 + c2) / (n1_int + n2_int)
        var = p_pool * (1.0 - p_pool) * (1.0 / n1_int + 1.0 / n2_int)
    else:
        var = (p1 * (1.0 - p1)) / n1_int + (p2 * (1.0 - p2)) / n2_int

    se = math.sqrt(max(0.0, var))
    if se == 0.0:
        if diff == 0.0:
            return ZTestResult(z_stat=0.0, p_value=1.0)
        z_stat = math.copysign(math.inf, diff)
        if alternative in ("two-sided", "two_sided"):
            p_val = 0.0
        elif alternative in ("larger", "greater"):
            p_val = 0.0 if diff > 0.0 else 1.0
        else:
            p_val = 0.0 if diff < 0.0 else 1.0
        return ZTestResult(z_stat=z_stat, p_value=p_val)

    z_stat = diff / se
    if alternative in ("two-sided", "two_sided"):
        p_val = 2.0 * (1.0 - _STD_NORMAL.cdf(abs(z_stat)))
    elif alternative in ("larger", "greater"):
        p_val = 1.0 - _STD_NORMAL.cdf(z_stat)
    else:
        p_val = _STD_NORMAL.cdf(z_stat)

    p_val = max(0.0, min(1.0, float(p_val)))
    return ZTestResult(z_stat=float(z_stat), p_value=p_val)


# Bí danh tương thích với tên hàm trong statsmodels
proportions_ztest = two_proportion_ztest


def _make_numpy_rng(
    seed: int | random.Random | np.random.Generator | None,
) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, random.Random):
        return np.random.default_rng(seed.getrandbits(64))
    if isinstance(seed, bool):
        raise ValueError("seed không được là kiểu bool")
    if seed is None:
        return np.random.default_rng(0)
    if isinstance(seed, (int, np.integer)):
        return np.random.default_rng(int(seed))
    raise ValueError(f"seed không hợp lệ: {seed!r}")


def _to_1d_float_array(name: str, data: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} phải là dãy 1 chiều không rỗng, nhận được shape {arr.shape}")
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError(f"{name} không được chứa NaN hoặc Inf")
    return arr


def _compute_statistic(
    samples_2d: np.ndarray,
    statistic: Callable[[np.ndarray], float] | str,
) -> np.ndarray:
    if statistic == "mean" or statistic is np.mean:
        return samples_2d.mean(axis=1)
    if statistic == "median" or statistic is np.median:
        return np.median(samples_2d, axis=1)
    if callable(statistic):
        try:
            res = statistic(samples_2d, axis=1)  # type: ignore[call-arg]
            res_arr = np.asarray(res, dtype=np.float64)
            if res_arr.shape == (samples_2d.shape[0],):
                return res_arr
        except TypeError:
            pass
        return np.array([float(statistic(row)) for row in samples_2d], dtype=np.float64)
    raise ValueError(f"statistic không hợp lệ: {statistic!r}")


def bootstrap_ci(
    data: Sequence[float] | np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] | str = "mean",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | random.Random | np.random.Generator | None = 0,
    method: str = "percentile",
) -> ConfidenceInterval:
    """Khoảng tin cậy bootstrap phi tham số cho một mẫu 1 chiều.

    Tham số:
      - `data`: dãy giá trị quan sát (ví dụ điểm số mỗi ván `{0.0, 0.5, 1.0}`, thời gian, số nút).
      - `statistic`: `"mean"` (mặc định), `"median"`, hoặc một hàm `f(sample_1d) -> float`.
      - `confidence`: mức tin cậy trong `(0, 1)`, mặc định `0.95`.
      - `n_resamples`: số mẫu bootstrap `B >= 1`, mặc định `10_000`.
      - `seed`: hạt giống ngẫu nhiên tường minh để tái lập kết quả.
      - `method`: `"percentile"` (mặc định, khớp `scipy.stats.bootstrap(..., method="percentile")`)
        hoặc `"basic"`.
    """
    arr = _to_1d_float_array("data", data)
    conf = _check_confidence(confidence)
    b_count = _check_int("n_resamples", n_resamples, minimum=1)
    if method not in BOOTSTRAP_METHODS:
        raise ValueError(f"method phải thuộc {BOOTSTRAP_METHODS}, nhận được {method!r}")

    n = arr.size
    if n == 1 or np.all(arr == arr[0]):
        val = float(_compute_statistic(arr.reshape(1, -1), statistic)[0])
        return ConfidenceInterval(low=val, high=val)

    rng = _make_numpy_rng(seed)
    indices = rng.integers(0, n, size=(b_count, n))
    boot_stats = _compute_statistic(arr[indices], statistic)

    alpha = 1.0 - conf
    q_low, q_high = np.quantile(boot_stats, [0.5 * alpha, 1.0 - 0.5 * alpha])

    if method == "percentile":
        return ConfidenceInterval(low=float(q_low), high=float(q_high))

    theta_hat = float(_compute_statistic(arr.reshape(1, -1), statistic)[0])
    return ConfidenceInterval(
        low=float(2.0 * theta_hat - q_high),
        high=float(2.0 * theta_hat - q_low),
    )


def bootstrap_diff_ci(
    data_a: Sequence[float] | np.ndarray,
    data_b: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | random.Random | np.random.Generator | None = 0,
    paired: bool = False,
    method: str = "percentile",
) -> ConfidenceInterval:
    """Khoảng tin cậy bootstrap cho hiệu hai kỳ vọng `E[A] - E[B]`.

    Nếu `paired=True`, `data_a` và `data_b` phải có cùng độ dài và hàm sẽ bootstrap trên
    dãy hiệu ghép cặp `d_i = a_i - b_i`. Nếu `paired=False` (mặc định), hai mẫu được lấy
    mẫu lại độc lập.
    """
    arr_a = _to_1d_float_array("data_a", data_a)
    arr_b = _to_1d_float_array("data_b", data_b)
    if not isinstance(paired, bool):
        raise ValueError(f"paired phải là kiểu bool, nhận được {paired!r}")

    if paired:
        if arr_a.size != arr_b.size:
            raise ValueError(
                f"Khi paired=True, data_a và data_b phải cùng độ dài ({arr_a.size} != {arr_b.size})"
            )
        return bootstrap_ci(
            arr_a - arr_b,
            statistic="mean",
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
            method=method,
        )

    conf = _check_confidence(confidence)
    b_count = _check_int("n_resamples", n_resamples, minimum=1)
    if method not in BOOTSTRAP_METHODS:
        raise ValueError(f"method phải thuộc {BOOTSTRAP_METHODS}, nhận được {method!r}")

    rng = _make_numpy_rng(seed)
    idx_a = rng.integers(0, arr_a.size, size=(b_count, arr_a.size))
    idx_b = rng.integers(0, arr_b.size, size=(b_count, arr_b.size))
    diffs = arr_a[idx_a].mean(axis=1) - arr_b[idx_b].mean(axis=1)

    alpha = 1.0 - conf
    q_low, q_high = np.quantile(diffs, [0.5 * alpha, 1.0 - 0.5 * alpha])
    if method == "percentile":
        return ConfidenceInterval(low=float(q_low), high=float(q_high))

    theta_hat = float(arr_a.mean() - arr_b.mean())
    return ConfidenceInterval(
        low=float(2.0 * theta_hat - q_high),
        high=float(2.0 * theta_hat - q_low),
    )


def summarize_matchup(
    scores_a: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Tóm tắt thống kê một cặp đấu từ dãy điểm `score_a` (`1.0` thắng, `0.5` hòa, `0.0` thua).

    Trả về dict gồm:
      - `games`, `wins`, `draws`, `losses`
      - `win_rate`, `win_wilson_low`, `win_wilson_high` (tỷ lệ thắng chặt và khoảng Wilson)
      - `score_mean`, `score_boot_low`, `score_boot_high` (điểm kỳ vọng và khoảng bootstrap)
    """
    arr = _to_1d_float_array("scores_a", scores_a)
    n = int(arr.size)
    wins = int(np.sum(np.isclose(arr, 1.0)))
    draws = int(np.sum(np.isclose(arr, 0.5)))
    losses = int(np.sum(np.isclose(arr, 0.0)))
    if wins + draws + losses != n:
        raise ValueError("scores_a chỉ được chứa các giá trị 0.0 (thua), 0.5 (hòa), 1.0 (thắng)")

    win_ci = wilson_ci(wins, n, confidence=confidence)
    score_ci = bootstrap_ci(
        arr,
        statistic="mean",
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    return {
        "games": float(n),
        "wins": float(wins),
        "draws": float(draws),
        "losses": float(losses),
        "win_rate": wins / n,
        "win_wilson_low": win_ci.low,
        "win_wilson_high": win_ci.high,
        "score_mean": float(arr.mean()),
        "score_boot_low": score_ci.low,
        "score_boot_high": score_ci.high,
    }
