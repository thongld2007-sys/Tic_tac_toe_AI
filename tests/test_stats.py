"""Unit test cho module thống kê thực nghiệm (P3.2a)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from experiments.stats import (
    ConfidenceInterval,
    ZTestResult,
    bootstrap_ci,
    bootstrap_diff_ci,
    normal_z,
    proportions_ztest,
    summarize_matchup,
    two_proportion_ztest,
    wilson_ci,
)

# ---------------------------------------------------------------------------
# Giá trị tham chiếu chính xác (khớp 1e-12 với scipy.stats và statsmodels)
# ---------------------------------------------------------------------------


def test_normal_z_matches_scipy_norm_ppf() -> None:
    """normal_z khớp với scipy.stats.norm.ppf(1 - alpha/2) trên 3 mức tin cậy chuẩn."""
    assert normal_z(0.90) == pytest.approx(1.6448536269514722, rel=1e-12)
    assert normal_z(0.95) == pytest.approx(1.959963984540054, rel=1e-12)
    assert normal_z(0.99) == pytest.approx(2.5758293035489004, rel=1e-12)


@pytest.mark.parametrize(
    ("k", "n", "confidence", "expected_low", "expected_high"),
    [
        # Ví dụ 1: 140 thắng / 200 ván ở mức tin cậy 95%
        # statsmodels: proportion_confint(140, 200, alpha=0.05, method="wilson")
        (140, 200, 0.95, 0.6332093162728074, 0.7592525531762859),
        # Ví dụ 2: Biên trái k = 0 / 50 ở mức tin cậy 95%
        (0, 50, 0.95, 0.0, 0.0713475991333587),
        # Ví dụ 3: Biên phải k = 50 / 50 ở mức tin cậy 95%
        (50, 50, 0.95, 0.9286524008666414, 1.0),
        # Ví dụ 4: 45 thắng / 60 ván ở mức tin cậy 99%
        (45, 60, 0.99, 0.5862217415306675, 0.8639928086565422),
    ],
)
def test_wilson_ci_matches_statsmodels_reference_examples(
    k: int,
    n: int,
    confidence: float,
    expected_low: float,
    expected_high: float,
) -> None:
    """Tiêu chí P3.2a: wilson_ci khớp với statsmodels trên >= 3 ví dụ."""
    ci = wilson_ci(k, n, confidence=confidence)
    assert isinstance(ci, ConfidenceInterval)
    low, high = ci
    assert low == pytest.approx(expected_low, abs=1e-12)
    assert high == pytest.approx(expected_high, abs=1e-12)
    assert ci.center == pytest.approx(0.5 * (expected_low + expected_high), abs=1e-12)
    assert ci.half_width == pytest.approx(0.5 * (expected_high - expected_low), abs=1e-12)
    assert ci.width == pytest.approx(expected_high - expected_low, abs=1e-12)


@pytest.mark.parametrize(
    ("k1", "n1", "k2", "n2", "alternative", "pooled", "expected_z", "expected_p"),
    [
        # Ví dụ 1: Kiểm định hai phía (two-sided), phương sai gộp (pooled)
        # statsmodels: proportions_ztest([150, 120], [200, 200], alternative="two-sided")
        (150, 200, 120, 200, "two-sided", True, 3.202563076101743, 0.0013621046715841523),
        # Ví dụ 2: Kiểm định một phía lớn hơn (larger / greater)
        # statsmodels: proportions_ztest([150, 120], [200, 200], alternative="larger")
        (150, 200, 120, 200, "larger", True, 3.202563076101743, 0.0006810523357920761),
        # Ví dụ 3: Kiểm định một phía nhỏ hơn (smaller / less)
        # statsmodels: proportions_ztest([45, 62], [100, 100], alternative="smaller")
        (45, 100, 62, 100, "smaller", True, -2.4100749907751173, 0.007974620933046683),
        # Ví dụ 4: Phương sai không gộp (pooled=False)
        # statsmodels: proportions_ztest([80, 55], [100, 90], prop_var=False)
        (80, 100, 55, 90, "two-sided", False, 2.90063160574278, 0.0037241141761386043),
    ],
)
def test_two_proportion_ztest_matches_statsmodels_reference_examples(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    alternative: str,
    pooled: bool,
    expected_z: float,
    expected_p: float,
) -> None:
    """Tiêu chí P3.2a: two_proportion_ztest khớp với statsmodels trên >= 3 ví dụ."""
    res = two_proportion_ztest(k1, n1, k2, n2, alternative=alternative, pooled=pooled)
    assert isinstance(res, ZTestResult)
    z_stat, p_val = res
    assert z_stat == pytest.approx(expected_z, rel=1e-12)
    assert p_val == pytest.approx(expected_p, rel=1e-12)
    assert res.statistic == pytest.approx(expected_z, rel=1e-12)
    assert res.pvalue == pytest.approx(expected_p, rel=1e-12)

    # Bí danh proportions_ztest trả về đúng cùng kết quả
    assert proportions_ztest(k1, n1, k2, n2, alternative=alternative, pooled=pooled) == res


def test_two_proportion_ztest_degenerate_and_alias_cases() -> None:
    """Kiểm tra các trường hợp suy biến (SE = 0 hoặc hai tỷ lệ bằng nhau)."""
    # Hai nhóm cùng 0/50 -> SE = 0, diff = 0 -> z = 0, p = 1
    assert two_proportion_ztest(0, 50, 0, 50) == ZTestResult(0.0, 1.0)
    # Hai nhóm cùng 50/50 -> SE = 0, diff = 0 -> z = 0, p = 1
    assert two_proportion_ztest(50, 50, 50, 50) == ZTestResult(0.0, 1.0)
    # Hai nhóm cùng tỷ lệ 25/50 và 50/100 -> diff = 0 -> z = 0, p = 1 (two-sided)
    assert two_proportion_ztest(25, 50, 50, 100) == ZTestResult(0.0, 1.0)

    # Nhóm 1 thắng tuyệt đối 50/50, nhóm 2 thua tuyệt đối 0/50 với pooled=False -> SE = 0, diff = 1
    res_inf = two_proportion_ztest(50, 50, 0, 50, pooled=False, alternative="greater")
    assert math.isinf(res_inf.z_stat) and res_inf.z_stat > 0
    assert res_inf.p_value == 0.0

    res_less = two_proportion_ztest(50, 50, 0, 50, pooled=False, alternative="less")
    assert res_less.p_value == 1.0


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_ci_matches_percentile_reference_on_3_examples() -> None:
    """Kiểm tra bootstrap_ci trên 3 ví dụ: điểm ván đấu, thời gian, và thống kê median."""
    # Ví dụ 1: Mẫu điểm ván đấu {1.0, 0.5, 0.0} (120 thắng, 40 hòa, 40 thua -> mean = 0.70)
    scores = np.array([1.0] * 120 + [0.5] * 40 + [0.0] * 40)
    ci_scores = bootstrap_ci(scores, confidence=0.95, n_resamples=20_000, seed=42)
    # Độ lệch chuẩn mẫu của scores là sqrt(0.16) = 0.4, SE = 0.4 / sqrt(200) = 0.028284
    # Khoảng 95% lý thuyết xấp xỉ [0.70 - 1.96*0.028284, 0.70 + 1.96*0.028284] = [0.6446, 0.7554]
    assert ci_scores.low == pytest.approx(0.6446, abs=0.005)
    assert ci_scores.high == pytest.approx(0.7554, abs=0.005)

    # Ví dụ 2: Mẫu chuẩn N(10, 2^2) cỡ mẫu 400 -> SE = 2 / 20 = 0.1
    rng = np.random.default_rng(2026)
    sample_norm = rng.normal(loc=10.0, scale=2.0, size=400)
    sample_mean = float(sample_norm.mean())
    sample_se = float(sample_norm.std(ddof=1) / math.sqrt(400))
    ci_norm = bootstrap_ci(sample_norm, confidence=0.95, n_resamples=20_000, seed=7)
    assert ci_norm.low == pytest.approx(sample_mean - 1.96 * sample_se, abs=0.01)
    assert ci_norm.high == pytest.approx(sample_mean + 1.96 * sample_se, abs=0.01)

    # Ví dụ 3: Thống kê trung vị (median) và hàm tùy chọn (callable) + phương pháp basic
    data_skew = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 20.0]
    ci_med = bootstrap_ci(data_skew, statistic="median", confidence=0.90, n_resamples=5_000, seed=9)
    ci_fn = bootstrap_ci(
        data_skew,
        statistic=lambda x: float(np.median(x)),
        confidence=0.90,
        n_resamples=5_000,
        seed=9,
    )
    assert ci_med == ci_fn
    assert 2.0 <= ci_med.low <= 3.5 <= ci_med.high <= 5.0

    ci_basic = bootstrap_ci(scores, confidence=0.95, n_resamples=10_000, seed=42, method="basic")
    assert ci_basic.low == pytest.approx(ci_scores.low, abs=0.01)
    assert ci_basic.high == pytest.approx(ci_scores.high, abs=0.01)


def test_bootstrap_reproducibility_and_constant_sample() -> None:
    """Cùng seed hoặc random.Random(seed) phải cho kết quả bootstrap_ci tái lập."""
    data = [0.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0]
    assert bootstrap_ci(data, seed=123) == bootstrap_ci(data, seed=123)
    assert bootstrap_ci(data, seed=random.Random(99)) == bootstrap_ci(data, seed=random.Random(99))

    # Mẫu hằng số hoặc 1 phần tử trả về khoảng suy biến [c, c]
    assert bootstrap_ci([1.0, 1.0, 1.0], seed=1) == ConfidenceInterval(1.0, 1.0)
    assert bootstrap_ci([0.5], seed=1) == ConfidenceInterval(0.5, 0.5)


def test_bootstrap_diff_ci_independent_and_paired() -> None:
    """Kiểm tra bootstrap_diff_ci cho mẫu độc lập và mẫu ghép cặp (paired=True)."""
    a = np.array([1.0] * 80 + [0.0] * 20)  # mean = 0.80
    b = np.array([1.0] * 50 + [0.0] * 50)  # mean = 0.50

    ci_ind = bootstrap_diff_ci(a, b, confidence=0.95, n_resamples=10_000, seed=11, paired=False)
    assert ci_ind.low > 0.15
    assert ci_ind.center == pytest.approx(0.30, abs=0.02)

    ci_paired = bootstrap_diff_ci(a, b, confidence=0.95, n_resamples=10_000, seed=11, paired=True)
    assert ci_paired.low > 0.15

    ci_basic = bootstrap_diff_ci(a, b, confidence=0.95, n_resamples=5_000, seed=11, method="basic")
    assert ci_basic.low > 0.15


def test_summarize_matchup_computes_all_fields() -> None:
    """summarize_matchup trả về đầy đủ số ván, thắng/hòa/thua, Wilson CI và Bootstrap CI."""
    scores = [1.0] * 70 + [0.5] * 20 + [0.0] * 10
    summary = summarize_matchup(scores, confidence=0.95, n_resamples=4_000, seed=42)
    assert summary["games"] == 100.0
    assert (summary["wins"], summary["draws"], summary["losses"]) == (70.0, 20.0, 10.0)
    assert summary["win_rate"] == pytest.approx(0.70)
    assert summary["score_mean"] == pytest.approx(0.80)
    assert 0.60 < summary["win_wilson_low"] < 0.70 < summary["win_wilson_high"] < 0.79
    assert 0.73 < summary["score_boot_low"] < 0.80 < summary["score_boot_high"] < 0.87


def test_invalid_inputs_raise_value_error() -> None:
    """Mọi đầu vào sai kiểu hoặc ngoài miền hợp lệ đều phải ném ValueError."""
    with pytest.raises(ValueError):
        wilson_ci(10, 0)
    with pytest.raises(ValueError):
        wilson_ci(-1, 10)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)
    with pytest.raises(ValueError):
        wilson_ci(True, 10)
    with pytest.raises(ValueError):
        wilson_ci(5, True)
    with pytest.raises(ValueError):
        wilson_ci(5, 10, confidence=1.0)
    with pytest.raises(ValueError):
        wilson_ci(5, 10, confidence=0.0)
    with pytest.raises(ValueError):
        wilson_ci(5, 10, confidence=True)

    with pytest.raises(ValueError):
        two_proportion_ztest(5, 10, 5, 10, alternative="wrong")
    with pytest.raises(ValueError):
        two_proportion_ztest(5, 10, 5, 10, pooled=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, float("nan")])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], n_resamples=0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], method="bca")
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], statistic="unknown")
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], seed=True)
    with pytest.raises(ValueError):
        bootstrap_diff_ci([1.0, 0.0], [1.0], paired=True)
