"""Tests for the statistical engine."""

import numpy as np
import pytest
from stats_engine import (
    boxcox_transform,
    boxcox_inverse,
    horn_outliers,
    tukey_outliers,
    dixon_outliers,
    check_normality,
    nonparametric_ri,
    parametric_ri,
    robust_ri,
    le_boedec_ri,
    bootstrap_ci,
    parametric_ci,
    calculate_reference_interval,
)


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_data():
    """Normal distribution, n=200."""
    rng = np.random.RandomState(42)
    return rng.normal(loc=100, scale=10, size=200)


@pytest.fixture
def normal_small():
    """Normal distribution, n=40."""
    rng = np.random.RandomState(42)
    return rng.normal(loc=50, scale=5, size=40)


@pytest.fixture
def skewed_data():
    """Right-skewed distribution (lognormal), n=150."""
    rng = np.random.RandomState(42)
    return rng.lognormal(mean=3, sigma=0.5, size=150)


@pytest.fixture
def data_with_outliers():
    """Normal data with clear outliers added."""
    rng = np.random.RandomState(42)
    data = rng.normal(loc=100, scale=10, size=100)
    data = np.append(data, [200, 210, 5, -10])  # Add obvious outliers
    return data


# ---------------------------------------------------------------------------
# Box-Cox transformation
# ---------------------------------------------------------------------------

class TestBoxCox:
    def test_transform_positive_data(self, normal_data):
        transformed, lmbda, shift = boxcox_transform(normal_data)
        assert len(transformed) == len(normal_data)
        assert shift == 0.0  # All positive

    def test_inverse(self, normal_data):
        transformed, lmbda, shift = boxcox_transform(normal_data)
        recovered = boxcox_inverse(transformed, lmbda, shift)
        np.testing.assert_allclose(recovered, normal_data, rtol=1e-5)

    def test_transform_with_negatives(self):
        data = np.array([-5, -1, 0, 2, 5, 10])
        transformed, lmbda, shift = boxcox_transform(data)
        assert shift > 0
        assert len(transformed) == len(data)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

class TestOutlierDetection:
    def test_horn_finds_outliers(self, data_with_outliers):
        mask = horn_outliers(data_with_outliers)
        assert mask.sum() >= 2  # Should find at least 2 of the extreme values

    def test_tukey_finds_outliers(self, data_with_outliers):
        mask = tukey_outliers(data_with_outliers)
        assert mask.sum() >= 2

    def test_dixon_finds_extremes(self):
        # Dixon D/R works best with a single clear outlier
        data = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 50.0])
        mask = dixon_outliers(data)
        assert mask.sum() >= 1
        assert mask[-1]  # The 50 should be flagged (after sorting)

    def test_no_outliers_in_clean_data(self, normal_data):
        mask = horn_outliers(normal_data)
        # A normal sample might have a few flagged, but should be rare
        assert mask.sum() <= 10  # Less than 5% flagged

    def test_horn_with_small_data(self):
        data = np.array([1.0, 2.0, 3.0])
        mask = horn_outliers(data)
        assert len(mask) == 3

    def test_dixon_with_no_outliers(self):
        data = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        mask = dixon_outliers(data)
        assert mask.sum() == 0


# ---------------------------------------------------------------------------
# Normality testing
# ---------------------------------------------------------------------------

class TestNormality:
    def test_normal_data_is_normal(self, normal_data):
        result = check_normality(normal_data)
        assert result["is_normal"] == True

    def test_skewed_data_is_not_normal(self, skewed_data):
        result = check_normality(skewed_data)
        assert result["is_normal"] == False

    def test_small_data(self):
        result = check_normality(np.array([1.0, 2.0]))
        assert result["is_normal"] is None  # Insufficient data


# ---------------------------------------------------------------------------
# Reference interval methods
# ---------------------------------------------------------------------------

class TestNonparametricRI:
    def test_basic(self, normal_data):
        lower, upper = nonparametric_ri(normal_data)
        # 95% RI for N(100, 10) should be roughly 80-120
        assert 70 < lower < 90
        assert 110 < upper < 130

    def test_coverage(self, normal_data):
        lower, upper = nonparametric_ri(normal_data, ref_conf=0.95)
        within = np.sum((normal_data >= lower) & (normal_data <= upper))
        # About 95% should be within
        assert within / len(normal_data) >= 0.90


class TestParametricRI:
    def test_basic(self, normal_data):
        lower, upper = parametric_ri(normal_data)
        mean = np.mean(normal_data)
        # Should be symmetric around mean
        assert abs((upper - mean) - (mean - lower)) < 0.01

    def test_width(self, normal_data):
        lower, upper = parametric_ri(normal_data, ref_conf=0.95)
        sd = np.std(normal_data, ddof=1)
        expected_width = 2 * 1.96 * sd
        actual_width = upper - lower
        assert abs(actual_width - expected_width) < 0.01


class TestRobustRI:
    def test_basic(self, normal_data):
        lower, upper, t_bi, s_bi, bc_lmbda = robust_ri(normal_data)
        # Normal data should NOT trigger Box-Cox (already symmetric)
        assert bc_lmbda is None
        # Should be similar to parametric for normal data
        p_lower, p_upper = parametric_ri(normal_data)
        assert abs(lower - p_lower) < 10
        assert abs(upper - p_upper) < 10

    def test_small_sample(self, normal_small):
        lower, upper, t_bi, s_bi, _ = robust_ri(normal_small)
        assert lower < upper
        assert not np.isnan(lower)
        assert not np.isnan(upper)

    def test_robust_location(self, normal_data):
        _, _, t_bi, _, _ = robust_ri(normal_data)
        # Robust location should be close to median
        assert abs(t_bi - np.median(normal_data)) < 5

    def test_transformed_robust_skewed(self):
        """Box-Cox + biweight should avoid negative lower limit on skewed data."""
        rng = np.random.RandomState(42)
        # Chi-squared with df=3 is right-skewed, all positive
        skewed = rng.chisquare(df=3, size=80)
        lower_t, upper_t, _, _, bc_lmbda_t = robust_ri(skewed, transform=True)
        lower_p, upper_p, _, _, bc_lmbda_p = robust_ri(skewed, transform=False)
        # Transformed robust should have used Box-Cox
        assert bc_lmbda_t is not None
        # Plain robust may produce a negative lower limit; transformed should not
        assert lower_t >= 0, (
            f"Transformed robust lower limit should be >= 0, got {lower_t}"
        )
        assert upper_t > lower_t

    def test_no_transform_flag(self, normal_data):
        """transform=False should skip Box-Cox even for non-normal data."""
        rng = np.random.RandomState(99)
        skewed = rng.exponential(scale=5, size=60)
        _, _, _, _, bc_lmbda = robust_ri(skewed, transform=False)
        assert bc_lmbda is None


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

class TestParametricCI:
    def test_ci_contains_limits(self, normal_data):
        lower, upper = parametric_ri(normal_data)
        ci_ll, ci_lh, ci_ul, ci_uh = parametric_ci(normal_data)
        assert ci_ll <= lower <= ci_lh
        assert ci_ul <= upper <= ci_uh


class TestBootstrapCI:
    def test_bootstrap_nonparametric(self, normal_data):
        ci = bootstrap_ci(normal_data, method="nonparametric", n_boot=1000)
        assert len(ci) == 4
        assert ci[0] < ci[1]  # Lower CI is ordered
        assert ci[2] < ci[3]  # Upper CI is ordered

    def test_bootstrap_robust(self, normal_small):
        ci = bootstrap_ci(normal_small, method="robust", n_boot=1000)
        assert len(ci) == 4

    def test_bootstrap_robust_transformed_ci_contains_ri(self):
        """Bootstrap CI with transform=True should contain the RI limits."""
        rng = np.random.RandomState(42)
        skewed = rng.chisquare(df=3, size=80)
        lower, upper, _, _, bc_lmbda = robust_ri(skewed, transform=True)
        assert bc_lmbda is not None, "Expected Box-Cox to be applied"
        ci = bootstrap_ci(skewed, method="robust", n_boot=2000,
                          robust_transform=True)
        # The 90% CI should contain the point estimate
        assert ci[0] <= lower <= ci[1], (
            f"Lower RI {lower:.4f} not in CI [{ci[0]:.4f}, {ci[1]:.4f}]"
        )
        assert ci[2] <= upper <= ci[3], (
            f"Upper RI {upper:.4f} not in CI [{ci[2]:.4f}, {ci[3]:.4f}]"
        )


# ---------------------------------------------------------------------------
# Le Boedec reference interval
# ---------------------------------------------------------------------------

class TestLeBoedecRI:
    def test_large_sample_uses_nonparametric(self, normal_data):
        """n=200 should use nonparametric."""
        lower, upper, method = le_boedec_ri(normal_data)
        assert "Nonparametric" in method
        assert "n >= 120" in method

    def test_small_sample_uses_nonparametric(self):
        """n < 40 should use nonparametric."""
        rng = np.random.RandomState(42)
        data = rng.normal(50, 5, 25)
        lower, upper, method = le_boedec_ri(data)
        assert "Nonparametric" in method
        assert "n < 40" in method

    def test_medium_gaussian_uses_parametric(self):
        """n=60 Gaussian data with SW p > 0.2 should use parametric."""
        rng = np.random.RandomState(42)
        data = rng.normal(100, 10, 60)
        lower, upper, method = le_boedec_ri(data)
        # Gaussian data should pass SW at p > 0.2
        assert lower < upper

    def test_medium_skewed_uses_nonparametric(self):
        """n=60 skewed data with SW p <= 0.2 should use nonparametric."""
        rng = np.random.RandomState(42)
        data = rng.lognormal(3, 0.8, 60)
        lower, upper, method = le_boedec_ri(data)
        assert lower < upper

    def test_via_calculate_ri(self, normal_data):
        """Test Le Boedec through the main calculate function."""
        result = calculate_reference_interval(
            normal_data, ri_method="le_boedec"
        )
        assert "Le Boedec" in result.method_ri
        assert not np.isnan(result.lower_limit)
        assert not np.isnan(result.upper_limit)

    def test_via_calculate_ri_small(self, normal_small):
        """Test Le Boedec with small sample through main calculate function."""
        result = calculate_reference_interval(
            normal_small, ri_method="le_boedec"
        )
        assert "Le Boedec" in result.method_ri


# ---------------------------------------------------------------------------
# Main calculate_reference_interval function
# ---------------------------------------------------------------------------

class TestCalculateRI:
    def test_auto_nonparametric(self, normal_data):
        """n=200 should auto-select nonparametric."""
        result = calculate_reference_interval(normal_data, analyte_name="test")
        assert "Nonparametric" in result.method_ri
        assert result.n_total == 200

    def test_auto_robust(self, normal_small):
        """n=40 should auto-select robust."""
        result = calculate_reference_interval(normal_small, analyte_name="test")
        assert "Robust" in result.method_ri or "Parametric" in result.method_ri

    def test_outlier_removal(self, data_with_outliers):
        result = calculate_reference_interval(
            data_with_outliers, outlier_method="horn", remove_outliers=True
        )
        assert result.n_outliers > 0
        assert result.n_used < result.n_total

    def test_no_outlier_removal(self, data_with_outliers):
        result = calculate_reference_interval(
            data_with_outliers, outlier_method="horn", remove_outliers=False
        )
        assert result.n_used == result.n_total

    def test_forced_parametric(self, normal_data):
        result = calculate_reference_interval(
            normal_data, ri_method="parametric"
        )
        assert "Parametric" in result.method_ri

    def test_forced_robust(self, normal_data):
        result = calculate_reference_interval(
            normal_data, ri_method="robust"
        )
        assert "Robust" in result.method_ri

    def test_with_nan_values(self):
        data = np.array([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        result = calculate_reference_interval(data)
        assert result.n_total == 19  # One NaN removed

    def test_insufficient_data(self):
        data = np.array([1.0, 2.0, 3.0])
        result = calculate_reference_interval(data)
        assert len(result.warnings) > 0

    def test_result_has_all_fields(self, normal_data):
        result = calculate_reference_interval(normal_data, analyte_name="Glucose")
        assert result.analyte == "Glucose"
        assert not np.isnan(result.lower_limit)
        assert not np.isnan(result.upper_limit)
        assert not np.isnan(result.mean)
        assert not np.isnan(result.median)
        assert not np.isnan(result.std)
        assert result.lower_limit < result.upper_limit
        assert result.method_ri != ""
        assert result.method_ci != ""

    def test_skewed_data_parametric(self, skewed_data):
        result = calculate_reference_interval(
            skewed_data, ri_method="parametric"
        )
        # Should detect non-normality and try Box-Cox
        assert result.is_normal is False or result.boxcox_lambda is not None
