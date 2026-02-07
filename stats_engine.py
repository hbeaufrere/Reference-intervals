"""
Statistical engine for reference interval calculation.

Implements methods following ASVCP guidelines (Friedrichs et al., 2012) and
algorithms from the CLSI EP28-A3c document. Inspired by the R package
'referenceIntervals' by Daniel Finnegan (CRAN).

Methods
-------
- Nonparametric reference intervals (recommended when n >= 120)
- Parametric reference intervals (requires Gaussian distribution)
- Robust reference intervals using Horn's method / CLSI C28-A3 Appendix B
- Outlier detection: Tukey (Horn) and Dixon/Reed
- Normality testing: Shapiro-Wilk and Anderson-Darling
- Box-Cox transformation
- 90% confidence intervals (parametric, nonparametric rank-based, bootstrap)
"""

import warnings
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReferenceIntervalResult:
    """Result of a reference interval calculation for a single analyte."""
    analyte: str
    n_total: int
    n_outliers: int
    n_used: int
    outlier_values: list = field(default_factory=list)
    outlier_indices: list = field(default_factory=list)
    method_ri: str = ""
    method_ci: str = ""
    lower_limit: float = np.nan
    upper_limit: float = np.nan
    lower_ci_low: float = np.nan
    lower_ci_high: float = np.nan
    upper_ci_low: float = np.nan
    upper_ci_high: float = np.nan
    ref_conf: float = 0.95
    limit_conf: float = 0.90
    mean: float = np.nan
    median: float = np.nan
    std: float = np.nan
    min_val: float = np.nan
    max_val: float = np.nan
    shapiro_p: float = np.nan
    anderson_stat: float = np.nan
    anderson_critical: float = np.nan
    is_normal: Optional[bool] = None
    normality_test: str = ""
    boxcox_lambda: Optional[float] = None
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Box-Cox transformation
# ---------------------------------------------------------------------------

def boxcox_transform(data):
    """Apply Box-Cox transformation to data, finding optimal lambda via MLE.

    Parameters
    ----------
    data : array-like
        Must be strictly positive.

    Returns
    -------
    transformed : ndarray
        Box-Cox transformed data.
    lmbda : float
        Optimal lambda value.
    """
    data = np.asarray(data, dtype=float)
    if np.any(data <= 0):
        shift = np.abs(data.min()) + 1.0
        data = data + shift
    else:
        shift = 0.0
    transformed, lmbda = stats.boxcox(data)
    return transformed, lmbda, shift


def boxcox_inverse(y, lmbda, shift=0.0):
    """Inverse Box-Cox transformation."""
    if np.abs(lmbda) < 1e-10:
        x = np.exp(y)
    else:
        x = inv_boxcox(y, lmbda)
    return x - shift


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def horn_outliers(data):
    """Detect outliers using Horn's method (Box-Cox + Tukey IQR fences).

    Following the algorithm in the R referenceIntervals package:
    1. Box-Cox transform the data.
    2. Compute Q1, Q3, IQR on transformed data.
    3. Flag points outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Parameters
    ----------
    data : array-like

    Returns
    -------
    outlier_mask : ndarray of bool
        True for outlier positions.
    """
    data = np.asarray(data, dtype=float)
    if len(data) < 4:
        return np.zeros(len(data), dtype=bool)
    try:
        transformed, _, _ = boxcox_transform(data)
    except Exception:
        transformed = data.copy()

    q1 = np.percentile(transformed, 25)
    q3 = np.percentile(transformed, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_mask = (transformed < lower_fence) | (transformed > upper_fence)
    return outlier_mask


def tukey_outliers(data):
    """Detect outliers using Tukey's fences on untransformed data.

    Parameters
    ----------
    data : array-like

    Returns
    -------
    outlier_mask : ndarray of bool
    """
    data = np.asarray(data, dtype=float)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    return (data < lower_fence) | (data > upper_fence)


def dixon_outliers(data):
    """Detect outliers using the Dixon/Reed D/R method.

    Tests both the minimum and maximum values.
    D/R >= 1/3 indicates an outlier.

    Parameters
    ----------
    data : array-like

    Returns
    -------
    outlier_mask : ndarray of bool
    """
    data = np.asarray(data, dtype=float)
    if len(data) < 3:
        return np.zeros(len(data), dtype=bool)

    sorted_data = np.sort(data)
    n = len(sorted_data)
    outlier_mask = np.zeros(len(data), dtype=bool)
    data_range = sorted_data[-1] - sorted_data[0]

    if data_range == 0:
        return outlier_mask

    # Test minimum
    d_low = sorted_data[1] - sorted_data[0]
    if d_low / data_range >= 1.0 / 3.0:
        idx = np.where(data == sorted_data[0])[0]
        if len(idx) > 0:
            outlier_mask[idx[0]] = True

    # Test maximum
    d_high = sorted_data[-1] - sorted_data[-2]
    if d_high / data_range >= 1.0 / 3.0:
        idx = np.where(data == sorted_data[-1])[0]
        if len(idx) > 0:
            outlier_mask[idx[0]] = True

    return outlier_mask


# ---------------------------------------------------------------------------
# Normality testing
# ---------------------------------------------------------------------------

def check_normality(data, alpha=0.05):
    """Test normality using Shapiro-Wilk and Anderson-Darling tests.

    Parameters
    ----------
    data : array-like
    alpha : float
        Significance level.

    Returns
    -------
    dict with keys:
        shapiro_stat, shapiro_p, anderson_stat, anderson_critical,
        is_normal, test_used
    """
    data = np.asarray(data, dtype=float)
    result = {
        "shapiro_stat": np.nan,
        "shapiro_p": np.nan,
        "anderson_stat": np.nan,
        "anderson_critical": np.nan,
        "is_normal": None,
        "test_used": "",
    }

    n = len(data)
    if n < 3:
        result["is_normal"] = None
        result["test_used"] = "insufficient data"
        return result

    # Shapiro-Wilk (valid for 3 <= n <= 5000)
    if n <= 5000:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sw_stat, sw_p = stats.shapiro(data)
        result["shapiro_stat"] = sw_stat
        result["shapiro_p"] = sw_p

    # Anderson-Darling
    ad_result = stats.anderson(data, dist="norm")
    result["anderson_stat"] = ad_result.statistic
    # Use the critical value at 5% significance
    idx_5pct = list(ad_result.significance_level).index(5)
    result["anderson_critical"] = ad_result.critical_values[idx_5pct]

    # Decision: prefer Shapiro-Wilk when available
    if not np.isnan(result["shapiro_p"]):
        result["is_normal"] = result["shapiro_p"] >= alpha
        result["test_used"] = "Shapiro-Wilk"
    else:
        result["is_normal"] = result["anderson_stat"] < result["anderson_critical"]
        result["test_used"] = "Anderson-Darling"

    return result


# ---------------------------------------------------------------------------
# Nonparametric reference interval
# ---------------------------------------------------------------------------

def nonparametric_ri(data, ref_conf=0.95):
    """Compute nonparametric reference interval.

    Uses rank-based percentile method. Linearly interpolates when ranks
    are non-integer.

    Parameters
    ----------
    data : array-like
    ref_conf : float
        Coverage probability (default 0.95 for 95% RI).

    Returns
    -------
    lower, upper : float
    """
    data = np.sort(np.asarray(data, dtype=float))
    n = len(data)
    lower_p = (1 - ref_conf) / 2
    upper_p = (1 + ref_conf) / 2
    lower = np.percentile(data, lower_p * 100)
    upper = np.percentile(data, upper_p * 100)
    return lower, upper


def nonparametric_ci(data, ref_conf=0.95, limit_conf=0.90):
    """Compute nonparametric confidence intervals for reference limits.

    Uses the rank-based method. For n >= 120 uses exact integer ranks;
    otherwise falls back to bootstrap.

    Parameters
    ----------
    data : array-like
    ref_conf : float
    limit_conf : float

    Returns
    -------
    lower_ci_low, lower_ci_high, upper_ci_low, upper_ci_high : float
    """
    data = np.sort(np.asarray(data, dtype=float))
    n = len(data)

    if n < 120:
        return bootstrap_ci(data, method="nonparametric", ref_conf=ref_conf,
                            limit_conf=limit_conf)

    alpha = 1 - limit_conf
    p_low = (1 - ref_conf) / 2
    p_high = (1 + ref_conf) / 2

    # Use the binomial distribution to find rank-based CI
    def _rank_ci(p, n, alpha):
        """Find ranks r1, r2 such that P(X_{r1} <= x_p <= X_{r2}) >= 1-alpha."""
        from scipy.stats import binom
        # Lower rank
        r1 = 1
        for j in range(1, n + 1):
            if binom.cdf(j - 1, n, p) <= alpha / 2:
                r1 = j
            else:
                break
        # Upper rank
        r2 = n
        for j in range(n, 0, -1):
            if 1 - binom.cdf(j - 1, n, p) <= alpha / 2:
                r2 = j
            else:
                break
        return max(r1, 1), min(r2, n)

    r1_low, r2_low = _rank_ci(p_low, n, alpha)
    r1_high, r2_high = _rank_ci(p_high, n, alpha)

    # Convert to 0-based index
    lower_ci_low = data[max(r1_low - 1, 0)]
    lower_ci_high = data[min(r2_low - 1, n - 1)]
    upper_ci_low = data[max(r1_high - 1, 0)]
    upper_ci_high = data[min(r2_high - 1, n - 1)]

    return lower_ci_low, lower_ci_high, upper_ci_low, upper_ci_high


# ---------------------------------------------------------------------------
# Parametric reference interval
# ---------------------------------------------------------------------------

def parametric_ri(data, ref_conf=0.95):
    """Compute parametric reference interval (mean +/- z * sd).

    Parameters
    ----------
    data : array-like
    ref_conf : float

    Returns
    -------
    lower, upper : float
    """
    data = np.asarray(data, dtype=float)
    m = np.mean(data)
    s = np.std(data, ddof=1)
    z = stats.norm.ppf((1 + ref_conf) / 2)
    return m - z * s, m + z * s


def parametric_ci(data, ref_conf=0.95, limit_conf=0.90):
    """Compute parametric confidence intervals for reference limits.

    SE = s * sqrt(1/n + z^2 / (2*(n-1)))
    CI = limit +/- t * SE

    Parameters
    ----------
    data : array-like
    ref_conf : float
    limit_conf : float

    Returns
    -------
    lower_ci_low, lower_ci_high, upper_ci_low, upper_ci_high : float
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    m = np.mean(data)
    s = np.std(data, ddof=1)
    z = stats.norm.ppf((1 + ref_conf) / 2)
    se = s * np.sqrt(1.0 / n + z ** 2 / (2.0 * (n - 1)))
    t_val = stats.t.ppf((1 + limit_conf) / 2, n - 1)

    lower_limit = m - z * s
    upper_limit = m + z * s

    lower_ci_low = lower_limit - t_val * se
    lower_ci_high = lower_limit + t_val * se
    upper_ci_low = upper_limit - t_val * se
    upper_ci_high = upper_limit + t_val * se

    return lower_ci_low, lower_ci_high, upper_ci_low, upper_ci_high


# ---------------------------------------------------------------------------
# Robust reference interval (Horn / CLSI C28-A3 Appendix B)
# ---------------------------------------------------------------------------

def _biweight(data, ref_conf=0.95, max_iter=50, tol=1e-6):
    """Core biweight estimator (CLSI C28-A3 Appendix B).

    Computes the iterative Tukey biweight location and scale estimates,
    then derives the reference interval as t_bi +/- z * s_bi.

    Parameters
    ----------
    data : array-like
    ref_conf : float
    max_iter : int
    tol : float

    Returns
    -------
    lower, upper, t_bi, s_bi : float
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    # Initial estimates
    t_bi = np.median(data)
    mad = np.median(np.abs(data - t_bi))

    if mad < 1e-10:
        # Degenerate case: all values essentially identical
        s = np.std(data, ddof=1)
        z = stats.norm.ppf((1 + ref_conf) / 2)
        return t_bi - z * s, t_bi + z * s, t_bi, s

    # Iterative biweight location
    for _ in range(max_iter):
        u = (data - t_bi) / (3.7 * mad)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        w_sum = np.sum(w)
        if w_sum == 0:
            break
        t_bi_new = t_bi + np.sum(w * (data - t_bi)) / w_sum
        if np.abs(t_bi_new - t_bi) < tol:
            t_bi = t_bi_new
            break
        t_bi = t_bi_new

    # Biweight midvariance (scale)
    u = (data - t_bi) / (205.6 * mad)
    mask = np.abs(u) < 1
    if np.sum(mask) < 2:
        s_bi = np.std(data, ddof=1)
    else:
        numer = np.sum(((data[mask] - t_bi) ** 2) * (1 - u[mask] ** 2) ** 4)
        denom = np.sum((1 - u[mask] ** 2) * (1 - 5 * u[mask] ** 2))
        denom = np.abs(denom)
        if denom < 1e-10:
            s_bi = np.std(data, ddof=1)
        else:
            s_bi = np.sqrt(n * numer) / denom

    z = stats.norm.ppf((1 + ref_conf) / 2)
    lower = t_bi - z * s_bi
    upper = t_bi + z * s_bi
    return lower, upper, t_bi, s_bi


def robust_ri(data, ref_conf=0.95, max_iter=50, tol=1e-6, transform=True):
    """Compute robust reference interval using the CLSI biweight method.

    Implements the iterative Tukey biweight algorithm from CLSI C28-A3
    Appendix B.  When *transform* is True (default) and the data are
    non-normal, a Box-Cox transformation is applied first to achieve
    symmetry before computing the biweight — this is the **transformed
    robust** variant described by Horn, Pesce & Copeland (1998).  The
    back-transformation naturally constrains limits to the positive
    domain, preventing biologically implausible negative lower limits.

    If the data are already symmetric (Shapiro-Wilk p >= 0.05) or
    Box-Cox fails, the plain (untransformed) biweight is used instead.

    Parameters
    ----------
    data : array-like
    ref_conf : float
    max_iter : int
    tol : float
    transform : bool
        If True, apply Box-Cox before biweight for non-normal data
        (Horn 1998 "transformed robust").  If False, always use the
        plain CLSI C28-A3 biweight on native data.

    Returns
    -------
    lower, upper : float
    t_bi : float
        Robust location estimate (on the original scale for transformed
        robust this is back-transformed from the median of the biweight).
    s_bi : float
        Robust scale estimate (original-scale approximation).
    boxcox_lambda : float or None
        Lambda used for Box-Cox, or None if no transformation was applied.
    """
    data = np.asarray(data, dtype=float)

    used_transform = False
    lmbda = None
    shift = 0.0

    if transform:
        # Only transform when data appear non-normal (skewed)
        norm = check_normality(data, alpha=0.05)
        if not norm["is_normal"]:
            try:
                transformed, lmbda, shift = boxcox_transform(data)
                lower_t, upper_t, t_bi_t, s_bi_t = _biweight(
                    transformed, ref_conf, max_iter, tol,
                )
                # Back-transform limits to original scale
                lower = boxcox_inverse(lower_t, lmbda, shift)
                upper = boxcox_inverse(upper_t, lmbda, shift)
                t_bi = boxcox_inverse(t_bi_t, lmbda, shift)
                s_bi = s_bi_t  # scale on transformed domain
                used_transform = True
            except Exception:
                pass  # fall through to plain biweight

    if not used_transform:
        lower, upper, t_bi, s_bi = _biweight(data, ref_conf, max_iter, tol)

    return lower, upper, t_bi, s_bi, lmbda


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(data, method="nonparametric", ref_conf=0.95, limit_conf=0.90,
                 n_boot=5000, seed=42):
    """Compute bootstrap confidence intervals for reference limits.

    Parameters
    ----------
    data : array-like
    method : str
        "nonparametric", "parametric", or "robust"
    ref_conf : float
    limit_conf : float
    n_boot : int
    seed : int

    Returns
    -------
    lower_ci_low, lower_ci_high, upper_ci_low, upper_ci_high : float
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data, dtype=float)
    n = len(data)
    boot_lower = np.empty(n_boot)
    boot_upper = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        if method == "nonparametric":
            lo, hi = nonparametric_ri(sample, ref_conf)
        elif method == "parametric":
            lo, hi = parametric_ri(sample, ref_conf)
        elif method == "robust":
            lo, hi, _, _, _ = robust_ri(sample, ref_conf, transform=False)
        else:
            lo, hi = nonparametric_ri(sample, ref_conf)
        boot_lower[i] = lo
        boot_upper[i] = hi

    alpha = (1 - limit_conf) / 2
    lower_ci_low = np.percentile(boot_lower, alpha * 100)
    lower_ci_high = np.percentile(boot_lower, (1 - alpha) * 100)
    upper_ci_low = np.percentile(boot_upper, alpha * 100)
    upper_ci_high = np.percentile(boot_upper, (1 - alpha) * 100)

    return lower_ci_low, lower_ci_high, upper_ci_low, upper_ci_high


# ---------------------------------------------------------------------------
# Le Boedec reference interval (2016 / 2019 adjusted strategy)
# ---------------------------------------------------------------------------

def le_boedec_ri(data, ref_conf=0.95):
    """Compute reference interval using the Le Boedec (2019) strategy.

    Decision algorithm based on Le Boedec K, Vet Clin Pathol 2019;48:335-346:
      - n >= 120: nonparametric
      - 40 <= n < 120: Shapiro-Wilk conditional with raised threshold P > 0.2
          - If SW p > 0.2 (data appears Gaussian): parametric
          - If SW p <= 0.2 (non-Gaussian):
              - Lower limit: nonparametric
              - Upper limit at n <= 40: Box-Cox + parametric
              - Upper limit at n > 40: nonparametric
      - n < 40: nonparametric for both limits

    The raised P > 0.2 threshold (instead of 0.05) is based on Le Boedec K,
    Vet Clin Pathol 2016;45:648-656, which showed that the standard alpha=0.05
    has poor specificity (~50%) at small sample sizes, leading to erroneous
    application of parametric methods to non-Gaussian data.

    Parameters
    ----------
    data : array-like
    ref_conf : float

    Returns
    -------
    lower : float
    upper : float
    method_description : str
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    # n >= 120: standard nonparametric (same as ASVCP)
    if n >= 120:
        lower, upper = nonparametric_ri(data, ref_conf)
        return lower, upper, "Nonparametric (n >= 120)"

    # n < 40: nonparametric for both limits
    if n < 40:
        lower, upper = nonparametric_ri(data, ref_conf)
        return lower, upper, "Nonparametric (n < 40, Le Boedec)"

    # 40 <= n < 120: Shapiro-Wilk conditional with P > 0.2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, sw_p = stats.shapiro(data)

    if sw_p > 0.2:
        # Data appears Gaussian with the raised threshold
        lower, upper = parametric_ri(data, ref_conf)
        return lower, upper, f"Parametric (Shapiro-Wilk p={sw_p:.3f} > 0.2)"

    # Non-Gaussian: lower limit is always nonparametric
    lower, _ = nonparametric_ri(data, ref_conf)

    if n <= 40:
        # Special case: Box-Cox + parametric for upper limit at n ~ 40
        try:
            transformed, lmbda, shift = boxcox_transform(data)
            _, t_upper = parametric_ri(transformed, ref_conf)
            upper = boxcox_inverse(t_upper, lmbda, shift)
            method = (f"Lower: nonparametric; Upper: Box-Cox parametric "
                      f"(n={n}, SW p={sw_p:.3f} <= 0.2)")
        except Exception:
            _, upper = nonparametric_ri(data, ref_conf)
            method = (f"Nonparametric (Box-Cox failed, n={n}, "
                      f"SW p={sw_p:.3f} <= 0.2)")
    else:
        _, upper = nonparametric_ri(data, ref_conf)
        method = f"Nonparametric (SW p={sw_p:.3f} <= 0.2)"

    return lower, upper, method


# ---------------------------------------------------------------------------
# Main reference interval calculation
# ---------------------------------------------------------------------------

def calculate_reference_interval(
    data,
    analyte_name="analyte",
    outlier_method="horn",
    remove_outliers=True,
    ri_method="auto",
    ci_method="auto",
    ref_conf=0.95,
    limit_conf=0.90,
    n_boot=5000,
):
    """Calculate a reference interval for a single analyte.

    Parameters
    ----------
    data : array-like
        Numeric values for one analyte.
    analyte_name : str
        Name of the analyte.
    outlier_method : str
        "horn" (Box-Cox + Tukey), "tukey" (Tukey without transform),
        "dixon" (Dixon/Reed D/R), or "none".
    remove_outliers : bool
        Whether to remove detected outliers before RI calculation.
    ri_method : str
        "auto", "nonparametric", "parametric", or "robust".
        "auto" selects based on sample size and normality.
    ci_method : str
        "auto", "parametric", "nonparametric", or "bootstrap".
    ref_conf : float
        Reference interval coverage (default 0.95).
    limit_conf : float
        Confidence level for CI around reference limits (default 0.90).
    n_boot : int
        Number of bootstrap resamples.

    Returns
    -------
    ReferenceIntervalResult
    """
    raw = np.asarray(data, dtype=float)
    # Remove NaN/Inf
    valid_mask = np.isfinite(raw)
    clean = raw[valid_mask]
    n_total = len(clean)

    result = ReferenceIntervalResult(
        analyte=analyte_name,
        n_total=n_total,
        n_outliers=0,
        n_used=n_total,
        ref_conf=ref_conf,
        limit_conf=limit_conf,
    )

    if n_total < 10:
        result.warnings.append(
            f"Insufficient sample size (n={n_total}). Minimum 10 required."
        )
        return result

    # --- Outlier detection ---
    outlier_mask = np.zeros(len(clean), dtype=bool)
    if outlier_method == "horn":
        outlier_mask = horn_outliers(clean)
    elif outlier_method == "tukey":
        outlier_mask = tukey_outliers(clean)
    elif outlier_method == "dixon":
        outlier_mask = dixon_outliers(clean)

    result.n_outliers = int(np.sum(outlier_mask))
    result.outlier_values = clean[outlier_mask].tolist()
    result.outlier_indices = np.where(outlier_mask)[0].tolist()

    if remove_outliers and result.n_outliers > 0:
        analysis_data = clean[~outlier_mask]
    else:
        analysis_data = clean.copy()

    result.n_used = len(analysis_data)

    if result.n_used < 10:
        result.warnings.append(
            f"After outlier removal, only {result.n_used} values remain."
        )
        return result

    # --- Descriptive statistics ---
    result.mean = float(np.mean(analysis_data))
    result.median = float(np.median(analysis_data))
    result.std = float(np.std(analysis_data, ddof=1))
    result.min_val = float(np.min(analysis_data))
    result.max_val = float(np.max(analysis_data))

    # --- Normality testing ---
    norm_result = check_normality(analysis_data)
    result.shapiro_p = norm_result["shapiro_p"]
    result.anderson_stat = norm_result["anderson_stat"]
    result.anderson_critical = norm_result["anderson_critical"]
    result.is_normal = norm_result["is_normal"]
    result.normality_test = norm_result["test_used"]

    # --- Select RI method ---
    n = result.n_used
    if ri_method == "le_boedec":
        chosen_ri = "le_boedec"
    elif ri_method == "auto":
        if n >= 120:
            chosen_ri = "nonparametric"
        elif n >= 20:
            chosen_ri = "robust"
        else:
            chosen_ri = "parametric"

        # If the user has enough data for nonparametric but data is normal,
        # nonparametric is still preferred per ASVCP guidelines
    else:
        chosen_ri = ri_method

    # Additional warnings per ASVCP guidelines
    if chosen_ri != "le_boedec":
        if n < 40:
            result.warnings.append(
                f"Sample size (n={n}) is below 40. ASVCP recommends n >= 40 "
                f"for robust methods. Results should be treated with caution."
            )
        elif n < 120 and chosen_ri == "nonparametric":
            result.warnings.append(
                f"Sample size (n={n}) is below 120 recommended for nonparametric "
                f"method. Consider using robust method instead."
            )
            if ri_method == "auto":
                chosen_ri = "robust"

    # --- Compute reference interval ---
    if chosen_ri == "le_boedec":
        lower, upper, method_desc = le_boedec_ri(analysis_data, ref_conf)
        result.method_ri = f"Le Boedec: {method_desc}"
    elif chosen_ri == "nonparametric":
        lower, upper = nonparametric_ri(analysis_data, ref_conf)
        result.method_ri = "Nonparametric (rank-based percentile)"
    elif chosen_ri == "parametric":
        if result.is_normal:
            lower, upper = parametric_ri(analysis_data, ref_conf)
            result.method_ri = "Parametric (Gaussian)"
        else:
            # Try Box-Cox transformation
            try:
                transformed, lmbda, shift = boxcox_transform(analysis_data)
                norm_t = check_normality(transformed)
                if norm_t["is_normal"]:
                    t_lower, t_upper = parametric_ri(transformed, ref_conf)
                    lower = boxcox_inverse(t_lower, lmbda, shift)
                    upper = boxcox_inverse(t_upper, lmbda, shift)
                    result.boxcox_lambda = float(lmbda)
                    result.method_ri = (
                        f"Parametric after Box-Cox transformation "
                        f"(lambda={lmbda:.3f})"
                    )
                else:
                    lower, upper = parametric_ri(analysis_data, ref_conf)
                    result.method_ri = "Parametric (non-normal data, use with caution)"
                    result.warnings.append(
                        "Data not normal even after Box-Cox transformation. "
                        "Consider nonparametric or robust method."
                    )
            except Exception:
                lower, upper = parametric_ri(analysis_data, ref_conf)
                result.method_ri = "Parametric (Gaussian assumed)"
    elif chosen_ri == "robust":
        lower, upper, t_bi, s_bi, bc_lmbda = robust_ri(analysis_data, ref_conf)
        if bc_lmbda is not None:
            result.boxcox_lambda = float(bc_lmbda)
            result.method_ri = (
                f"Robust transformed (Box-Cox lambda={bc_lmbda:.3f} + "
                f"CLSI biweight)"
            )
        else:
            result.method_ri = "Robust (CLSI C28-A3 biweight)"
    else:
        lower, upper = nonparametric_ri(analysis_data, ref_conf)
        result.method_ri = "Nonparametric (rank-based percentile)"

    result.lower_limit = float(lower)
    result.upper_limit = float(upper)

    # --- Compute confidence intervals ---
    if ci_method == "auto":
        if chosen_ri == "le_boedec":
            # Le Boedec mixes methods; bootstrap is safest
            if n >= 120:
                chosen_ci = "nonparametric"
            else:
                chosen_ci = "bootstrap"
        elif chosen_ri == "nonparametric" and n >= 120:
            chosen_ci = "nonparametric"
        elif chosen_ri == "parametric" and result.is_normal:
            chosen_ci = "parametric"
        else:
            chosen_ci = "bootstrap"
    else:
        chosen_ci = ci_method

    try:
        if chosen_ci == "nonparametric":
            ci_vals = nonparametric_ci(analysis_data, ref_conf, limit_conf)
            result.method_ci = "Nonparametric (rank-based)"
        elif chosen_ci == "parametric":
            ci_vals = parametric_ci(analysis_data, ref_conf, limit_conf)
            result.method_ci = "Parametric"
        else:
            ci_vals = bootstrap_ci(
                analysis_data, method=chosen_ri, ref_conf=ref_conf,
                limit_conf=limit_conf, n_boot=n_boot
            )
            result.method_ci = f"Bootstrap (n={n_boot})"

        result.lower_ci_low = float(ci_vals[0])
        result.lower_ci_high = float(ci_vals[1])
        result.upper_ci_low = float(ci_vals[2])
        result.upper_ci_high = float(ci_vals[3])
    except Exception as e:
        result.warnings.append(f"CI computation failed: {e}")

    return result
