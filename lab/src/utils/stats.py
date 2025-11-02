"""Statistical utilities for multi-seed aggregation."""

from math import sqrt

# Two-tailed 95% t critical values (df -> t) for n <= 30.
# See: https://en.wikipedia.org/wiki/Student%27s_t-distribution#Table_of_selected_values
_T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _critical_value(df):
    """Return the 95% critical value for df degrees of freedom."""
    if df <= 0:
        return 0.0
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    # Normal approximation once we are beyond the table.
    return 1.96


def mean_ci(xs, alpha=0.05):
    """Calculate mean and 95% confidence interval.

    Args:
        xs: List of numeric values.
        alpha: Significance level (unused; 0.05 fixed for now).

    Returns:
        (mean, (lower_bound, upper_bound)) or (mean, None) if n < 2.
    """
    n = len(xs)
    if n == 0:
        return 0.0, (0.0, 0.0)

    m = sum(xs) / n
    if n < 2:
        return m, None  # Cannot compute CI with a single sample.

    # Sample variance with Bessel correction.
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = sqrt(var / n)

    t = _critical_value(n - 1)
    margin = t * se

    return m, (m - margin, m + margin)
