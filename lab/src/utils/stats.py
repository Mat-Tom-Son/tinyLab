"""Statistical utilities for multi-seed aggregation."""


def mean_ci(xs, alpha=0.05):
    """Calculate mean and 95% confidence interval.

    Args:
        xs: List of numeric values
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (mean, (lower_bound, upper_bound)) or (mean, None) if n < 2
    """
    n = len(xs)
    if n == 0:
        return 0.0, (0.0, 0.0)

    m = sum(xs) / n
    if n < 2:
        return m, None  # Cannot compute CI

    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = (var / n) ** 0.5

    # Use 1.96 (z-score for 95% CI)
    # TODO: Use t-distribution for small n if more rigor needed
    z = 1.96

    return m, (m - z * se, m + z * se)
