"""
temporal/trend.py
Trend detection and short-term forecasting.

Mann-Kendall test: non-parametric test for monotonic trends in time series.
  - No assumption of normality (important for skewed surveillance data)
  - Standard in environmental/epidemiological monitoring
  - Returns: trend direction, tau (strength), p-value

Exponential smoothing: simple forecast for next expected risk score.
  - Single (Holt's) exponential smoothing with trend component
  - Projects 1–4 weeks ahead with confidence intervals

Sen's slope: robust estimate of the rate of change per time step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TrendResult:
    site: str
    trend: str              # "increasing" | "decreasing" | "stable"
    tau: float              # Kendall's tau [-1, 1]; strength of trend
    p_value: float          # significance of trend
    sens_slope: float       # rate of change per time step (Sen's slope)
    is_significant: bool    # p < 0.05
    forecast_next: float    # predicted score for next time step
    forecast_ci_low: float  # 80% CI lower bound
    forecast_ci_high: float # 80% CI upper bound
    n_points: int
    summary: str            # human-readable one-liner


def mann_kendall(scores: list[float]) -> tuple[float, float, float]:
    """
    Compute Mann-Kendall trend test.
    Returns (tau, p_value, sens_slope).
    """
    n = len(scores)
    if n < 4:
        return 0.0, 1.0, 0.0

    arr = np.array(scores)

    # Mann-Kendall S statistic
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = arr[j] - arr[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Variance of S
    var_s = n * (n - 1) * (2 * n + 5) / 18.0

    # Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    tau = s / (n * (n - 1) / 2)

    # Sen's slope (median of all pairwise slopes)
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if j != i:
                slopes.append((arr[j] - arr[i]) / (j - i))
    sens_slope = float(np.median(slopes)) if slopes else 0.0

    return float(tau), p_value, sens_slope


def exponential_smoothing_forecast(
    scores: list[float],
    alpha: float = 0.3,
    beta: float = 0.1,
    steps_ahead: int = 1,
) -> tuple[float, float, float]:
    """
    Holt's double exponential smoothing (level + trend).
    Returns (forecast, ci_low, ci_high) for `steps_ahead` steps.
    alpha: smoothing factor for level (0–1)
    beta: smoothing factor for trend (0–1)
    """
    if len(scores) < 2:
        val = scores[0] if scores else 0.0
        return val, max(0.0, val - 0.1), min(1.0, val + 0.1)

    arr = np.array(scores, dtype=float)

    # Initialize
    level = arr[0]
    trend = arr[1] - arr[0]

    residuals = []
    for t in range(1, len(arr)):
        prev_level = level
        level = alpha * arr[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        residuals.append(float(arr[t] - (prev_level + trend)))

    # Forecast
    forecast = float(level + steps_ahead * trend)
    forecast = float(np.clip(forecast, 0.0, 1.0))

    # 80% CI from residual std
    res_std = float(np.std(residuals)) if residuals else 0.1
    z80 = 1.282
    ci_low = float(np.clip(forecast - z80 * res_std * np.sqrt(steps_ahead), 0.0, 1.0))
    ci_high = float(np.clip(forecast + z80 * res_std * np.sqrt(steps_ahead), 0.0, 1.0))

    return forecast, ci_low, ci_high


def analyze_trend(
    site: str,
    scores: list[float],
    alpha_threshold: float = 0.05,
    tau_threshold: float = 0.3,
) -> TrendResult:
    """
    Full trend analysis for a site's risk score series.
    """
    n = len(scores)

    if n < 3:
        return TrendResult(
            site=site, trend="stable", tau=0.0, p_value=1.0,
            sens_slope=0.0, is_significant=False,
            forecast_next=scores[-1] if scores else 0.0,
            forecast_ci_low=0.0, forecast_ci_high=1.0,
            n_points=n, summary="Insufficient history for trend analysis.",
        )

    tau, p_value, sens_slope = mann_kendall(scores)
    forecast, ci_low, ci_high = exponential_smoothing_forecast(scores)

    significant = p_value < alpha_threshold
    if significant and tau >= tau_threshold:
        trend = "increasing"
    elif significant and tau <= -tau_threshold:
        trend = "decreasing"
    else:
        trend = "stable"

    # Human-readable summary
    slope_per_week = abs(sens_slope)
    if trend == "increasing" and significant:
        weeks_to_high = None
        current = scores[-1]
        if sens_slope > 0 and current < 0.6:
            weeks_to_high = int((0.6 - current) / sens_slope) if sens_slope > 0 else None
        time_str = f" — projected HIGH in ~{weeks_to_high} weeks" if weeks_to_high and weeks_to_high < 12 else ""
        summary = (
            f"Significant upward trend (τ={tau:.2f}, p={p_value:.3f}, "
            f"+{slope_per_week:.3f}/run){time_str}. Forecast: {forecast:.3f} [{ci_low:.3f}–{ci_high:.3f}]"
        )
    elif trend == "decreasing" and significant:
        summary = (
            f"Significant downward trend (τ={tau:.2f}, p={p_value:.3f}, "
            f"-{slope_per_week:.3f}/run). Forecast: {forecast:.3f} [{ci_low:.3f}–{ci_high:.3f}]"
        )
    else:
        summary = (
            f"No significant trend (τ={tau:.2f}, p={p_value:.3f}). "
            f"Forecast: {forecast:.3f} [{ci_low:.3f}–{ci_high:.3f}]"
        )

    return TrendResult(
        site=site,
        trend=trend,
        tau=round(tau, 4),
        p_value=round(p_value, 4),
        sens_slope=round(sens_slope, 5),
        is_significant=significant,
        forecast_next=round(forecast, 4),
        forecast_ci_low=round(ci_low, 4),
        forecast_ci_high=round(ci_high, 4),
        n_points=n,
        summary=summary,
    )


def analyze_trends_all_sites(
    store,
    current_risk_scores: list,
    window: int = 16,
) -> dict[str, TrendResult]:
    """Analyze trends for all sites, pulling history from the store."""
    results = {}
    for rs in current_risk_scores:
        history = store.get_site_history(rs.sample_name, last_n=window)
        past = [h["risk_score"] for h in history]
        full_series = past + [rs.score]
        results[rs.sample_name] = analyze_trend(
            site=rs.sample_name,
            scores=full_series,
        )
    return results
