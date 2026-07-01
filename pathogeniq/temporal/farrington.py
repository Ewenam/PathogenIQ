"""
temporal/farrington.py
Farrington-style aberration detection — the public-health standard for flagging
a current observation above its historical baseline.

Implements the core of Farrington et al. (1996) / Farrington-flexible (Noufaily
et al. 2013) as used by ECDC/UKHSA and the R `surveillance` package:

  1. Fit a log-linear quasi-Poisson trend to the historical baseline.
  2. Estimate overdispersion (phi) from Pearson residuals.
  3. Down-weight past outliers via Anscombe residuals and refit once, so a
     previous outbreak doesn't inflate the expected baseline.
  4. Predict the current period's expected value + a one-sided upper threshold
     using the 2/3-power (variance-stabilizing) transformation.
  5. Report an exceedance z and an alarm (observed > threshold).

Seasonal reference-window selection from prior years is intentionally omitted:
wastewater series here are sub-annual, so we use a trailing baseline + trend
(the honest choice given the data). Everything else follows the method.

Works on counts (Poisson-appropriate). For a continuous signal (e.g. a risk
score in [0,1]) pass `scale` to convert to pseudo-counts.
"""
from __future__ import annotations

import warnings

import numpy as np


def _squash(z: float, k: float = 2.0) -> float:
    z = max(0.0, z)
    return z / (z + k)


def farrington_detect(
    history: list[float],
    current: float,
    alpha: float = 0.05,
    reweight: bool = True,
    scale: float = 1.0,
    min_history: int = 4,
) -> dict:
    """Test whether `current` is an aberration above the baseline `history`.

    Returns a dict: expected, threshold, dispersion, trend_slope, z, exceedance
    (0-1), alarm (bool), note.
    """
    import statsmodels.api as sm
    from scipy import stats

    y = np.asarray(history, dtype=float) * scale
    y0 = float(current) * scale
    n = len(y)
    if n < min_history or np.all(y == y[0]):
        # too little history, or a perfectly flat baseline (no dispersion to fit)
        base = float(np.mean(y)) if n else 0.0
        thr = base * 1.5 + 1e-6
        return {"expected": round(base / scale, 6), "threshold": round(thr / scale, 6),
                "dispersion": None, "trend_slope": None,
                "z": 0.0, "exceedance": 0.0, "alarm": y0 > thr,
                "note": "insufficient/flat history — fallback threshold"}

    t = np.arange(n, dtype=float)
    X = sm.add_constant(t)

    def _fit(var_weights=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.GLM(y, X, family=sm.families.Poisson(), var_weights=var_weights)
            return m.fit()

    try:
        res = _fit()
        phi = max(1.0, float(res.pearson_chi2 / max(res.df_resid, 1)))
        if reweight:
            mu = np.clip(res.mu, 1e-6, None)
            anscombe = 1.5 * (y ** (2/3) - mu ** (2/3)) / (mu ** (1/6) + 1e-9)
            anscombe /= np.sqrt(phi)
            gamma = 2.58
            w = np.where(np.abs(anscombe) > gamma, (gamma / np.abs(anscombe)) ** 2, 1.0)
            res = _fit(var_weights=w)
            phi = max(1.0, float(res.pearson_chi2 / max(res.df_resid, 1)))
        x0 = np.array([1.0, float(n)])
        eta0 = float(x0 @ res.params)
        mu0 = float(np.exp(eta0))
        var_eta = float(x0 @ res.cov_params() @ x0)
        slope = float(res.params[1])
    except Exception:
        # robust fallback: log-linear OLS on log(y+0.5)
        b = np.polyfit(t, np.log(y + 0.5), 1)
        mu0 = float(np.exp(np.polyval(b, n)))
        phi = max(1.0, float(np.var(y) / (np.mean(y) + 1e-9)))
        var_eta = 0.0
        slope = float(b[0])

    # total predictive variance: overdispersion + estimation uncertainty
    V = phi * mu0 + (mu0 ** 2) * var_eta
    z = stats.norm.ppf(1 - alpha)
    # 2/3-power (variance-stabilizing) transform → symmetric, then back-transform
    w0 = mu0 ** (2/3)
    sd_w = (2/3) * mu0 ** (-1/3) * np.sqrt(max(V, 1e-12))
    U = float((w0 + z * sd_w) ** (3/2))
    z_obs = float((y0 ** (2/3) - w0) / (sd_w + 1e-12))

    return {
        "expected": round(mu0 / scale, 6),
        "threshold": round(U / scale, 6),
        "dispersion": round(phi, 4),
        "trend_slope": round(slope, 5),
        "z": round(z_obs, 4),
        "exceedance": round(_squash(z_obs), 4),
        "alarm": bool(y0 > U),
        "note": "ok",
    }


def run_farrington_all_sites(store, current_risk_scores: list, window: int = 16,
                             alpha: float = 0.05, scale: float = 100.0) -> dict[str, dict]:
    """Drop-in mirror of run_cusum_all_sites: pull each site's score history
    from the store and test the current score as a Farrington aberration.
    Scores are continuous in [0,1], so the default scale=100 maps them to
    pseudo-counts for the Poisson model."""
    results: dict[str, dict] = {}
    for rs in current_risk_scores:
        history = store.get_site_history(rs.sample_name, last_n=window)
        past = [h_["risk_score"] for h_ in history]
        results[rs.sample_name] = farrington_detect(past, rs.score, alpha=alpha,
                                                     scale=scale)
    return results


def farrington_series(values: list[float], alpha: float = 0.05, scale: float = 1.0,
                      min_history: int = 4) -> list[dict]:
    """Run the detector prospectively over a series: each point tested against
    all points before it. Points before `min_history` get a null result."""
    out = []
    for i in range(len(values)):
        if i < min_history:
            out.append({"expected": None, "threshold": None, "z": 0.0,
                        "exceedance": 0.0, "alarm": False, "note": "warmup"})
        else:
            out.append(farrington_detect(values[:i], values[i], alpha=alpha,
                                         scale=scale, min_history=min_history))
    return out
