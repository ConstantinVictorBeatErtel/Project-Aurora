"""
Parameter Estimators Module

Converts real-world data into Bayesian posterior distributions.
These functions fetch data and fit it to appropriate probability models.

WHAT THIS DOES:
- Takes baseline values and real economic data
- Returns posterior distributions that capture uncertainty
- Used by samplers to generate realistic cost scenarios
"""

from typing import Callable

import numpy as np

from .data_fetching import (
    fetch_fred_series,
    fetch_china_wages_manufacturing,
)
from .posterior_models import (
    BetaPosterior,
    NormalPosterior,
    fit_normal_posterior,
)


def estimate_raw_material_posterior(baseline: float) -> NormalPosterior:
    """
    Fit posterior for raw material costs from PPI data.
    Uses plastics PPI as proxy for automotive components.
    """
    ppi = fetch_fred_series("PCU325211325211P", months=24)

    if len(ppi) < 2:
        print("⚠️  No PPI data, using hand-picked parameters")
        # Fallback: create posterior that mimics Normal(baseline, baseline*0.1)
        return NormalPosterior(
            mu=baseline, kappa=100, alpha=50, beta=50 * (baseline * 0.1) ** 2
        )

    # Convert PPI index to dollar values centered at baseline
    ppi_normalized = (ppi / ppi.mean()) * baseline

    return fit_normal_posterior(ppi_normalized, prior_mean=baseline)


def estimate_fx_posterior(country: str) -> Callable[[int], np.ndarray]:
    """
    Fit FX volatility from exchange rate data.
    Returns a MULTIPLIER function: cost → cost * (1 ± FX_change)
    """
    series_map = {
        "Mexico": "DEXMXUS",  # Peso per USD
        "China": "DEXCHUS",  # Yuan per USD
    }

    if country == "US":
        return lambda n: np.ones(n)  # No FX risk

    if country not in series_map:
        print(f"⚠️  No FX mapping for {country}")
        return lambda n: np.ones(n)

    fx_series = fetch_fred_series(series_map[country], months=12)

    if len(fx_series) < 2:
        print(f"⚠️  No FX data for {country}, using zero volatility")
        return lambda n: np.ones(n)

    # Calculate log returns (percent changes)
    returns = np.log(fx_series / fx_series.shift(1)).dropna()

    # Fit posterior to returns
    posterior = fit_normal_posterior(returns, prior_mean=0.0)

    # Return sampler that adds FX volatility
    def fx_multiplier(n: int) -> np.ndarray:
        fx_shocks = posterior.sample_predictive(n)
        return 1.0 + fx_shocks

    return fx_multiplier


def estimate_yield_posterior(
    baseline_yield: float, uncertainty: str = "medium"
) -> BetaPosterior:
    """
    Create Beta posterior for manufacturing yield.

    Args:
        baseline_yield: Expected yield (e.g., 0.80 for 80%)
        uncertainty: 'low' (tight), 'medium', 'high' (wide spread)
                    Use 'high' for Mexico ("questionable skills")
    """
    # Uncertainty maps to total pseudo-observations
    uncertainty_map = {
        "low": 100,  # Very confident (China mature facility)
        "medium": 50,  # Moderate confidence (US new facility)
        "high": 15,  # Low confidence (Mexico skill issues)
    }

    total = uncertainty_map.get(uncertainty, 50)
    alpha = baseline_yield * total
    beta = (1 - baseline_yield) * total

    return BetaPosterior(alpha, beta)


# -----------------------------
# Labor posterior (US/Mexico/China)
# -----------------------------
def _lognormal_mean(mu_log: float, sigma_log: float) -> float:
    """
    Convert log-space (mu, sigma) to arithmetic mean of a lognormal.
      E[X] = exp(mu + sigma^2 / 2)
    """
    return float(np.exp(mu_log + (sigma_log**2) / 2.0))


def estimate_labor_posterior(country: str, baseline_per_lamp: float) -> NormalPosterior:
    """
    Fit a Normal–Inverse-Gamma posterior for labor cost per lamp.
    Strategy:
      - Fetch a country-appropriate series (level or index)
      - Normalize to baseline_per_lamp so units become $/lamp
      - Fit N-IG posterior → Student-t predictive at sample time
    """
    series = None

    if country == "US":
        # BLS via FRED: CES3000000003 (AHE Manufacturing, USD/hour, monthly)
        s = fetch_fred_series("CES3000000003", months=24)
        series = s
    elif country == "Mexico":
        # OECD via FRED: LCEAMN01MXM661S (Hourly Earnings Index 2015=100, monthly)
        s = fetch_fred_series("LCEAMN01MXM661S", months=24)
        series = s
    elif country == "China":
        # TradingEconomics (annual, CNY/year). Keep annual; Student-t handles small n.
        s = fetch_china_wages_manufacturing()
        # Keep last ~10 years if available to avoid ancient regimes
        if len(s) > 0:
            s = s.iloc[-10:]
        series = s
    else:
        print(f"⚠️  No labor series mapping for {country}")
        series = None

    if series is None or len(series) < 2:
        # Fallback: weakly-informative posterior centered at baseline
        print(f"⚠️  Labor posterior fallback for {country}; using weak N-IG prior at ${baseline_per_lamp:.2f}")
        return NormalPosterior(
            mu=baseline_per_lamp,      # center on baseline
            kappa=20.0,                # weakish confidence in mean
            alpha=10.0,                # df = 20
            beta=(baseline_per_lamp * 0.10) ** 2 * 10.0,  # ~10% coeffvar guess
        )

    # Normalize to $/lamp baseline (same approach as PPI above)
    # This converts whatever units the series has ($/hour, index, $/year)
    # into $/lamp by scaling relative to mean
    series_normalized = (series / series.mean()) * baseline_per_lamp
    return fit_normal_posterior(series_normalized, prior_mean=baseline_per_lamp)


def estimate_logistics_posterior(
    country: str, baseline_per_lamp: float, mode: str = "truck"
) -> NormalPosterior:
    """
    Fit a Normal–Inverse-Gamma posterior for logistics cost per lamp.
    
    Strategy:
      - Fetch freight/transport price index for the country/mode
      - Normalize to baseline_per_lamp so units become $/lamp
      - Fit N-IG posterior → Student-t predictive at sample time
      
    Args:
        country: "US", "Mexico", or "China"
        baseline_per_lamp: Expected logistics cost per lamp in USD
        mode: "truck" (US/Mexico), "ocean" (China), or "air" (China)
    """
    series = None
    
    if country in ("US", "Mexico"):
        # FRED PPI: General Freight Trucking, Long-Distance, Truckload
        # PCU4841214841212 (monthly)
        # Used for both US domestic and Mexico cross-border (priced off NA market)
        s = fetch_fred_series("PCU4841214841212", months=24)
        series = s
        if len(s) > 0:
            print(f"  → Using US long-distance trucking PPI for {country}")
    
    elif country == "China":
        if mode == "ocean":
            # FRED PPI: Deep-sea Freight Transportation PCU483111483111 (monthly)
            s = fetch_fred_series("PCU483111483111", months=24)
            series = s
            if len(s) > 0:
                print("  → Using deep-sea freight PPI for China")
        elif mode == "air":
            # FRED Inbound Price Index: Air Freight for Asia IC1312 (monthly)
            s = fetch_fred_series("IC1312", months=24)
            series = s
            if len(s) > 0:
                print("  → Using air freight index for China")
        else:
            # Default to ocean for China
            s = fetch_fred_series("PCU483111483111", months=24)
            series = s
    
    if series is None or len(series) < 2:
        # Fallback: weakly-informative posterior centered at baseline
        print(f"⚠️  Logistics posterior fallback for {country}; using weak N-IG prior at ${baseline_per_lamp:.2f}")
        return NormalPosterior(
            mu=baseline_per_lamp,
            kappa=20.0,                # moderate confidence in mean
            alpha=10.0,                # df = 20
            beta=(baseline_per_lamp * 0.15) ** 2 * 10.0,  # ~15% coeffvar (logistics is volatile)
        )
    
    # Normalize to $/lamp baseline (same approach as raw/labor)
    series_normalized = (series / series.mean()) * baseline_per_lamp
    return fit_normal_posterior(series_normalized, prior_mean=baseline_per_lamp)


def estimate_indirect_posterior(country: str, baseline_per_lamp: float) -> NormalPosterior:
    """
    Fit a Normal–Inverse-Gamma posterior for indirect/overhead cost per lamp.
    
    Strategy:
      - US: FRED Employment Cost Index for office/admin (tracks overhead wage pressure)
      - Mexico/China: World Bank CPI (widely used overhead inflator)
      - Normalize to baseline_per_lamp so units become $/lamp
      - Fit N-IG posterior → Student-t predictive at sample time
      
    Args:
        country: "US", "Mexico", or "China"
        baseline_per_lamp: Expected indirect cost per lamp in USD
    """
    from .data_fetching import fetch_worldbank_indicator
    
    series = None
    
    if country == "US":
        # FRED Employment Cost Index: Private Industry, Office & Administrative Support
        # ECIOCC52 (quarterly; stable long history)
        # Good proxy for back-office overhead wage pressure
        s = fetch_fred_series("ECIOCC52", months=48)
        series = s
        if len(s) > 0:
            print(f"  → Using ECI office/admin for US indirect costs")
    else:
        # World Bank CPI (All items, 2010=100) FP.CPI.TOTL
        # Annual, but perfectly serviceable for conservative overhead inflator
        # Student-t predictive will reflect small-n uncertainty
        country_code = {"Mexico": "MEX", "China": "CHN"}.get(country, "USA")
        s = fetch_worldbank_indicator(country_code, "FP.CPI.TOTL", years=15)
        series = s
        if len(s) > 0:
            print(f"  → Using World Bank CPI for {country} indirect costs")
    
    if series is None or len(series) < 2:
        # Fallback: weakly-informative posterior centered at baseline
        print(f"⚠️  Indirect posterior fallback for {country}; using weak N-IG prior at ${baseline_per_lamp:.2f}")
        return NormalPosterior(
            mu=baseline_per_lamp,
            kappa=10.0,                # moderate confidence in mean
            alpha=8.0,                 # df = 16
            beta=(baseline_per_lamp * 0.10) ** 2 * 8.0,  # ~10% coeffvar
        )
    
    # Normalize to $/lamp baseline (same approach as raw/labor/logistics)
    series_normalized = (series / series.mean()) * baseline_per_lamp
    return fit_normal_posterior(series_normalized, prior_mean=baseline_per_lamp)

