"""
Samplers Module

Builds country-specific sampler collections that draw from posterior distributions.
Each sampler function takes a number of runs and returns that many samples.

WHY SAMPLERS?
- Monte Carlo simulation needs random samples
- Each sample represents one possible future scenario
- Drawing from posteriors (not just point estimates) gives realistic uncertainty
"""

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from scipy.stats import beta as beta_dist

from .parameter_estimators import (
    estimate_fx_posterior,
    estimate_raw_material_posterior,
    estimate_yield_posterior,
    estimate_labor_posterior,
    estimate_logistics_posterior,
    estimate_indirect_posterior,
    estimate_electricity_posterior,
    estimate_depreciation_posterior,
    estimate_working_capital_posterior,
)


@dataclass
class CountrySamplers:
    """
    Collection of sampler functions for a country.
    Each function takes n_runs and returns n_runs samples from posterior predictive.
    """

    raw_material: Callable[[int], np.ndarray]
    labor: Callable[[int], np.ndarray]
    logistics: Callable[[int], np.ndarray]
    indirect: Callable[[int], np.ndarray]
    electricity: Callable[[int], np.ndarray]
    depreciation: Callable[[int], np.ndarray]
    working_capital: Callable[[int], np.ndarray]
    fx_multiplier: Callable[[int], np.ndarray]
    yield_rate: Callable[[int], np.ndarray]


def build_samplers_for_country(country: str, baseline_params: Dict) -> CountrySamplers:
    """
    Build posterior samplers from real data for a country.
    Falls back to baseline parameters if data unavailable.
    """
    print(f"\n🔧 Building samplers for {country}...")

    # Raw materials
    try:
        raw_post = estimate_raw_material_posterior(baseline_params["raw"]["mean"])

        def raw_sampler(n):
            return raw_post.sample_predictive(n)

    except Exception:
        raw_mean = baseline_params["raw"]["mean"]
        raw_std = baseline_params["raw"]["std"]

        def raw_sampler(n):
            return np.random.normal(raw_mean, raw_std, n)

    # Labor (now using posterior from real data; fallback = lognormal baseline)
    try:
        # baseline_params["labor"] uses lognormal params (mu_log, sigma_log)
        mu_log = baseline_params["labor"]["mean"]
        sigma_log = baseline_params["labor"]["std"]
        baseline_labor_mean = float(np.exp(mu_log + (sigma_log**2) / 2.0))
        labor_post = estimate_labor_posterior(country, baseline_labor_mean)

        def labor_sampler(n):
            return labor_post.sample_predictive(n)

        print("  ✓ Labor: Student-t predictive from real series")
    except Exception as e:
        # Robust fallback to the *correct* lognormal, not normal
        print(f"  ⚠️  Labor posterior failed for {country}: {e}. Using baseline lognormal.")
        mu_log = baseline_params["labor"]["mean"]
        sigma_log = baseline_params["labor"]["std"]
        labor_sampler = lambda n: np.random.lognormal(mu_log, sigma_log, n)

    # Logistics (now using posterior from freight/transport data)
    try:
        # Determine transport mode based on country
        mode = "truck" if country in ("US", "Mexico") else "ocean"
        
        # Get baseline logistics cost (handle both dict formats)
        if isinstance(baseline_params["logistics"], dict):
            if baseline_params["logistics"].get("dist") == "lognormal":
                # For lognormal, compute arithmetic mean from parameters
                log_mean = baseline_params["logistics"]["mean"]
                log_std = baseline_params["logistics"]["std"]
                baseline_logistics = log_mean  # Use mean directly as baseline
            else:
                baseline_logistics = baseline_params["logistics"]["mean"]
        else:
            baseline_logistics = baseline_params["logistics"]
        
        logistics_post = estimate_logistics_posterior(country, baseline_logistics, mode=mode)
        
        def logistics_sampler(n):
            return logistics_post.sample_predictive(n)
        
        print(f"  ✓ Logistics: Student-t predictive from {mode} freight data")
    except Exception as e:
        # Robust fallback to baseline parameters
        print(f"  ⚠️  Logistics posterior failed for {country}: {e}. Using baseline.")
        if baseline_params["logistics"]["dist"] == "lognormal":
            log_mean = baseline_params["logistics"]["mean"]
            log_std = baseline_params["logistics"]["std"]
            sigma = np.sqrt(np.log(1 + (log_std**2 / log_mean**2)))
            mu = np.log(log_mean) - (sigma**2 / 2)
            logistics_sampler = lambda n: np.random.lognormal(mu, sigma, n)
        else:
            log_mean = baseline_params["logistics"]["mean"]
            log_std = baseline_params["logistics"]["std"]
            logistics_sampler = lambda n: np.random.normal(log_mean, log_std, n)

    # Indirect (now using posterior from ECI/CPI data)
    try:
        # Get baseline indirect cost (handle different dist formats)
        if isinstance(baseline_params["indirect"], dict):
            if baseline_params["indirect"].get("dist") == "gamma":
                # For gamma, mean = shape * scale
                shape = baseline_params["indirect"]["shape"]
                scale = baseline_params["indirect"]["scale"]
                baseline_indirect = shape * scale
            elif "mean" in baseline_params["indirect"]:
                baseline_indirect = baseline_params["indirect"]["mean"]
            else:
                # Fallback to 10 if no mean specified
                baseline_indirect = 10.0
        else:
            baseline_indirect = baseline_params["indirect"]
        
        indirect_post = estimate_indirect_posterior(country, baseline_indirect)
        
        def indirect_sampler(n):
            return indirect_post.sample_predictive(n)
        
        print(f"  ✓ Indirect: Student-t predictive from ECI/CPI data")
    except Exception as e:
        # Robust fallback to baseline parameters
        print(f"  ⚠️  Indirect posterior failed for {country}: {e}. Using baseline.")
        if baseline_params["indirect"]["dist"] == "gamma":
            shape = baseline_params["indirect"]["shape"]
            scale = baseline_params["indirect"]["scale"]
            indirect_sampler = lambda n: np.random.gamma(shape, scale, n)
        elif baseline_params["indirect"]["dist"] == "normal":
            mean = baseline_params["indirect"]["mean"]
            std = baseline_params["indirect"]["std"]
            indirect_sampler = lambda n: np.random.normal(mean, std, n)
        else:
            # Generic fallback
            mean = baseline_params["indirect"].get("mean", 10.0)
            indirect_sampler = lambda n: np.full(n, mean)

    # Electricity (now using posterior from energy price data)
    try:
        # Get baseline electricity cost (handle different formats)
        if isinstance(baseline_params["electricity"], dict):
            baseline_electricity = baseline_params["electricity"].get("mean", 4.0)
        else:
            baseline_electricity = baseline_params["electricity"]
        
        electricity_post = estimate_electricity_posterior(country, baseline_electricity)
        
        def electricity_sampler(n):
            return electricity_post.sample_predictive(n)
        
        print(f"  ✓ Electricity: Student-t predictive from energy price data")
    except Exception as e:
        # Robust fallback to baseline parameters
        print(f"  ⚠️  Electricity posterior failed for {country}: {e}. Using baseline.")
        if isinstance(baseline_params["electricity"], dict):
            if baseline_params["electricity"].get("dist") == "gamma":
                shape = baseline_params["electricity"]["shape"]
                scale = baseline_params["electricity"]["scale"]
                electricity_sampler = lambda n: np.random.gamma(shape, scale, n)
            elif baseline_params["electricity"].get("dist") == "normal":
                mean = baseline_params["electricity"]["mean"]
                std = baseline_params["electricity"].get("std", 0.4)
                electricity_sampler = lambda n: np.random.normal(mean, std, n)
            else:
                # Use mean as constant
                mean = baseline_params["electricity"].get("mean", 4.0)
                electricity_sampler = lambda n: np.full(n, mean)
        else:
            electricity_sampler = lambda n: np.full(n, baseline_params["electricity"])

    # Depreciation (now using posterior from machinery/capital goods price data)
    try:
        # Get baseline depreciation cost (handle different formats)
        if isinstance(baseline_params["depreciation"], dict):
            baseline_depreciation = baseline_params["depreciation"].get("mean", 5.0)
        else:
            baseline_depreciation = baseline_params["depreciation"]
        
        depreciation_post = estimate_depreciation_posterior(country, baseline_depreciation)
        
        def depreciation_sampler(n):
            return depreciation_post.sample_predictive(n)
        
        print(f"  ✓ Depreciation: Student-t predictive from machinery/investment price data")
    except Exception as e:
        # Robust fallback to baseline parameters
        print(f"  ⚠️  Depreciation posterior failed for {country}: {e}. Using baseline.")
        if isinstance(baseline_params["depreciation"], dict):
            if baseline_params["depreciation"].get("dist") == "gamma":
                shape = baseline_params["depreciation"]["shape"]
                scale = baseline_params["depreciation"]["scale"]
                depreciation_sampler = lambda n: np.random.gamma(shape, scale, n)
            elif baseline_params["depreciation"].get("dist") == "normal":
                mean = baseline_params["depreciation"]["mean"]
                std = baseline_params["depreciation"].get("std", 0.5)
                depreciation_sampler = lambda n: np.random.normal(mean, std, n)
            else:
                # Use mean as constant
                mean = baseline_params["depreciation"].get("mean", 5.0)
                depreciation_sampler = lambda n: np.full(n, mean)
        else:
            depreciation_sampler = lambda n: np.full(n, baseline_params["depreciation"])

    # Working Capital (now using posterior from interest rate data)
    try:
        # Get baseline working capital cost (handle different formats)
        if isinstance(baseline_params["working_capital"], dict):
            baseline_working_capital = baseline_params["working_capital"].get("mean", 5.0)
        else:
            baseline_working_capital = baseline_params["working_capital"]
        
        working_capital_post = estimate_working_capital_posterior(country, baseline_working_capital)
        
        def working_capital_sampler(n):
            return working_capital_post.sample_predictive(n)
        
        print(f"  ✓ Working Capital: Student-t predictive from interest rate data")
    except Exception as e:
        # Robust fallback to baseline parameters
        print(f"  ⚠️  Working capital posterior failed for {country}: {e}. Using baseline.")
        if isinstance(baseline_params["working_capital"], dict):
            if baseline_params["working_capital"].get("dist") == "gamma":
                shape = baseline_params["working_capital"]["shape"]
                scale = baseline_params["working_capital"]["scale"]
                working_capital_sampler = lambda n: np.random.gamma(shape, scale, n)
            elif baseline_params["working_capital"].get("dist") == "normal":
                mean = baseline_params["working_capital"]["mean"]
                std = baseline_params["working_capital"].get("std", 0.5)
                working_capital_sampler = lambda n: np.random.normal(mean, std, n)
            else:
                # Use mean as constant
                mean = baseline_params["working_capital"].get("mean", 5.0)
                working_capital_sampler = lambda n: np.full(n, mean)
        else:
            working_capital_sampler = lambda n: np.full(n, baseline_params["working_capital"])

    # FX volatility
    try:
        fx_sampler = estimate_fx_posterior(country)
        print("  ✓ FX: Posterior from FRED exchange rate data")
    except Exception:
        fx_std = baseline_params["currency_std"]

        def fx_sampler(n):
            return 1 + np.random.normal(0, fx_std, n)

        print(f"  ⚠️  FX: Using baseline volatility {fx_std}")

    # Manufacturing yield
    try:
        yield_baseline = baseline_params["yield_params"]["a"] / (
            baseline_params["yield_params"]["a"] + baseline_params["yield_params"]["b"]
        )
        uncertainty_map = {"US": "medium", "Mexico": "high", "China": "low"}
        yield_post = estimate_yield_posterior(yield_baseline, uncertainty_map[country])

        def yield_sampler(n):
            return yield_post.sample_predictive(n).clip(0.01, 0.99)

        print(f"  ✓ Yield: Beta posterior (uncertainty={uncertainty_map[country]})")
    except Exception:
        a = baseline_params["yield_params"]["a"]
        b = baseline_params["yield_params"]["b"]

        def yield_sampler(n):
            return beta_dist.rvs(a, b, size=n)

        print(f"  ⚠️  Yield: Using baseline Beta({a}, {b})")

    return CountrySamplers(
        raw_material=raw_sampler,
        labor=labor_sampler,
        logistics=logistics_sampler,
        indirect=indirect_sampler,
        electricity=electricity_sampler,
        depreciation=depreciation_sampler,
        working_capital=working_capital_sampler,
        fx_multiplier=fx_sampler,
        yield_rate=yield_sampler,
    )
