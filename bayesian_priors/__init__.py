"""
Bayesian Parameter Estimation for Tesla Sourcing Simulation

WHY BAYESIAN?
- With limited data (24 months), we're uncertain about TRUE parameters
- Naive approach: Use sample mean/std (pretends we know true values)
- Bayesian approach: Account for parameter uncertainty via posterior predictive
- Result: More conservative risk estimates (wider tails = Student-t vs Normal)

WHY THIS INCREASES SUPPLY CHAIN READINESS:
- More realistic tail risk: Fatter tails in student-t distribution = better prepared for extreme cost scenarios
- Data-driven: Uses actual PPI/FX data instead of guesses
- Conservative planning: Wider confidence intervals → less likely to underestimate costs

WHAT THIS DOES:
1. Fetches real data (PPI, FX rates, etc.)
2. Fits Bayesian posterior from data
3. Returns SAMPLER FUNCTIONS that draw from posterior predictive distributions
4. These samplers replace your hand-picked parameters in app.py

VARIABLES WE ARE DOING THIS FOR (with real data):
- [x] raw material (PPI data → Student-t samples)
- [x] labor 
- [x] logistics
- [x] fx (FRED exchange rates → Student-t samples)
- [x] yield (uncertainty-adjusted Beta)
- [x] indirect (ECI for US, World Bank CPI for Mexico/China → Student-t samples)
- [x] electricity (US: $/kWh, Mexico/China: CPI energy → Student-t samples)
- [x] depreciation (US: Machinery PPI, Mexico/China: WB investment price level → Student-t samples)
- [x] working capital (US: Fed Funds, Mexico: WB lending rate, China: 3-month interbank → Student-t samples)

USAGE:
    from bayesian_priors import create_bayesian_simulator
    
    bayesian_sims = create_bayesian_simulator(countries)
    
    # Use instead of simulate_country()
    us_costs = bayesian_sims['US'](n_runs=10000)
"""

from typing import Callable, Dict

import numpy as np

from .samplers import build_samplers_for_country

# Expose main API functions
__all__ = ["create_bayesian_simulator", "build_samplers_for_country"]


def create_bayesian_simulator(countries_dict: Dict) -> Dict[str, Callable]:
    """
    Create Bayesian simulators for all countries.
    
    NOTE: This function is deprecated. Use simulation.run_monte_carlo() directly
    instead, which provides better architecture and returns both costs and lost units.

    Returns:
        Dict mapping country -> simulator function

    Usage in app.py:
        from bayesian_priors import create_bayesian_simulator

        bayesian_sims = create_bayesian_simulator(countries)

        # Use instead of simulate_country()
        us_costs = bayesian_sims['US'](n_runs=10000)
    """
    print("\n" + "=" * 60)
    print("BUILDING BAYESIAN POSTERIOR SAMPLERS")
    print("=" * 60)
    print("⚠️  WARNING: create_bayesian_simulator() is deprecated.")
    print("    Use simulation.run_monte_carlo() instead for full functionality.")
    print("=" * 60 + "\n")

    # This function is now deprecated but kept for backwards compatibility
    # The main simulation.py has the correct implementation
    raise DeprecationWarning(
        "create_bayesian_simulator() is deprecated. "
        "Use simulation.run_monte_carlo() which provides better separation "
        "of concerns and returns both total costs and lost units."
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Minimal test
    test_params = {
        "US": {
            "raw": {"mean": 40, "std": 4},
            "labor": {"mean": 12, "std": 0.6},
            "indirect": {"mean": 10},
            "logistics": {"dist": "normal", "mean": 9, "std": 0},
            "electricity": {"mean": 4},
            "depreciation": {"mean": 5},
            "working_capital": {"mean": 5},
            "yield_params": {"a": 79, "b": 20},
            "currency_std": 0,
            "tariff": {"fixed": 0},
            "tariff_escal": {"mean": 0, "std": 0},
            "disruption_prob": 0.05,
            "disruption_impact": 10,
            "border_mean": 0,
            "border_std": 0,
            "border_threshold": 2,
            "border_cost_per_hr": 10,
            "damage_prob": 0.01,
            "damage_impact": 20,
            "skills_mean": 0,
            "skills_std": 0,
            "cancellation_prob": 0,
            "cancellation_impact": 50,
        }
    }

    simulators = create_bayesian_simulator(test_params)
    costs = simulators["US"](1000)

    print("\nTest simulation complete:")
    print(f"  Mean: ${costs.mean():.2f}")
    print(f"  Std:  ${costs.std():.2f}")
    print(f"  95th: ${np.percentile(costs, 95):.2f}")

