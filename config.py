"""
# Supply Chain Risk Analysis Configuration

This module contains all configuration parameters for a Monte Carlo simulation
analyzing supply chain risks across different manufacturing locations (China, US, Mexico).

## GLOBAL PARAMETERS

- MONTE_CARLO_SIMULATIONS: Number of simulation runs for Monte Carlo analysis (50,000)
- MODEL_Y_PRICE: Selling price per Model Y vehicle (USD)
- MODEL_Y_MANUFACTURING_COST: Manufacturing cost per Model Y vehicle (USD)
- MODEL_Y_PROFIT: Profit per Model Y vehicle (calculated from price - manufacturing cost)
- WACC: Weighted Average Cost of Capital (8.77%)
- EXPEDITED_SHIPPING_COST_PER_HEADLAMP: Cost for expedited shipping per headlamp unit (USD)
- FED_FUNDS_RATE: Current Federal Funds Rate (fetched from live data)

## COUNTRIES

The COUNTRIES dictionary contains probabilistic parameters for each location,
categorized into:

### FACTORY COST VARIABLES (Continuous distributions):
- raw: Raw materials cost per unit (USD)
- labor: Labor cost per unit (USD)
- indirect: Indirect manufacturing costs per unit (USD)
- logistics: Transportation and logistics costs per unit (USD)
- electricity: Energy costs per unit (USD)
- depreciation: Equipment depreciation costs per unit (USD)
- working_capital: Working capital requirements per unit (USD)
- yield_params: Manufacturing yield efficiency (0-1 scale, higher is better)
- tariff: Fixed tariff rate as percentage of product value
- currency_std: Currency volatility (standard deviation)

### RISK & DISCRETE EVENT VARIABLES:
- damage_probability: Probability of shipment damage
- defective_probability: Probability of defective products
- quality_days_delayed: Days of delay for either damage or defective issue
- disruption_lambda: Average number of supply disruptions per shipment
- disruption_days_delayed: Days of production delay per disruption
- disruption_min/max_impact: Range of units lost per disruption event
- tariff_escal: (CHINA / MEXICO ONLY) A probability that a tariff escalation event occurs
- border_delay_lambda: (CHINA / MEXICO ONLY) Average number of border delays per shipment
- border_min/max_impact: (CHINA / MEXICO ONLY) Range of units lost per border delay
- border_days_delayed: (CHINA / MEXICO ONLY) Days of delay per border issue
- cancellation_probability: (CHINA ONLY) Probability of order cancellation
- cancellation_days_delayed: (CHINA ONLY) Days of delay for cancellation events

NOTE: Some variables are estimated by bayesian_priors/samplers.py using
historical data, replacing the need for modeling assumptions.

## SENSITIVITY FACTORS

The `SENSITIVITY_FACTORS` mapping defines, per country, which parameters are
varied during one-at-a-time sensitivity analysis. Each item in the mapping is a
tuple of:
- a human-readable label shown in charts; and
- a tuple path used to locate the parameter inside `COUNTRIES` (for example,
  ("raw", "mean") or ("disruption_lambda",)).

Only the factors listed under each country are perturbed when generating
impact/tornado charts; all other parameters remain fixed at their baseline
distributions.
"""

from live_data import get_most_recent_fed_funds_rate

MONTE_CARLO_SIMULATIONS = 50000
MODEL_Y_PRICE = 41630
MODEL_Y_MANUFACTURING_COST = 38000
MODEL_Y_PROFIT = MODEL_Y_PRICE - MODEL_Y_MANUFACTURING_COST
WACC = 0.0877
EXPEDITED_SHIPPING_COST_PER_HEADLAMP = 50.71
FED_FUNDS_RATE = get_most_recent_fed_funds_rate()


COUNTRIES = {
    "China": {
        # --- FACTORY COSTS ---
        "raw": {"dist": "normal", "mean": 30, "std": 3},
        "labor": {
            "dist": "lognormal",
            "mean": 1.379,
            "std": 0.120,
        },  # mean ~4, std ~0.5
        "indirect": {"dist": "gamma", "shape": 16.0, "scale": 0.25},  # mean 4, std 1
        "logistics": {"dist": "lognormal", "mean": 12, "std": 8},
        "electricity": {"dist": "triangular", "min": 3.60, "mode": 4.00, "max": 4.40},
        "depreciation": {"dist": "normal", "mean": 5, "std": 0.25},
        "working_capital": {"dist": "normal", "mean": 10, "std": 1},
        "yield_params": {
            "dist": "beta",
            "a": 49,
            "b": 3,
        },  # Approx for mean 0.95, std 0.03
        "tariff": {"fixed": 0.25},
        "currency_std": 0.03,
        # --- DISCRETE VARIABLES ---
        "tariff_escal": 0.15,
        "disruption_lambda": 0.15,  # NEW: Avg 0.2 disruptive events per shipment
        "disruption_min_impact": 100,
        "disruption_max_impact": 1000,
        "disruption_days_delayed": 10,
        # Border Delay Risks are impossible for China
        "border_delay_lambda": 0,
        "border_min_impact": 100,
        "border_max_impact": 1000,
        "border_days_delayed": 0,
        # Quality Risks (Binomial)
        "damage_probability": 0.02,
        "defective_probability": 0,  # NEW: Added a separate probability for defects
        "quality_days_delayed": 15,  # NEW: A single delay for any quality issue
        # Cancellation Risk (Bernoulli)
        "cancellation_probability": 0.15,
        "cancellation_days_delayed": 90,
        # --- NEW DISCRETE VARIABLES ---
    },
    "US": {
        # --- FACTORY COSTS ---
        "raw": {"dist": "normal", "mean": 40, "std": 4},
        "labor": {"dist": "lognormal", "mean": 2.48, "std": 0.15},  # mean ~12, std ~2
        "indirect": {"dist": "gamma", "shape": 25.0, "scale": 0.40},  # mean 10, std 2
        "logistics": {"dist": "normal", "mean": 9, "std": 0},
        "electricity": {"dist": "triangular", "min": 3.5, "mode": 4.0, "max": 4.5},
        "depreciation": {"dist": "normal", "mean": 5, "std": 0.25},
        "working_capital": {"dist": "normal", "mean": 5, "std": 0.5},
        "yield_params": {
            "dist": "beta",
            "a": 79,
            "b": 20,
        },  # Approx for mean 0.8, std 0.04
        "tariff": {"fixed": 0},
        "currency_std": 0,
        # --- RISK & DISCRETE EVENT VARIABLES ---
        "tariff_escal": 0,
        "disruption_lambda": 0.002,
        "disruption_min_impact": 5000,
        "disruption_max_impact": 15000,
        "disruption_days_delayed": 20,
        "border_delay_lambda": 0.0,
        "border_min_impact": 0,
        "border_max_impact": 0,
        "border_days_delayed": 0,
        "damage_probability": 0.01,
        "defective_probability": 0,
        "quality_days_delayed": 15,
        "cancellation_probability": 0.0001,
        "cancellation_days_delayed": 90,
    },
    "Mexico": {
        # --- FACTORY COSTS ---
        "raw": {"dist": "normal", "mean": 35, "std": 3.5},
        "labor": {
            "dist": "lognormal",
            "mean": 2.0635,
            "std": 0.1786,
        },  # mean ~8, std ~1.5
        "indirect": {
            "dist": "gamma",
            "shape": 20.66,
            "scale": 0.387,
        },  # mean 8, std 1.75
        "logistics": {"dist": "normal", "mean": 7, "std": 0.056},
        "electricity": {"dist": "triangular", "min": 2.5, "mode": 3.0, "max": 3.5},
        "depreciation": {"dist": "normal", "mean": 1, "std": 0.05},
        "working_capital": {"dist": "normal", "mean": 6, "std": 0.6},
        "yield_params": {
            "dist": "beta",
            "a": 12,
            "b": 1,
        },  # Approx for mean 0.9, std 0.08
        "tariff": {"fixed": 0.25},
        "tariff_escal": 0.1,
        "currency_std": 0.08,
        # --- RISK & DISCRETE EVENT VARIAB;ES ---
        "disruption_lambda": 0.1,
        "disruption_min_impact": 500,
        "disruption_max_impact": 1500,
        "disruption_days_delayed": 5,
        "border_delay_lambda": 0.83,
        "border_min_impact": 100,
        "border_max_impact": 1000,
        "border_days_delayed": 20,
        "damage_probability": 0.015,
        "defective_probability": 0.05,
        "quality_days_delayed": 5,
        "cancellation_probability": 0.0001,
        "cancellation_days_delayed": 90,
    },
}

SENSITIVITY_FACTORS = {
    "US": [
        ("Raw Material Mean", ("raw", "mean")),
        # ('Raw Material Std', ('raw', 'std')),
        ("Labor Mean", ("labor", "mean")),
        # ('Labor Std', ('labor', 'std')),
        # ('Indirect Shape', ('indirect', 'shape')),
        # ('Indirect Scale', ('indirect', 'scale')),
        # ('Logistics Mean', ('logistics', 'mean')),
        # ('Depreciation Mean', ('depreciation', 'mean')),
        # ('Depreciation Std', ('depreciation', 'std')),
        # ('Working Capital Mean', ('working_capital', 'mean')),
        # ('Working Capital Std', ('working_capital', 'std')),
        # ('Manufacturing Yield (a)', ('yield_params', 'a')),
        # ('Manufacturing Yield (b)', ('yield_params', 'b')),
        ("Disruption Lambda", ("disruption_lambda",)),
        # ('Disruption Min Impact', ('disruption_min_impact',)),
        # ('Disruption Max Impact', ('disruption_max_impact',)),
        ("Disruption Days Delayed", ("disruption_days_delayed",)),
        ("Damage Probability", ("damage_probability",)),
        ("Quality Days Delayed", ("quality_days_delayed",)),
    ],
    "Mexico": [
        ("Raw Material Mean", ("raw", "mean")),
        # ('Raw Material Std', ('raw', 'std')),
        ("Labor Mean", ("labor", "mean")),
        # ('Labor Std', ('labor', 'std')),
        # ('Indirect Shape', ('indirect', 'shape')),
        # ('Indirect Scale', ('indirect', 'scale')),
        # ('Logistics Mean', ('logistics', 'mean')),
        # ('Logistics Std', ('logistics', 'std')),
        # ('Depreciation Mean', ('depreciation', 'mean')),
        # ('Depreciation Std', ('depreciation', 'std')),
        # ('Working Capital Mean', ('working_capital', 'mean')),
        # ('Working Capital Std', ('working_capital', 'std')),
        # ('Manufacturing Yield (a)', ('yield_params', 'a')),
        # ('Manufacturing Yield (b)', ('yield_params', 'b')),
        ("Tariff (Fixed)", ("tariff", "fixed")),
        ("Tariff Escalation", ("tariff_escal",)),
        ("Currency Volatility", ("currency_std",)),
        ("Disruption Lambda", ("disruption_lambda",)),
        # ('Disruption Min Impact', ('disruption_min_impact',)),
        # ('Disruption Max Impact', ('disruption_max_impact',)),
        ("Disruption Days Delayed", ("disruption_days_delayed",)),
        ("Border Delay Lambda", ("border_delay_lambda",)),
        # ('Border Min Impact', ('border_min_impact',)),
        # ('Border Max Impact', ('border_max_impact',)),
        ("Border Days Delayed", ("border_days_delayed",)),
        ("Damage Probability", ("damage_probability",)),
        ("Defective Probability", ("defective_probability",)),
        ("Quality Days Delayed", ("quality_days_delayed",)),
    ],
    "China": [
        ("Raw Material Mean", ("raw", "mean")),
        # ('Raw Material Std', ('raw', 'std')),
        ("Labor Mean", ("labor", "mean")),
        # ('Labor Std', ('labor', 'std')),
        # ('Indirect Shape', ('indirect', 'shape')),
        # ('Indirect Scale', ('indirect', 'scale')),
        # ('Logistics Mean', ('logistics', 'mean')),
        # ('Logistics Std', ('logistics', 'std')),
        # ('Depreciation Mean', ('depreciation', 'mean')),
        # ('Depreciation Std', ('depreciation', 'std')),
        # ('Working Capital Mean', ('working_capital', 'mean')),
        # ('Working Capital Std', ('working_capital', 'std')),
        # ('Manufacturing Yield (a)', ('yield_params', 'a')),
        # ('Manufacturing Yield (b)', ('yield_params', 'b')),
        ("Tariff (Fixed)", ("tariff", "fixed")),
        ("Tariff Escalation", ("tariff_escal",)),
        ("Currency Volatility", ("currency_std",)),
        ("Disruption Lambda", ("disruption_lambda",)),
        # ('Disruption Min Impact', ('disruption_min_impact',)),
        # ('Disruption Max Impact', ('disruption_max_impact',)),
        ("Disruption Days Delayed", ("disruption_days_delayed",)),
        ("Border Delay Lambda", ("border_delay_lambda",)),
        # ('Border Min Impact', ('border_min_impact',)),
        # ('Border Max Impact', ('border_max_impact',)),
        ("Border Days Delayed", ("border_days_delayed",)),
        ("Damage Probability", ("damage_probability",)),
        ("Defective Probability", ("defective_probability",)),
        ("Quality Days Delayed", ("quality_days_delayed",)),
        ("Cancellation Probability", ("cancellation_probability",)),
        ("Cancellation Days Delayed", ("cancellation_days_delayed",)),
    ],
}
