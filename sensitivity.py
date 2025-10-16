import copy

import numpy as np
import pandas as pd

from simulation import run_monte_carlo


def run_sensitivity_analysis(
    country, base_params, factors_to_test, order_size, swing=0.20
):
    """
    Runs a one-at-a-time sensitivity analysis for a given set of factors.
    Returns a dataframe with the results and the baseline mean cost.
    """
    results = []

    # Calculate baseline mean cost
    base_results = run_monte_carlo(country, base_params, order_size)
    baseline_mean = np.mean(base_results["total_cost"])

    for factor_name, param_path in factors_to_test:
        params_low = copy.deepcopy(base_params)
        params_high = copy.deepcopy(base_params)

        # Get the base value using the path
        base_value = base_params
        for key in param_path:
            base_value = base_value[key]

        # Skip parameters that are 0 (can't do ±20% of 0)
        if base_value == 0:
            continue

        low_value = base_value * (1 - swing)
        high_value = base_value * (1 + swing)

        # Set the low and high values in the copied params
        temp_low = params_low
        temp_high = params_high
        for i, key in enumerate(param_path):
            if i == len(param_path) - 1:
                temp_low[key] = low_value
                temp_high[key] = high_value
            else:
                temp_low = temp_low[key]
                temp_high = temp_high[key]

        try:
            # Simulate with low and high values
            low_results = run_monte_carlo(country, params_low, order_size)
            high_results = run_monte_carlo(country, params_high, order_size)

            mean_low = np.mean(low_results["total_cost"])
            mean_high = np.mean(high_results["total_cost"])

            results.append(
                {
                    "Factor": factor_name,
                    "Low Cost": mean_low,
                    "High Cost": mean_high,
                    "Impact": mean_high - mean_low,
                }
            )
        except Exception as e:
            # Skip factors that cause errors (e.g., invalid parameter combinations)
            print(f"WARNING: Skipping {factor_name}: {str(e)}")
            continue

    return pd.DataFrame(results), baseline_mean
