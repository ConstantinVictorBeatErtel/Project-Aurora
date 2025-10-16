import numpy as np
from scipy.optimize import minimize


def _generic_optimize(
    all_costs,
    lambda_risk,
    constraints,
    objective_func,
    diagnostic_fn=None,
    return_diagnostics=False,
    **objective_kwargs,
):
    """
    Generic optimization function that handles the common optimization logic.

    Args:
        all_costs: dict of {country: cost_array}
        lambda_risk: risk aversion parameter
        constraints: dict of {country: (min_alloc, max_alloc)} or None
        objective_func: function that takes (weights, all_costs, lambda_risk, **kwargs) and returns objective value
        **objective_kwargs: additional arguments to pass to objective_func

    Returns:
        optimization result dict or None
    """
    countries_list = list(all_costs.keys())
    n_countries = len(countries_list)

    def objective(weights):
        return objective_func(weights, all_costs, lambda_risk, **objective_kwargs)

    # Constraints: weights sum to 1, all weights >= 0
    constraint_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = []

    # Add custom constraints if provided
    if constraints:
        for country in countries_list:
            if country in constraints:
                min_alloc, max_alloc = constraints[country]
                bounds.append((min_alloc, max_alloc))
            else:
                bounds.append((0, 1))
    else:
        bounds = [(0, 1) for _ in range(n_countries)]

    # Initial guess: equal allocation
    x0 = np.ones(n_countries) / n_countries

    # Optimize
    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=[constraint_sum]
    )

    if result.success:
        optimal_weights = result.x
        portfolio = np.zeros(len(list(all_costs.values())[0]))
        for i, country in enumerate(countries_list):
            portfolio += optimal_weights[i] * all_costs[country]

        out = {
            "allocations": {
                countries_list[i]: w for i, w in enumerate(optimal_weights)
            },
            "expected_cost": np.mean(portfolio),
            "std_cost": np.std(portfolio),
            "portfolio_costs": portfolio,
        }

        if return_diagnostics:
            diag_fn = diagnostic_fn or _compute_diagnostics
            out["diagnostics"] = diag_fn(all_costs, optimal_weights)

        return out
    else:
        return None


def _objective_without_yield(weights, all_costs, lambda_risk):
    """
    Mean–variance objective for costs: minimize E[cost] + lambda_risk * SD(cost).

    - This balances low average cost with low variability. As lambda_risk increases,
      the solution prioritizes stability (lower standard deviation) more heavily.
    - It is not strictly the Global Minimum Variance (GMV) objective (which minimizes
      variance alone), but as lambda_risk → ∞, it converges toward the GMV portfolio
      under the same constraints.
    - Because country cost series are not perfectly correlated, the variance-minimizing
      solution is typically diversified: mixing countries reduces portfolio variability
      via covariance effects.
    - Despite an individual country's variance being lower, this solution will natively
      prefer a diversified solution

    Args:
        weights: array-like of portfolio weights (summing to 1 under constraints).
        all_costs: dict[str, np.ndarray], per-country cost series aligned by time/scenario.
        lambda_risk: float, risk aversion scaling the standard deviation penalty.

    Returns:
        float: objective value E[cost] + lambda_risk * SD(cost) for the given weights.
    """
    countries_list = list(all_costs.keys())
    portfolio = np.zeros(len(list(all_costs.values())[0]))

    for i, country in enumerate(countries_list):
        portfolio += weights[i] * all_costs[country]

    cost_per_good_unit = portfolio
    expected_cost = np.mean(cost_per_good_unit)
    std_cost = np.std(cost_per_good_unit)

    return expected_cost + lambda_risk * std_cost


def _objective_with_yield(weights, all_costs, lambda_risk, yields):
    """Objective function for optimization with yield consideration."""
    countries_list = list(all_costs.keys())
    portfolio = np.zeros(len(list(all_costs.values())[0]))
    portfolio_yield = 0

    for i, country in enumerate(countries_list):
        portfolio += weights[i] * all_costs[country]
        portfolio_yield += weights[i] * yields[country]

    # GUARD: Avoid division by zero if yield is somehow zero
    if portfolio_yield <= 0:
        return np.inf

    cost_per_good_unit = portfolio / portfolio_yield
    expected_cost = np.mean(cost_per_good_unit)
    std_cost = np.std(cost_per_good_unit)

    return expected_cost + lambda_risk * std_cost


def optimize_without_yield(all_costs, lambda_risk, constraints=None):
    """
    Optimizes portfolio allocation to minimize E[Cost] + lambda * SD[Cost]

    Args:
        all_costs: dict of {country: cost_array}
        lambda_risk: risk aversion parameter
        constraints: dict of {country: (min_alloc, max_alloc)} or None

    Returns:
        optimal allocations, expected cost, std dev
    """
    return _generic_optimize(
        all_costs,
        lambda_risk,
        constraints,
        _objective_without_yield,
        return_diagnostics=True,
    )


def optimize_portfolio(all_costs, yields, lambda_risk, constraints=None):
    """
    Optimizes portfolio allocation to minimize E[Cost / Yield] + lambda * SD[Cost / Yield]

    Args:
        all_costs: dict of {country: cost_array}
        yields: dict of {country: yield_value}
        lambda_risk: risk aversion parameter
        constraints: dict of {country: (min_alloc, max_alloc)} or None

    Returns:
        optimal allocations, expected cost, std dev
    """
    return _generic_optimize(
        all_costs, lambda_risk, constraints, _objective_with_yield, yields=yields
    )


def _compute_diagnostics(all_costs: dict[str, np.ndarray], w: np.ndarray):
    countries = list(all_costs.keys())
    X = np.column_stack([all_costs[c] for c in countries])  # shape (T, N)
    mu = X.mean(axis=0)  # per-country mean
    Sigma = np.cov(X, rowvar=False)  # N x N covariance
    stds = X.std(axis=0)
    corr = np.corrcoef(X, rowvar=False)

    # Portfolio stats
    mu_p = float(w @ mu)
    var_p = float(w @ Sigma @ w)
    std_p = var_p**0.5

    return {
        "countries": countries,
        "means": dict(zip(countries, mu)),
        "stds": dict(zip(countries, stds)),
        "corr": corr,
        "portfolio_mean": mu_p,
        "portfolio_std": std_p,
        "Sigma": Sigma,
    }
