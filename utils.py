import numpy as np
from scipy.optimize import minimize

from structs import DiscreteRisksParams


def sample_from_spec(spec, n):
    dist = spec.get("dist", "normal").lower()
    if dist == "normal":
        return np.random.normal(spec["mean"], spec["std"], n)
    if dist == "lognormal":  # expects log-space μ, σ
        return np.random.lognormal(spec["mean"], spec["std"], n)
    if dist == "triangular":  # expects min, mode, max
        return np.random.triangular(spec["min"], spec["mode"], spec["max"], n)
    if dist == "gamma":  # expects shape k, scale θ
        return np.random.gamma(spec["shape"], spec["scale"], n)
    if dist == "beta":  # expects min, max
        return np.random.beta(spec["a"], spec["b"], n)
    raise ValueError(f"Unsupported dist: {dist}")


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
    countries_list = list(all_costs.keys())
    n_countries = len(countries_list)

    def objective(weights):
        # Calculate portfolio cost for all runs
        portfolio = np.zeros(len(list(all_costs.values())[0]))
        # portfolio_yield = 0

        for i, country in enumerate(countries_list):
            portfolio += weights[i] * all_costs[country]
            # portfolio_yield += weights[i] * yields[country]

        # GUARD: Avoid division by zero if yield is somehow zero
        # if portfolio_yield <= 0:
        #     return np.inf

        cost_per_good_unit = portfolio

        expected_cost = np.mean(cost_per_good_unit)
        std_cost = np.std(cost_per_good_unit)

        return expected_cost + lambda_risk * std_cost

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

        return {
            "allocations": {
                countries_list[i]: w for i, w in enumerate(optimal_weights)
            },
            "expected_cost": np.mean(portfolio),
            "std_cost": np.std(portfolio),
            "portfolio_costs": portfolio,
        }
    else:
        return None


def optimize_portfolio(all_costs, yields, lambda_risk, constraints=None):
    """
    Optimizes portfolio allocation to minimize E[Cost] + lambda * SD[Cost]

    Args:
        all_costs: dict of {country: cost_array}
        lambda_risk: risk aversion parameter
        constraints: dict of {country: (min_alloc, max_alloc)} or None

    Returns:
        optimal allocations, expected cost, std dev
    """
    countries_list = list(all_costs.keys())
    n_countries = len(countries_list)

    def objective(weights):
        # Calculate portfolio cost for all runs
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

        return {
            "allocations": {
                countries_list[i]: w for i, w in enumerate(optimal_weights)
            },
            "expected_cost": np.mean(portfolio),
            "std_cost": np.std(portfolio),
            "portfolio_costs": portfolio,
        }
    else:
        return None


def create_params_from_dict(country_dict: dict, order_size: int) -> DiscreteRisksParams:
    """
    Reads a dictionary of parameters for a country and creates a
    structured DiscreteRisksParams object.
    """
    print(country_dict)
    return DiscreteRisksParams(
        order_size=order_size,
        disruption_lambda=country_dict["disruption_lambda"],
        disruption_min=country_dict["disruption_min_impact"],
        disruption_max=country_dict["disruption_max_impact"],
        disruption_days_delayed=country_dict["disruption_days_delayed"],
        border_delay_lambda=country_dict["border_delay_lambda"],
        border_delay_min=country_dict["border_min_impact"],
        border_delay_max=country_dict["border_max_impact"],
        border_delay_days_delayed=country_dict["border_days_delayed"],
        damage_probability=country_dict["damage_probability"],
        defective_probability=country_dict["defective_probability"],
        quality_days_delayed=country_dict["quality_days_delayed"],
        cancellation_probability=country_dict["cancellation_probability"],
        cancellation_days_delayed=country_dict["cancellation_days_delayed"],
        tariff_escalation=country_dict["tariff_escal"],
    )
