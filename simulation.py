import numpy as np

from bayesian_priors.samplers import CountrySamplers, build_samplers_for_country
from config import MONTE_CARLO_SIMULATIONS
from discrete import (
    create_params_from_dict,
    generate_border_delay_risk,
    generate_damaged_risk,
    generate_defective_risk,
    generate_disruption_risk,
    generate_last_minute_cancellation_risk,
)
from structs import DiscreteRisks, DiscreteRisksParams


def factory_costs_with_bayesian_priors(
    samplers: CountrySamplers, risk_params: dict, n_runs: int
) -> np.ndarray:
    """
    Run Monte Carlo simulation using Bayesian posterior samplers.
    
    ALL cost components now use Bayesian priors with real data:
    - raw material (PPI data → Student-t samples)
    - labor (wage data → Student-t samples)
    - logistics (CPI transport → Student-t samples)
    - indirect (ECI/CPI → Student-t samples)
    - electricity ($/kWh or CPI energy → Student-t samples)
    - depreciation (Machinery PPI/WB investment → Student-t samples)
    - working capital (interest rates → Student-t samples)
    - fx (FRED exchange rates → Student-t samples)
    - yield (uncertainty-adjusted Beta)
    """
    # Sample ALL components from posterior predictives
    raw = samplers.raw_material(n_runs)
    labor = samplers.labor(n_runs)
    logistics = samplers.logistics(n_runs)
    indirect = samplers.indirect(n_runs)
    electricity = samplers.electricity(n_runs)
    depreciation = samplers.depreciation(n_runs)
    working = samplers.working_capital(n_runs)
    fx_mult = samplers.fx_multiplier(n_runs)
    yield_rate = samplers.yield_rate(n_runs)

    # Calculate base cost
    base = raw + labor + indirect + logistics + electricity + depreciation + working
    base = base * fx_mult  # Apply FX volatility

    # Apply yield and tariff
    base_tariff = risk_params["tariff"]["fixed"]
    
    # Handle tariff escalation (can be {"fixed": 0} or {"mean": x, "std": y})
    if "fixed" in risk_params["tariff_escal"]:
        tariff_escal = risk_params["tariff_escal"]["fixed"]
    else:
        tariff_escal = np.random.normal(
            risk_params["tariff_escal"]["mean"], 
            risk_params["tariff_escal"]["std"], 
            n_runs
        )
    
    tariff = base_tariff + tariff_escal
    total = base / yield_rate + tariff

    return total


def generate_discrete_risks(params: DiscreteRisksParams) -> DiscreteRisks:
    damaged = generate_damaged_risk(
        params.order_size, params.damage_probability, params.quality_days_delayed
    )
    defective = generate_defective_risk(
        params.order_size, params.defective_probability, params.quality_days_delayed
    )
    cancelled = generate_last_minute_cancellation_risk(
        params.cancellation_probability,
        params.order_size,
        params.cancellation_days_delayed,
    )
    border_delay = generate_border_delay_risk(
        params.border_delay_lambda,
        params.border_delay_min,
        params.border_delay_max,
        params.border_delay_days_delayed,
    )
    disruption = generate_disruption_risk(
        params.disruption_lambda,
        params.disruption_min,
        params.disruption_max,
        params.disruption_days_delayed,
    )

    return DiscreteRisks(
        disruptions=disruption,
        border_delays=border_delay,
        damaged=damaged,
        defectives=defective,
        last_minute_cancellations=cancelled,
    )


def run_monte_carlo(country: str, params: dict, order_size: int) -> np.ndarray:
    """
    The main orchestrator for running simulations.
    Returns a distribution of the TOTAL COST for an entire order.
    """
    samplers = build_samplers_for_country(country, params)
    # 1. GENERATE THE DISTRIBUTION OF TOTAL BASE COSTS
    # this is PER-UNIT costs, one for each simulation run.
    base_cost_per_unit_dist = factory_costs_with_bayesian_priors(
        samplers, params, MONTE_CARLO_SIMULATIONS
    )

    # Then, scale it by the order size to get the TOTAL base cost for the order
    # for each of the simulation runs.
    total_base_cost_dist = base_cost_per_unit_dist * order_size

    # 2. GENERATE THE DISTRIBUTION OF TOTAL RISK COSTS
    risk_params = create_params_from_dict(params, order_size)
    risk_costs_for_order = []
    lost_units_for_order = []

    for _ in range(MONTE_CARLO_SIMULATIONS):
        risk_scenario = generate_discrete_risks(risk_params)
        total_risk_cost = (
            risk_scenario.disruptions.cost
            + risk_scenario.border_delays.cost
            + risk_scenario.damaged.cost
            + risk_scenario.defectives.cost
            + risk_scenario.last_minute_cancellations.cost
        )
        risk_costs_for_order.append(total_risk_cost)

        total_lost_units = (
            risk_scenario.disruptions.delayed_units
            + risk_scenario.border_delays.delayed_units
            + risk_scenario.damaged.delayed_units
            + risk_scenario.defectives.delayed_units
            + risk_scenario.last_minute_cancellations.delayed_units
        )
        lost_units_for_order.append(total_lost_units)

    total_risk_cost_dist = np.array(risk_costs_for_order)
    total_lost_units_dist = np.array(lost_units_for_order)

    # 3. COMBINE THE DISTRIBUTIONS
    # Add the two arrays element-wise. Each element represents one
    # complete, simulated future (one base cost scenario + one risk scenario).
    total_order_cost_dist = total_base_cost_dist + total_risk_cost_dist

    return {
        "total_cost": total_order_cost_dist,
        "lost_units": total_lost_units_dist,
    }
