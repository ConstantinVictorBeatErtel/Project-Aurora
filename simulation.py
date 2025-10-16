import numpy as np
from numpy.typing import NDArray

from bayesian_priors.samplers import CountrySamplers, build_samplers_for_country
from config import MONTE_CARLO_SIMULATIONS
from discrete import (
    generate_border_delay_risk,
    generate_damaged_risk,
    generate_defective_risk,
    generate_disruption_risk,
    generate_last_minute_cancellation_risk,
    generate_tariff_escalation,
)
from structs import DiscreteRisks, DiscreteRisksParams
from utils import create_params_from_dict, sample_from_spec


def factory_costs_with_bayesian_priors(
    samplers: CountrySamplers, risk_params: dict, order_size: int
) -> NDArray:
    """Sample factory costs from bayesian priors (if known) or assumed distributions
    Returns:
      - An array of costs for"""
    # Sample from posterior predictives
    raw = samplers.raw_material(order_size)
    labor = samplers.labor(order_size)
    logistics = samplers.logistics(order_size)
    fx_mult = samplers.fx_multiplier(order_size)
    yield_rate = samplers.yield_rate(order_size)

    # Constants (no public data)
    indirect = sample_from_spec(risk_params["indirect"], order_size)
    electricity = sample_from_spec(risk_params["electricity"], order_size)
    depreciation = sample_from_spec(risk_params["depreciation"], order_size)
    working = sample_from_spec(risk_params["working_capital"], order_size)

    # Calculate base cost
    base = raw + labor + indirect + logistics + electricity + depreciation + working
    base = base * fx_mult  # Apply FX volatility

    # Apply yield and tariff
    tariff = risk_params["tariff"]["fixed"]

    # Calculate total cost
    total = base / yield_rate * (1 + tariff)

    return total


def generate_discrete_risks(params: DiscreteRisksParams) -> DiscreteRisks:
    """Wrapper for calling all discrete risk events

    Returns:
      - DiscreteRisks: an object containing the $ cost and estimated lost units
        for each discrete risk event"""
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
    tariffs = generate_tariff_escalation(params.tariff_escalation)

    return DiscreteRisks(
        disruptions=disruption,
        border_delays=border_delay,
        damaged=damaged,
        defectives=defective,
        last_minute_cancellations=cancelled,
        tariff_cost=tariffs,
    )


def run_monte_carlo(country: str, params: dict, order_size: int) -> dict[str, NDArray]:
    """
    Runs Monte Carlo simulation using:
      - Bayesian posterior samplers where we possess some data distribution
      - Assumed model parameters when we possess no priors
      - Modeled risk events (with assumed model parameters)

    This is the main orchestrator for running simulations.

    Returns:
      - a dictionary of the TOTAL COST & LOST UNITS for simulated orders
    """
    samplers = build_samplers_for_country(country, params)

    # 1. GENERATE THE DISTRIBUTION OF TOTAL BASE COSTS
    # this is PER-UNIT costs, one for each simulation run.
    base_cost_per_unit_dist = factory_costs_with_bayesian_priors(
        samplers, params, MONTE_CARLO_SIMULATIONS
    )

    # Then, scale it by the order size to get the TOTAL base cost for the order
    # for each of the simulation runs.
    # this is a matrix of dimension (order_size x MONTE_CARLO_SIMULATIONS)
    total_base_cost_dist = base_cost_per_unit_dist * order_size

    # 2. GENERATE THE DISTRIBUTION OF TOTAL RISK COSTS
    risk_params = create_params_from_dict(params, order_size)
    risk_costs_for_order = []
    lost_units_for_order = []
    tariff_escalations_for_order = []

    for _ in range(MONTE_CARLO_SIMULATIONS):
        risk_scenario = generate_discrete_risks(risk_params)

        # model tariff escalation (not a risk cost, but applied to the whole order)
        tariff_escalation = risk_scenario.tariff_cost
        tariff_escalations_for_order.append(tariff_escalation)

        # sum up the risk premiums
        total_risk_cost = (
            risk_scenario.disruptions.cost
            + risk_scenario.border_delays.cost
            + risk_scenario.damaged.cost
            + risk_scenario.defectives.cost
            + risk_scenario.last_minute_cancellations.cost
        )
        risk_costs_for_order.append(total_risk_cost)

        # track the lost units
        total_lost_units = (
            risk_scenario.disruptions.delayed_units
            + risk_scenario.border_delays.delayed_units
            + risk_scenario.damaged.delayed_units
            + risk_scenario.defectives.delayed_units
            + risk_scenario.last_minute_cancellations.delayed_units
        )
        lost_units_for_order.append(total_lost_units)

    # convert these to np.arrays for matrix math later
    total_risk_cost_dist = np.array(risk_costs_for_order)
    total_lost_units_dist = np.array(lost_units_for_order)
    total_tariff_escalation_dist = np.array(tariff_escalations_for_order)

    # 3. COMBINE THE DISTRIBUTIONS
    # Add the two arrays element-wise. Each element represents one
    # complete, simulated future (one base cost scenario + one risk scenario).

    # NOTE: we're applying the tariff escalation math here
    total_order_cost_dist = (
        total_base_cost_dist * (1 + total_tariff_escalation_dist)
    ) + total_risk_cost_dist

    return {
        "total_cost": total_order_cost_dist,
        "lost_units": total_lost_units_dist,
    }
