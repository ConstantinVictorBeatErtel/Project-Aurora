import numpy as np

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


def create_params_from_dict(country_dict: dict, order_size: int) -> DiscreteRisksParams:
    """
    Reads a dictionary of parameters for a country and creates a
    structured DiscreteRisksParams object.
    """
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
