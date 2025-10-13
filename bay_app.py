import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import beta
import copy
from bayesian_tesla import create_bayesian_simulator

# --- Parameters ---
# Parameters updated based on the provided table
countries = {
    'US': {
        'raw': {'dist': 'normal', 'mean': 40, 'std': 4},
        'labor': {'dist': 'normal', 'mean': 12, 'std': 0.6},
        'indirect': {'dist': 'normal', 'mean': 10, 'std': 0.5},
        'logistics': {'dist': 'normal', 'mean': 9, 'std': 0},
        'electricity': {'dist': 'normal', 'mean': 4, 'std': 0.4},
        'depreciation': {'dist': 'normal', 'mean': 5, 'std': 0.25},
        'working_capital': {'dist': 'normal', 'mean': 5, 'std': 0.5},
        'yield_params': {'a': 79, 'b': 20},  # Approx for mean 0.8, std 0.04
        'tariff': {'fixed': 0},
        'tariff_escal': {'mean': 0, 'std': 0},
        'currency_std': 0,
        'disruption_prob': 0.05,
        'disruption_impact': 10,
        'border_mean': 0,
        'border_std': 0,
        'border_threshold': 2,
        'border_cost_per_hr': 10,
        'damage_prob': 0.001,
        'damage_impact': 20,
        'skills_mean': 0,
        'skills_std': 0,
        'cancellation_prob': 0,
        'cancellation_impact': 50
    },
    'Mexico': {
        'raw': {'dist': 'normal', 'mean': 35, 'std': 3.5},
        'labor': {'dist': 'normal', 'mean': 8, 'std': 0.4},
        'indirect': {'dist': 'normal', 'mean': 8, 'std': 0.4},
        'logistics': {'dist': 'normal', 'mean': 7, 'std': 0.056},
        'electricity': {'dist': 'normal', 'mean': 3, 'std': 0.3},
        'depreciation': {'dist': 'normal', 'mean': 1, 'std': 0.05},
        'working_capital': {'dist': 'normal', 'mean': 6, 'std': 0.6},
        'yield_params': {'a': 12, 'b': 1},  # Approx for mean 0.9, std 0.08
        'tariff': {'fixed': 15.5},
        'tariff_escal': {'mean': 0, 'std': 5},
        'currency_std': 0.08,
        'disruption_prob': 0.1,
        'disruption_impact': 10,
        'border_mean': 0.83,
        'border_std': 0.67,
        'border_threshold': 2,
        'border_cost_per_hr': 10,
        'damage_prob': 0.01,
        'damage_impact': 20,
        'skills_mean': 0,
        'skills_std': 0.05,
        'cancellation_prob': 0,
        'cancellation_impact': 50
    },
    'China': {
        'raw': {'dist': 'normal', 'mean': 30, 'std': 3},
        'labor': {'dist': 'normal', 'mean': 4, 'std': 0.2},
        'indirect': {'dist': 'normal', 'mean': 4, 'std': 0.2},
        'logistics': {'dist': 'normal', 'mean': 12, 'std': 0.936},
        'electricity': {'dist': 'normal', 'mean': 4, 'std': 0.4},
        'depreciation': {'dist': 'normal', 'mean': 5, 'std': 0.25},
        'working_capital': {'dist': 'normal', 'mean': 10, 'std': 1},
        'yield_params': {'a': 49, 'b': 3},  # Approx for mean 0.95, std 0.03
        'tariff': {'fixed': 15},
        'tariff_escal': {'mean': 0, 'std': 20},
        'currency_std': 0.10,
        'disruption_prob': 0.2,
        'disruption_impact': 10,
        'border_mean': 0,
        'border_std': 0,
        'border_threshold': 2,
        'border_cost_per_hr': 10,
        'damage_prob': 0.03,
        'damage_impact': 20,
        'skills_mean': 0,
        'skills_std': 0,
        'cancellation_prob': 0.05,
        'cancellation_impact': 50
    }
}

# --- Simulation Function ---
def simulate_country(params, n_runs):
    """Runs a Monte Carlo simulation for a single country's sourcing cost."""
    # Sample costs from distributions
    raw = np.random.normal(params['raw']['mean'], params['raw']['std'], n_runs)
    labor = np.random.normal(params['labor']['mean'], params['labor']['std'], n_runs)
    indirect = np.random.normal(params['indirect']['mean'], params['indirect']['std'], n_runs)
    electricity = np.random.normal(params['electricity']['mean'], params['electricity']['std'], n_runs)
    depreciation = np.random.normal(params['depreciation']['mean'], params['depreciation']['std'], n_runs)
    working = np.random.normal(params['working_capital']['mean'], params['working_capital']['std'], n_runs)
    yield_ = beta.rvs(params['yield_params']['a'], params['yield_params']['b'], size=n_runs)

    # Handle lognormal distribution for logistics
    if params['logistics'].get('dist') == 'lognormal':
        m, s = params['logistics']['mean'], params['logistics']['std']
        sigma = np.sqrt(np.log(1 + (s**2 / m**2)))
        mu = np.log(m) - (sigma**2 / 2)
        logistics = np.random.lognormal(mu, sigma, n_runs)
    else:
        logistics = np.random.normal(params['logistics']['mean'], params['logistics']['std'], n_runs)

    # Calculate base cost
    base = raw + labor + indirect + logistics + electricity + depreciation + working

    # Apply currency fluctuation
    base *= (1 + np.random.normal(0, params['currency_std'], n_runs))

    # Apply tariff and potential escalation
    tariff = np.full(n_runs, params['tariff']['fixed']) + np.random.normal(params['tariff_escal']['mean'], params['tariff_escal']['std'], n_runs)

    # Calculate total cost before discrete risks
    total = base / yield_ + tariff

    # --- Add Discrete Risk Events ---
    disruption = np.random.binomial(1, params['disruption_prob'], n_runs)
    total += disruption * params['disruption_impact']
    border_time = np.random.normal(params['border_mean'], params['border_std'], n_runs)
    border_cost = np.maximum(0, border_time - params['border_threshold']) * params['border_cost_per_hr']
    total += border_cost
    damage = np.random.binomial(1, params['damage_prob'], n_runs)
    total += damage * params['damage_impact']
    skills_adj = np.random.normal(params['skills_mean'], params['skills_std'], n_runs)
    total *= (1 + skills_adj)
    cancellation = np.random.binomial(1, params['cancellation_prob'], n_runs)
    total += cancellation * params['cancellation_impact']

    return total

# --- Sensitivity Analysis Functions ---
def get_sensitivity_factors(params):
    """
    Dynamically creates a list of ALL non-zero numerical factors to test.
    This is a more comprehensive approach.
    """
    factors = []
    # Exclude keys that are not independent variables for sensitivity analysis
    exclude_keys = ['b', 'dist', 'border_threshold'] 

    def recurse_dict(d, path=(), name_path=()):
        for key, value in d.items():
            if key in exclude_keys:
                continue
            
            new_path = path + (key,)
            # Create a more readable name for the factor
            new_name_path = name_path + (key.replace('_', ' ').title(),)
            
            if isinstance(value, dict):
                recurse_dict(value, new_path, new_name_path)
            elif isinstance(value, (int, float)) and value != 0:
                # Join the path to create a unique, descriptive name
                factor_name = " ".join(new_name_path)
                factors.append((factor_name, new_path))

    recurse_dict(params)
    return factors

def run_sensitivity_analysis(base_params, factors_to_test, n_runs, swing=0.20):
    """Runs a one-at-a-time sensitivity analysis for a given set of factors."""
    results = []
    base_costs = simulate_country(base_params, n_runs)
    baseline_mean = np.mean(base_costs)

    for factor_name, param_path in factors_to_test:
        params_low = copy.deepcopy(base_params)
        params_high = copy.deepcopy(base_params)
        base_value = base_params
        for key in param_path:
            base_value = base_value[key]
        
        if base_value == 0:
            continue

        low_value = base_value * (1 - swing)
        high_value = base_value * (1 + swing)
        temp_low = params_low
        temp_high = params_high
        for i, key in enumerate(param_path):
            if i == len(param_path) - 1:
                temp_low[key] = low_value
                temp_high[key] = high_value
            else:
                temp_low = temp_low[key]
                temp_high = temp_high[key]
        mean_low = np.mean(simulate_country(params_low, n_runs))
        mean_high = np.mean(simulate_country(params_high, n_runs))
        results.append({
            'Factor': factor_name,
            'Low Cost': round(mean_low, 2),
            'High Cost': round(mean_high, 2),
            'Impact': round(abs(mean_high - mean_low), 2)
        })
    return pd.DataFrame(results), baseline_mean

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Tesla Sourcing Monte Carlo Simulation Dashboard")
st.write("This dashboard simulates total costs per lamp for sourcing from the US, Mexico, or China, accounting for uncertainties in costs, yields, and risks.")

# Toggle for Bayesian approach
use_bayesian = st.checkbox(
    "Use Bayesian Priors (fetch real data from FRED)",
    value=False,
    help="Enable to use real economic data (PPI, FX rates) with Bayesian posterior estimation. This accounts for parameter uncertainty and gives more conservative risk estimates.",
    key="main_bayesian_checkbox"
)

n_runs = st.number_input("Number of Simulation Runs", min_value=1000, max_value=100000, value=10000, step=1000)

if st.button("Run Global Simulation"):
    with st.spinner("Running simulations for all countries..."):
        posterior_params = None
        if use_bayesian:
            try:
                # Use Bayesian simulators with real data
                bayesian_sims = create_bayesian_simulator(countries)
                all_costs = {country: bayesian_sims[country](n_runs) for country in countries.keys()}
                posterior_params = None  # bayesian_tesla doesn't return posterior_params
            except ImportError:
                all_costs = {country: simulate_country(params, n_runs) for country, params in countries.items()}
                st.error("Could not import from `bay_app.py`. Please ensure the file is in the same directory and contains the `create_bayesian_simulator` function. Running with standard simulation.", icon="🚨")
        else:
            all_costs = {country: simulate_country(params, n_runs) for country, params in countries.items()}

        st.subheader("Summary Statistics")
        cols = st.columns(len(countries))
        for i, (country, costs) in enumerate(all_costs.items()):
            with cols[i]:
                st.markdown(f"### {country}")
                st.metric(label="Expected Cost", value=f"${np.mean(costs):.2f}")
                st.metric(label="Standard Deviation", value=f"${np.std(costs):.2f}")
                st.metric(label="5th Percentile Cost", value=f"${np.percentile(costs, 5):.2f}")
                st.metric(label="95th Percentile Cost", value=f"${np.percentile(costs, 95):.2f}")

        # --- Display Bayesian Updates if applicable ---
        if posterior_params:
            st.subheader("Bayesian Posterior Updates")
            st.write("The following parameters were updated from their initial (prior) estimates using real-world data.")
            update_data = []
            for country, params in posterior_params.items():
                # Compare Raw Mean
                if params['raw']['mean'] != countries[country]['raw']['mean']:
                    update_data.append([country, "Raw Material Mean", f"${countries[country]['raw']['mean']:.2f}", f"${params['raw']['mean']:.2f}"])
                # Compare Raw Std Dev
                if params['raw']['std'] != countries[country]['raw']['std']:
                    update_data.append([country, "Raw Material Std Dev", f"${countries[country]['raw']['std']:.2f}", f"${params['raw']['std']:.2f}"])
                # Compare Currency Std Dev
                if params['currency_std'] != countries[country]['currency_std']:
                     update_data.append([country, "Currency Volatility (Std Dev)", f"{countries[country]['currency_std']:.2%}", f"{params['currency_std']:.2%}"])
            
            df_updates = pd.DataFrame(update_data, columns=["Country", "Parameter", "Original (Prior)", "Updated (Posterior)"])
            st.dataframe(df_updates, use_container_width=True)


        percentiles = np.arange(1, 101)
        percentile_data = []
        for country, costs in all_costs.items():
            percentile_values = np.percentile(costs, percentiles)
            for p, v in zip(percentiles, percentile_values):
                percentile_data.append({'Country': country, 'Percentile': p, 'Cost ($)': v})
        df_percentiles = pd.DataFrame(percentile_data)

        st.subheader("Cost Distribution by Percentile")
        fig = px.line(df_percentiles, x='Percentile', y='Cost ($)', color='Country', title='Cost Distribution by Percentile', labels={'Percentile': 'Cost Percentile', 'Cost ($)': 'Total Cost ($/lamp)'}, hover_data={'Cost ($)': ':.2f'})
        fig.update_layout(legend_title_text='Country', yaxis_title="Total Cost ($/lamp)", xaxis_title="Cost Percentile (%)")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
# --- Sensitivity Analysis UI ---
st.header("Sensitivity Analysis")
st.write("Analyze which factors have the biggest impact on the total cost for a specific country. This chart shows how a +/- 20% change in each input variable affects the expected total cost.")

sa_col1, sa_col2 = st.columns([1, 3])
with sa_col1:
    sa_country = st.selectbox("Select Country to Analyze", list(countries.keys()), key="sa_country")
    run_sa = st.button("Run Sensitivity Analysis")

if run_sa:
    with st.spinner(f"Running sensitivity analysis for {sa_country}..."):
        base_params = countries[sa_country]
        factors_to_test = get_sensitivity_factors(base_params)
        sa_results, baseline_mean = run_sensitivity_analysis(base_params, factors_to_test, n_runs)
        sa_results = sa_results.sort_values(by='Impact', ascending=True)

        # Create Tornado Plot
        fig = go.Figure()

        high_impact = (sa_results['High Cost'] - baseline_mean).round(2)
        low_impact = (sa_results['Low Cost'] - baseline_mean).round(2)

        fig.add_trace(go.Bar(
            y=sa_results['Factor'],
            x=high_impact,
            name='High Estimate (Input +20%)',
            orientation='h',
            marker_color='indianred',
            text=[f"${x:+.2f}" for x in high_impact],
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            y=sa_results['Factor'],
            x=low_impact,
            name='Low Estimate (Input -20%)',
            orientation='h',
            marker_color='lightblue',
            text=[f"${x:+.2f}" for x in low_impact],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f'Tornado Plot for {sa_country} (Baseline Cost: ${baseline_mean:.2f})',
            xaxis_title='Impact on Total Cost ($/lamp)',
            yaxis_title='Sensitivity Factor',
            barmode='relative',
            yaxis_autorange='reversed',
            height=max(400, len(sa_results) * 30), # Dynamically adjust height
            legend=dict(x=0.01, y=0.01, traceorder='normal'),
            margin=dict(l=250), # Increased left margin for longer factor names
            uniformtext_minsize=8, 
            uniformtext_mode='hide'
        )
        
        with sa_col2:
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
# --- Bayesian Prior Comparison ---
st.header("Bayesian Prior Impact Analysis")
st.write("This section shows how using Bayesian priors (with real economic data) changes the estimated parameters compared to hand-picked baseline values.")

if st.button("Compare Bayesian vs Baseline Parameters"):
    with st.spinner("Fetching real data and building Bayesian posteriors..."):
        # Build Bayesian samplers to get posterior distributions
        from bayesian_tesla import build_samplers_for_country

        comparison_data = []
        for country, params in countries.items():
            samplers = build_samplers_for_country(country, params)

            # Sample from posteriors to get mean estimates
            n_samples = 1000

            # Raw material comparison
            baseline_raw = params['raw']['mean']
            bayesian_raw_samples = samplers.raw_material(n_samples)
            bayesian_raw_mean = np.mean(bayesian_raw_samples)
            bayesian_raw_std = np.std(bayesian_raw_samples)

            comparison_data.append({
                'Country': country,
                'Parameter': 'Raw Material Cost ($/lamp)',
                'Baseline Mean': f"${baseline_raw:.2f}",
                'Bayesian Mean': f"${bayesian_raw_mean:.2f}",
                'Bayesian Std': f"${bayesian_raw_std:.2f}",
                'Difference': f"${bayesian_raw_mean - baseline_raw:+.2f}"
            })

            # FX multiplier comparison (only for non-US)
            if country != 'US':
                baseline_fx = 1.0  # baseline assumes no FX change
                bayesian_fx_samples = samplers.fx_multiplier(n_samples)
                bayesian_fx_mean = np.mean(bayesian_fx_samples)
                bayesian_fx_std = np.std(bayesian_fx_samples)

                comparison_data.append({
                    'Country': country,
                    'Parameter': 'FX Multiplier (relative to baseline)',
                    'Baseline Mean': f"{baseline_fx:.4f}",
                    'Bayesian Mean': f"{bayesian_fx_mean:.4f}",
                    'Bayesian Std': f"{bayesian_fx_std:.4f}",
                    'Difference': f"{bayesian_fx_mean - baseline_fx:+.4f}"
                })

            # Yield comparison
            baseline_yield_mean = params['yield_params']['a'] / (params['yield_params']['a'] + params['yield_params']['b'])
            bayesian_yield_samples = samplers.yield_rate(n_samples)
            bayesian_yield_mean = np.mean(bayesian_yield_samples)
            bayesian_yield_std = np.std(bayesian_yield_samples)

            comparison_data.append({
                'Country': country,
                'Parameter': 'Manufacturing Yield (%)',
                'Baseline Mean': f"{baseline_yield_mean*100:.2f}%",
                'Bayesian Mean': f"{bayesian_yield_mean*100:.2f}%",
                'Bayesian Std': f"{bayesian_yield_std*100:.2f}%",
                'Difference': f"{(bayesian_yield_mean - baseline_yield_mean)*100:+.2f}%"
            })

        df_comparison = pd.DataFrame(comparison_data)

        st.subheader("Parameter Comparison Table")
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

        # Create visualization showing differences
        st.subheader("Parameter Differences: Bayesian vs Baseline")

        # Extract numeric differences for plotting
        plot_data = []
        for _, row in df_comparison.iterrows():
            diff_str = row['Difference']
            # Parse the difference value
            if '$' in diff_str:
                diff_val = float(diff_str.replace('$', '').replace('+', ''))
            elif '%' in diff_str:
                diff_val = float(diff_str.replace('%', '').replace('+', ''))
            else:
                diff_val = float(diff_str.replace('+', ''))

            plot_data.append({
                'Country': row['Country'],
                'Parameter': row['Parameter'],
                'Difference': diff_val
            })

        df_plot = pd.DataFrame(plot_data)

        fig_comparison = px.bar(
            df_plot,
            x='Difference',
            y='Parameter',
            color='Country',
            barmode='group',
            title='Bayesian Posterior Adjustments (Difference from Baseline)',
            labels={'Difference': 'Change from Baseline', 'Parameter': ''},
            orientation='h'
        )
        fig_comparison.update_layout(
            height=max(400, len(df_plot) * 50),
            xaxis_title="Difference (Bayesian - Baseline)",
            yaxis={'categoryorder': 'total ascending'},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

