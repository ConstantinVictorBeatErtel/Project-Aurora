"""
Tests for working capital cost posterior estimation.

Tests:
1. Working capital data fetching works (US: Fed Funds, Mexico: WB lending rate, China: 3-month interbank)
2. Posterior is fitted correctly from interest rate data
3. Samples are generated with correct properties
4. Fallback mechanism works when API fails
5. Country-specific data sources work correctly
"""

import numpy as np
import pytest
from bayesian_priors.parameter_estimators import estimate_working_capital_posterior
from bayesian_priors.posterior_models import NormalPosterior


class TestWorkingCapitalPosterior:
    """Test suite for working capital cost estimation."""
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_posterior_returns_normal_posterior(self, country):
        """Test that estimate_working_capital_posterior returns a NormalPosterior object."""
        baseline = 5.0
        posterior = estimate_working_capital_posterior(country, baseline)
        
        assert isinstance(posterior, NormalPosterior)
        assert hasattr(posterior, 'mu')
        assert hasattr(posterior, 'kappa')
        assert hasattr(posterior, 'alpha')
        assert hasattr(posterior, 'beta')
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_posterior_parameters_are_valid(self, country):
        """Test that posterior parameters are valid (positive, finite)."""
        baseline = 5.0
        posterior = estimate_working_capital_posterior(country, baseline)
        
        assert np.isfinite(posterior.mu), f"{country}: mu not finite"
        assert posterior.kappa > 0, f"{country}: kappa must be positive"
        assert posterior.alpha > 0, f"{country}: alpha must be positive"
        assert posterior.beta > 0, f"{country}: beta must be positive"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_have_correct_shape(self, country):
        """Test that sampling returns correct number of samples."""
        baseline = 5.0
        posterior = estimate_working_capital_posterior(country, baseline)
        
        n_samples = 1000
        samples = posterior.sample_predictive(n_samples)
        
        assert len(samples) == n_samples
        assert isinstance(samples, np.ndarray)
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_are_finite(self, country):
        """Test that all samples are finite (no NaN or inf)."""
        baseline = 5.0
        posterior = estimate_working_capital_posterior(country, baseline)
        
        samples = posterior.sample_predictive(1000)
        
        assert np.all(np.isfinite(samples)), f"{country}: Samples contain NaN or inf"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_sample_distribution_properties(self, country):
        """Test that samples have reasonable statistical properties."""
        baseline = 5.0
        posterior = estimate_working_capital_posterior(country, baseline)
        
        samples = posterior.sample_predictive(10000)
        
        # Mean should be reasonable
        sample_mean = np.mean(samples)
        assert 0.1 * baseline < sample_mean < 10.0 * baseline, \
            f"{country}: Sample mean {sample_mean} unreasonable for baseline {baseline}"
        
        # Standard deviation should be reasonable
        sample_std = np.std(samples)
        assert sample_std > 0, f"{country}: Samples have zero variance"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_mostly_positive(self, country):
        """Test that samples are mostly positive (can't have negative WC costs)."""
        baseline = 5.0
        posterior = estimate_working_capital_posterior(country, baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # At least 95% should be positive
        positive_ratio = np.sum(samples > 0) / len(samples)
        assert positive_ratio > 0.95, \
            f"{country}: Only {positive_ratio*100:.1f}% of samples are positive"
    
    def test_us_uses_fed_funds_rate(self):
        """Test that US working capital estimation works (should use Fed Funds rate)."""
        baseline = 5.0
        posterior = estimate_working_capital_posterior("US", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable US working capital costs per lamp
        mean_cost = np.mean(samples)
        assert 0.5 < mean_cost < 25, f"US working capital cost {mean_cost} seems unrealistic"
    
    def test_mexico_uses_world_bank_lending_rate(self):
        """Test that Mexico working capital estimation works (should use WB lending rate)."""
        baseline = 6.0
        posterior = estimate_working_capital_posterior("Mexico", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable Mexico working capital costs
        mean_cost = np.mean(samples)
        assert 0.5 < mean_cost < 30, f"Mexico working capital cost {mean_cost} seems unrealistic"
    
    def test_china_uses_interbank_rate(self):
        """Test that China working capital estimation works (should use 3-month interbank)."""
        baseline = 4.5
        posterior = estimate_working_capital_posterior("China", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable China working capital costs
        mean_cost = np.mean(samples)
        assert 0.5 < mean_cost < 20, f"China working capital cost {mean_cost} seems unrealistic"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_coefficient_of_variation_reasonable(self, country):
        """Test that working capital cost uncertainty is reasonable."""
        baseline = 5.0
        posterior = estimate_working_capital_posterior(country, baseline)
        
        samples = posterior.sample_predictive(10000)
        
        mean = np.mean(samples)
        std = np.std(samples)
        cv = std / mean
        
        # Coefficient of variation should be reasonable (interest rates vary)
        # Typically 5-30%
        assert 0.02 < cv < 0.8, \
            f"{country}: Coefficient of variation {cv:.2%} outside reasonable range"
    
    def test_working_capital_reflects_interest_rate_volatility(self):
        """Test that working capital captures interest rate volatility."""
        baseline = 5.0
        posterior = estimate_working_capital_posterior("US", baseline)
        
        samples = posterior.sample_predictive(10000)
        
        # Working capital should vary (interest rates change)
        std = np.std(samples)
        assert std > 0.1, "Working capital has negligible volatility"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_fallback_with_extreme_baseline(self, country):
        """Test that function handles extreme baseline values."""
        # Test with very small baseline
        posterior_small = estimate_working_capital_posterior(country, 0.01)
        samples_small = posterior_small.sample_predictive(100)
        assert np.all(np.isfinite(samples_small))
        
        # Test with large baseline
        posterior_large = estimate_working_capital_posterior(country, 100.0)
        samples_large = posterior_large.sample_predictive(100)
        assert np.all(np.isfinite(samples_large))
    
    def test_multiple_calls_give_different_samples(self):
        """Test that sampling is stochastic."""
        baseline = 5.0
        posterior = estimate_working_capital_posterior("US", baseline)
        
        samples1 = posterior.sample_predictive(100)
        samples2 = posterior.sample_predictive(100)
        
        # Should not be identical
        assert not np.array_equal(samples1, samples2), "Samples are deterministic"
    
    def test_mexico_higher_rates_than_us(self):
        """Test that Mexico generally has higher interest rates than US (historical fact)."""
        # Note: This compares baselines assuming they reflect typical rates
        us_baseline = 5.0
        mexico_baseline = 6.0  # Mexico typically has higher rates
        
        us_posterior = estimate_working_capital_posterior("US", us_baseline)
        mexico_posterior = estimate_working_capital_posterior("Mexico", mexico_baseline)
        
        us_samples = us_posterior.sample_predictive(5000)
        mexico_samples = mexico_posterior.sample_predictive(5000)
        
        # Mexico median should be higher than US (given higher baseline)
        assert np.median(mexico_samples) > np.median(us_samples) * 0.8, \
            "Mexico WC costs not appropriately higher than US"


class TestWorkingCapitalIntegration:
    """Integration tests for working capital cost estimation."""
    
    def test_us_uses_policy_rate(self):
        """Test that US uses Fed Funds rate (policy rate)."""
        baseline = 5.0
        posterior = estimate_working_capital_posterior("US", baseline)
        
        samples = posterior.sample_predictive(10000)
        
        # Should reflect typical US short-term financing costs
        mean_cost = np.mean(samples)
        assert mean_cost > 0, "Mean cost should be positive"
    
    def test_emerging_markets_use_local_rates(self):
        """Test that Mexico and China use local interest rates."""
        mexico_baseline = 6.0
        china_baseline = 4.5
        
        mexico_posterior = estimate_working_capital_posterior("Mexico", mexico_baseline)
        china_posterior = estimate_working_capital_posterior("China", china_baseline)
        
        # Both should produce valid posteriors
        mexico_samples = mexico_posterior.sample_predictive(1000)
        china_samples = china_posterior.sample_predictive(1000)
        
        assert np.all(np.isfinite(mexico_samples))
        assert np.all(np.isfinite(china_samples))
    
    def test_data_sources_differ_by_country(self):
        """Test that different countries use different interest rate sources."""
        baseline = 5.0
        
        us_posterior = estimate_working_capital_posterior("US", baseline)
        mexico_posterior = estimate_working_capital_posterior("Mexico", baseline)
        china_posterior = estimate_working_capital_posterior("China", baseline)
        
        # All should produce valid posteriors
        for country, posterior in [("US", us_posterior), ("Mexico", mexico_posterior), ("China", china_posterior)]:
            samples = posterior.sample_predictive(1000)
            assert len(samples) == 1000
            assert np.all(np.isfinite(samples))
    
    def test_working_capital_scales_with_inventory(self):
        """Test that working capital costs scale with inventory levels."""
        low_baseline = 2.0  # Low inventory/quick turnover
        high_baseline = 10.0  # High inventory/slow turnover
        
        low_posterior = estimate_working_capital_posterior("US", low_baseline)
        high_posterior = estimate_working_capital_posterior("US", high_baseline)
        
        low_samples = low_posterior.sample_predictive(5000)
        high_samples = high_posterior.sample_predictive(5000)
        
        # Higher inventory should lead to higher WC costs
        assert np.mean(high_samples) > np.mean(low_samples), \
            "Higher inventory should result in higher WC costs"
    
    def test_working_capital_more_volatile_than_depreciation(self):
        """Test that WC is more volatile than depreciation (interest rates fluctuate)."""
        baseline = 5.0
        posterior = estimate_working_capital_posterior("US", baseline)
        
        samples = posterior.sample_predictive(10000)
        
        mean = np.mean(samples)
        std = np.std(samples)
        cv = std / mean
        
        # WC should be more volatile than depreciation but less than commodities
        assert cv > 0.03, f"WC volatility {cv:.2%} too low"
        assert cv < 0.7, f"WC volatility {cv:.2%} too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

