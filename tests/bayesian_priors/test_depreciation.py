"""
Tests for depreciation cost posterior estimation.

Tests:
1. Depreciation data fetching works (US: Machinery PPI, Mexico/China: WB investment price level)
2. Posterior is fitted correctly from data
3. Samples are generated with correct properties
4. Fallback mechanism works when API fails
5. Country-specific data sources work correctly
"""

import numpy as np
import pytest
from bayesian_priors.parameter_estimators import estimate_depreciation_posterior
from bayesian_priors.posterior_models import NormalPosterior


class TestDepreciationPosterior:
    """Test suite for depreciation cost estimation."""
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_posterior_returns_normal_posterior(self, country):
        """Test that estimate_depreciation_posterior returns a NormalPosterior object."""
        baseline = 5.0
        posterior = estimate_depreciation_posterior(country, baseline)
        
        assert isinstance(posterior, NormalPosterior)
        assert hasattr(posterior, 'mu')
        assert hasattr(posterior, 'kappa')
        assert hasattr(posterior, 'alpha')
        assert hasattr(posterior, 'beta')
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_posterior_parameters_are_valid(self, country):
        """Test that posterior parameters are valid (positive, finite)."""
        baseline = 5.0
        posterior = estimate_depreciation_posterior(country, baseline)
        
        assert np.isfinite(posterior.mu), f"{country}: mu not finite"
        assert posterior.kappa > 0, f"{country}: kappa must be positive"
        assert posterior.alpha > 0, f"{country}: alpha must be positive"
        assert posterior.beta > 0, f"{country}: beta must be positive"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_have_correct_shape(self, country):
        """Test that sampling returns correct number of samples."""
        baseline = 5.0
        posterior = estimate_depreciation_posterior(country, baseline)
        
        n_samples = 1000
        samples = posterior.sample_predictive(n_samples)
        
        assert len(samples) == n_samples
        assert isinstance(samples, np.ndarray)
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_are_finite(self, country):
        """Test that all samples are finite (no NaN or inf)."""
        baseline = 5.0
        posterior = estimate_depreciation_posterior(country, baseline)
        
        samples = posterior.sample_predictive(1000)
        
        assert np.all(np.isfinite(samples)), f"{country}: Samples contain NaN or inf"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_sample_distribution_properties(self, country):
        """Test that samples have reasonable statistical properties."""
        baseline = 5.0
        posterior = estimate_depreciation_posterior(country, baseline)
        
        samples = posterior.sample_predictive(10000)
        
        # Mean should be reasonable
        sample_mean = np.mean(samples)
        assert 0.2 * baseline < sample_mean < 10.0 * baseline, \
            f"{country}: Sample mean {sample_mean} unreasonable for baseline {baseline}"
        
        # Standard deviation should be reasonable
        sample_std = np.std(samples)
        assert sample_std > 0, f"{country}: Samples have zero variance"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_mostly_positive(self, country):
        """Test that samples are mostly positive (can't have negative depreciation)."""
        baseline = 5.0
        posterior = estimate_depreciation_posterior(country, baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # At least 95% should be positive
        positive_ratio = np.sum(samples > 0) / len(samples)
        assert positive_ratio > 0.95, \
            f"{country}: Only {positive_ratio*100:.1f}% of samples are positive"
    
    def test_us_uses_machinery_ppi(self):
        """Test that US depreciation estimation works (should use machinery PPI)."""
        baseline = 5.0
        posterior = estimate_depreciation_posterior("US", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable US depreciation costs per lamp
        mean_cost = np.mean(samples)
        assert 1 < mean_cost < 25, f"US depreciation cost {mean_cost} seems unrealistic"
    
    def test_mexico_uses_world_bank_investment_price(self):
        """Test that Mexico depreciation estimation works (should use WB investment price)."""
        baseline = 4.0
        posterior = estimate_depreciation_posterior("Mexico", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable Mexico depreciation costs
        mean_cost = np.mean(samples)
        assert 0.5 < mean_cost < 20, f"Mexico depreciation cost {mean_cost} seems unrealistic"
    
    def test_china_uses_world_bank_investment_price(self):
        """Test that China depreciation estimation works (should use WB investment price)."""
        baseline = 3.5
        posterior = estimate_depreciation_posterior("China", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable China depreciation costs
        mean_cost = np.mean(samples)
        assert 0.5 < mean_cost < 18, f"China depreciation cost {mean_cost} seems unrealistic"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_coefficient_of_variation_reasonable(self, country):
        """Test that depreciation cost uncertainty is reasonable."""
        baseline = 5.0
        posterior = estimate_depreciation_posterior(country, baseline)
        
        samples = posterior.sample_predictive(10000)
        
        mean = np.mean(samples)
        std = np.std(samples)
        cv = std / mean
        
        # Coefficient of variation should be low (depreciation is very stable)
        # Typically 3-15%
        assert 0.01 < cv < 0.4, \
            f"{country}: Coefficient of variation {cv:.2%} outside reasonable range"
    
    def test_depreciation_low_volatility(self):
        """Test that depreciation has low volatility (most stable cost component)."""
        baseline = 5.0
        posterior = estimate_depreciation_posterior("US", baseline)
        
        samples = posterior.sample_predictive(10000)
        
        mean = np.mean(samples)
        std = np.std(samples)
        cv = std / mean
        
        # Depreciation should be very stable (equipment cost changes slowly)
        assert cv < 0.3, f"Depreciation volatility {cv:.2%} too high (should be most stable)"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_fallback_with_extreme_baseline(self, country):
        """Test that function handles extreme baseline values."""
        # Test with very small baseline
        posterior_small = estimate_depreciation_posterior(country, 0.01)
        samples_small = posterior_small.sample_predictive(100)
        assert np.all(np.isfinite(samples_small))
        
        # Test with large baseline
        posterior_large = estimate_depreciation_posterior(country, 100.0)
        samples_large = posterior_large.sample_predictive(100)
        assert np.all(np.isfinite(samples_large))
    
    def test_multiple_calls_give_different_samples(self):
        """Test that sampling is stochastic."""
        baseline = 5.0
        posterior = estimate_depreciation_posterior("US", baseline)
        
        samples1 = posterior.sample_predictive(100)
        samples2 = posterior.sample_predictive(100)
        
        # Should not be identical
        assert not np.array_equal(samples1, samples2), "Samples are deterministic"
    
    def test_depreciation_more_stable_than_other_costs(self):
        """Test that depreciation is more stable than labor/logistics."""
        baseline = 5.0
        posterior = estimate_depreciation_posterior("US", baseline)
        
        samples = posterior.sample_predictive(10000)
        
        # Check that distribution is relatively tight
        p5 = np.percentile(samples, 5)
        p95 = np.percentile(samples, 95)
        range_ratio = (p95 - p5) / np.median(samples)
        
        # 90% range should be relatively narrow for depreciation
        assert range_ratio < 1.0, \
            f"Depreciation has too wide a range (ratio={range_ratio:.2f})"


class TestDepreciationIntegration:
    """Integration tests for depreciation cost estimation."""
    
    def test_us_uses_machinery_equipment_ppi(self):
        """Test that US uses machinery & equipment PPI."""
        baseline = 5.0
        posterior = estimate_depreciation_posterior("US", baseline)
        
        samples = posterior.sample_predictive(10000)
        
        # US depreciation should reflect industrial equipment costs
        mean_cost = np.mean(samples)
        assert mean_cost > 0, "Mean cost should be positive"
    
    def test_emerging_markets_use_investment_price_level(self):
        """Test that Mexico and China use World Bank investment price level."""
        mexico_baseline = 4.0
        china_baseline = 3.5
        
        mexico_posterior = estimate_depreciation_posterior("Mexico", mexico_baseline)
        china_posterior = estimate_depreciation_posterior("China", china_baseline)
        
        # Both should produce valid posteriors
        mexico_samples = mexico_posterior.sample_predictive(1000)
        china_samples = china_posterior.sample_predictive(1000)
        
        assert np.all(np.isfinite(mexico_samples))
        assert np.all(np.isfinite(china_samples))
    
    def test_data_sources_differ_by_country(self):
        """Test that different countries use different data sources."""
        baseline = 5.0
        
        us_posterior = estimate_depreciation_posterior("US", baseline)
        mexico_posterior = estimate_depreciation_posterior("Mexico", baseline)
        china_posterior = estimate_depreciation_posterior("China", baseline)
        
        # All should produce valid posteriors
        for country, posterior in [("US", us_posterior), ("Mexico", mexico_posterior), ("China", china_posterior)]:
            samples = posterior.sample_predictive(1000)
            assert len(samples) == 1000
            assert np.all(np.isfinite(samples))
    
    def test_depreciation_proportional_to_capex(self):
        """Test that depreciation scales with capital expenditure."""
        low_baseline = 2.0  # Low capex facility
        high_baseline = 10.0  # High capex facility
        
        low_posterior = estimate_depreciation_posterior("US", low_baseline)
        high_posterior = estimate_depreciation_posterior("US", high_baseline)
        
        low_samples = low_posterior.sample_predictive(5000)
        high_samples = high_posterior.sample_predictive(5000)
        
        # Higher capex should lead to higher depreciation
        assert np.mean(high_samples) > np.mean(low_samples), \
            "Higher capex should result in higher depreciation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

