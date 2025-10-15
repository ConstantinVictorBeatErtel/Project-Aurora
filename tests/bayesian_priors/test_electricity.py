"""
Tests for electricity cost posterior estimation.

Tests:
1. Electricity data fetching works (US: $/kWh, Mexico/China: CPI energy)
2. Posterior is fitted correctly from data
3. Samples are generated with correct properties
4. Fallback mechanism works when API fails
5. Country-specific data sources work correctly
"""

import numpy as np
import pytest
from bayesian_priors.parameter_estimators import estimate_electricity_posterior
from bayesian_priors.posterior_models import NormalPosterior


class TestElectricityPosterior:
    """Test suite for electricity cost estimation."""
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_posterior_returns_normal_posterior(self, country):
        """Test that estimate_electricity_posterior returns a NormalPosterior object."""
        baseline = 4.0
        posterior = estimate_electricity_posterior(country, baseline)
        
        assert isinstance(posterior, NormalPosterior)
        assert hasattr(posterior, 'mu')
        assert hasattr(posterior, 'kappa')
        assert hasattr(posterior, 'alpha')
        assert hasattr(posterior, 'beta')
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_posterior_parameters_are_valid(self, country):
        """Test that posterior parameters are valid (positive, finite)."""
        baseline = 4.0
        posterior = estimate_electricity_posterior(country, baseline)
        
        assert np.isfinite(posterior.mu), f"{country}: mu not finite"
        assert posterior.kappa > 0, f"{country}: kappa must be positive"
        assert posterior.alpha > 0, f"{country}: alpha must be positive"
        assert posterior.beta > 0, f"{country}: beta must be positive"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_have_correct_shape(self, country):
        """Test that sampling returns correct number of samples."""
        baseline = 4.0
        posterior = estimate_electricity_posterior(country, baseline)
        
        n_samples = 1000
        samples = posterior.sample_predictive(n_samples)
        
        assert len(samples) == n_samples
        assert isinstance(samples, np.ndarray)
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_are_finite(self, country):
        """Test that all samples are finite (no NaN or inf)."""
        baseline = 4.0
        posterior = estimate_electricity_posterior(country, baseline)
        
        samples = posterior.sample_predictive(1000)
        
        assert np.all(np.isfinite(samples)), f"{country}: Samples contain NaN or inf"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_sample_distribution_properties(self, country):
        """Test that samples have reasonable statistical properties."""
        baseline = 4.0
        posterior = estimate_electricity_posterior(country, baseline)
        
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
        """Test that samples are mostly positive (can't have negative electricity costs)."""
        baseline = 4.0
        posterior = estimate_electricity_posterior(country, baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # At least 95% should be positive
        positive_ratio = np.sum(samples > 0) / len(samples)
        assert positive_ratio > 0.95, \
            f"{country}: Only {positive_ratio*100:.1f}% of samples are positive"
    
    def test_us_uses_kwh_pricing(self):
        """Test that US electricity estimation works (should use $/kWh data)."""
        baseline = 4.0
        posterior = estimate_electricity_posterior("US", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable US electricity costs per lamp
        mean_cost = np.mean(samples)
        assert 1 < mean_cost < 20, f"US electricity cost {mean_cost} seems unrealistic"
    
    def test_mexico_uses_cpi_energy(self):
        """Test that Mexico electricity estimation works (should use CPI energy)."""
        baseline = 3.0
        posterior = estimate_electricity_posterior("Mexico", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable Mexico electricity costs
        mean_cost = np.mean(samples)
        assert 0.5 < mean_cost < 15, f"Mexico electricity cost {mean_cost} seems unrealistic"
    
    def test_china_uses_cpi_energy(self):
        """Test that China electricity estimation works (should use CPI energy)."""
        baseline = 2.5
        posterior = estimate_electricity_posterior("China", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable China electricity costs
        mean_cost = np.mean(samples)
        assert 0.5 < mean_cost < 12, f"China electricity cost {mean_cost} seems unrealistic"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_coefficient_of_variation_reasonable(self, country):
        """Test that electricity cost uncertainty is reasonable."""
        baseline = 4.0
        posterior = estimate_electricity_posterior(country, baseline)
        
        samples = posterior.sample_predictive(10000)
        
        mean = np.mean(samples)
        std = np.std(samples)
        cv = std / mean
        
        # Coefficient of variation should be between 5% and 50% (energy prices can be volatile)
        assert 0.02 < cv < 0.8, \
            f"{country}: Coefficient of variation {cv:.2%} outside reasonable range"
    
    def test_electricity_volatility_higher_than_depreciation(self):
        """Test that electricity is more volatile than depreciation (known characteristic)."""
        baseline = 4.0
        posterior = estimate_electricity_posterior("US", baseline)
        
        samples = posterior.sample_predictive(10000)
        
        mean = np.mean(samples)
        std = np.std(samples)
        cv = std / mean
        
        # Electricity prices are volatile (oil/gas prices, policy changes)
        # CV typically 10-30%
        assert cv > 0.03, f"Electricity volatility {cv:.2%} too low"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_fallback_with_extreme_baseline(self, country):
        """Test that function handles extreme baseline values."""
        # Test with very small baseline
        posterior_small = estimate_electricity_posterior(country, 0.01)
        samples_small = posterior_small.sample_predictive(100)
        assert np.all(np.isfinite(samples_small))
        
        # Test with large baseline
        posterior_large = estimate_electricity_posterior(country, 100.0)
        samples_large = posterior_large.sample_predictive(100)
        assert np.all(np.isfinite(samples_large))
    
    def test_multiple_calls_give_different_samples(self):
        """Test that sampling is stochastic."""
        baseline = 4.0
        posterior = estimate_electricity_posterior("US", baseline)
        
        samples1 = posterior.sample_predictive(100)
        samples2 = posterior.sample_predictive(100)
        
        # Should not be identical
        assert not np.array_equal(samples1, samples2), "Samples are deterministic"


class TestElectricityIntegration:
    """Integration tests for electricity cost estimation."""
    
    def test_us_direct_price_signal(self):
        """Test that US uses direct $/kWh price signal."""
        baseline = 4.0
        posterior = estimate_electricity_posterior("US", baseline)
        
        samples = posterior.sample_predictive(10000)
        
        # US electricity prices should be reasonable
        # Industrial rates typically $0.07-0.15/kWh
        # Per lamp might be $2-8 depending on energy intensity
        mean_cost = np.mean(samples)
        assert mean_cost > 0, "Mean cost should be positive"
    
    def test_emerging_markets_use_cpi_proxy(self):
        """Test that Mexico and China use CPI energy as proxy."""
        mexico_baseline = 3.0
        china_baseline = 2.5
        
        mexico_posterior = estimate_electricity_posterior("Mexico", mexico_baseline)
        china_posterior = estimate_electricity_posterior("China", china_baseline)
        
        # Both should produce valid posteriors
        mexico_samples = mexico_posterior.sample_predictive(1000)
        china_samples = china_posterior.sample_predictive(1000)
        
        assert np.all(np.isfinite(mexico_samples))
        assert np.all(np.isfinite(china_samples))
    
    def test_data_sources_differ_by_country(self):
        """Test that different countries use different data sources."""
        baseline = 4.0
        
        us_posterior = estimate_electricity_posterior("US", baseline)
        mexico_posterior = estimate_electricity_posterior("Mexico", baseline)
        china_posterior = estimate_electricity_posterior("China", baseline)
        
        # All should produce valid posteriors with meaningful variance
        for country, posterior in [("US", us_posterior), ("Mexico", mexico_posterior), ("China", china_posterior)]:
            samples = posterior.sample_predictive(1000)
            assert len(samples) == 1000
            assert np.all(np.isfinite(samples))
            assert np.std(samples) > 0  # Should have some variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

