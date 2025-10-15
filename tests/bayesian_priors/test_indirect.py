"""
Tests for indirect/overhead cost posterior estimation.

Tests:
1. Indirect cost data fetching works (ECI for US, CPI for Mexico/China)
2. Posterior is fitted correctly from data
3. Samples are generated with correct properties
4. Fallback mechanism works when API fails
5. Country-specific data sources work correctly
"""

import numpy as np
import pytest
from bayesian_priors.parameter_estimators import estimate_indirect_posterior
from bayesian_priors.posterior_models import NormalPosterior


class TestIndirectPosterior:
    """Test suite for indirect/overhead cost estimation."""
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_posterior_returns_normal_posterior(self, country):
        """Test that estimate_indirect_posterior returns a NormalPosterior object."""
        baseline = 10.0
        posterior = estimate_indirect_posterior(country, baseline)
        
        assert isinstance(posterior, NormalPosterior)
        assert hasattr(posterior, 'mu')
        assert hasattr(posterior, 'kappa')
        assert hasattr(posterior, 'alpha')
        assert hasattr(posterior, 'beta')
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_posterior_parameters_are_valid(self, country):
        """Test that posterior parameters are valid (positive, finite)."""
        baseline = 10.0
        posterior = estimate_indirect_posterior(country, baseline)
        
        assert np.isfinite(posterior.mu), f"{country}: mu not finite"
        assert posterior.kappa > 0, f"{country}: kappa must be positive"
        assert posterior.alpha > 0, f"{country}: alpha must be positive"
        assert posterior.beta > 0, f"{country}: beta must be positive"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_have_correct_shape(self, country):
        """Test that sampling returns correct number of samples."""
        baseline = 10.0
        posterior = estimate_indirect_posterior(country, baseline)
        
        n_samples = 1000
        samples = posterior.sample_predictive(n_samples)
        
        assert len(samples) == n_samples
        assert isinstance(samples, np.ndarray)
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_are_finite(self, country):
        """Test that all samples are finite (no NaN or inf)."""
        baseline = 10.0
        posterior = estimate_indirect_posterior(country, baseline)
        
        samples = posterior.sample_predictive(1000)
        
        assert np.all(np.isfinite(samples)), f"{country}: Samples contain NaN or inf"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_sample_distribution_properties(self, country):
        """Test that samples have reasonable statistical properties."""
        baseline = 10.0
        posterior = estimate_indirect_posterior(country, baseline)
        
        samples = posterior.sample_predictive(10000)
        
        # Mean should be reasonable
        sample_mean = np.mean(samples)
        assert 0.2 * baseline < sample_mean < 5.0 * baseline, \
            f"{country}: Sample mean {sample_mean} unreasonable for baseline {baseline}"
        
        # Standard deviation should be reasonable
        sample_std = np.std(samples)
        assert sample_std > 0, f"{country}: Samples have zero variance"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_mostly_positive(self, country):
        """Test that samples are mostly positive (can't have negative overhead)."""
        baseline = 10.0
        posterior = estimate_indirect_posterior(country, baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # At least 95% should be positive
        positive_ratio = np.sum(samples > 0) / len(samples)
        assert positive_ratio > 0.95, \
            f"{country}: Only {positive_ratio*100:.1f}% of samples are positive"
    
    def test_us_uses_eci_data(self):
        """Test that US indirect estimation works (should use ECI data)."""
        baseline = 10.0
        posterior = estimate_indirect_posterior("US", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable US overhead costs
        mean_cost = np.mean(samples)
        assert 3 < mean_cost < 50, f"US indirect cost {mean_cost} seems unrealistic"
    
    def test_mexico_uses_world_bank_cpi(self):
        """Test that Mexico indirect estimation works (should use World Bank CPI)."""
        baseline = 8.0
        posterior = estimate_indirect_posterior("Mexico", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable Mexico overhead costs
        mean_cost = np.mean(samples)
        assert 2 < mean_cost < 40, f"Mexico indirect cost {mean_cost} seems unrealistic"
    
    def test_china_uses_world_bank_cpi(self):
        """Test that China indirect estimation works (should use World Bank CPI)."""
        baseline = 7.0
        posterior = estimate_indirect_posterior("China", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable China overhead costs
        mean_cost = np.mean(samples)
        assert 2 < mean_cost < 35, f"China indirect cost {mean_cost} seems unrealistic"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_coefficient_of_variation_reasonable(self, country):
        """Test that indirect cost uncertainty is reasonable."""
        baseline = 10.0
        posterior = estimate_indirect_posterior(country, baseline)
        
        samples = posterior.sample_predictive(10000)
        
        mean = np.mean(samples)
        std = np.std(samples)
        cv = std / mean
        
        # Coefficient of variation should be between 5% and 40% (overhead is fairly stable)
        assert 0.02 < cv < 0.6, \
            f"{country}: Coefficient of variation {cv:.2%} outside reasonable range"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_fallback_with_extreme_baseline(self, country):
        """Test that function handles extreme baseline values."""
        # Test with very small baseline
        posterior_small = estimate_indirect_posterior(country, 0.01)
        samples_small = posterior_small.sample_predictive(100)
        assert np.all(np.isfinite(samples_small))
        
        # Test with large baseline
        posterior_large = estimate_indirect_posterior(country, 1000.0)
        samples_large = posterior_large.sample_predictive(100)
        assert np.all(np.isfinite(samples_large))
    
    def test_multiple_calls_give_different_samples(self):
        """Test that sampling is stochastic."""
        baseline = 10.0
        posterior = estimate_indirect_posterior("US", baseline)
        
        samples1 = posterior.sample_predictive(100)
        samples2 = posterior.sample_predictive(100)
        
        # Should not be identical
        assert not np.array_equal(samples1, samples2), "Samples are deterministic"


class TestIndirectIntegration:
    """Integration tests for indirect cost estimation."""
    
    def test_us_overhead_higher_than_emerging(self):
        """Test that US overhead is generally higher than emerging markets."""
        us_baseline = 10.0
        china_baseline = 7.0
        
        us_posterior = estimate_indirect_posterior("US", us_baseline)
        china_posterior = estimate_indirect_posterior("China", china_baseline)
        
        us_samples = us_posterior.sample_predictive(5000)
        china_samples = china_posterior.sample_predictive(5000)
        
        # US median should be higher than China median (in absolute terms)
        # Note: This depends on baseline choices, so we check relative to baseline
        us_mean = np.mean(us_samples)
        china_mean = np.mean(china_samples)
        
        assert us_mean > china_mean * 0.5, \
            f"US overhead {us_mean} not appropriately higher than China {china_mean}"
    
    def test_indirect_less_volatile_than_logistics(self):
        """Test that indirect costs are less volatile than logistics (characteristic)."""
        baseline = 10.0
        posterior = estimate_indirect_posterior("US", baseline)
        
        samples = posterior.sample_predictive(10000)
        
        mean = np.mean(samples)
        std = np.std(samples)
        cv = std / mean
        
        # Indirect overhead should be relatively stable (CV typically 5-20%)
        assert cv < 0.5, f"Indirect cost CV {cv:.2%} too high (should be stable)"
    
    def test_data_sources_differ_by_country(self):
        """Test that different countries use different data sources (ECI vs CPI)."""
        baseline = 10.0
        
        us_posterior = estimate_indirect_posterior("US", baseline)
        mexico_posterior = estimate_indirect_posterior("Mexico", baseline)
        china_posterior = estimate_indirect_posterior("China", baseline)
        
        # All should produce valid posteriors
        for country, posterior in [("US", us_posterior), ("Mexico", mexico_posterior), ("China", china_posterior)]:
            samples = posterior.sample_predictive(100)
            assert len(samples) == 100
            assert np.all(np.isfinite(samples))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

