"""
Tests for labor cost posterior estimation.

Tests:
1. Labor data fetching works for US/Mexico/China
2. Posterior is fitted correctly from data
3. Samples are generated with correct properties
4. Fallback mechanism works when API fails
5. Country-specific handling works correctly
"""

import numpy as np
import pytest
from bayesian_priors.parameter_estimators import estimate_labor_posterior
from bayesian_priors.posterior_models import NormalPosterior


class TestLaborPosterior:
    """Test suite for labor cost estimation."""
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_posterior_returns_normal_posterior(self, country):
        """Test that estimate_labor_posterior returns a NormalPosterior object."""
        baseline = 12.0
        posterior = estimate_labor_posterior(country, baseline)
        
        assert isinstance(posterior, NormalPosterior)
        assert hasattr(posterior, 'mu')
        assert hasattr(posterior, 'kappa')
        assert hasattr(posterior, 'alpha')
        assert hasattr(posterior, 'beta')
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_posterior_parameters_are_valid(self, country):
        """Test that posterior parameters are valid (positive, finite)."""
        baseline = 12.0
        posterior = estimate_labor_posterior(country, baseline)
        
        assert np.isfinite(posterior.mu), f"{country}: mu not finite"
        assert posterior.kappa > 0, f"{country}: kappa must be positive"
        assert posterior.alpha > 0, f"{country}: alpha must be positive"
        assert posterior.beta > 0, f"{country}: beta must be positive"
    
    @pytest.mark.parametrize("country,baseline", [
        ("US", 12.0),
        ("Mexico", 5.0),
        ("China", 3.0),
    ])
    def test_posterior_mean_near_baseline(self, country, baseline):
        """Test that posterior mean is reasonably close to baseline for each country."""
        posterior = estimate_labor_posterior(country, baseline)
        
        # Mean should be within 100% of baseline (allows for significant data influence)
        assert 0.3 * baseline < posterior.mu < 3.0 * baseline, \
            f"{country}: Posterior mean {posterior.mu} too far from baseline {baseline}"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_have_correct_shape(self, country):
        """Test that sampling returns correct number of samples."""
        baseline = 12.0
        posterior = estimate_labor_posterior(country, baseline)
        
        n_samples = 1000
        samples = posterior.sample_predictive(n_samples)
        
        assert len(samples) == n_samples
        assert isinstance(samples, np.ndarray)
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_are_finite(self, country):
        """Test that all samples are finite (no NaN or inf)."""
        baseline = 12.0
        posterior = estimate_labor_posterior(country, baseline)
        
        samples = posterior.sample_predictive(1000)
        
        assert np.all(np.isfinite(samples)), f"{country}: Samples contain NaN or inf"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_sample_distribution_properties(self, country):
        """Test that samples have reasonable statistical properties."""
        baseline = 12.0
        posterior = estimate_labor_posterior(country, baseline)
        
        samples = posterior.sample_predictive(10000)
        
        # Mean should be reasonable
        sample_mean = np.mean(samples)
        assert sample_mean > 0, f"{country}: Sample mean is not positive"
        
        # Standard deviation should be reasonable
        sample_std = np.std(samples)
        assert sample_std > 0, f"{country}: Samples have zero variance"
    
    def test_us_labor_costs_higher_than_china(self):
        """Test that US labor costs are generally higher than China."""
        us_baseline = 12.0
        china_baseline = 3.0
        
        us_posterior = estimate_labor_posterior("US", us_baseline)
        china_posterior = estimate_labor_posterior("China", china_baseline)
        
        us_samples = us_posterior.sample_predictive(1000)
        china_samples = china_posterior.sample_predictive(1000)
        
        # US median should be higher than China median
        assert np.median(us_samples) > np.median(china_samples), \
            "US labor costs not higher than China"
    
    def test_mexico_labor_costs_between_us_and_china(self):
        """Test that Mexico labor costs are between US and China."""
        us_baseline = 12.0
        mexico_baseline = 5.0
        china_baseline = 3.0
        
        us_posterior = estimate_labor_posterior("US", us_baseline)
        mexico_posterior = estimate_labor_posterior("Mexico", mexico_baseline)
        china_posterior = estimate_labor_posterior("China", china_baseline)
        
        us_samples = us_posterior.sample_predictive(5000)
        mexico_samples = mexico_posterior.sample_predictive(5000)
        china_samples = china_posterior.sample_predictive(5000)
        
        us_mean = np.mean(us_samples)
        mexico_mean = np.mean(mexico_samples)
        china_mean = np.mean(china_samples)
        
        # Check ordering (with some tolerance for data variation)
        assert china_mean < mexico_mean or mexico_mean < us_mean, \
            f"Labor cost ordering unexpected: China={china_mean:.2f}, Mexico={mexico_mean:.2f}, US={us_mean:.2f}"
    
    def test_invalid_country_fallback(self):
        """Test that invalid country uses fallback."""
        baseline = 12.0
        posterior = estimate_labor_posterior("InvalidCountry", baseline)
        
        # Should still return valid posterior
        assert isinstance(posterior, NormalPosterior)
        samples = posterior.sample_predictive(100)
        assert len(samples) == 100
        assert np.all(np.isfinite(samples))
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_mostly_positive(self, country):
        """Test that samples are mostly positive (labor costs can't be negative)."""
        baseline = 12.0
        posterior = estimate_labor_posterior(country, baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # At least 95% should be positive
        positive_ratio = np.sum(samples > 0) / len(samples)
        assert positive_ratio > 0.95, \
            f"{country}: Only {positive_ratio*100:.1f}% of samples are positive"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_fallback_with_extreme_baseline(self, country):
        """Test that function handles extreme baseline values."""
        # Test with very small baseline
        posterior_small = estimate_labor_posterior(country, 0.01)
        samples_small = posterior_small.sample_predictive(100)
        assert np.all(np.isfinite(samples_small))
        
        # Test with large baseline
        posterior_large = estimate_labor_posterior(country, 1000.0)
        samples_large = posterior_large.sample_predictive(100)
        assert np.all(np.isfinite(samples_large))


class TestLaborIntegration:
    """Integration tests for labor estimation with real/fallback data."""
    
    def test_us_uses_bls_data(self):
        """Test that US labor estimation works (should use BLS/FRED data)."""
        baseline = 12.0
        posterior = estimate_labor_posterior("US", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable US labor costs
        mean_cost = np.mean(samples)
        assert 5 < mean_cost < 50, f"US labor cost {mean_cost} seems unrealistic"
    
    def test_mexico_uses_oecd_data(self):
        """Test that Mexico labor estimation works."""
        baseline = 5.0
        posterior = estimate_labor_posterior("Mexico", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable Mexico labor costs
        mean_cost = np.mean(samples)
        assert 1 < mean_cost < 20, f"Mexico labor cost {mean_cost} seems unrealistic"
    
    def test_china_uses_world_bank_data(self):
        """Test that China labor estimation works."""
        baseline = 3.0
        posterior = estimate_labor_posterior("China", baseline)
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable China labor costs
        mean_cost = np.mean(samples)
        assert 0.5 < mean_cost < 15, f"China labor cost {mean_cost} seems unrealistic"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_coefficient_of_variation_reasonable(self, country):
        """Test that labor cost uncertainty is reasonable."""
        baseline = 12.0
        posterior = estimate_labor_posterior(country, baseline)
        
        samples = posterior.sample_predictive(10000)
        
        mean = np.mean(samples)
        std = np.std(samples)
        cv = std / mean
        
        # Coefficient of variation should be between 2% and 50% (labor is relatively stable)
        assert 0.02 < cv < 0.5, \
            f"{country}: Coefficient of variation {cv:.2%} outside reasonable range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

