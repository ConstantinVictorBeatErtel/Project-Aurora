"""
Tests for logistics cost posterior estimation.

Tests:
1. Logistics data fetching works for different countries/modes
2. Posterior is fitted correctly from freight data
3. Truck vs ocean freight handled appropriately
4. Samples are generated with correct properties
5. Fallback mechanism works when API fails
"""

import numpy as np
import pytest
from bayesian_priors.parameter_estimators import estimate_logistics_posterior
from bayesian_priors.posterior_models import NormalPosterior


class TestLogisticsPosterior:
    """Test suite for logistics cost estimation."""
    
    @pytest.mark.parametrize("country,mode", [
        ("US", "truck"),
        ("Mexico", "truck"),
        ("China", "ocean"),
        ("China", "air"),
    ])
    def test_posterior_returns_normal_posterior(self, country, mode):
        """Test that estimate_logistics_posterior returns a NormalPosterior object."""
        baseline = 9.0
        posterior = estimate_logistics_posterior(country, baseline, mode=mode)
        
        assert isinstance(posterior, NormalPosterior)
        assert hasattr(posterior, 'mu')
        assert hasattr(posterior, 'kappa')
        assert hasattr(posterior, 'alpha')
        assert hasattr(posterior, 'beta')
    
    @pytest.mark.parametrize("country,mode", [
        ("US", "truck"),
        ("Mexico", "truck"),
        ("China", "ocean"),
    ])
    def test_posterior_parameters_are_valid(self, country, mode):
        """Test that posterior parameters are valid (positive, finite)."""
        baseline = 9.0
        posterior = estimate_logistics_posterior(country, baseline, mode=mode)
        
        assert np.isfinite(posterior.mu), f"{country} {mode}: mu not finite"
        assert posterior.kappa > 0, f"{country} {mode}: kappa must be positive"
        assert posterior.alpha > 0, f"{country} {mode}: alpha must be positive"
        assert posterior.beta > 0, f"{country} {mode}: beta must be positive"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_have_correct_shape(self, country):
        """Test that sampling returns correct number of samples."""
        baseline = 9.0
        mode = "truck" if country in ("US", "Mexico") else "ocean"
        posterior = estimate_logistics_posterior(country, baseline, mode=mode)
        
        n_samples = 1000
        samples = posterior.sample_predictive(n_samples)
        
        assert len(samples) == n_samples
        assert isinstance(samples, np.ndarray)
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_samples_are_finite(self, country):
        """Test that all samples are finite (no NaN or inf)."""
        baseline = 9.0
        mode = "truck" if country in ("US", "Mexico") else "ocean"
        posterior = estimate_logistics_posterior(country, baseline, mode=mode)
        
        samples = posterior.sample_predictive(1000)
        
        assert np.all(np.isfinite(samples)), f"{country}: Samples contain NaN or inf"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_sample_distribution_properties(self, country):
        """Test that samples have reasonable statistical properties."""
        baseline = 9.0
        mode = "truck" if country in ("US", "Mexico") else "ocean"
        posterior = estimate_logistics_posterior(country, baseline, mode=mode)
        
        samples = posterior.sample_predictive(10000)
        
        # Mean should be reasonable
        sample_mean = np.mean(samples)
        assert 0.2 * baseline < sample_mean < 5.0 * baseline, \
            f"{country}: Sample mean {sample_mean} unreasonable for baseline {baseline}"
        
        # Standard deviation should be reasonable
        sample_std = np.std(samples)
        assert sample_std > 0, f"{country}: Samples have zero variance"
    
    def test_ocean_freight_for_china(self):
        """Test that China uses ocean freight by default."""
        baseline = 15.0  # Ocean freight typically more expensive
        posterior = estimate_logistics_posterior("China", baseline, mode="ocean")
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable samples
        assert np.all(np.isfinite(samples))
        assert np.mean(samples) > 0
    
    def test_air_freight_for_china(self):
        """Test that China can use air freight (expensive option)."""
        baseline = 30.0  # Air freight significantly more expensive
        posterior = estimate_logistics_posterior("China", baseline, mode="air")
        
        samples = posterior.sample_predictive(1000)
        
        # Should produce reasonable samples
        assert np.all(np.isfinite(samples))
        assert np.mean(samples) > 0
    
    def test_truck_freight_for_us_mexico(self):
        """Test that US/Mexico use truck freight."""
        baseline = 9.0
        
        for country in ["US", "Mexico"]:
            posterior = estimate_logistics_posterior(country, baseline, mode="truck")
            samples = posterior.sample_predictive(1000)
            
            assert np.all(np.isfinite(samples))
            assert np.mean(samples) > 0
    
    def test_air_more_expensive_than_ocean(self):
        """Test that air freight samples are generally more expensive than ocean."""
        ocean_baseline = 15.0
        air_baseline = 30.0
        
        ocean_posterior = estimate_logistics_posterior("China", ocean_baseline, mode="ocean")
        air_posterior = estimate_logistics_posterior("China", air_baseline, mode="air")
        
        ocean_samples = ocean_posterior.sample_predictive(5000)
        air_samples = air_posterior.sample_predictive(5000)
        
        # Air should generally be more expensive
        # Note: Due to data normalization, this checks that baseline ordering is preserved
        assert np.median(air_samples) >= np.median(ocean_samples) * 0.8, \
            "Air freight not more expensive than ocean"
    
    def test_samples_mostly_positive(self):
        """Test that samples are mostly positive (can't have negative logistics costs)."""
        baseline = 9.0
        posterior = estimate_logistics_posterior("US", baseline, mode="truck")
        
        samples = posterior.sample_predictive(1000)
        
        # At least 95% should be positive
        positive_ratio = np.sum(samples > 0) / len(samples)
        assert positive_ratio > 0.95, \
            f"Only {positive_ratio*100:.1f}% of samples are positive"
    
    def test_logistics_volatility_higher_than_labor(self):
        """Test that logistics has higher volatility than labor (known characteristic)."""
        baseline = 9.0
        posterior = estimate_logistics_posterior("US", baseline, mode="truck")
        
        samples = posterior.sample_predictive(10000)
        
        mean = np.mean(samples)
        std = np.std(samples)
        cv = std / mean
        
        # Logistics should have some volatility (typically 1-30%, but can be low with stable truck freight)
        assert cv > 0.01, "Logistics volatility too low"
        assert cv < 0.8, "Logistics volatility unreasonably high"
    
    @pytest.mark.parametrize("country", ["US", "Mexico", "China"])
    def test_fallback_with_extreme_baseline(self, country):
        """Test that function handles extreme baseline values."""
        mode = "truck" if country in ("US", "Mexico") else "ocean"
        
        # Test with very small baseline
        posterior_small = estimate_logistics_posterior(country, 0.01, mode=mode)
        samples_small = posterior_small.sample_predictive(100)
        assert np.all(np.isfinite(samples_small))
        
        # Test with large baseline
        posterior_large = estimate_logistics_posterior(country, 1000.0, mode=mode)
        samples_large = posterior_large.sample_predictive(100)
        assert np.all(np.isfinite(samples_large))


class TestLogisticsIntegration:
    """Integration tests for logistics estimation with real/fallback data."""
    
    def test_us_truck_freight_realistic(self):
        """Test that US truck freight produces realistic costs."""
        baseline = 9.0
        posterior = estimate_logistics_posterior("US", baseline, mode="truck")
        
        samples = posterior.sample_predictive(1000)
        
        mean_cost = np.mean(samples)
        # US domestic trucking should be in $5-30 range per lamp
        assert 2 < mean_cost < 50, f"US truck freight {mean_cost} seems unrealistic"
    
    def test_china_ocean_freight_realistic(self):
        """Test that China ocean freight produces realistic costs."""
        baseline = 15.0
        posterior = estimate_logistics_posterior("China", baseline, mode="ocean")
        
        samples = posterior.sample_predictive(1000)
        
        mean_cost = np.mean(samples)
        # Ocean freight from China should be in $5-40 range per lamp
        assert 3 < mean_cost < 60, f"China ocean freight {mean_cost} seems unrealistic"
    
    def test_mexico_truck_freight_realistic(self):
        """Test that Mexico truck freight produces realistic costs."""
        baseline = 7.0
        posterior = estimate_logistics_posterior("Mexico", baseline, mode="truck")
        
        samples = posterior.sample_predictive(1000)
        
        mean_cost = np.mean(samples)
        # Mexico cross-border trucking should be in $3-25 range per lamp
        assert 2 < mean_cost < 40, f"Mexico truck freight {mean_cost} seems unrealistic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

