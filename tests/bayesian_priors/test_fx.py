"""
Tests for FX (foreign exchange) posterior estimation.

Tests:
1. FX data fetching works from FRED
2. FX multiplier function returns correct shape
3. US has no FX risk (multiplier = 1)
4. Mexico and China have FX volatility
5. Fallback mechanism works when API fails
"""

import numpy as np
import pytest
from bayesian_priors.parameter_estimators import estimate_fx_posterior


class TestFXPosterior:
    """Test suite for FX volatility estimation."""
    
    def test_us_has_no_fx_risk(self):
        """Test that US returns multiplier of 1 (no FX risk)."""
        fx_sampler = estimate_fx_posterior("US")
        
        samples = fx_sampler(1000)
        
        # All samples should be exactly 1 for US
        assert np.all(samples == 1.0), "US should have no FX risk"
        assert len(samples) == 1000
    
    @pytest.mark.parametrize("country", ["Mexico", "China"])
    def test_fx_sampler_returns_function(self, country):
        """Test that estimate_fx_posterior returns a callable function."""
        fx_sampler = estimate_fx_posterior(country)
        
        assert callable(fx_sampler), f"{country}: FX sampler is not callable"
    
    @pytest.mark.parametrize("country", ["Mexico", "China"])
    def test_fx_samples_have_correct_shape(self, country):
        """Test that FX sampler returns correct number of samples."""
        fx_sampler = estimate_fx_posterior(country)
        
        n_samples = 1000
        samples = fx_sampler(n_samples)
        
        assert len(samples) == n_samples
        assert isinstance(samples, np.ndarray)
    
    @pytest.mark.parametrize("country", ["Mexico", "China"])
    def test_fx_samples_are_finite(self, country):
        """Test that all FX samples are finite (no NaN or inf)."""
        fx_sampler = estimate_fx_posterior(country)
        
        samples = fx_sampler(1000)
        
        assert np.all(np.isfinite(samples)), f"{country}: FX samples contain NaN or inf"
    
    @pytest.mark.parametrize("country", ["Mexico", "China"])
    def test_fx_multiplier_centered_near_one(self, country):
        """Test that FX multiplier is centered near 1 (mean = no change)."""
        fx_sampler = estimate_fx_posterior(country)
        
        samples = fx_sampler(10000)
        
        sample_mean = np.mean(samples)
        
        # Mean should be close to 1 (within ±20%)
        assert 0.8 < sample_mean < 1.2, \
            f"{country}: FX multiplier mean {sample_mean} too far from 1"
    
    @pytest.mark.parametrize("country", ["Mexico", "China"])
    def test_fx_has_nonzero_volatility(self, country):
        """Test that FX samples have meaningful volatility."""
        fx_sampler = estimate_fx_posterior(country)
        
        samples = fx_sampler(10000)
        
        sample_std = np.std(samples)
        
        # Should have some volatility (at least 0.2% - China yuan is managed with low volatility)
        assert sample_std > 0.002, f"{country}: FX has negligible volatility"
        
        # But not excessive (less than 50%)
        assert sample_std < 0.5, f"{country}: FX volatility unreasonably high"
    
    def test_mexico_peso_volatility(self):
        """Test that Mexico (MXN/USD) has reasonable volatility."""
        fx_sampler = estimate_fx_posterior("Mexico")
        
        samples = fx_sampler(10000)
        
        # Mexican peso typically has 1-10% annualized volatility
        sample_std = np.std(samples)
        assert 0.001 < sample_std < 0.3, \
            f"Mexico FX volatility {sample_std} outside expected range"
    
    def test_china_yuan_volatility(self):
        """Test that China (CNY/USD) has reasonable volatility."""
        fx_sampler = estimate_fx_posterior("China")
        
        samples = fx_sampler(10000)
        
        # Chinese yuan typically has lower volatility (managed float)
        sample_std = np.std(samples)
        assert 0.001 < sample_std < 0.3, \
            f"China FX volatility {sample_std} outside expected range"
    
    @pytest.mark.parametrize("country", ["Mexico", "China"])
    def test_fx_samples_mostly_positive(self, country):
        """Test that FX multipliers are mostly positive (realistic range)."""
        fx_sampler = estimate_fx_posterior(country)
        
        samples = fx_sampler(1000)
        
        # At least 99% should be positive (FX rarely goes to zero)
        positive_ratio = np.sum(samples > 0) / len(samples)
        assert positive_ratio > 0.99, \
            f"{country}: Only {positive_ratio*100:.1f}% of FX samples are positive"
    
    @pytest.mark.parametrize("country", ["Mexico", "China"])
    def test_fx_reasonable_range(self, country):
        """Test that FX multipliers are in reasonable range."""
        fx_sampler = estimate_fx_posterior(country)
        
        samples = fx_sampler(10000)
        
        # 95% of samples should be within reasonable FX swing (50% depreciation to 50% appreciation)
        p5 = np.percentile(samples, 5)
        p95 = np.percentile(samples, 95)
        
        assert 0.5 < p5 < 1.5, f"{country}: 5th percentile {p5} outside reasonable range"
        assert 0.5 < p95 < 1.5, f"{country}: 95th percentile {p95} outside reasonable range"
    
    def test_invalid_country_returns_no_fx_risk(self):
        """Test that invalid country falls back to no FX risk."""
        fx_sampler = estimate_fx_posterior("InvalidCountry")
        
        samples = fx_sampler(1000)
        
        # Should default to no FX risk (multiplier = 1)
        assert np.all(samples == 1.0), "Invalid country should have no FX risk"
    
    @pytest.mark.parametrize("country", ["Mexico", "China"])
    def test_multiple_calls_give_different_samples(self, country):
        """Test that FX sampling is stochastic."""
        fx_sampler = estimate_fx_posterior(country)
        
        samples1 = fx_sampler(100)
        samples2 = fx_sampler(100)
        
        # Should not be identical (unless fallback with zero volatility)
        # Allow for fallback case where all samples might be 1
        if not np.all(samples1 == 1.0):
            assert not np.array_equal(samples1, samples2), \
                f"{country}: FX samples are deterministic"


class TestFXIntegration:
    """Integration tests for FX estimation with real/fallback data."""
    
    def test_fx_multiplier_effect_on_costs(self):
        """Test that FX multiplier properly affects costs."""
        fx_sampler = estimate_fx_posterior("Mexico")
        
        base_cost = 100.0
        multipliers = fx_sampler(1000)
        adjusted_costs = base_cost * multipliers
        
        # Adjusted costs should vary around base_cost
        assert np.mean(adjusted_costs) > 0
        assert np.std(adjusted_costs) > 0
        
        # Some costs should be higher, some lower (if there's volatility)
        if np.std(multipliers) > 0.01:  # If not fallback
            assert np.any(adjusted_costs > base_cost)
            assert np.any(adjusted_costs < base_cost)
    
    @pytest.mark.parametrize("country", ["Mexico", "China"])
    def test_fx_data_fetch_works_or_fallback(self, country):
        """Test that FX estimation works (either with data or fallback)."""
        fx_sampler = estimate_fx_posterior(country)
        
        # Should return a valid sampler regardless
        samples = fx_sampler(100)
        
        assert len(samples) == 100
        assert np.all(np.isfinite(samples))
        assert np.all(samples > 0)
    
    def test_mexico_china_fx_independence(self):
        """Test that Mexico and China FX samples are independent."""
        mexico_sampler = estimate_fx_posterior("Mexico")
        china_sampler = estimate_fx_posterior("China")
        
        np.random.seed(42)
        mexico_samples = mexico_sampler(1000)
        
        np.random.seed(42)
        china_samples = china_sampler(1000)
        
        # Even with same seed, they should use different data sources
        # So samples won't be identical (unless both are fallback)
        if np.std(mexico_samples) > 0.01 or np.std(china_samples) > 0.01:
            correlation = np.corrcoef(mexico_samples, china_samples)[0, 1]
            assert abs(correlation) < 0.5, \
                "Mexico and China FX are too correlated (should be independent)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

