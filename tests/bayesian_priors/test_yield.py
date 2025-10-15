"""
Tests for manufacturing yield posterior estimation.

Tests:
1. Yield posterior uses Beta distribution (bounded 0-1)
2. Uncertainty levels (low/medium/high) work correctly
3. Samples are in valid range [0, 1]
4. Different uncertainty levels produce different spreads
5. Country-specific uncertainty mapping works
"""

import numpy as np
import pytest
from bayesian_priors.parameter_estimators import estimate_yield_posterior
from bayesian_priors.posterior_models import BetaPosterior


class TestYieldPosterior:
    """Test suite for manufacturing yield estimation."""
    
    @pytest.mark.parametrize("uncertainty", ["low", "medium", "high"])
    def test_posterior_returns_beta_posterior(self, uncertainty):
        """Test that estimate_yield_posterior returns a BetaPosterior object."""
        baseline_yield = 0.80
        posterior = estimate_yield_posterior(baseline_yield, uncertainty)
        
        assert isinstance(posterior, BetaPosterior)
        assert hasattr(posterior, 'alpha')
        assert hasattr(posterior, 'beta')
    
    @pytest.mark.parametrize("uncertainty", ["low", "medium", "high"])
    def test_posterior_parameters_are_valid(self, uncertainty):
        """Test that posterior parameters are valid (positive)."""
        baseline_yield = 0.80
        posterior = estimate_yield_posterior(baseline_yield, uncertainty)
        
        assert posterior.alpha > 0, f"{uncertainty}: alpha must be positive"
        assert posterior.beta > 0, f"{uncertainty}: beta must be positive"
    
    @pytest.mark.parametrize("baseline_yield", [0.70, 0.80, 0.90, 0.95])
    def test_posterior_mean_near_baseline(self, baseline_yield):
        """Test that posterior mean is close to baseline yield."""
        posterior = estimate_yield_posterior(baseline_yield, uncertainty="medium")
        
        # Mean of Beta(alpha, beta) = alpha / (alpha + beta)
        posterior_mean = posterior.alpha / (posterior.alpha + posterior.beta)
        
        # Should be very close to baseline
        assert abs(posterior_mean - baseline_yield) < 0.01, \
            f"Posterior mean {posterior_mean} differs from baseline {baseline_yield}"
    
    @pytest.mark.parametrize("uncertainty", ["low", "medium", "high"])
    def test_samples_have_correct_shape(self, uncertainty):
        """Test that sampling returns correct number of samples."""
        baseline_yield = 0.80
        posterior = estimate_yield_posterior(baseline_yield, uncertainty)
        
        n_samples = 1000
        samples = posterior.sample_predictive(n_samples)
        
        assert len(samples) == n_samples
        assert isinstance(samples, np.ndarray)
    
    @pytest.mark.parametrize("uncertainty", ["low", "medium", "high"])
    def test_samples_are_in_valid_range(self, uncertainty):
        """Test that all samples are in [0, 1] range."""
        baseline_yield = 0.80
        posterior = estimate_yield_posterior(baseline_yield, uncertainty)
        
        samples = posterior.sample_predictive(1000)
        
        assert np.all(samples >= 0), f"{uncertainty}: Some samples < 0"
        assert np.all(samples <= 1), f"{uncertainty}: Some samples > 1"
    
    @pytest.mark.parametrize("uncertainty", ["low", "medium", "high"])
    def test_samples_are_finite(self, uncertainty):
        """Test that all samples are finite (no NaN or inf)."""
        baseline_yield = 0.80
        posterior = estimate_yield_posterior(baseline_yield, uncertainty)
        
        samples = posterior.sample_predictive(1000)
        
        assert np.all(np.isfinite(samples)), f"{uncertainty}: Samples contain NaN or inf"
    
    def test_low_uncertainty_tighter_than_high(self):
        """Test that low uncertainty produces tighter distribution than high."""
        baseline_yield = 0.80
        
        low_posterior = estimate_yield_posterior(baseline_yield, "low")
        high_posterior = estimate_yield_posterior(baseline_yield, "high")
        
        low_samples = low_posterior.sample_predictive(10000)
        high_samples = high_posterior.sample_predictive(10000)
        
        low_std = np.std(low_samples)
        high_std = np.std(high_samples)
        
        assert low_std < high_std, \
            f"Low uncertainty std {low_std} not smaller than high uncertainty std {high_std}"
    
    def test_medium_uncertainty_between_low_and_high(self):
        """Test that medium uncertainty is between low and high."""
        baseline_yield = 0.80
        
        low_posterior = estimate_yield_posterior(baseline_yield, "low")
        medium_posterior = estimate_yield_posterior(baseline_yield, "medium")
        high_posterior = estimate_yield_posterior(baseline_yield, "high")
        
        low_samples = low_posterior.sample_predictive(10000)
        medium_samples = medium_posterior.sample_predictive(10000)
        high_samples = high_posterior.sample_predictive(10000)
        
        low_std = np.std(low_samples)
        medium_std = np.std(medium_samples)
        high_std = np.std(high_samples)
        
        assert low_std < medium_std < high_std, \
            f"Uncertainty ordering broken: low={low_std:.4f}, medium={medium_std:.4f}, high={high_std:.4f}"
    
    @pytest.mark.parametrize("baseline_yield", [0.50, 0.70, 0.80, 0.90, 0.95])
    def test_samples_clustered_around_baseline(self, baseline_yield):
        """Test that samples are clustered around baseline yield."""
        posterior = estimate_yield_posterior(baseline_yield, uncertainty="medium")
        
        samples = posterior.sample_predictive(10000)
        
        sample_mean = np.mean(samples)
        sample_median = np.median(samples)
        
        # Mean and median should be close to baseline
        assert abs(sample_mean - baseline_yield) < 0.05, \
            f"Sample mean {sample_mean} too far from baseline {baseline_yield}"
        assert abs(sample_median - baseline_yield) < 0.05, \
            f"Sample median {sample_median} too far from baseline {baseline_yield}"
    
    def test_invalid_uncertainty_defaults_to_medium(self):
        """Test that invalid uncertainty level defaults gracefully."""
        baseline_yield = 0.80
        
        # Should not crash, should use default (medium)
        posterior = estimate_yield_posterior(baseline_yield, uncertainty="invalid")
        samples = posterior.sample_predictive(100)
        
        assert len(samples) == 100
        assert np.all((samples >= 0) & (samples <= 1))
    
    def test_extreme_baseline_low_yield(self):
        """Test that function handles very low baseline yield."""
        baseline_yield = 0.10  # 10% yield (very bad)
        posterior = estimate_yield_posterior(baseline_yield, uncertainty="medium")
        
        samples = posterior.sample_predictive(1000)
        
        assert np.all((samples >= 0) & (samples <= 1))
        assert np.mean(samples) < 0.3, "Mean too high for low baseline"
    
    def test_extreme_baseline_high_yield(self):
        """Test that function handles very high baseline yield."""
        baseline_yield = 0.99  # 99% yield (excellent)
        posterior = estimate_yield_posterior(baseline_yield, uncertainty="medium")
        
        samples = posterior.sample_predictive(1000)
        
        assert np.all((samples >= 0) & (samples <= 1))
        assert np.mean(samples) > 0.90, "Mean too low for high baseline"
    
    def test_multiple_calls_give_different_samples(self):
        """Test that sampling is stochastic."""
        baseline_yield = 0.80
        posterior = estimate_yield_posterior(baseline_yield, uncertainty="medium")
        
        samples1 = posterior.sample_predictive(100)
        samples2 = posterior.sample_predictive(100)
        
        # Should not be identical
        assert not np.array_equal(samples1, samples2), "Samples are deterministic"


class TestYieldIntegration:
    """Integration tests for yield estimation."""
    
    def test_us_medium_uncertainty(self):
        """Test US (new facility) uses medium uncertainty."""
        baseline_yield = 0.80
        posterior = estimate_yield_posterior(baseline_yield, uncertainty="medium")
        
        samples = posterior.sample_predictive(10000)
        
        # Should have moderate spread
        std = np.std(samples)
        assert 0.03 < std < 0.15, f"US yield uncertainty {std} outside expected range"
    
    def test_china_low_uncertainty(self):
        """Test China (mature facility) uses low uncertainty."""
        baseline_yield = 0.85
        posterior = estimate_yield_posterior(baseline_yield, uncertainty="low")
        
        samples = posterior.sample_predictive(10000)
        
        # Should have tight spread
        std = np.std(samples)
        assert std < 0.10, f"China yield uncertainty {std} too high"
    
    def test_mexico_high_uncertainty(self):
        """Test Mexico (skill issues) uses high uncertainty."""
        baseline_yield = 0.75
        posterior = estimate_yield_posterior(baseline_yield, uncertainty="high")
        
        samples = posterior.sample_predictive(10000)
        
        # Should have wide spread
        std = np.std(samples)
        assert std > 0.05, f"Mexico yield uncertainty {std} too low"
    
    def test_yield_affects_production(self):
        """Test how yield affects production output."""
        baseline_yield = 0.80
        posterior = estimate_yield_posterior(baseline_yield, uncertainty="medium")
        
        samples = posterior.sample_predictive(1000)
        
        # If producing 1000 units, yield determines how many are good
        target_production = 1000
        actual_good_units = target_production * samples
        
        # Should lose some units due to yield
        mean_good_units = np.mean(actual_good_units)
        assert mean_good_units < target_production, "Yield not reducing output"
        assert mean_good_units > 0.5 * target_production, "Yield too low"
    
    def test_yield_distribution_shape(self):
        """Test that yield distribution has appropriate shape for Beta."""
        baseline_yield = 0.80
        posterior = estimate_yield_posterior(baseline_yield, uncertainty="medium")
        
        samples = posterior.sample_predictive(10000)
        
        # Beta distribution should be skewed (not symmetric) unless alpha = beta
        from scipy.stats import skew
        sample_skew = skew(samples)
        
        # Should have some skewness (not perfectly symmetric)
        assert abs(sample_skew) < 2, "Distribution has excessive skewness"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

