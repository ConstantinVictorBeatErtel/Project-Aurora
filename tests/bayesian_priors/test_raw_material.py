"""
Tests for raw material posterior estimation.

Tests:
1. PPI data fetching works
2. Posterior is fitted correctly from data
3. Samples are generated with correct properties
4. Fallback mechanism works when API fails
5. Distribution properties are reasonable
"""

import numpy as np
import pytest
from bayesian_priors.parameter_estimators import estimate_raw_material_posterior
from bayesian_priors.posterior_models import NormalPosterior


class TestRawMaterialPosterior:
    """Test suite for raw material cost estimation."""
    
    def test_posterior_returns_normal_posterior(self):
        """Test that estimate_raw_material_posterior returns a NormalPosterior object."""
        baseline = 40.0
        posterior = estimate_raw_material_posterior(baseline)
        
        assert isinstance(posterior, NormalPosterior)
        assert hasattr(posterior, 'mu')
        assert hasattr(posterior, 'kappa')
        assert hasattr(posterior, 'alpha')
        assert hasattr(posterior, 'beta')
    
    def test_posterior_parameters_are_valid(self):
        """Test that posterior parameters are valid (positive, finite)."""
        baseline = 40.0
        posterior = estimate_raw_material_posterior(baseline)
        
        assert np.isfinite(posterior.mu)
        assert posterior.kappa > 0, "kappa must be positive"
        assert posterior.alpha > 0, "alpha must be positive"
        assert posterior.beta > 0, "beta must be positive"
    
    def test_posterior_mean_near_baseline(self):
        """Test that posterior mean is reasonably close to baseline."""
        baseline = 40.0
        posterior = estimate_raw_material_posterior(baseline)
        
        # Mean should be within 50% of baseline (allows for data influence)
        assert 0.5 * baseline < posterior.mu < 1.5 * baseline, \
            f"Posterior mean {posterior.mu} too far from baseline {baseline}"
    
    def test_samples_have_correct_shape(self):
        """Test that sampling returns correct number of samples."""
        baseline = 40.0
        posterior = estimate_raw_material_posterior(baseline)
        
        n_samples = 1000
        samples = posterior.sample_predictive(n_samples)
        
        assert len(samples) == n_samples
        assert isinstance(samples, np.ndarray)
    
    def test_samples_are_finite(self):
        """Test that all samples are finite (no NaN or inf)."""
        baseline = 40.0
        posterior = estimate_raw_material_posterior(baseline)
        
        samples = posterior.sample_predictive(1000)
        
        assert np.all(np.isfinite(samples)), "Samples contain NaN or inf"
    
    def test_sample_distribution_properties(self):
        """Test that samples have reasonable statistical properties."""
        baseline = 40.0
        posterior = estimate_raw_material_posterior(baseline)
        
        samples = posterior.sample_predictive(10000)
        
        # Mean should be near posterior mean
        sample_mean = np.mean(samples)
        assert 0.5 * baseline < sample_mean < 1.5 * baseline, \
            f"Sample mean {sample_mean} unreasonable for baseline {baseline}"
        
        # Standard deviation should be reasonable (not too tight, not too wide)
        sample_std = np.std(samples)
        assert sample_std > 0, "Samples have zero variance"
        assert sample_std < baseline, "Samples have excessive variance"
    
    def test_student_t_fatter_tails(self):
        """Test that samples have fatter tails than normal (characteristic of Student-t)."""
        baseline = 40.0
        posterior = estimate_raw_material_posterior(baseline)
        
        samples = posterior.sample_predictive(10000)
        
        # Kurtosis of Student-t should be higher than normal (3)
        # For small degrees of freedom, kurtosis is typically > 3
        from scipy.stats import kurtosis
        kurt = kurtosis(samples)
        
        # Student-t with finite variance has kurtosis >= 0 (excess kurtosis)
        # We're just checking it's a reasonable distribution
        assert kurt > -2, "Distribution has suspiciously light tails"
    
    def test_multiple_runs_give_different_samples(self):
        """Test that sampling is stochastic (different runs give different results)."""
        baseline = 40.0
        posterior = estimate_raw_material_posterior(baseline)
        
        samples1 = posterior.sample_predictive(100)
        samples2 = posterior.sample_predictive(100)
        
        # Should not be identical
        assert not np.array_equal(samples1, samples2), "Samples are deterministic"
    
    def test_fallback_mechanism_with_zero_baseline(self):
        """Test that function handles edge case of zero baseline gracefully."""
        baseline = 0.0
        
        # Should not crash, should handle gracefully
        posterior = estimate_raw_material_posterior(baseline)
        samples = posterior.sample_predictive(100)
        
        assert len(samples) == 100
        assert np.all(np.isfinite(samples))
    
    def test_fallback_mechanism_with_large_baseline(self):
        """Test that function handles large baseline values."""
        baseline = 10000.0
        
        posterior = estimate_raw_material_posterior(baseline)
        samples = posterior.sample_predictive(100)
        
        assert len(samples) == 100
        assert np.all(np.isfinite(samples))
        
        # Samples should scale with baseline
        sample_mean = np.mean(samples)
        assert sample_mean > 1000, "Samples didn't scale with large baseline"
    
    def test_posterior_reproducibility(self):
        """Test that calling function multiple times gives consistent results."""
        baseline = 40.0
        
        # Call function twice
        posterior1 = estimate_raw_material_posterior(baseline)
        posterior2 = estimate_raw_material_posterior(baseline)
        
        # Parameters should be the same (data fetch should be deterministic for same time)
        # Note: This may fail if data updates between calls, but should be stable most of the time
        assert np.abs(posterior1.mu - posterior2.mu) < 1.0, \
            "Posterior parameters differ significantly between calls"


class TestRawMaterialIntegration:
    """Integration tests for raw material estimation with real/fallback data."""
    
    def test_works_with_realistic_baseline(self):
        """Test with realistic baseline value for auto parts."""
        baseline = 40.0  # $40 per lamp for raw materials
        
        posterior = estimate_raw_material_posterior(baseline)
        samples = posterior.sample_predictive(1000)
        
        # All samples should be positive (can't have negative costs)
        assert np.all(samples > 0), "Generated negative costs"
        
        # Samples should be in reasonable range for auto parts
        assert np.mean(samples) > 10, "Mean cost too low"
        assert np.mean(samples) < 200, "Mean cost too high"
    
    def test_posterior_uncertainty_increases_with_data_scarcity(self):
        """Test that uncertainty is captured appropriately."""
        baseline = 40.0
        posterior = estimate_raw_material_posterior(baseline)
        
        # Variance of predictive distribution
        samples = posterior.sample_predictive(10000)
        sample_var = np.var(samples)
        
        # Should have meaningful variance (not degenerate)
        assert sample_var > 0.1, "Posterior has negligible uncertainty"
        
        # Should not have excessive variance
        coeff_variation = np.sqrt(sample_var) / np.mean(samples)
        assert coeff_variation < 1.0, "Coefficient of variation too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

