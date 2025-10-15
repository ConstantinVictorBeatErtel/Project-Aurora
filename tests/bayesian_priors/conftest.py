"""
Pytest configuration for Bayesian priors tests.

This file provides shared fixtures and configuration for all tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to Python path so we can import bayesian_priors
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def typical_baselines():
    """Fixture providing typical baseline values for testing."""
    return {
        "US": {
            "raw_material": 40.0,
            "labor": 12.0,
            "logistics": 9.0,
            "indirect": 10.0,
            "electricity": 4.0,
            "depreciation": 5.0,
            "working_capital": 5.0,
            "yield": 0.80,
        },
        "Mexico": {
            "raw_material": 38.0,
            "labor": 5.0,
            "logistics": 7.0,
            "indirect": 8.0,
            "electricity": 3.0,
            "depreciation": 4.0,
            "working_capital": 6.0,
            "yield": 0.75,
        },
        "China": {
            "raw_material": 35.0,
            "labor": 3.0,
            "logistics": 15.0,  # Higher due to ocean freight
            "indirect": 7.0,
            "electricity": 2.5,
            "depreciation": 3.5,
            "working_capital": 4.5,
            "yield": 0.85,
        },
    }


@pytest.fixture
def sample_sizes():
    """Fixture providing standard sample sizes for different test types."""
    return {
        "quick": 100,
        "standard": 1000,
        "statistical": 10000,
    }


# ============================================================================
# Reusable test helpers to eliminate redundancy across test files
# ============================================================================

def assert_samples_have_correct_shape(posterior, n_samples=1000):
    """Helper: Test that sampling returns correct number of samples."""
    samples = posterior.sample_predictive(n_samples)
    assert len(samples) == n_samples
    assert isinstance(samples, np.ndarray)
    return samples


def assert_samples_are_finite(posterior, n_samples=1000):
    """Helper: Test that all samples are finite (no NaN or inf)."""
    samples = posterior.sample_predictive(n_samples)
    assert np.all(np.isfinite(samples)), "Samples contain NaN or inf"
    return samples


def assert_samples_mostly_positive(posterior, n_samples=1000, threshold=0.95):
    """Helper: Test that samples are mostly positive."""
    samples = posterior.sample_predictive(n_samples)
    positive_ratio = np.sum(samples > 0) / len(samples)
    assert positive_ratio > threshold, \
        f"Only {positive_ratio*100:.1f}% of samples are positive (expected >{threshold*100:.0f}%)"
    return samples


def assert_posterior_parameters_valid(posterior, param_names=['mu', 'kappa', 'alpha', 'beta']):
    """Helper: Test that posterior parameters are valid (positive, finite)."""
    for param in param_names:
        if hasattr(posterior, param):
            value = getattr(posterior, param)
            if param in ['kappa', 'alpha', 'beta']:
                assert value > 0, f"{param} must be positive"
            assert np.isfinite(value), f"{param} not finite"


def assert_extreme_baseline_handling(estimator_func, country, *args):
    """Helper: Test that function handles extreme baseline values."""
    # Test with very small baseline
    posterior_small = estimator_func(country, 0.01, *args)
    samples_small = posterior_small.sample_predictive(100)
    assert np.all(np.isfinite(samples_small))
    
    # Test with large baseline
    posterior_large = estimator_func(country, 1000.0, *args)
    samples_large = posterior_large.sample_predictive(100)
    assert np.all(np.isfinite(samples_large))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that make API calls (deselect with '-m \"not api\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

