"""
Tests for built-in Zeeman spatiotemporal functions.
"""
import numpy as np
import pytest
import micromagneticmodel as mm


class TestZeemanBuiltInTemporalFunctions:
    """Tests for built-in temporal functions in Zeeman class."""

    def test_sin_function(self):
        """Test sinusoidal temporal function."""
        zeeman = mm.Zeeman()
        
        # Test at t=0
        result = zeeman.sin(0, amplitude=1e5, frequency=1e9)
        assert np.isclose(result, 0, atol=1e-10)
        
        # Test at t=0.25 ns (quarter period)
        t = 0.25e-9  # 1/(4*f)
        result = zeeman.sin(t, amplitude=1e5, frequency=1e9)
        assert np.isclose(result, 1e5, rtol=1e-10)
        
        # Test with phase
        result = zeeman.sin(0, amplitude=1e5, frequency=1e9, phase=np.pi/2)
        assert np.isclose(result, 1e5, rtol=1e-10)

    def test_cos_function(self):
        """Test cosinusoidal temporal function."""
        zeeman = mm.Zeeman()
        
        # Test at t=0
        result = zeeman.cos(0, amplitude=1e5, frequency=1e9)
        assert np.isclose(result, 1e5, rtol=1e-10)
        
        # Test at t=0.25 ns (quarter period)
        t = 0.25e-9
        result = zeeman.cos(t, amplitude=1e5, frequency=1e9)
        assert np.isclose(result, 0, atol=1e-10)

    def test_constant_function(self):
        """Test constant temporal function."""
        zeeman = mm.Zeeman()
        
        result = zeeman.constant(0, amplitude=1e5)
        assert np.isclose(result, 1e5, rtol=1e-10)
        
        result = zeeman.constant(1e-9, amplitude=1e5)
        assert np.isclose(result, 1e5, rtol=1e-10)

    def test_gaussian_function(self):
        """Test Gaussian pulse temporal function."""
        zeeman = mm.Zeeman()
        
        # Test at center
        result = zeeman.gaussian(0, amplitude=1e5, center=0, sigma=1e-12)
        assert np.isclose(result, 1e5, rtol=1e-10)
        
        # Test away from center
        result = zeeman.gaussian(2e-12, amplitude=1e5, center=0, sigma=1e-12)
        expected = 1e5 * np.exp(-2**2 / 2)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_exponential_function(self):
        """Test exponential decay temporal function."""
        zeeman = mm.Zeeman()
        
        # Test at t=0
        result = zeeman.exponential(0, amplitude=1e5, tau=1e-12)
        assert np.isclose(result, 1e5, rtol=1e-10)
        
        # Test at t=tau
        result = zeeman.exponential(1e-12, amplitude=1e5, tau=1e-12)
        expected = 1e5 * np.exp(-1)
        assert np.isclose(result, expected, rtol=1e-10)


class TestZeemanBuiltInSpatialMasks:
    """Tests for built-in spatial masks in Zeeman class."""

    def test_uniform_mask(self):
        """Test uniform spatial mask."""
        zeeman = mm.Zeeman()
        
        result = zeeman.uniform(0, 0, 0)
        assert np.isclose(result, 1.0, rtol=1e-10)
        
        result = zeeman.uniform(1e-6, 1e-6, 1e-6)
        assert np.isclose(result, 1.0, rtol=1e-10)

    def test_cos_mask(self):
        """Test cosine spatial mask."""
        zeeman = mm.Zeeman()
        
        # Test at x=0
        result = zeeman.cos_mask(0, 0, 0, k=1)
        assert np.isclose(result, 1.0, rtol=1e-10)
        
        # Test at kx = pi/2
        x = np.pi / 2
        result = zeeman.cos_mask(x, 0, 0, k=1)
        assert np.isclose(result, 0, atol=1e-10)
        
        # Test different axes
        result_x = zeeman.cos_mask(1, 0, 0, k=1, axis='x')
        result_y = zeeman.cos_mask(0, 1, 0, k=1, axis='y')
        result_z = zeeman.cos_mask(0, 0, 1, k=1, axis='z')
        assert np.isclose(result_x, result_y)
        assert np.isclose(result_y, result_z)

    def test_sin_mask(self):
        """Test sine spatial mask."""
        zeeman = mm.Zeeman()
        
        # Test at x=0
        result = zeeman.sin_mask(0, 0, 0, k=1)
        assert np.isclose(result, 0, atol=1e-10)
        
        # Test at kx = pi/2
        x = np.pi / 2
        result = zeeman.sin_mask(x, 0, 0, k=1)
        assert np.isclose(result, 1.0, rtol=1e-10)

    def test_gaussian_mask(self):
        """Test Gaussian spatial mask."""
        zeeman = mm.Zeeman()
        
        # Test at center
        result = zeeman.gaussian_mask(0, 0, 0, sigma=50e-9)
        assert np.isclose(result, 1.0, rtol=1e-10)
        
        # Test away from center (x = sigma)
        result = zeeman.gaussian_mask(50e-9, 0, 0, sigma=50e-9)
        expected = np.exp(-1**2 / 2)  # x/sigma = 1
        assert np.isclose(result, expected, rtol=1e-10)
        
        # Test with custom center
        result = zeeman.gaussian_mask(50e-9, 0, 0, sigma=50e-9, center=(50e-9, 0, 0))
        assert np.isclose(result, 1.0, rtol=1e-10)

    def test_step_mask(self):
        """Test step function spatial mask."""
        zeeman = mm.Zeeman()
        
        # Test below threshold
        result = zeeman.step_mask(0, 0, 0, threshold=50e-9)
        assert result == 0.0
        
        # Test above threshold
        result = zeeman.step_mask(100e-9, 0, 0, threshold=50e-9)
        assert result == 1.0


class TestZeemanStageCount:
    """Tests for automatic stage_count feature."""

    def test_stage_count_none_by_default(self):
        """Test that stage_count is None by default."""
        zeeman = mm.Zeeman()
        assert zeeman._stage_count is None

    def test_stage_count_can_be_set(self):
        """Test that stage_count can be explicitly set."""
        zeeman = mm.Zeeman(stage_count=50)
        assert zeeman._stage_count == 50

    def test_stage_count_with_spatiotemporal_terms(self):
        """Test stage_count with spatiotemporal terms."""
        zeeman = mm.Zeeman(stage_count=100, dt=1e-14)
        zeeman.add_time_term(lambda t: np.sin(t))
        
        assert zeeman._stage_count == 100
        assert zeeman._dt == 1e-14
        assert zeeman.has_time_terms
