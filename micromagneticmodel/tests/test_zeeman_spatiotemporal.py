"""Tests for spatiotemporal Zeeman functionality."""

import numpy as np
import pytest

import micromagneticmodel as mm


class TestZeemanSpatiotemporal:
    """Tests for spatiotemporal Zeeman functionality."""

    def test_init_default(self):
        """Test default initialization."""
        zeeman = mm.Zeeman()
        assert zeeman.H == (0, 0, 0)
        assert zeeman._terms == []
        assert not zeeman.has_time_terms

    def test_init_static_field(self):
        """Test initialization with static field."""
        zeeman = mm.Zeeman(H=(1e6, 0, 0))
        assert zeeman.H == (1e6, 0, 0)
        assert not zeeman.has_time_terms

    def test_init_spatiotemporal_terms(self):
        """Test initialization with spatiotemporal terms."""
        terms = [(lambda t: np.sin(t), None)]
        zeeman = mm.Zeeman(spatiotemporal_terms=terms)
        assert zeeman.has_time_terms
        assert len(zeeman._terms) == 1

    def test_add_time_term_uniform(self):
        """Test adding uniform time-dependent term."""
        zeeman = mm.Zeeman()
        zeeman.add_time_term(lambda t: (np.sin(t), 0, 0))
        assert zeeman.has_time_terms
        assert len(zeeman._terms) == 1

    def test_add_time_term_with_mask(self):
        """Test adding term with spatial mask."""
        zeeman = mm.Zeeman()
        zeeman.add_time_term(
            func=lambda t: np.sin(t),
            mask=lambda x, y, z: np.exp(-x**2)
        )
        assert zeeman.has_time_terms
        assert len(zeeman._terms) == 1

    def test_clear_time_terms(self):
        """Test clearing time-dependent terms."""
        zeeman = mm.Zeeman()
        zeeman.add_time_term(lambda t: (1, 0, 0))
        zeeman.clear_time_terms()
        assert not zeeman.has_time_terms
        assert zeeman._terms == []

    def test_multiple_terms(self):
        """Test adding multiple time-dependent terms."""
        zeeman = mm.Zeeman()
        zeeman.add_time_term(lambda t: (np.sin(t), 0, 0))
        zeeman.add_time_term(lambda t: (0, np.cos(t), 0))
        assert len(zeeman._terms) == 2

    def test_traveling_wave(self):
        """Test traveling wave configuration."""
        H0 = 1e6
        omega = 2 * np.pi * 1e9
        k = 2 * np.pi / 100e-9

        zeeman = mm.Zeeman()
        zeeman.add_time_term(
            func=lambda t: H0 * np.sin(omega * t),
            mask=lambda x, y, z: np.cos(k * x)
        )
        zeeman.add_time_term(
            func=lambda t: H0 * np.cos(omega * t),
            mask=lambda x, y, z: -np.sin(k * x)
        )

        assert len(zeeman._terms) == 2

    def test_spatiotemporal_terms_property(self):
        """Test spatiotemporal_terms property."""
        zeeman = mm.Zeeman()
        assert zeeman._terms == []

        zeeman.add_time_term(lambda t: (1, 0, 0))
        assert len(zeeman._terms) == 1

    def test_combined_static_and_spatiotemporal(self):
        """Test combining static field with spatiotemporal terms."""
        zeeman = mm.Zeeman(H=(1e6, 0, 0))
        zeeman.add_time_term(lambda t: (1e3 * np.sin(2 * np.pi * 1e9 * t), 0, 0))

        assert zeeman.H == (1e6, 0, 0)
        assert zeeman.has_time_terms
        assert len(zeeman._terms) == 1
