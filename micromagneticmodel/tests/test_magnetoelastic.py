import re

import discretisedfield as df
import pytest

import micromagneticmodel as mm
from .checks import check_term


class TestMagnetoElastic:
    def setup_method(self):
        mesh = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), cell=(1, 1, 1))
        B1field = df.Field(mesh, nvdim=1, value=5e6)
        B2field = df.Field(mesh, nvdim=1, value=7e6)
        e_diagfield = df.Field(mesh, nvdim=3, value=(1, 0, 0))
        e_offdiagfield = df.Field(mesh, nvdim=3, value=(1, 1, 0))

        self.valid_args = [
            (1, 1, (0, 1, 0), (2, 3, 7)),
            (2e6, 3e6, (1e6, 1, 0.1), (2, 3, 7)),
            (7e6, -2e-6, (0, 1e-8, 0), (2, 3.14, 7)),
            (B1field, B2field, e_diagfield, e_offdiagfield),
        ]
        self.invalid_args = [
            ((0, 1), 1, (0, 1, 0), (2, 3, 7)),
            (2e6, "5", (1e6, 1, 0.1), (2, 3, 7)),
            (7e6, -2e-6, (0, 1e-8, 3.14), (1, 7)),
            (7e6, -2e-6, 5, 2),
        ]

    def test_init_valid_args(self):
        for B1, B2, e_diag, e_offdiag in self.valid_args:
            term = mm.MagnetoElastic(B1=B1, B2=B2, e_diag=e_diag, e_offdiag=e_offdiag)
            check_term(term)
            assert hasattr(term, "B1")
            assert hasattr(term, "B2")
            assert hasattr(term, "e_diag")
            assert hasattr(term, "e_offdiag")
            assert term.name == "magnetoelastic"
            assert re.search(
                (r"^MagnetoElastic\(B1=.+, B2=.+, " r"e_diag=.+\, e_offdiag=.+\)$"),
                repr(term),
            )

    def test_init_invalid_args(self):
        for B1, B2, e_diag, e_offdiag in self.invalid_args:
            with pytest.raises((TypeError, ValueError)):
                mm.MagnetoElastic(B1=B1, B2=B2, e_diag=e_diag, e_offdiag=e_offdiag)

        with pytest.raises(AttributeError):
            mm.MagnetoElastic(wrong=1)


class TestMagnetoElasticStatic:
    """Tests for static magneto-elastic energy (YY_FixedMEL)."""

    def test_static_classmethod(self):
        """Test MagnetoElastic.static() classmethod."""
        mel = mm.MagnetoElastic.static(
            B1=1e7, B2=1e7, e_diag=(1e-3, 1e-3, 1e-3), e_offdiag=(0, 0, 0)
        )
        check_term(mel)
        assert mel._mel_class == "YY_FixedMEL"
        assert mel.B1 == 1e7
        assert mel.B2 == 1e7
        assert mel.e_diag == (1e-3, 1e-3, 1e-3)
        assert mel.e_offdiag == (0, 0, 0)

    def test_static_equivalent_to_init(self):
        """Test that static() is equivalent to direct initialization."""
        mel1 = mm.MagnetoElastic.static(
            B1=1e7, B2=1e7, e_diag=(1e-3, 1e-3, 1e-3), e_offdiag=(0, 0, 0)
        )
        mel2 = mm.MagnetoElastic(
            B1=1e7, B2=1e7, e_diag=(1e-3, 1e-3, 1e-3), e_offdiag=(0, 0, 0)
        )
        assert mel1.B1 == mel2.B1
        assert mel1.B2 == mel2.B2
        assert mel1.e_diag == mel2.e_diag
        assert mel1.e_offdiag == mel2.e_offdiag


class TestMagnetoElasticStage:
    """Tests for stage-based magneto-elastic energy (YY_StageMEL)."""

    def test_stage_classmethod(self):
        """Test MagnetoElastic.stage() classmethod."""
        e_diag_files = ["strain_0.ovf", "strain_1.ovf", "strain_2.ovf"]
        e_offdiag_files = ["off_0.ovf", "off_1.ovf", "off_2.ovf"]

        mel = mm.MagnetoElastic.stage(
            B1=1e7,
            B2=1e7,
            e_diag_files=e_diag_files,
            e_offdiag_files=e_offdiag_files,
            stage_count=3,
        )
        check_term(mel)
        assert mel._mel_class == "YY_StageMEL"
        assert mel.e_diag_files == e_diag_files
        assert mel.e_offdiag_files == e_offdiag_files
        assert mel.stage_count == 3

    def test_stage_automatic_stage_count(self):
        """Test that stage_count is inferred from file list length."""
        e_diag_files = ["strain_0.ovf", "strain_1.ovf"]
        e_offdiag_files = ["off_0.ovf", "off_1.ovf"]

        mel = mm.MagnetoElastic.stage(
            B1=1e7, B2=1e7, e_diag_files=e_diag_files, e_offdiag_files=e_offdiag_files
        )
        assert mel.stage_count == 2

    def test_stage_mismatched_file_lengths(self):
        """Test that mismatched file list lengths raise an error."""
        e_diag_files = ["strain_0.ovf", "strain_1.ovf"]
        e_offdiag_files = ["off_0.ovf"]  # Different length

        with pytest.raises(ValueError, match="same length"):
            mm.MagnetoElastic.stage(
                B1=1e7, B2=1e7, e_diag_files=e_diag_files, e_offdiag_files=e_offdiag_files
            )

    def test_stage_missing_files(self):
        """Test that missing file lists raise an error."""
        with pytest.raises(ValueError, match="both e_diag_files"):
            mm.MagnetoElastic.stage(
                B1=1e7, B2=1e7, e_diag_files=["strain_0.ovf"], e_offdiag_files=None
            )


class TestMagnetoElasticTransform:
    """Tests for transformation-based magneto-elastic energy (YY_TransformStageMEL)."""

    def test_transform_classmethod(self):
        """Test MagnetoElastic.transform() classmethod."""
        import numpy as np

        def oscillating_transform(t):
            f = 10e9
            eps = 1e-3
            coef = eps * np.sin(2 * np.pi * f * t)
            dcoef = eps * 2 * np.pi * f * np.cos(2 * np.pi * f * t)
            return [coef, coef, coef, dcoef, dcoef, dcoef]

        mel = mm.MagnetoElastic.transform(
            B1=1e7,
            B2=1e7,
            e_diag=(1, 0.3, 0.3),
            e_offdiag=(0, 0, 0),
            transform_script=oscillating_transform,
            transform_type="diagonal",
        )
        check_term(mel)
        assert mel._mel_class == "YY_TransformStageMEL"
        assert mel.transform_type == "diagonal"
        assert mel.transform_script == oscillating_transform
        assert mel.transform_script_args == "total_time"  # default

    def test_transform_default_type(self):
        """Test that transform_type defaults to 'diagonal'."""
        mel = mm.MagnetoElastic.transform(
            B1=1e7,
            B2=1e7,
            e_diag=(1, 0.3, 0.3),
            e_offdiag=(0, 0, 0),
            transform_script=lambda t: [1, 1, 1, 0, 0, 0],
        )
        assert mel.transform_type == "diagonal"

    def test_transform_all_types(self):
        """Test all transformation types."""
        for transform_type in ["identity", "diagonal", "symmetric", "general"]:
            mel = mm.MagnetoElastic.transform(
                B1=1e7,
                B2=1e7,
                e_diag=(1, 0.3, 0.3),
                e_offdiag=(0, 0, 0),
                transform_script=lambda t: [1] * 18,
                transform_type=transform_type,
            )
            assert mel.transform_type == transform_type

    def test_transform_invalid_type(self):
        """Test that invalid transform_type raises an error."""
        with pytest.raises(ValueError):
            mm.MagnetoElastic.transform(
                B1=1e7,
                B2=1e7,
                e_diag=(1, 0.3, 0.3),
                e_offdiag=(0, 0, 0),
                transform_script=lambda t: [1] * 18,
                transform_type="invalid_type",
            )

    def test_transform_direct_substitution_no_base_strain(self):
        """Test direct substitution mode: func without e_diag/e_offdiag.

        When only transform_script is provided (without e_diag/e_offdiag),
        the function returns full strain values (direct substitution mode).
        """
        mel = mm.MagnetoElastic.transform(
            B1=1e7,
            B2=1e7,
            e_diag=None,  # Not needed for direct substitution
            e_offdiag=None,
            transform_script=lambda t: [1e-3, 1e-3, 1e-3, 0, 0, 0],
        )
        assert mel._mel_class == "YY_TransformStageMEL"
        assert mel.transform_script is not None
        # e_diag/e_offdiag should be set to defaults
        assert mel.e_diag == (0, 0, 0)
        assert mel.e_offdiag == (0, 0, 0)

    def test_transform_matrix_mode_with_base_strain(self):
        """Test matrix transformation mode: func + e_diag/e_offdiag.

        When both transform_script and e_diag/e_offdiag are provided,
        the function returns transformation matrix M(t), and final strain
        is computed as: e_final = M(t) × e_base × M(t)ᵀ
        """
        mel = mm.MagnetoElastic.transform(
            B1=1e7,
            B2=1e7,
            e_diag=(1e-3, 1e-3, 1e-3),  # Base strain
            e_offdiag=(0, 0, 0),
            transform_script=lambda t: [1.0, 1.0, 1.0, 0, 0, 0],  # M(t)
        )
        assert mel._mel_class == "YY_TransformStageMEL"
        assert mel.transform_script is not None
        assert mel.e_diag == (1e-3, 1e-3, 1e-3)  # Base strain set
        assert mel.e_offdiag == (0, 0, 0)


class TestMagnetoElasticValidation:
    """Tests for validation of conflicting parameters."""

    def test_conflicting_static_and_files(self):
        """Test that static and file-based strain cannot be mixed."""
        with pytest.raises(ValueError, match="Cannot specify both static"):
            mm.MagnetoElastic(
                B1=1e7,
                B2=1e7,
                e_diag=(1e-3, 1e-3, 1e-3),  # Static
                e_offdiag=(0, 0, 0),
                e_diag_files=["strain_0.ovf"],  # Files
                e_offdiag_files=["off_0.ovf"],
            )

    def test_conflicting_transform_and_files(self):
        """Test that transform and file-based strain cannot be mixed."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            mm.MagnetoElastic(
                B1=1e7,
                B2=1e7,
                e_diag=(1e-3, 1e-3, 1e-3),
                e_offdiag=(0, 0, 0),
                e_diag_files=["strain_0.ovf"],  # Files
                e_offdiag_files=["off_0.ovf"],
                transform_script=lambda t: [1] * 6,  # Transform
            )

    def test_repr_with_files(self):
        """Test repr for stage-based magnetoelastic."""
        mel = mm.MagnetoElastic.stage(
            B1=1e7,
            B2=1e7,
            e_diag_files=["strain_0.ovf", "strain_1.ovf"],
            e_offdiag_files=["off_0.ovf", "off_1.ovf"],
        )
        assert "MagnetoElastic" in repr(mel)

    def test_repr_with_transform(self):
        """Test repr for transform-based magnetoelastic."""
        mel = mm.MagnetoElastic.transform(
            B1=1e7,
            B2=1e7,
            e_diag=(1, 0.3, 0.3),
            e_offdiag=(0, 0, 0),
            transform_script=lambda t: [1] * 6,
        )
        assert "MagnetoElastic" in repr(mel)
