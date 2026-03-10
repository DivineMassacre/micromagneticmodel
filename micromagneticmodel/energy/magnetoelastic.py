import collections.abc

import discretisedfield as df
import ubermagutil as uu
import ubermagutil.typesystem as ts

from .energyterm import EnergyTerm


@uu.inherit_docs
@ts.typesystem(
    B1=ts.Parameter(descriptor=ts.Scalar(), otherwise=df.Field),
    B2=ts.Parameter(descriptor=ts.Scalar(), otherwise=df.Field),
    e_diag=ts.Parameter(descriptor=ts.Vector(size=3), otherwise=df.Field),
    e_offdiag=ts.Parameter(descriptor=ts.Vector(size=3), otherwise=df.Field),
)
class MagnetoElastic(EnergyTerm):
    r"""Magneto-elastic energy term.

    .. math::

        w = B_{1}\sum_{i} m_{i}\epsilon_{ii} + B_{2}\sum_{i}\sum_{j\ne i}
        m_{i}m_{j}\epsilon_{ij}

    The magneto-elastic energy term allows defining static as well as
    time-dependent strain. If only ``e_diag`` and ``e_offdiag`` are passed,
    a time-constant strain is defined (using ``YY_FixedMEL`` in OOMMF).

    For time-dependent strain, two methods are available:

    - **Stage-based**: specify a list of OVf files (one per stage) using
      ``e_diag_files`` and ``e_offdiag_files`` (uses ``YY_StageMEL``)
    - **Transformation-based**: specify a base strain and a transformation
      function using ``transform_script`` (uses ``YY_TransformStageMEL``)

    Parameters
    ----------
    B1, B2 : numbers.Real, dict, discretisedfield.Field

        If a single value ``numbers.Real`` is passed, a spatially constant
        parameter is defined. For a spatially varying parameter, either a
        dictionary, e.g. ``B1={'region1': 1e7, 'region2': 5e7}`` (if the
        parameter is defined "per region") or ``discretisedfield.Field`` is
        passed.

    e_diag/e_offdiag : (3,) array_like, dict, discretisedfield.Field

        Symmetric strain matrix is assembled from the values of the vector e,
        so that eps11 = e_diag[0], eps22=e_diag[1], eps33=e_diag[2],
        eps23=eps32=e_offdiag[0], eps13=eps31=e_offdiag[1],
        eps12=eps21=e_offdiag[2].

        If a single length-3 array_like (tuple, list, ``numpy.ndarray``) is
        passed, which consists of ``numbers.Real``, a spatially constant
        parameter is defined. For a spatially varying parameter, either a
        dictionary, e.g. ``e={'region1': (1, 1, 1), 'region2': (1, 1, 1)}`` (if
        the parameter is defined "per region") or ``discretisedfield.Field`` is
        passed.

        **Note**: For time-dependent strain using ``transform_script``, these
        define the base (reference) strain that is transformed at each time step.

    e_diag_files/e_offdiag_files : list of str, optional

        List of paths to OVf files containing the diagonal/off-diagonal strain
        components for each stage. The number of files determines the number
        of stages. Only one of (``e_diag``, ``e_diag_files``) should be specified.

    stage_count : int, optional

        Number of stages for stage-based strain. If not specified, it is
        inferred from the length of ``e_diag_files``.

    transform_type : {'identity', 'diagonal', 'symmetric', 'general'}, optional

        Type of transformation matrix for ``YY_TransformStageMEL``:

        - ``'identity'``: no transformation (identity matrix)
        - ``'diagonal'``: diagonal matrix (6 values: 3 diagonal + 3 time derivatives)
        - ``'symmetric'``: symmetric matrix (12 values)
        - ``'general'``: general 3x3 matrix (18 values)

    transform_script : callable, optional

        Function that returns the transformation matrix values at each time step.
        The function signature depends on ``transform_script_args``:

        - ``func(t)`` → returns 6, 12, or 18 values depending on ``transform_type``
        - ``func(stage, stage_time, total_time)`` → same return

        The returned values define the transformation matrix M and its time
        derivative dM/dt. Strain is transformed as ε' = M·ε·Mᵀ.

    transform_script_args : str, optional

        Arguments passed to ``transform_script``. Common values:

        - ``'total_time'``: function receives total simulation time
        - ``'stage_time'``: function receives time within current stage
        - ``'stage stage_time total_time'``: function receives all three

    Examples
    --------
    1. Defining the magneto-elastic energy term using single values
       (static strain, ``YY_FixedMEL``).

    >>> import micromagneticmodel as mm
    ...
    >>> mel = mm.MagnetoElastic(B1=1e7, B2=1e7, e_diag=(1e-3, 1e-3, 1e-3),
    ...                         e_offdiag=(0, 0, 0))

    2. Defining the magneto-elastic energy term using dictionary.

    >>> B1 = B2 = {'region1': 1e7, 'region2': 2e7}
    >>> e_diag = {'region1': (1e-3, 1e-3, 1e-3), 'region2': (2e-3, 2e-3, 2e-3)}
    >>> e_offdiag = {'region1': (0, 0, 0), 'region2': (0, 0, 0)}
    >>> mel = mm.MagnetoElastic(B1=B1, B2=B2, e_diag=e_diag,
    ...                         e_offdiag=e_offdiag)

    3. Defining the magneto-elastic energy term using
    ``discretisedfield.Field``.

    >>> import discretisedfield as df
    ...
    >>> region = df.Region(p1=(0, 0, 0), p2=(5e-9, 5e-9, 5e-9))
    >>> mesh = df.Mesh(region=region, n=(5, 5, 5))
    >>> B1 = B2 = df.Field(mesh, nvdim=1, value=1e6)
    >>> e_diag = df.Field(mesh, nvdim=3, value=(1e-3, 1e-3, 1e-3))
    >>> mel = mm.MagnetoElastic(B1=B1, B2=B2, e_diag=e_diag,
    ...                         e_offdiag=(0, 0, 0))

    4. Defining stage-based time-dependent strain (``YY_StageMEL``).

    >>> e_diag_files = ['strain_0.ovf', 'strain_1.ovf', 'strain_2.ovf']
    >>> e_offdiag_files = ['strain_off_0.ovf', 'strain_off_1.ovf', 'strain_off_2.ovf']
    >>> mel = mm.MagnetoElastic(
    ...     B1=1e7, B2=1e7,
    ...     e_diag_files=e_diag_files,
    ...     e_offdiag_files=e_offdiag_files,
    ...     stage_count=3
    ... )

    5. Defining transformation-based time-dependent strain
       (``YY_TransformStageMEL``) with oscillating diagonal strain.

    >>> import numpy as np
    >>> def oscillating_transform(t):
    ...     f = 10e9  # 10 GHz
    ...     eps = 1e-3
    ...     coef = eps * np.sin(2 * np.pi * f * t)
    ...     dcoef = eps * 2 * np.pi * f * np.cos(2 * np.pi * f * t)
    ...     return [coef, coef, coef, dcoef, dcoef, dcoef]
    >>> mel = mm.MagnetoElastic(
    ...     B1=1e7, B2=1e7,
    ...     e_diag=(1, 0.3, 0.3),
    ...     e_offdiag=(0, 0, 0),
    ...     transform_type='diagonal',
    ...     transform_script=oscillating_transform,
    ...     transform_script_args='total_time'
    ... )

    6. An attempt to define the magneto-elastic energy term using a wrong
    value.

    >>> # length-4 e value
    >>> mel = mm.MagnetoElastic(B1=1e7, B2=2e7, e_diag=(1, 1, 1, 1))
    Traceback (most recent call last):
    ...
    ValueError: ...

    See Also
    --------
    Zeeman : Zeeman energy term with similar time-dependence support
    """

    _allowed_attributes = [
        "B1",
        "B2",
        "e_diag",
        "e_offdiag",
        "e_diag_files",
        "e_offdiag_files",
        "stage_count",
        "transform_type",
        "transform_script",
        "transform_script_args",
        "transform_dt",
        "transform_n_points",
    ]
    _reprlatex = (
        r"B_{1}\sum_{i} m_{i}\epsilon_{ii} + "
        r"B_{2}\sum_{i}\sum_{j\ne i} m_{i}m_{j}\epsilon_{ij}"
    )

    def __init__(
        self,
        B1=None,
        B2=None,
        e_diag=None,
        e_offdiag=None,
        e_diag_files=None,
        e_offdiag_files=None,
        stage_count=None,
        transform_type=None,
        transform_script=None,
        transform_script_args=None,
        transform_dt=None,
        transform_n_points=None,
        **kwargs,
    ):
        # Validate that only one strain specification method is used
        has_static = e_diag is not None or e_offdiag is not None
        has_files = e_diag_files is not None or e_offdiag_files is not None
        has_transform = transform_script is not None

        if has_static and has_files:
            raise ValueError(
                "Cannot specify both static strain (e_diag/e_offdiag) "
                "and stage-based strain (e_diag_files/e_offdiag_files). "
                "Use only one method."
            )

        if has_transform and has_files:
            raise ValueError(
                "Cannot specify both transformation-based strain (transform_script) "
                "and stage-based strain (e_diag_files/e_offdiag_files). "
                "Use only one method."
            )

        if has_files:
            # Validate file lists
            if e_diag_files is None or e_offdiag_files is None:
                raise ValueError(
                    "For stage-based strain, both e_diag_files and "
                    "e_offdiag_files must be specified."
                )
            if len(e_diag_files) != len(e_offdiag_files):
                raise ValueError(
                    "e_diag_files and e_offdiag_files must have the same length."
                )
            if stage_count is None:
                stage_count = len(e_diag_files)
            # For stage-based, e_diag and e_offdiag are not required
            # Set them to zero tuples to pass type validation (length 3 required)
            e_diag = e_diag if e_diag is not None else (0, 0, 0)
            e_offdiag = e_offdiag if e_offdiag is not None else (0, 0, 0)

        if has_transform:
            # Validate transform parameters
            if transform_type is None:
                transform_type = "diagonal"
            # Validate transform_type
            valid_types = {"identity", "diagonal", "symmetric", "general"}
            if transform_type not in valid_types:
                raise ValueError(
                    f"Invalid transform_type '{transform_type}'. "
                    f"Must be one of {valid_types}."
                )
            if transform_script_args is None:
                transform_script_args = "total_time"
            if e_diag is None or e_offdiag is None:
                raise ValueError(
                    "For transformation-based strain, both e_diag and "
                    "e_offdiag (base strain) must be specified."
                )

        # Check for conflicting parameters (transform + files)
        if has_transform and has_files:
            raise ValueError(
                "Cannot specify both transformation-based strain (transform_script) "
                "and stage-based strain (e_diag_files/e_offdiag_files). "
                "Use only one method."
            )

        # Call parent __init__ with kwargs only
        # (kwargs validation happens in parent class - this catches unknown attributes)
        # Core parameters (B1, B2, e_diag, e_offdiag) are set via typesystem
        # but we need to handle None values carefully
        super().__init__(**kwargs)

        # Set core attributes manually to avoid type validation errors with None
        object.__setattr__(self, 'B1', B1)
        object.__setattr__(self, 'B2', B2)
        object.__setattr__(self, 'e_diag', e_diag)
        object.__setattr__(self, 'e_offdiag', e_offdiag)

        # Set new attributes manually after calling super().__init__()
        # (to avoid type system validation issues with None values)
        object.__setattr__(self, 'e_diag_files', e_diag_files)
        object.__setattr__(self, 'e_offdiag_files', e_offdiag_files)
        object.__setattr__(self, 'stage_count', stage_count)
        object.__setattr__(self, 'transform_type', transform_type)
        object.__setattr__(self, 'transform_script', transform_script)
        object.__setattr__(self, 'transform_script_args', transform_script_args)
        object.__setattr__(self, 'transform_dt', transform_dt)
        object.__setattr__(self, 'transform_n_points', transform_n_points)

    @classmethod
    def static(cls, B1, B2, e_diag, e_offdiag, **kwargs):
        """Create static magneto-elastic energy term (YY_FixedMEL).

        Parameters
        ----------
        B1, B2 : numbers.Real, dict, discretisedfield.Field
            Magneto-elastic coefficients.
        e_diag, e_offdiag : (3,) array_like, dict, discretisedfield.Field
            Diagonal and off-diagonal strain components.
        **kwargs
            Additional keyword arguments passed to ``MagnetoElastic``.

        Returns
        -------
        MagnetoElastic
            Static magneto-elastic energy term.

        Examples
        --------
        >>> mel = mm.MagnetoElastic.static(
        ...     B1=1e7, B2=1e7,
        ...     e_diag=(1e-3, 1e-3, 1e-3),
        ...     e_offdiag=(0, 0, 0)
        ... )
        """
        return cls(B1=B1, B2=B2, e_diag=e_diag, e_offdiag=e_offdiag, **kwargs)

    @classmethod
    def stage(
        cls, B1, B2, e_diag_files, e_offdiag_files, stage_count=None, **kwargs
    ):
        """Create stage-based magneto-elastic energy term (YY_StageMEL).

        Parameters
        ----------
        B1, B2 : numbers.Real, dict, discretisedfield.Field
            Magneto-elastic coefficients.
        e_diag_files, e_offdiag_files : list of str
            Lists of paths to OVf files containing strain components for each stage.
        stage_count : int, optional
            Number of stages. If not specified, inferred from file list length.
        **kwargs
            Additional keyword arguments passed to ``MagnetoElastic``.

        Returns
        -------
        MagnetoElastic
            Stage-based magneto-elastic energy term.

        Examples
        --------
        >>> mel = mm.MagnetoElastic.stage(
        ...     B1=1e7, B2=1e7,
        ...     e_diag_files=['strain_0.ovf', 'strain_1.ovf'],
        ...     e_offdiag_files=['off_0.ovf', 'off_1.ovf'],
        ...     stage_count=2
        ... )
        """
        return cls(
            B1=B1,
            B2=B2,
            e_diag_files=e_diag_files,
            e_offdiag_files=e_offdiag_files,
            stage_count=stage_count,
            **kwargs,
        )

    @classmethod
    def transform(
        cls,
        B1,
        B2,
        e_diag,
        e_offdiag,
        transform_script,
        transform_type="diagonal",
        transform_script_args="total_time",
        **kwargs,
    ):
        """Create transformation-based magneto-elastic energy term (YY_TransformStageMEL).

        Parameters
        ----------
        B1, B2 : numbers.Real, dict, discretisedfield.Field
            Magneto-elastic coefficients.
        e_diag, e_offdiag : (3,) array_like, dict, discretisedfield.Field
            Base (reference) strain components.
        transform_script : callable
            Function returning transformation matrix values.
        transform_type : {'identity', 'diagonal', 'symmetric', 'general'}
            Type of transformation matrix.
        transform_script_args : str, optional
            Arguments passed to transform_script (default: 'total_time').
        **kwargs
            Additional keyword arguments passed to ``MagnetoElastic``.

        Returns
        -------
        MagnetoElastic
            Transformation-based magneto-elastic energy term.

        Examples
        --------
        >>> import numpy as np
        >>> def oscillating(t):
        ...     coef = 1e-3 * np.sin(2 * np.pi * 10e9 * t)
        ...     dcoef = 1e-3 * 2 * np.pi * 10e9 * np.cos(2 * np.pi * 10e9 * t)
        ...     return [coef, coef, coef, dcoef, dcoef, dcoef]
        >>> mel = mm.MagnetoElastic.transform(
        ...     B1=1e7, B2=1e7,
        ...     e_diag=(1, 0.3, 0.3),
        ...     e_offdiag=(0, 0, 0),
        ...     transform_script=oscillating,
        ...     transform_type='diagonal'
        ... )
        """
        return cls(
            B1=B1,
            B2=B2,
            e_diag=e_diag,
            e_offdiag=e_offdiag,
            transform_type=transform_type,
            transform_script=transform_script,
            transform_script_args=transform_script_args,
            **kwargs,
        )

    @property
    def _mel_class(self):
        """Return the OOMMF class name for this magneto-elastic term."""
        if self.transform_script is not None:
            return "YY_TransformStageMEL"
        elif self.e_diag_files is not None:
            return "YY_StageMEL"
        else:
            return "YY_FixedMEL"

    def effective_field(self, m):
        raise NotImplementedError
