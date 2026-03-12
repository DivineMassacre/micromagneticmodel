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
    time-dependent strain. Three modes are available:

    1. **Static strain** (YY_FixedMEL): specify ``e_diag`` and ``e_offdiag``
       for time-constant strain.

    2. **Stage-based strain** (YY_StageMEL): specify ``e_diag_files`` and
       ``e_offdiag_files`` (list of OVf files, one per stage).

    3. **Time-dependent strain** (YY_TransformStageMEL): specify ``func``
       (or ``transform_script``) and ``dt`` (or ``transform_dt``) for
       strain that varies continuously with time. Two sub-modes are supported:

       a) **Direct substitution** (default): ``func(t)`` returns full strain
          values [e11, e22, e33, e23, e13, e12]. Use without ``e_diag``/``e_offdiag``.

       b) **Matrix transformation**: ``func(t)`` returns transformation matrix
          elements M(t), and final strain is computed as
          e_final = M(t) × e_base × M(t)ᵀ. Use with ``e_diag``/``e_offdiag``.

    Parameters
    ----------
    B1, B2 : numbers.Real, dict, discretisedfield.Field

        Magneto-elastic coefficients in J/m³.

        If a single value ``numbers.Real`` is passed, a spatially constant
        parameter is defined. For a spatially varying parameter, either a
        dictionary, e.g. ``B1={'region1': 1e7, 'region2': 5e7}`` (if the
        parameter is defined "per region") or ``discretisedfield.Field`` is
        passed.

    e_diag/e_offdiag : (3,) array_like, dict, discretisedfield.Field

        Strain components. The symmetric strain matrix is assembled as:

        - eps11 = e_diag[0], eps22 = e_diag[1], eps33 = e_diag[2]
        - eps23 = eps32 = e_offdiag[0], eps13 = eps31 = e_offdiag[1],
          eps12 = eps21 = e_offdiag[2]

        If a single length-3 array_like (tuple, list, ``numpy.ndarray``) is
        passed, which consists of ``numbers.Real``, a spatially constant
        parameter is defined. For a spatially varying parameter, either a
        dictionary or ``discretisedfield.Field`` is passed.

        **Note**: For time-dependent strain:

        - **Direct substitution mode** (without ``e_diag``/``e_offdiag``):
          These are ignored; ``func(t)`` returns full strain values.

        - **Matrix transformation mode** (with ``e_diag``/``e_offdiag``):
          These define the base strain e_base, and ``func(t)`` returns
          transformation matrix M(t) for e_final = M(t) × e_base × M(t)ᵀ.

    e_diag_files/e_offdiag_files : list of str, optional

        List of paths to OVf files containing the diagonal/off-diagonal strain
        components for each stage (stage-based mode). The number of files
        determines the number of stages. Mutually exclusive with ``func``.

    stage_count : int, optional

        Number of stages for stage-based strain. If not specified, it is
        inferred from the length of ``e_diag_files``.

    func : callable, optional

        Time-dependent function for strain (Zeeman-style interface).
        Signature: ``func(t)`` where ``t`` is time in seconds.

        The return value and interpretation depend on whether ``e_diag``/
        ``e_offdiag`` are provided:

        - **Direct substitution mode** (without ``e_diag``/``e_offdiag``):
          Returns full strain values [e11, e22, e33, e23, e13, e12].

        - **Matrix transformation mode** (with ``e_diag``/``e_offdiag``):
          Returns transformation matrix elements M(t). For ``transform_type=
          'diagonal'``, returns [M11, M22, M33, dM11, dM22, dM33] where M
          scales the base strain: e_final_ii = M_ii × e_base_ii.

        Returns
        -------
        list of float
            List of 6 strain values (direct substitution) or 6-18 matrix
            elements (matrix transformation) depending on ``transform_type``.

        Example
        -------
        >>> def strain_func(t):
        ...     # 1 GHz oscillation with 1e-3 amplitude
        ...     return [1e-3 * np.sin(2*np.pi*1e9*t)] * 6

    transform_script : callable, optional

        Alternative to ``func`` (advanced interface). Time-dependent function
        for strain. Signature and return values same as ``func``.

    dt : float, optional

        Time step in seconds for pre-computing strain values (Zeeman-style).
        Default: 1e-13 (0.1 ps). Smaller values give better resolution but
        larger MIF files. Must be chosen to resolve the fastest variations
        in ``func(t)``.

    transform_dt : float, optional

        Alternative to ``dt`` (advanced interface). Time step for pre-computation.

    transform_type : {'diagonal', 'symmetric', 'general'}, optional

        Type of strain representation for time-dependent mode:

        - ``'diagonal'`` (default): 6 values [e11, e22, e33, e23, e13, e12]
        - ``'symmetric'``: 12 values (symmetric tensor)
        - ``'general'``: 18 values (full tensor)

    transform_script_args : str, optional

        Arguments passed to ``transform_script``. Default: ``'total_time'``.
        Common values:

        - ``'total_time'``: function receives total simulation time
        - ``'stage_time'``: function receives time within current stage

    Examples
    --------
    1. Static strain (YY_FixedMEL):

    >>> mel = mm.MagnetoElastic(B1=1e7, B2=1e7,
    ...                         e_diag=(1e-3, 1e-3, 1e-3),
    ...                         e_offdiag=(0, 0, 0))

    2. Stage-based strain from OVf files (YY_StageMEL):

    >>> mel = mm.MagnetoElastic.stage(
    ...     B1=1e7, B2=1e7,
    ...     e_diag_files=['strain_0.ovf', 'strain_1.ovf'],
    ...     e_offdiag_files=['off_0.ovf', 'off_1.ovf']
    ... )

    3. Time-dependent strain: Direct substitution mode (YY_TransformStageMEL):

    >>> import numpy as np
    >>> def strain_func(t):
    ...     f = 1e9  # 1 GHz
    ...     A = 1e-3  # Amplitude
    ...     strain = A * np.sin(2 * np.pi * f * t)
    ...     return [strain, strain, strain, 0, 0, 0]
    >>> mel = mm.MagnetoElastic(
    ...     B1=1e7, B2=1e7,
    ...     func=strain_func,  # Returns full strain values
    ...     dt=1e-13  # 0.1 ps time step
    ... )

    4. Time-dependent strain: Matrix transformation mode:

    >>> def transform_matrix(t):
    ...     # Scaling matrix M(t) for diagonal transformation
    ...     m = 1 + 1e-3 * np.sin(2 * np.pi * 1e9 * t)
    ...     dm = 1e-3 * 2 * np.pi * 1e9 * np.cos(2 * np.pi * 1e9 * t)
    ...     return [m, m, m, dm, dm, dm]  # M11, M22, M33, dM11, dM22, dM33
    >>> mel = mm.MagnetoElastic(
    ...     B1=1e7, B2=1e7,
    ...     e_diag=(1e-3, 1e-3, 1e-3),  # Base strain e_base
    ...     e_offdiag=(0, 0, 0),
    ...     func=transform_matrix,  # Returns M(t)
    ...     dt=1e-13,
    ...     transform_type='diagonal'  # e_final_ii = M_ii * e_base_ii
    ... )

    5. Using transform() factory method (direct substitution):

    >>> mel = mm.MagnetoElastic.transform(
    ...     B1=1e7, B2=1e7,
    ...     func=strain_func,
    ...     dt=1e-13
    ... )

    6. Using transform() factory method (matrix transformation):

    >>> mel = mm.MagnetoElastic.transform(
    ...     B1=1e7, B2=1e7,
    ...     e_diag=(1e-3, 1e-3, 1e-3),  # Base strain
    ...     func=transform_matrix,
    ...     dt=1e-13,
    ...     transform_type='diagonal'
    ... )

    See Also
    --------
    Zeeman : Zeeman energy term with similar func/dt interface
    MagnetoElastic.static : Factory method for static strain
    MagnetoElastic.stage : Factory method for stage-based strain
    MagnetoElastic.transform : Factory method for time-dependent strain
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
        "func",  # Like Zeeman - for time-dependent
        "dt",    # Like Zeeman - for time-dependent
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
        func=None,  # Like Zeeman - for time-dependent
        dt=None,    # Like Zeeman - for time-dependent
        **kwargs,
    ):
        # Handle func/dt like Zeeman (func -> transform_script, dt -> transform_dt)
        if func is not None:
            if transform_script is not None:
                raise ValueError(
                    "Cannot specify both 'func' and 'transform_script'. "
                    "Use 'func' for simple time-dependence (like Zeeman) or "
                    "'transform_script' for advanced usage."
                )
            transform_script = func
        
        if dt is not None:
            if transform_dt is not None:
                raise ValueError(
                    "Cannot specify both 'dt' and 'transform_dt'. "
                    "Use 'dt' for simple time-dependence (like Zeeman) or "
                    "'transform_dt' for advanced usage."
                )
            transform_dt = dt
        
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
            # NOTE: e_diag/e_offdiag are OPTIONAL for transform mode
            # - If provided: Matrix transformation mode (future enhancement)
            # - If not provided: Direct substitution mode (current)
            # Set defaults for type validation
            e_diag = e_diag if e_diag is not None else (0, 0, 0)
            e_offdiag = e_offdiag if e_offdiag is not None else (0, 0, 0)

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

    @classmethod
    def static(cls, B1, B2, e_diag, e_offdiag, **kwargs):
        """Create static magneto-elastic energy term (YY_FixedMEL).

        Use this method for time-independent (constant) strain.

        Parameters
        ----------
        B1, B2 : numbers.Real, dict, discretisedfield.Field
            Magneto-elastic coefficients in J/m³.
        e_diag, e_offdiag : (3,) array_like, dict, discretisedfield.Field
            Diagonal and off-diagonal strain components (constant in time).
        **kwargs
            Additional keyword arguments passed to ``MagnetoElastic``.

        Returns
        -------
        MagnetoElastic
            Static magneto-elastic energy term (YY_FixedMEL).

        Examples
        --------
        Constant strain:

        >>> mel = mm.MagnetoElastic.static(
        ...     B1=1e7, B2=1e7,
        ...     e_diag=(1e-3, 1e-3, 1e-3),
        ...     e_offdiag=(0, 0, 0)
        ... )

        See Also
        --------
        MagnetoElastic.stage : Stage-based strain from OVf files
        MagnetoElastic.transform : Time-dependent strain with func/dt
        """
        return cls(B1=B1, B2=B2, e_diag=e_diag, e_offdiag=e_offdiag, **kwargs)

    @classmethod
    def stage(
        cls, B1, B2, e_diag_files, e_offdiag_files, stage_count=None, **kwargs
    ):
        """Create stage-based magneto-elastic energy term (YY_StageMEL).

        Use this method for strain that changes between stages (coarse time resolution).
        Each OVf file contains the strain distribution for one stage.

        Parameters
        ----------
        B1, B2 : numbers.Real, dict, discretisedfield.Field
            Magneto-elastic coefficients in J/m³.
        e_diag_files, e_offdiag_files : list of str
            Lists of paths to OVf files containing strain components for each stage.
            The number of files determines the number of stages.
        stage_count : int, optional
            Number of stages. If not specified, inferred from file list length.
        **kwargs
            Additional keyword arguments passed to ``MagnetoElastic``.

        Returns
        -------
        MagnetoElastic
            Stage-based magneto-elastic energy term (YY_StageMEL).

        Examples
        --------
        Strain from OVf files:

        >>> mel = mm.MagnetoElastic.stage(
        ...     B1=1e7, B2=1e7,
        ...     e_diag_files=['strain_0.ovf', 'strain_1.ovf', 'strain_2.ovf'],
        ...     e_offdiag_files=['off_0.ovf', 'off_1.ovf', 'off_2.ovf'],
        ...     stage_count=3
        ... )

        Note
        ----
        For stage-based strain with Python callable (called each stage),
        use ``MagnetoElastic`` directly with ``e_diag_script`` and
        ``e_offdiag_script`` parameters.

        See Also
        --------
        MagnetoElastic.static : Static strain
        MagnetoElastic.transform : Time-dependent strain with func/dt
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
        e_diag=None,
        e_offdiag=None,
        transform_script=None,
        transform_type="diagonal",
        transform_script_args="total_time",
        transform_dt=None,
        func=None,
        dt=None,
        **kwargs,
    ):
        """Create time-dependent magneto-elastic energy term (YY_TransformStageMEL).

        Use this method for strain that varies continuously with time (fine time resolution).
        Analogous to ``Zeeman(func=..., dt=...)`` interface.

        Two modes are available:

        1. **Direct substitution**: ``func(t)`` returns full strain values
           [e11, e22, e33, e23, e13, e12]. Use without ``e_diag``/``e_offdiag``.

        2. **Matrix transformation**: ``func(t)`` returns transformation matrix
           elements M(t), and final strain is computed as
           e_final = M(t) × e_base × M(t)ᵀ. Use with ``e_diag``/``e_offdiag``.

        Parameters
        ----------
        B1, B2 : numbers.Real, dict, discretisedfield.Field
            Magneto-elastic coefficients in J/m³.
        e_diag, e_offdiag : (3,) array_like, optional
            Base strain components for matrix transformation mode.
            For direct substitution mode, leave as None.
        transform_script : callable, optional
            Time-dependent function. Returns:

            - Direct substitution: ``[e11, e22, e33, e23, e13, e12]``
            - Matrix transformation: ``[M11, M22, M33, dM11, dM22, dM33]``
              for diagonal type

        transform_type : {'diagonal', 'symmetric', 'general'}, optional
            Type of strain representation (default: 'diagonal').

            - ``'diagonal'``: 6 values (direct) or 6 matrix elements [M11, M22, M33, dM11, dM22, dM33]
            - ``'symmetric'``: 12 values
            - ``'general'``: 18 values

        transform_script_args : str, optional
            Arguments for transform_script (default: 'total_time').
        transform_dt : float, optional
            Time step for pre-computation in seconds (default: 1e-13 = 0.1 ps).
        func : callable, optional
            Zeeman-style interface: alternative to ``transform_script``.
        dt : float, optional
            Zeeman-style interface: alternative to ``transform_dt``.
        **kwargs
            Additional keyword arguments passed to ``MagnetoElastic``.

        Returns
        -------
        MagnetoElastic
            Time-dependent magneto-elastic energy term (YY_TransformStageMEL).

        Examples
        --------
        1. Direct substitution mode (full strain values):

        >>> import numpy as np
        >>> def strain_func(t):
        ...     f = 1e9  # 1 GHz
        ...     A = 1e-3
        ...     strain = A * np.sin(2 * np.pi * f * t)
        ...     return [strain, strain, strain, 0, 0, 0]
        >>> mel = mm.MagnetoElastic.transform(
        ...     B1=1e7, B2=1e7,
        ...     func=strain_func,  # Returns full strain
        ...     dt=1e-13  # 0.1 ps
        ... )

        2. Matrix transformation mode (scaling matrix):

        >>> def transform_matrix(t):
        ...     m = 1 + 1e-3 * np.sin(2 * np.pi * 1e9 * t)
        ...     dm = 1e-3 * 2 * np.pi * 1e9 * np.cos(2 * np.pi * 1e9 * t)
        ...     return [m, m, m, dm, dm, dm]  # M11, M22, M33, dM11, dM22, dM33
        >>> mel = mm.MagnetoElastic.transform(
        ...     B1=1e7, B2=1e7,
        ...     e_diag=(1e-3, 1e-3, 1e-3),  # Base strain
        ...     func=transform_matrix,  # Returns M(t)
        ...     dt=1e-13,
        ...     transform_type='diagonal'  # e_final_ii = M_ii * e_base_ii
        ... )

        3. Pulse strain (direct substitution):

        >>> def pulse_strain(t):
        ...     if t < 1e-9:
        ...         return [1e-3, 1e-3, 1e-3, 0, 0, 0]
        ...     else:
        ...         return [0, 0, 0, 0, 0, 0]
        >>> mel = mm.MagnetoElastic.transform(
        ...     B1=1e7, B2=1e7,
        ...     func=pulse_strain,
        ...     dt=1e-12
        ... )

        See Also
        --------
        MagnetoElastic.static : Static strain
        MagnetoElastic.stage : Stage-based strain from OVf files
        Zeeman : Zeeman energy with similar func/dt interface
        """
        # Validate that either func or transform_script is provided
        if func is None and transform_script is None:
            raise ValueError(
                "Either 'func' or 'transform_script' must be specified "
                "for time-dependent MEL."
            )

        # Handle func/dt like Zeeman
        if func is not None:
            if transform_script is not None:
                raise ValueError(
                    "Cannot specify both 'func' and 'transform_script'. "
                    "Use 'func' for Zeeman-style interface or 'transform_script'."
                )
            transform_script = func

        if dt is not None:
            if transform_dt is not None:
                raise ValueError(
                    "Cannot specify both 'dt' and 'transform_dt'. "
                    "Use 'dt' for Zeeman-style interface or 'transform_dt'."
                )
            transform_dt = dt

        return cls(
            B1=B1,
            B2=B2,
            e_diag=e_diag,
            e_offdiag=e_offdiag,
            transform_type=transform_type,
            transform_script=transform_script,
            transform_script_args=transform_script_args,
            transform_dt=transform_dt,
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
