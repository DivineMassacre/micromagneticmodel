import collections

import numpy as np
import discretisedfield as df
import ubermagutil as uu
import ubermagutil.typesystem as ts

from .energyterm import EnergyTerm


@uu.inherit_docs
@ts.typesystem(
    H=ts.Parameter(descriptor=ts.Vector(size=3), otherwise=df.Field),
    wave=ts.Subset(sample_set={"sin", "sinc"}, unpack=False),
    f=ts.Scalar(positive=True),
    t0=ts.Scalar(),
    func=ts.Parameter(
        descriptor=ts.Subset(sample_set={"sin", "sinc"}, unpack=False),
        otherwise=collections.abc.Callable,
    ),
    dt=ts.Scalar(positive=True),
    tcl_strings=ts.Dictionary(
        key_descriptor=ts.Subset(
            sample_set=("script", "energy", "type", "script_args", "script_name"),
            unpack=False,
        ),
        value_descriptor=ts.Typed(expected_type=str),
    ),
    spatiotemporal_terms=ts.Parameter(
        descriptor=list,
        default=[],
    ),
)
class Zeeman(EnergyTerm):
    r"""Zeeman energy term.

    .. math::

        w = -\mu_{0}M_\text{s} \mathbf{m} \cdot \mathbf{H}

    Zeeman energy term allows defining time-dependent as well as
    time-independent external magnetic field. If only external magnetic field
    ``H`` is passed, a time-constant field is defined.

    The time-dependent field $H(t)$ is obtained by multiplying the
    time-independent field `H` with a time-dependent pre-factor $f(t)$:

    .. math::

        H(t) = f(t) \cdot H

    Three different methods are available to define the pre-factor for a
    time-dependent field:

    - pre-defined ``sine`` wave and ``sinc`` pulse
    - custom time-dependence via Python callable
    - custom ``tcl`` code passed directly to OOMMF

    There are two built-in functions to specify a time-dependent field. To use
    these a string must be passed to ``func``. ``func`` can be either
    ``'sine'`` or ``'sinc'``. If time-dependent external magnetic field is
    defined using ``func``, ``f`` and ``t0`` must be passed. For
    ``func='sine'``, energy density is:

    .. math::

        w = -\mu_{0}M_\text{s} \mathbf{m} \cdot \mathbf{H} \sin[2\pi
        f(t-t_{0})]

    whereas for ``func='sinc'``, the energy density is:

    .. math::

        w = -\mu_{0}M_\text{s} \mathbf{m} \cdot \mathbf{H}
        \text{sinc}[2\pi f(t-t_{0})]

    and ``f`` is a cut-off frequency.

    Arbitrary time-dependence can be specified by passing a callable to
    ``func``. Additionally ``dt`` (in seconds) must be provided. The function
    is evaluated at all time steps separated by ``dt`` (up to the desired
    run-time). Additionally, the derivative is computed internally (using
    central differences). Therefore, the function has to be differentiable. In
    order for this method to be stable a reasonable small time-step must be
    chosen. As a rough guideline start around ``dt=1e-13`` (s). The callable
    passed to ``func`` must either return a single number that is used to
    multiply the initial field ``H`` or a list of nine values that define a
    matrix ``M`` that is multiplied with the initial field vector. Ordering of
    the matrix elements is ``[M11, M12, M13, M21, M22, M23, M31, M32, M33]``.
    The matrix allows for more complicated processes, e.g. a rotating field
    (for more details see the OOMMF documentation:
    https://math.nist.gov/oommf/doc/userguide20a3/userguide/Standard_Oxs_Ext_Child_Clas.html#TZ).

    To have more control and use the full flexibility of OOMMF it is also
    possible to directly pass several tcl strings that are added to the ``mif``
    file without further processing. The dictionary must be passed to
    ``tcl_strings`` and must contain ``script``, ``energy``, ``type``,
    ``script_args``, and ``script_name``. Please refer to the OOMMF
    documentation for detailed explanations. In general, specifying
    ``time_dependence`` and ``tstep`` is easier for the user and should be
    preferred, if possible.

    **Spatiotemporal fields** (H(x,y,z,t)) can be defined using the
    ``spatiotemporal_terms`` parameter or the :meth:`add_time_term` method.
    This uses the Oxs_StageZeeman + Oxs_ScriptVectorField approach in OOMMF.
    The total field is computed as:

    .. math::

        \mathbf{H}(x,y,z,t) = \mathbf{H}_\text{static} + \sum_i f_i(t) \cdot \text{mask}_i(x,y,z)

    where :math:`f_i(t)` is a time-dependent function and :math:`\text{mask}_i(x,y,z)` is
    an optional spatial mask.

    Parameters
    ----------
    H : (3,) array_like, dict, discretisedfield.Field

        If a single length-3 array_like (tuple, list, ``numpy.ndarray``) is
        passed, which consists of ``numbers.Real``, a spatially constant
        parameter is defined. For a spatially varying parameter, either a
        dictionary, e.g. ``H={'region1': (0, 0, 3e6), 'region2': (0, 0,
        -3e6)}`` (if the parameter is defined "per region") or
        ``discretisedfield.Field`` is passed.

    f : numbers.Real, optional (required for ``func='sin'``/``'sinc'``)

        (Cut-off) frequency in Hz.

    t0 : numbers.Real, optional (required for ``func='sin'``/``'sinc'``)

        Time for adjusting the phase (time-shift) of a wave.

    func : str, callable, optional

        Predefined functions can be used by passing ``'sin'`` or ``'sinc'``.
        Callables can be used to define arbitrary time-dependence. Called at
        times that are multiples of ``dt``. Must return either a single
        number or a list of nine values.

    dt : numbers.Real, optional (required for callable ``func``)

        Time steps in seconds to evaluate callable ``func`` at.

    tcl_strings : dict, optional

        Dictionary of ``tcl`` strings to be included into the ``mif`` file for
        more control over specific time-dependencies. Must contain the
        following keys: ``script``, ``energy``, ``type``, ``script_args``, and
        ``script_name``. Refer to the OOMMF documentation for more details:
        https://math.nist.gov/oommf/doc/userguide20a3/userguide/Standard_Oxs_Ext_Child_Clas.html#SU.
        ``script_name`` refers to what is called ``script`` in the function
        definition on the OOMMF website.

    spatiotemporal_terms : list, optional

        List of ``(func, mask)`` tuples for spatiotemporal field terms.
        Each tuple defines a time-dependent term where:

        - ``func`` : callable
            Time-dependent function. Can return:

            - Scalar: ``float`` → ``f(t)`` (multiplied by mask)
            - Vector: ``float`` → ``(Hx, Hy, Hz)``

            .. warning::

               For proper MIF generation, ``func`` must be defined in a ``.py`` file
               (not in Jupyter notebook or REPL). The function can only use standard
               math functions (``sin``, ``cos``, ``exp``, ``sqrt``, etc.).

        - ``mask`` : callable, dict, or None, optional
            Spatial mask. Can be:

            - ``None``: uniform mask (1 for all cells)
            - callable: ``(x, y, z) → scalar`` or ``(Hx, Hy, Hz)``
            - dict: ``{'region_name': value}`` for regional masks

            .. warning::

               For proper MIF generation, ``mask`` must be defined in a ``.py`` file
               (not in notebook/REPL). The mask is **fixed in time** and cannot depend
               on ``t``.

        Default is empty list (no spatiotemporal terms).

        .. seealso::

            :meth:`add_time_term` : Method for adding spatiotemporal terms.
            :meth:`add_harmonic_term` : Convenience method for harmonic fields.

    stage_count : int, optional

        Number of stages for spatiotemporal field updates.
        If ``None`` (default), automatically taken from :class:`TimeDriver` 
        parameter ``n``. This is the recommended usage.

        .. warning::

           The ``stage_count`` in Zeeman must match the number of stages ``n`` 
           passed to :meth:`TimeDriver.drive()`. If you explicitly set 
           ``stage_count``, ensure it matches the driver's ``n``.

        Default is ``None``.

    dt : numbers.Real, optional

        Time step for spatiotemporal field updates (seconds).
        The field is updated every ``dt`` seconds.
        Default is 1e-13 (0.1 ps).

        .. tip::

           Choose ``dt`` small enough to resolve the fastest temporal variations
           in your field. As a rule of thumb, use at least 10-20 steps per period
           of the highest frequency component.

    Examples
    --------
    1. Defining the Zeeman energy term using a vector.

    >>> import micromagneticmodel as mm
    ...
    >>> zeeman = mm.Zeeman(H=(0, 0, 1e6))

    2. Defining the Zeeman energy term using dictionary.

    >>> zeeman = mm.Zeeman(H={'region1': (0, 0, 1e6), 'region2': (0, 0, -1e6)})

    3. Defining the Zeeman energy term using ``discretisedfield.Field``.

    >>> import discretisedfield as df
    ...
    >>> region = df.Region(p1=(0, 0, 0), p2=(5e-9, 5e-9, 10e-9))
    >>> mesh = df.Mesh(region=region, n=(5, 5, 10))
    >>> H = df.Field(mesh, nvdim=3, value=(1e6, -1e6, 0))
    >>> zeeman = mm.Zeeman(H=H)

    4. Defining the Zeeman energy term using a vector which changes as a sine
       wave.

    >>> zeeman = mm.Zeeman(H=(0, 0, 1e6), func='sin', f=1e9, t0=0)

    5. Defining an exponentially decaying field.

    >>> import numpy as np
    >>> def decay(t):
    ...     t_0 = 1e-10
    ...     return np.exp(-t / t_0)
    >>> zeeman = mm.Zeeman(H=(0, 0, 1e6), func=decay, dt=1e-13)

    6. Defining a spatiotemporal field with uniform time dependence.

    >>> import numpy as np
    >>> zeeman = mm.Zeeman(H=(1e6, 0, 0))
    >>> zeeman.add_time_term(lambda t: (1e3 * np.sin(2*np.pi*1e9*t), 0, 0))

    7. Defining a spatiotemporal field with Gaussian spatial mask.

    >>> zeeman = mm.Zeeman()
    >>> zeeman.add_time_term(
    ...     func=lambda t: np.sin(2*np.pi*1e9*t),
    ...     mask=lambda x,y,z: (1e3 * np.exp(-x**2/1e-16), 0, 0)
    ... )

    8. Defining a traveling wave (two terms).

    >>> H0 = 1e6
    >>> omega = 2*np.pi * 1e9
    >>> k = 2*np.pi / 100e-9
    >>> zeeman = mm.Zeeman()
    >>> zeeman.add_time_term(
    ...     func=lambda t: H0 * np.sin(omega * t),
    ...     mask=lambda x,y,z: np.cos(k * x)
    ... )
    >>> zeeman.add_time_term(
    ...     func=lambda t: H0 * np.cos(omega * t),
    ...     mask=lambda x,y,z: -np.sin(k * x)
    ... )

    9. Using built-in harmonic function (quick method).

    >>> zeeman = mm.Zeeman()
    >>> zeeman.add_harmonic_term(amplitude=1e5, frequency=1e9, mask='cos', k=2*np.pi/100e-9)

    10. Using built-in temporal and spatial functions.

    >>> zeeman = mm.Zeeman()
    >>> zeeman.add_time_term(
    ...     func=zeeman.sin,
    ...     func_kwargs={'amplitude': 1e5, 'frequency': 1e9},
    ...     mask=zeeman.cos_mask,
    ...     mask_kwargs={'k': 2*np.pi/100e-9}
    ... )

    11. An attempt to define the Zeeman energy term using a wrong value.

    >>> zeeman = mm.Zeeman(H=(0, -1e7))  # length-2 vector
    Traceback (most recent call last):
    ...
    ValueError: ...

    """

    # 'wave' is replaced by 'func' (deprecated but kept for compatibility)
    _allowed_attributes = ["H", "wave", "f", "t0", "func", "dt", "tcl_strings", "spatiotemporal_terms"]

    def __init__(self, H=None, spatiotemporal_terms=None, stage_count=None, dt=None, **kwargs):
        """Initialize Zeeman term.

        Parameters
        ----------
        H : tuple, list, dict, or discretisedfield.Field, optional
            Static magnetic field. Default is (0, 0, 0).
        spatiotemporal_terms : list, optional
            List of (func, mask) tuples for spatiotemporal terms.
        stage_count : int, optional
            Number of stages for spatiotemporal field updates.
            If None, automatically taken from TimeDriver (parameter n).
            Default is None.
        dt : float, optional
            Time step for spatiotemporal field updates (seconds).
            Default is 1e-13 (0.1 ps).
        """
        self.H = H if H is not None else (0, 0, 0)
        self._terms = spatiotemporal_terms if spatiotemporal_terms is not None else []
        self._stage_count = stage_count  # None means auto from driver
        self._dt = dt if dt is not None else 1e-13
        super().__init__(**kwargs)

    def add_time_term(self, func, mask=None):
        """Add a time-dependent term to the Zeeman field.

        The total field is: H_total = H_static + Σᵢ [fᵢ(t) × maskᵢ(x,y,z)]

        .. note::

           **Important limitations on custom functions:**

           - ``func`` must be defined in a ``.py`` file (not in Jupyter notebook or REPL)
             for proper MIF generation. This is required because the converter uses
             ``inspect.getsource()`` to extract the function body.
           - ``func`` can only use standard math functions: ``sin``, ``cos``, ``tan``,
             ``exp``, ``log``, ``sqrt``, ``abs``, ``numpy.*`` equivalents.
           - ``func`` must be a simple expression or return statement. Complex control
             flow (if/else, loops) is not supported.
           - ``mask`` has the same limitations as ``func``, but takes ``(x, y, z)`` arguments.
           - ``mask`` is **fixed in time** - it cannot depend on ``t``.
           - For ``add_harmonic_term()``, the parameters are automatically extracted,
             but using explicit functions in a ``.py`` file is recommended for reliability.

        Parameters
        ----------
        func : callable
            Time-dependent function. Can return:

            - Scalar: ``float`` → f(t) (multiplied by mask)
            - Vector: ``float`` → (Hx, Hy, Hz)

            .. warning::

               The function must be defined in a ``.py`` file (not in notebook/REPL)
               and must use only supported math functions. See note above.

        mask : callable, dict, or None, optional
            Spatial mask. Can be:

            - ``None``: uniform mask (1 for all cells)
            - callable: ``(x, y, z) → scalar`` or ``(Hx, Hy, Hz)``
            - dict: ``{'region_name': value}`` for regional masks

            .. warning::

               The mask must be defined in a ``.py`` file (not in notebook/REPL)
               and cannot depend on time ``t``. See note above.

        Examples
        --------
        Uniform time-dependent field:

        >>> zeeman = mm.Zeeman(H=(1e6, 0, 0))
        >>> zeeman.add_time_term(lambda t: (1e3 * np.sin(2*np.pi*1e9*t), 0, 0))

        With spatial Gaussian mask:

        >>> zeeman.add_time_term(
        ...     func=lambda t: np.sin(2*np.pi*1e9*t),
        ...     mask=lambda x,y,z: (1e3 * np.exp(-x**2/1e-16), 0, 0)
        ... )

        Traveling wave (two terms):

        >>> zeeman.add_time_term(
        ...     func=lambda t: H0 * np.sin(omega * t),
        ...     mask=lambda x,y,z: np.cos(k * x)
        ... )
        >>> zeeman.add_time_term(
        ...     func=lambda t: H0 * np.cos(omega * t),
        ...     mask=lambda x,y,z: -np.sin(k * x)
        ... )
        """
        # Validate func
        if not callable(func):
            raise TypeError(
                f"func must be callable, got {type(func).__name__}. "
                f"Example: lambda t: np.sin(omega*t)"
            )

        # Validate mask
        if mask is not None:
            if not callable(mask) and not isinstance(mask, dict):
                raise TypeError(
                    f"mask must be callable, dict, or None, got {type(mask).__name__}. "
                    f"Example: lambda x,y,z: np.cos(k*x)"
                )

        # Validate func return type (test call at t=0)
        try:
            test_result = func(0)
            if isinstance(test_result, (tuple, list)):
                if len(test_result) != 3:
                    raise ValueError(
                        f"Vector func must return 3 components (Hx, Hy, Hz), got {len(test_result)}"
                    )
                # Check all components are numeric
                for i, val in enumerate(test_result):
                    if not isinstance(val, (int, float)):
                        raise TypeError(
                            f"Vector func component {i} must be numeric, got {type(val).__name__}"
                        )
            elif not isinstance(test_result, (int, float)):
                raise TypeError(
                    f"func must return scalar float or 3-tuple of floats, got {type(test_result).__name__}"
                )
        except ValueError:
            # Re-raise ValueError as-is (validation errors)
            raise
        except TypeError as e:
            # Re-raise TypeError with more context
            raise TypeError(f"Error evaluating func at t=0: {e}") from e
        except Exception as e:
            # Other errors (e.g., missing variables) - warn but don't fail
            # The error will be caught later during MIF generation if it persists
            import warnings
            warnings.warn(
                f"Could not validate func at t=0: {type(e).__name__}: {e}. "
                f"Ensure func is properly defined."
            )

        self._terms.append((func, mask))

    def clear_time_terms(self):
        """Remove all time-dependent terms."""
        self._terms = []

    @property
    def has_time_terms(self):
        """True if there are time-dependent terms."""
        return len(self._terms) > 0

    # ========== ВСТРОЕННЫЕ ВРЕМЕННЫЕ ФУНКЦИИ ==========

    @staticmethod
    def sin(t, amplitude=1, frequency=1, phase=0):
        """Sinusoidal temporal function.

        H(t) = amplitude * sin(2*pi*frequency*t + phase)

        Parameters
        ----------
        t : float
            Time (s)
        amplitude : float, optional
            Amplitude (A/m). Default is 1.
        frequency : float, optional
            Frequency (Hz). Default is 1.
        phase : float, optional
            Phase shift (rad). Default is 0.

        Returns
        -------
        float
            Field value at time t
        """
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)

    @staticmethod
    def cos(t, amplitude=1, frequency=1, phase=0):
        """Cosinusoidal temporal function.

        H(t) = amplitude * cos(2*pi*frequency*t + phase)

        Parameters
        ----------
        t : float
            Time (s)
        amplitude : float, optional
            Amplitude (A/m). Default is 1.
        frequency : float, optional
            Frequency (Hz). Default is 1.
        phase : float, optional
            Phase shift (rad). Default is 0.

        Returns
        -------
        float
            Field value at time t
        """
        return amplitude * np.cos(2 * np.pi * frequency * t + phase)

    @staticmethod
    def constant(t, amplitude=1):
        """Constant temporal function.

        H(t) = amplitude

        Parameters
        ----------
        t : float
            Time (s)
        amplitude : float, optional
            Amplitude (A/m). Default is 1.

        Returns
        -------
        float
            Field value at time t
        """
        return amplitude

    @staticmethod
    def gaussian(t, amplitude=1, center=0, sigma=1e-12):
        """Gaussian pulse temporal function.

        H(t) = amplitude * exp(-(t-center)^2 / (2*sigma^2))

        Parameters
        ----------
        t : float
            Time (s)
        amplitude : float, optional
            Amplitude (A/m). Default is 1.
        center : float, optional
            Center of the pulse (s). Default is 0.
        sigma : float, optional
            Width of the pulse (s). Default is 1e-12.

        Returns
        -------
        float
            Field value at time t
        """
        return amplitude * np.exp(-(t - center)**2 / (2 * sigma**2))

    @staticmethod
    def exponential(t, amplitude=1, tau=1e-12):
        """Exponential decay temporal function.

        H(t) = amplitude * exp(-t / tau)

        Parameters
        ----------
        t : float
            Time (s)
        amplitude : float, optional
            Amplitude (A/m). Default is 1.
        tau : float, optional
            Decay time constant (s). Default is 1e-12.

        Returns
        -------
        float
            Field value at time t
        """
        return amplitude * np.exp(-t / tau)

    # ========== ВСТРОЕННЫЕ ПРОСТРАНСТВЕННЫЕ МАСКИ ==========

    @staticmethod
    def uniform(x, y, z):
        """Uniform spatial mask (1 everywhere).

        Parameters
        ----------
        x, y, z : float
            Coordinates (m)

        Returns
        -------
        float
            Mask value (always 1.0)
        """
        return 1.0

    @staticmethod
    def cos_mask(x, y, z, k=1, axis='x'):
        """Cosine spatial mask.

        mask(x,y,z) = cos(k * axis_coord)

        Parameters
        ----------
        x, y, z : float
            Coordinates (m)
        k : float, optional
            Wave vector (rad/m). Default is 1.
        axis : str, optional
            Axis ('x', 'y', or 'z'). Default is 'x'.

        Returns
        -------
        float
            Mask value
        """
        coords = {'x': x, 'y': y, 'z': z}
        return np.cos(k * coords[axis])

    @staticmethod
    def sin_mask(x, y, z, k=1, axis='x'):
        """Sine spatial mask.

        mask(x,y,z) = sin(k * axis_coord)

        Parameters
        ----------
        x, y, z : float
            Coordinates (m)
        k : float, optional
            Wave vector (rad/m). Default is 1.
        axis : str, optional
            Axis ('x', 'y', or 'z'). Default is 'x'.

        Returns
        -------
        float
            Mask value
        """
        coords = {'x': x, 'y': y, 'z': z}
        return np.sin(k * coords[axis])

    @staticmethod
    def gaussian_mask(x, y, z, sigma=50e-9, center=(0, 0, 0)):
        """Gaussian spatial mask.

        mask(x,y,z) = exp(-((x-cx)^2 + (y-cy)^2 + (z-cz)^2) / (2*sigma^2))

        Parameters
        ----------
        x, y, z : float
            Coordinates (m)
        sigma : float, optional
            Width of the Gaussian (m). Default is 50e-9.
        center : tuple, optional
            Center of the Gaussian (cx, cy, cz) in meters. Default is (0, 0, 0).

        Returns
        -------
        float
            Mask value
        """
        dx = x - center[0]
        dy = y - center[1]
        dz = z - center[2]
        return np.exp(-(dx**2 + dy**2 + dz**2) / (2 * sigma**2))

    @staticmethod
    def step_mask(x, y, z, threshold=0, axis='x'):
        """Step function spatial mask.

        mask(x,y,z) = 1 if axis_coord > threshold else 0

        Parameters
        ----------
        x, y, z : float
            Coordinates (m)
        threshold : float, optional
            Threshold value (m). Default is 0.
        axis : str, optional
            Axis ('x', 'y', or 'z'). Default is 'x'.

        Returns
        -------
        float
            Mask value (0 or 1)
        """
        coords = {'x': x, 'y': y, 'z': z}
        return 1.0 if coords[axis] > threshold else 0.0

    # ========== МЕТОД ДЛЯ БЫСТРОГО ДОБАВЛЕНИЯ ==========

    def add_harmonic_term(self, amplitude, frequency, phase=0,
                          mask='uniform', **mask_kwargs):
        """Add harmonic (sinusoidal) term with one line.

        The total field contribution is:
        H_term = amplitude * sin(2*pi*frequency*t + phase) * mask(x,y,z)

        Parameters
        ----------
        amplitude : float
            Field amplitude (A/m)
        frequency : float
            Frequency (Hz)
        phase : float, optional
            Phase shift (rad). Default is 0.
        mask : str or callable, optional
            Spatial mask. Can be:

            - 'uniform': uniform mask (default)
            - 'cos': cosine mask
            - 'sin': sine mask
            - 'gaussian': Gaussian mask
            - 'step': step function mask
            - callable: custom mask function

        **mask_kwargs
            Additional arguments for mask function:

            - For 'cos'/'sin': k (wave vector), axis
            - For 'gaussian': sigma, center
            - For 'step': threshold, axis

        Examples
        --------
        Simple harmonic field:

        >>> zeeman = mm.Zeeman()
        >>> zeeman.add_harmonic_term(amplitude=1e5, frequency=1e9)

        Standing wave:

        >>> zeeman.add_harmonic_term(
        ...     amplitude=1e5,
        ...     frequency=1e9,
        ...     mask='cos',
        ...     k=2*np.pi/100e-9,
        ...     axis='x'
        ... )

        Traveling wave (two terms):

        >>> k = 2*np.pi/100e-9
        >>> zeeman = mm.Zeeman()
        >>> zeeman.add_harmonic_term(1e5, 1e9, phase=0, mask='sin', k=k, axis='x')
        >>> zeeman.add_harmonic_term(1e5, 1e9, phase=np.pi/2, mask='cos', k=k, axis='x')

        Gaussian spatial profile:

        >>> zeeman.add_harmonic_term(
        ...     amplitude=1e5,
        ...     frequency=1e9,
        ...     mask='gaussian',
        ...     sigma=50e-9
        ... )
        """
        func = lambda t: self.sin(t, amplitude=amplitude, frequency=frequency, phase=phase)

        if isinstance(mask, str):
            mask_funcs = {
                'uniform': self.uniform,
                'cos': self.cos_mask,
                'sin': self.sin_mask,
                'gaussian': self.gaussian_mask,
                'step': self.step_mask,
            }
            mask_func = mask_funcs.get(mask, self.uniform)
            mask = lambda x, y, z: mask_func(x, y, z, **mask_kwargs)

        self.add_time_term(func=func, mask=mask)

    @property
    def _reprlatex(self):
        if self.wave == "sin":
            return (
                r"-\mu_{0}M_\text{s} \mathbf{m}"
                r"\cdot \mathbf{H} \sin[2 \pi f (t-t_{0})]"
            )
        elif self.wave == "sinc":
            return (
                r"-\mu_{0}M_\text{s} \mathbf{m} \cdot \mathbf{H}\, "
                r"\text{sinc}[2 \pi f (t-t_{0})]"
            )
        elif (
            self.name != self.__class__.__name__.lower()
        ):  # Check for user defined name
            return (
                r"-\mu_{0}M_\text{s} \mathbf{m} \cdot \mathbf{H}_\text"
                + f"{{{self.name}}}"
            )

        else:
            return r"-\mu_{0}M_\text{s} \mathbf{m} \cdot \mathbf{H}"

    def effective_field(self, m):
        raise NotImplementedError
