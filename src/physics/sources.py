import numpy as np


class Source:
    """Base class for external forcing."""

    def initialize(self, grid):
        """Precompute any spatial quantities."""
        pass

    def __call__(self, t):
        raise NotImplementedError

class OscillatingGaussianSource(Source):
    """Oscillating Gaussian forcing."""

    __name__ = "oscillating_gaussian_source"

    def __init__(
        self,
        center_ratio=0.5,
        sigma=0.05,
        amplitude=1.0,
        omega=2 * np.pi,
        num_fields=1,
        active_field=0,
    ):
        self.center_ratio = center_ratio
        self.sigma = sigma
        self.amplitude = amplitude
        self.omega = omega
        self.num_fields = num_fields
        self.active_field = active_field

        self.profile = None

    def initialize(self, grid):

        if grid.ndim != 2:
            raise NotImplementedError(
                "OscillatingGaussianSource only supports 2D grids."
            )

        y, x = grid.coordinates

        yc = y.min() + self.center_ratio * (y.max() - y.min())
        xc = x.min() + self.center_ratio * (x.max() - x.min())

        r2 = (x - xc) ** 2 + (y - yc) ** 2

        gaussian = np.exp(-r2 / (2 * self.sigma**2))

        self.profile = np.zeros((self.num_fields, *grid.shape))
        self.profile[self.active_field] = gaussian

    def __call__(self, t):

        if self.profile is None:
            raise RuntimeError(
                "Source has not been initialized. "
                "Call source.initialize(grid) first."
            )

        return (
            self.amplitude
            * np.sin(self.omega * t)
            * self.profile
        )


class ZeroMeanGaussianSource(Source):
    """
    Localized oscillatory forcing with exactly zero domain-integrated PV.

    The spatial profile is the standard Gaussian minus its spatial mean::

        profile(x, y) = G(x, y) − ⟨G⟩

    where G is a Gaussian centred at ``center_ratio`` of the domain and
    ⟨G⟩ = (1/|Ω|) ∫∫ G dx dy is its spatial average over the grid.

    This guarantees::

        ∫∫ F(x, y, t) dx dy = A · sin(ωt) · ∫∫ (G − ⟨G⟩) dx dy = 0   ∀ t

    so no net PV is ever injected into the domain.  Without this property the
    plain-Gaussian source continuously shifts the domain-total PV up and down
    by up to  A · sin(ωt) · ∫∫ G dx dy  each half-period, which the Poisson
    solver absorbs as a large-scale linear tilt / background adjustment that
    obscures the wave-radiation pattern.

    Physical background
    -------------------
    Classical linear barotropic β-plane Rossby wave radiation experiments
    (Held et al. 1985; Vallis 2006 §6) prescribe a localized oscillatory
    vorticity source with zero domain integral.  The positive lobe near the
    centre injects cyclonic PV; the weak negative halo (−⟨G⟩ spread over the
    whole domain) removes exactly the same amount.  The Poisson solve of the
    resulting PV anomaly produces a localized streamfunction whose x-gradient
    drives westward-propagating Rossby wave trains from the forcing region.

    Parameters
    ----------
    center_ratio : float
        Fractional position of the source centre along each axis (0–1).
        A value > 0.5 places the source east of centre, leaving more domain
        for westward radiation to develop.
    sigma : float
        Gaussian half-width.  Must satisfy sigma / dx ≥ 4 to be resolved
        and sigma / L_domain ≤ 0.15 so the source is genuinely localised.
    amplitude : float
        Peak forcing magnitude.
    omega : float
        Angular frequency of temporal oscillation (rad / time unit).
    num_fields : int
        Number of state components (must match the equation's state vector).
    active_field : int
        Index of the component to force.
    """

    __name__ = "zero_mean_gaussian_source"

    def __init__(
        self,
        center_ratio=0.6,
        sigma=0.3,
        amplitude=1.0,
        omega=np.pi,
        num_fields=1,
        active_field=0,
    ):
        self.center_ratio = center_ratio
        self.sigma        = sigma
        self.amplitude    = amplitude
        self.omega        = omega
        self.num_fields   = num_fields
        self.active_field = active_field

        self.profile = None

    def initialize(self, grid):

        if grid.ndim != 2:
            raise NotImplementedError(
                "ZeroMeanGaussianSource only supports 2D grids."
            )

        y, x = grid.coordinates

        yc = y.min() + self.center_ratio * (y.max() - y.min())
        xc = x.min() + self.center_ratio * (x.max() - x.min())

        r2 = (x - xc) ** 2 + (y - yc) ** 2

        gaussian = np.exp(-r2 / (2 * self.sigma**2))

        # Subtract spatial mean → ∫∫ profile dx dy = 0 exactly
        gaussian_zero_mean = gaussian - gaussian.mean()

        self.profile = np.zeros((self.num_fields, *grid.shape))
        self.profile[self.active_field] = gaussian_zero_mean

    def __call__(self, t):

        if self.profile is None:
            raise RuntimeError(
                "Source has not been initialized. "
                "Call source.initialize(grid) first."
            )

        return (
            self.amplitude
            * np.sin(self.omega * t)
            * self.profile
        )