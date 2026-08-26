"""Tritium fuel-cycle dynamics (compartment model).

Linear box model of the deuterium-tritium fuel cycle after Abdou et al.
(Fusion Technology 9 (1986) 250; Nucl. Fusion 61 (2021) 013001). Four
inventories evolve as the linear time-invariant system ``dI/dt = A I + b``:

- ``blanket``: bred tritium awaiting extraction. Source ``TBR * N_b``
  (one triton is burned per D-T source neutron, so tritons bred per
  burned triton equals the TBR). Drains with residence time ``tau_B``;
  a fraction ``extraction_efficiency`` reaches processing, the rest is
  lost.
- ``exhaust``: unburned fuel pumped from the plasma. Fueling at rate
  ``N_b / f_b`` burns ``N_b`` (fractional burnup ``f_b``); the unburned
  ``(1 - f_b) N_b / f_b`` enters here and drains to processing.
- ``processing``: loses ``processing_loss`` per pass; the rest reaches
  storage.
- ``storage``: supplies the fueling withdrawal ``N_b / f_b``; nothing
  couples back except decay, so trajectories are exactly linear in the
  startup inventory.

Every compartment additionally decays with the tritium half-life
12.32 y (Lucas & Unterweger 2000). The in-vessel plasma inventory is
not tracked (its residence time is negligible against all others).

Units: durations in DAYS at the API surface (the fuel-cycle-literature
convention), rates in atoms/s, inventories in kg. All internal math is
SI seconds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.optimize

from fusionbench.constants import AVOGADRO, ArrayLike, as_float64, scalar_like
from fusionbench.errors import FusionbenchError
from fusionbench.materials import ATOMIC_MASS_U
from fusionbench.provenance import Provenance, build_provenance

MODEL_ID = "abdou-1986"
MODERN_MODEL_ID = "abdou-2021"
HALF_LIFE_MODEL_ID = "lucas-unterweger-2000"

SECONDS_PER_DAY: Final = 86_400.0
SECONDS_PER_YEAR: Final = 3.1536e7
TRITIUM_HALF_LIFE_YEARS: Final = 12.32
TRITIUM_DECAY_CONSTANT: Final = math.log(2.0) / (TRITIUM_HALF_LIFE_YEARS * SECONDS_PER_YEAR)
"""Tritium decay constant, 1/s (Lucas & Unterweger 2000)."""

TRITIUM_MOLAR_MASS: Final = ATOMIC_MASS_U["H3"]
"""Tritium molar mass, g/mol (AME2020)."""

KG_PER_ATOM: Final = TRITIUM_MOLAR_MASS / 1.0e3 / AVOGADRO

COMPARTMENTS: Final[tuple[str, str, str, str]] = ("blanket", "exhaust", "processing", "storage")
"""Tracked inventory compartments, in state-vector order."""


def atoms_to_kg(atoms: ArrayLike) -> ArrayLike:
    """Convert a tritium atom count to kilograms."""
    return scalar_like(as_float64(atoms) * KG_PER_ATOM, atoms)


def kg_to_atoms(kg: ArrayLike) -> ArrayLike:
    """Convert a tritium mass in kilograms to an atom count."""
    return scalar_like(as_float64(kg) / KG_PER_ATOM, kg)


@dataclass(frozen=True)
class CycleHistory:
    """Time histories of the fuel-cycle inventories.

    Attributes
    ----------
    times : ndarray
        Sample times in days, shape ``(n,)``.
    inventories : mapping of str to ndarray
        Inventory in kg per compartment (keys :data:`COMPARTMENTS`).
    provenance : Provenance
        The generating cycle's parameters and model citations.
    """

    times: npt.NDArray[np.float64]
    inventories: Mapping[str, npt.NDArray[np.float64]]
    provenance: Provenance

    def __post_init__(self) -> None:
        """Validate shapes and freeze the mapping."""
        for name, values in self.inventories.items():
            if values.shape != self.times.shape:
                raise FusionbenchError(
                    f"inventory {name!r} shape {values.shape} does not match "
                    f"times shape {self.times.shape}"
                )
        object.__setattr__(self, "inventories", MappingProxyType(dict(self.inventories)))

    def inventory(self, compartment: str) -> npt.NDArray[np.float64]:
        """Inventory history (kg) of one compartment."""
        try:
            return self.inventories[compartment]
        except KeyError:
            raise FusionbenchError(
                f"unknown compartment {compartment!r}; known: {COMPARTMENTS}"
            ) from None

    def total(self) -> npt.NDArray[np.float64]:
        """Total tracked inventory (kg) versus time."""
        return as_float64(sum(self.inventories.values(), np.zeros_like(self.times)))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation."""
        return {
            "times": self.times.tolist(),
            "inventories": {name: values.tolist() for name, values in self.inventories.items()},
        }


@dataclass(frozen=True)
class TritiumCycle:
    """A deuterium-tritium fuel cycle as a linear compartment model.

    Parameters
    ----------
    burn_rate : float
        Tritium burn rate ``N_b``, tritons/s.
    tbr : float
        Tritium breeding ratio: tritons bred per burned triton.
    fractional_burnup : float
        Fraction ``f_b`` of injected tritium burned per pass, in (0, 1].
    startup_inventory : float
        Initial storage inventory, kg.
    blanket_residence_days, exhaust_residence_days, processing_residence_days : float
        Mean residence times of the compartments, days.
    extraction_efficiency : float
        Fraction of tritium leaving the blanket delivered to processing,
        in (0, 1]; the remainder is lost.
    processing_loss : float
        Fraction lost to waste per processing pass, in [0, 1).
    reserve_inventory : float
        Operational storage floor (kg) used by
        :meth:`required_startup_inventory`.
    decay_constant : float
        Radioactive decay constant, 1/s. Defaults to tritium's; zero is
        allowed for verification runs (then no steady state exists).
    """

    burn_rate: float
    tbr: float
    fractional_burnup: float
    startup_inventory: float
    blanket_residence_days: float = 10.0
    exhaust_residence_days: float = 0.2
    processing_residence_days: float = 1.0
    extraction_efficiency: float = 1.0
    processing_loss: float = 0.0
    reserve_inventory: float = 0.0
    decay_constant: float = TRITIUM_DECAY_CONSTANT

    def __post_init__(self) -> None:
        """Validate all parameter bounds."""
        if self.burn_rate <= 0.0:
            raise FusionbenchError("burn_rate must be positive (tritons/s)")
        if self.tbr < 0.0:
            raise FusionbenchError("tbr must be non-negative")
        if not 0.0 < self.fractional_burnup <= 1.0:
            raise FusionbenchError("fractional_burnup must lie in (0, 1]")
        if self.startup_inventory < 0.0:
            raise FusionbenchError("startup_inventory must be non-negative (kg)")
        for name in (
            "blanket_residence_days",
            "exhaust_residence_days",
            "processing_residence_days",
        ):
            if getattr(self, name) <= 0.0:
                raise FusionbenchError(f"{name} must be positive (days)")
        if not 0.0 < self.extraction_efficiency <= 1.0:
            raise FusionbenchError("extraction_efficiency must lie in (0, 1]")
        if not 0.0 <= self.processing_loss < 1.0:
            raise FusionbenchError("processing_loss must lie in [0, 1)")
        if self.reserve_inventory < 0.0:
            raise FusionbenchError("reserve_inventory must be non-negative (kg)")
        if self.decay_constant < 0.0:
            raise FusionbenchError("decay_constant must be non-negative (1/s)")

    @classmethod
    def from_fusion_power(cls, power_mw: float, **kwargs: float) -> TritiumCycle:
        """Build a cycle from a D-T fusion power.

        ``burn_rate = P / Q_DT`` with the registry Q value (17.589 MeV
        per reaction); "fusion power" excludes blanket exothermics.

        Parameters
        ----------
        power_mw : float
            D-T fusion power, MW.
        **kwargs
            Remaining :class:`TritiumCycle` parameters (tbr,
            fractional_burnup, startup_inventory, ...).
        """
        from fusionbench.constants import KEV_TO_JOULE
        from fusionbench.reactions import REACTIONS

        if power_mw <= 0.0:
            raise FusionbenchError("power_mw must be positive")
        burn_rate = power_mw * 1.0e6 / (REACTIONS["DT"].q_value * KEV_TO_JOULE)
        return cls(burn_rate=burn_rate, **kwargs)

    @classmethod
    def from_blanket_result(
        cls,
        result: Any,
        *,
        fractional_burnup: float,
        startup_inventory: float,
        **kwargs: float,
    ) -> TritiumCycle:
        """Build a cycle from a blanket neutronics result.

        Wires ``burn_rate = result.total_rate`` (every D-T source neutron
        burns one triton — the documented assumption that the source is
        pure D-T) and ``tbr = result.tbr.value``.

        Parameters
        ----------
        result : BlanketResult
            Output of :meth:`fusionbench.blanket.Blanket.run_neutronics`.
        fractional_burnup, startup_inventory
            Required cycle parameters not derivable from neutronics.
        **kwargs
            Remaining :class:`TritiumCycle` parameters.
        """
        return cls(
            burn_rate=float(result.total_rate),
            tbr=float(result.tbr.value),
            fractional_burnup=fractional_burnup,
            startup_inventory=startup_inventory,
            **kwargs,
        )

    def _system(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Build the LTI system ``(A [1/s], b [atoms/s])`` over :data:`COMPARTMENTS`."""
        lam = self.decay_constant
        tau_b = self.blanket_residence_days * SECONDS_PER_DAY
        tau_p = self.exhaust_residence_days * SECONDS_PER_DAY
        tau_e = self.processing_residence_days * SECONDS_PER_DAY
        f_b = self.fractional_burnup
        matrix = np.array(
            [
                [-(1.0 / tau_b + lam), 0.0, 0.0, 0.0],
                [0.0, -(1.0 / tau_p + lam), 0.0, 0.0],
                [
                    self.extraction_efficiency / tau_b,
                    1.0 / tau_p,
                    -(1.0 / tau_e + lam),
                    0.0,
                ],
                [0.0, 0.0, (1.0 - self.processing_loss) / tau_e, -lam],
            ]
        )
        source = np.array(
            [
                self.tbr * self.burn_rate,
                (1.0 - f_b) / f_b * self.burn_rate,
                0.0,
                -self.burn_rate / f_b,
            ]
        )
        return matrix, source

    def _initial_atoms(self) -> npt.NDArray[np.float64]:
        state = np.zeros(4)
        state[3] = float(kg_to_atoms(self.startup_inventory))
        return state

    def simulate(self, days: float, *, n_points: int = 1001) -> CycleHistory:
        """Evolve the inventories over ``[0, days]`` on a uniform grid.

        The affine flow is evaluated exactly via the matrix exponential
        of the augmented block ``[[A, b], [0, 0]]`` (machine precision at
        every grid point; works even for ``decay_constant = 0``).

        Parameters
        ----------
        days : float
            Simulation horizon, days.
        n_points : int
            Number of grid points (>= 2), including t = 0.
        """
        if days <= 0.0:
            raise FusionbenchError("days must be positive")
        if n_points < 2:
            raise FusionbenchError("n_points must be at least 2")
        matrix, source = self._system()
        dt = days * SECONDS_PER_DAY / (n_points - 1)
        augmented = np.zeros((5, 5))
        augmented[:4, :4] = matrix * dt
        augmented[:4, 4] = source * dt
        flow = as_float64(scipy.linalg.expm(augmented))
        step, offset = flow[:4, :4], flow[:4, 4]

        states = np.empty((n_points, 4))
        states[0] = self._initial_atoms()
        for i in range(1, n_points):
            states[i] = step @ states[i - 1] + offset

        times = np.linspace(0.0, days, n_points)
        inventories = {
            name: as_float64(atoms_to_kg(states[:, j])) for j, name in enumerate(COMPARTMENTS)
        }
        return CycleHistory(times=times, inventories=inventories, provenance=self.provenance)

    def steady_state(self) -> Mapping[str, float]:
        """Steady-state inventories (kg) per compartment.

        ``I* = -A^{-1} b``; unique whenever ``decay_constant > 0``. The
        storage entry may be negative, signalling a tritium deficit (the
        cycle is not self-sufficient). Closed forms (back-substitution)::

            I_B* = TBR N_b tau_B / (1 + lam tau_B)
            I_P* = (1-f_b)/f_b N_b tau_P / (1 + lam tau_P)
            I_E* = (eta I_B*/tau_B + I_P*/tau_P) tau_E / (1 + lam tau_E)
            I_S* = ((1-eps) I_E*/tau_E - N_b/f_b) / lam
        """
        if self.decay_constant == 0.0:
            raise FusionbenchError(
                "no steady state without decay: the storage equation is singular"
            )
        matrix, source = self._system()
        atoms = np.linalg.solve(matrix, -source)
        return MappingProxyType(
            {name: float(atoms_to_kg(atoms[j])) for j, name in enumerate(COMPARTMENTS)}
        )

    def _state_at(
        self, t_seconds: float, initial: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluate the exact affine flow at one time (atoms)."""
        matrix, source = self._system()
        augmented = np.zeros((5, 5))
        augmented[:4, :4] = matrix * t_seconds
        augmented[:4, 4] = source * t_seconds
        flow = as_float64(scipy.linalg.expm(augmented))
        return as_float64(flow[:4, :4] @ initial + flow[:4, 4])

    def accumulation_rate(self) -> float:
        """Net storage accumulation rate at upstream steady state, kg/day.

        Closed form (excluding storage's own decay)::

            a = (1-eps)/(1+lam tau_E) * [eta TBR/(1+lam tau_B)
                + (1-f_b)/(f_b (1+lam tau_P))] * N_b  -  N_b/f_b

        which reduces to ``(TBR - 1) N_b`` for a lossless, decay-free
        cycle. Negative values signal a tritium deficit.
        """
        lam = self.decay_constant
        tau_b = self.blanket_residence_days * SECONDS_PER_DAY
        tau_p = self.exhaust_residence_days * SECONDS_PER_DAY
        tau_e = self.processing_residence_days * SECONDS_PER_DAY
        f_b = self.fractional_burnup
        atoms_per_s = (1.0 - self.processing_loss) / (1.0 + lam * tau_e) * (
            self.extraction_efficiency * self.tbr / (1.0 + lam * tau_b)
            + (1.0 - f_b) / (f_b * (1.0 + lam * tau_p))
        ) * self.burn_rate - self.burn_rate / f_b
        return atoms_per_s * KG_PER_ATOM * SECONDS_PER_DAY

    @property
    def self_sufficient(self) -> bool:
        """Whether storage accumulates tritium at steady state."""
        return self.accumulation_rate() > 0.0

    def doubling_time(self, *, max_days: float = 36_500.0) -> float | None:
        """Days until the storage inventory first reaches twice its start.

        The classic figure of merit: time to accumulate a second plant's
        startup inventory. Returns ``None`` if storage never doubles
        within ``max_days`` (increase the horizon for slowly-accumulating
        cycles; a cycle with negative :meth:`accumulation_rate` never
        doubles).
        """
        if self.startup_inventory <= 0.0:
            raise FusionbenchError("doubling_time requires a positive startup_inventory")
        initial = self._initial_atoms()
        target = 2.0 * initial[3]
        history_atoms = np.array(
            [
                float(kg_to_atoms(v))
                for v in self.simulate(days=max_days, n_points=4001).inventory("storage")
            ]
        )
        above = np.nonzero(history_atoms >= target)[0]
        if above.size == 0:
            return None
        first = int(above[0])
        if first == 0:
            return 0.0
        grid = np.linspace(0.0, max_days, 4001)

        def excess(t_days: float) -> float:
            return float(self._state_at(t_days * SECONDS_PER_DAY, initial)[3] - target)

        return float(
            scipy.optimize.brentq(excess, grid[first - 1], grid[first], xtol=1e-9 * max_days)
        )

    def required_startup_inventory(self, *, days: float = 3650.0) -> float:
        """Minimum startup inventory (kg) keeping storage above the reserve.

        Exact via superposition: storage from startup ``I0`` obeys
        ``I_S(t; I0) = I0 exp(-lam t) + I_S(t; 0)``, so the constraint
        ``I_S >= reserve_inventory`` over ``[0, days]`` gives
        ``I0 = max(0, max_t (reserve - I_S(t; 0)) exp(+lam t))``.

        The horizon is intrinsic: for a cycle that is not self-sufficient
        the requirement grows without bound as ``days`` increases — the
        physically correct statement that such a plant needs external
        tritium supply indefinitely.
        """
        if days <= 0.0:
            raise FusionbenchError("days must be positive")
        zero_start = np.zeros(4)
        reserve_atoms = float(kg_to_atoms(self.reserve_inventory))
        lam = self.decay_constant
        grid = np.linspace(0.0, days, 4001)
        matrix, source = self._system()
        dt = days * SECONDS_PER_DAY / (grid.size - 1)
        augmented = np.zeros((5, 5))
        augmented[:4, :4] = matrix * dt
        augmented[:4, 4] = source * dt
        flow = as_float64(scipy.linalg.expm(augmented))
        step, offset = flow[:4, :4], flow[:4, 4]
        state = zero_start.copy()
        storage = np.empty(grid.size)
        storage[0] = 0.0
        for i in range(1, grid.size):
            state = step @ state + offset
            storage[i] = state[3]

        def requirement(t_days: float, storage_atoms: float) -> float:
            return (reserve_atoms - storage_atoms) * math.exp(lam * t_days * SECONDS_PER_DAY)

        needs = np.array(
            [requirement(t, s) for t, s in zip(grid.tolist(), storage.tolist(), strict=True)]
        )
        peak = int(np.argmax(needs))
        low = grid[max(0, peak - 1)]
        high = grid[min(grid.size - 1, peak + 1)]

        def negated(t_days: float) -> float:
            storage_atoms = float(self._state_at(t_days * SECONDS_PER_DAY, zero_start)[3])
            return -requirement(t_days, storage_atoms)

        refined = scipy.optimize.minimize_scalar(negated, bounds=(low, high), method="bounded")
        best_atoms = max(float(np.max(needs)), -float(refined.fun))
        return max(0.0, best_atoms * KG_PER_ATOM)

    @property
    def provenance(self) -> Provenance:
        """Reproducibility record for this cycle configuration."""
        return build_provenance(
            models=[MODEL_ID, MODERN_MODEL_ID, HALF_LIFE_MODEL_ID],
            inputs=self.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe parameter record."""
        return {
            "burn_rate": self.burn_rate,
            "tbr": self.tbr,
            "fractional_burnup": self.fractional_burnup,
            "startup_inventory": self.startup_inventory,
            "blanket_residence_days": self.blanket_residence_days,
            "exhaust_residence_days": self.exhaust_residence_days,
            "processing_residence_days": self.processing_residence_days,
            "extraction_efficiency": self.extraction_efficiency,
            "processing_loss": self.processing_loss,
            "reserve_inventory": self.reserve_inventory,
            "decay_constant": self.decay_constant,
        }
