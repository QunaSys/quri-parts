# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod, abstractproperty
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple, Optional, Protocol, Sequence

import numpy as np

from quri_parts.chem.mol import get_core_and_active_orbital_indices

if TYPE_CHECKING:
    import numpy.typing as npt  # noqa: F401


class OrbitalType(Enum):
    """Type of orbital."""

    #: core orbital.
    CORE = auto()
    #: active orbital.
    ACTIVE = auto()
    #: virtual orbital.
    VIRTUAL = auto()


@dataclass(frozen=True)
class ActiveSpace:
    """An immutable (frozen) dataclass representing a active space."""

    #: number of active electrons.
    n_active_ele: int
    #: number of active orbitals.
    n_active_orb: int
    #: sequence of spatial orbital indices of the active orbitals.
    active_orbs_indices: Optional[Sequence[int]] = None


class MolecularOrbitals(Protocol):
    """Interface protocol for a data of the molecule."""

    @abstractproperty
    def n_electron(self) -> int:
        """Returns a number of electrons."""
        ...

    @abstractproperty
    def spin(self) -> int:
        """Returns the spin of the electron.

        Note:
            We follow the quantum chemistry convention where:

            .. math::
                N_{\\alpha} - N_{\\beta} = 2S

            and spin = 2S.
        """
        ...

    @abstractproperty
    def n_orb(self) -> int:
        """Returns the number of orbitals."""
        ...

    @abstractproperty
    def mo_coeff(self) -> "npt.NDArray[np.complex128]":
        """Returns molecular orbital coefficients."""
        ...


@dataclass(frozen=True)
class ActiveSpaceInfo:
    """Active space information containing.

    - Information related to the number of electrons:
        - n_electrons: Number of electrons
        - n_active_ele: Number of active electrons
        - n_core_ele: Number of core electrons
        - n_ele_alpha: Number of spin up electrons
        - n_ele_beta: Number of spin down electrons

    - Information related to the number of orbitals:
        - n_orb: Number of orbitals
        - n_core_orb: Number of core orbitals
        - n_vir_orb: Number of virtual orbitals
    """

    n_electron: int
    n_active_ele: int
    n_core_ele: int
    n_ele_alpha: int
    n_ele_beta: int
    n_orb: int
    n_active_orb: int
    n_core_orb: int
    n_vir_orb: int


class ActiveSpaceMolecularOrbitals(MolecularOrbitals):
    """Represents a data of the active space for the molecule.

    Args:
        mo: :class:`MolecularOrbitals` for the molecule.
        active_space: :class:`ActiveSpace` for the molecule.
    """

    def __init__(
        self,
        mo: MolecularOrbitals,
        active_space: ActiveSpace,
    ):
        self._mo = mo
        self._active_space = active_space

    @property
    def n_electron(self) -> int:
        """Returns a number of electrons."""
        return self._mo.n_electron

    @property
    def spin(self) -> int:
        """Returns the spin of the molecule."""
        return self._mo.spin

    @property
    def n_active_ele(self) -> int:
        """Returns the number of active electrons."""
        return self._active_space.n_active_ele

    @property
    def n_core_ele(self) -> int:
        """Returns the number of core electrons."""
        n_core_electron = self.n_electron - self.n_active_ele

        if n_core_electron % 2 == 1:
            raise ValueError("The number of electrons in core must be even.")

        return n_core_electron

    @property
    def n_ele_alpha(self) -> int:
        """Return the number of spin up electrons."""
        return self.n_active_ele - self.n_ele_beta

    @property
    def n_ele_beta(self) -> int:
        """Return the number of spin down electrons."""
        return (self.n_active_ele - self.spin) // 2

    @property
    def n_orb(self) -> int:
        """Returns the number of spatial orbitals."""
        return self._mo.n_orb

    @property
    def n_active_orb(self) -> int:
        """Returns the number of active orbitals."""
        return self._active_space.n_active_orb

    @property
    def n_core_orb(self) -> int:
        """Returns the number of core orbitals."""
        return self.n_core_ele // 2

    @property
    def n_vir_orb(self) -> int:
        """Returns the number of virtual orbitals."""
        return self._mo.n_orb - self.n_active_orb - self.n_core_orb

    @property
    def info(self) -> ActiveSpaceInfo:
        """Returns an instance of the `ActiveSpaceInfo` class."""
        active_space_info = ActiveSpaceInfo(
            n_electron=self.n_electron,
            n_active_ele=self.n_active_ele,
            n_core_ele=self.n_core_ele,
            n_ele_alpha=self.n_ele_alpha,
            n_ele_beta=self.n_ele_beta,
            n_orb=self.n_orb,
            n_active_orb=self.n_active_orb,
            n_core_orb=self.n_core_orb,
            n_vir_orb=self.n_vir_orb,
        )
        return active_space_info

    @property
    def mo_coeff(self) -> "npt.NDArray[np.complex128]":
        """Returns molecular orbital coefficients."""
        return self._mo.mo_coeff

    def get_core_and_active_orb(self) -> tuple[Sequence[int], Sequence[int]]:
        """Returns a set of core and active orbitals.

        The output orbitals are represented as the spatial orbital.
        """
        n_active_ele = self._active_space.n_active_ele
        n_active_orb = self._active_space.n_active_orb
        active_orbs_indices = self._active_space.active_orbs_indices
        core_orb_list, active_orb_list = get_core_and_active_orbital_indices(
            n_active_ele,
            n_active_orb,
            self._mo.n_electron,
            active_orbs_indices,
        )
        return core_orb_list, active_orb_list

    def orb_type(self, mo_index: int) -> OrbitalType:
        """Returns a type of the given orbital index.

        Args:
            mo_index: Orbital index.
        """
        core_orb_list, active_orb_list = self.get_core_and_active_orb()
        if mo_index in core_orb_list:
            return OrbitalType.CORE
        elif mo_index in active_orb_list:
            return OrbitalType.ACTIVE
        else:
            return OrbitalType.VIRTUAL


class AO1eIntBase(Protocol):
    """Interface protocol for an atomic orbital one-electron integral."""

    @abstractproperty
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns integral as numpy ndarray."""
        ...

    @abstractmethod
    def to_mo1int(self, mo_coeff: "npt.NDArray[np.complex128]") -> "MO1eInt":
        """This method converts an atomic orbital one-electron integral into
        the molecular orbital one-electron integral.

        Args:
            mo_coeff: molecular orbital coefficients.
        """
        ...


class AO2eIntBase(Protocol):
    """Interface protocol for an atomic orbital two-electron integral."""

    @abstractproperty
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns integral as numpy ndarray."""
        ...

    @abstractmethod
    def to_mo2int(self, mo_coeff: "npt.NDArray[np.complex128]") -> "MO2eInt":
        """This method converts an atomic orbital two-electron integral into
        the molecular orbital two-electron integral.

        Args:
            mo_coeff: molecular orbital coefficients.
        """
        ...


class AOeIntSetBase(NamedTuple):
    """NamedTuple holding a constant and instance of the ao electron
    integrals."""

    #: constant.
    constant: float
    #: non-relativistic atomic  orbital one-electron integral :class:`NRAO1eInt`.
    ao_1e_int: AO1eIntBase
    #: non-relativistic atomic  orbital two-electron integral :class:`NRAO2eInt`.
    ao_2e_int: AO2eIntBase


class MO1eInt(Protocol):
    """Interface protocol for a molecular orbital one-electron integral."""

    @abstractproperty
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns integral as numpy ndarray."""
        ...


class MO2eInt(Protocol):
    """Interface protocol for a molecular orbital two-electron integral."""

    @abstractproperty
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns integral as numpy ndarray."""
        ...


class MO1eIntArray(MO1eInt):
    """A class of a molecular orbital one-electron integral.

    This interface has an integral as numpy ndarray.
    """

    def __init__(self, array: "npt.NDArray[np.complex128]"):
        self._array = array

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns integral as numpy ndarray."""
        return self._array


class MO2eIntArray(MO2eInt):
    """A class of a molecular orbital two-electron integral.

    This interface has an integral as numpy ndarray.
    """

    def __init__(self, array: "npt.NDArray[np.complex128]"):
        self._array = array

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns integral as numpy ndarray."""
        return self._array


class MOeIntSet(NamedTuple):
    """MOeIntSet holds a constant and a set of molecular orbital electron
    integral."""

    #: constant.
    const: float
    #: molecular orbital one-electron integral :class:`MO1eInt`.
    mo_1e_int: MO1eInt
    #: molecular orbital two-electron integral :class:`MO2eInt`.
    mo_2e_int: MO2eInt
