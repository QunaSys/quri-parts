from abc import ABC, abstractmethod
from typing import Any

from quri_algo.problem.operators.hamiltonian import (
    FermionicHamiltonianInput,
    QubitHamiltonianInput,
)


class HamiltonianMixin(ABC):
    @abstractmethod
    def get_qubit_hamiltonian(self, *args: Any) -> QubitHamiltonianInput:
        ...

    @abstractmethod
    def get_fermionic_hamiltonian(self, *args: Any) -> FermionicHamiltonianInput:
        ...
