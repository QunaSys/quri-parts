from dataclasses import dataclass

from quri_algo.problem.operators.hamiltonian import (
    FermionicHamiltonianInput,
    QubitHamiltonianInput,
)

from .interface import HamiltonianMixin


@dataclass
class FermiHubbardSystem(HamiltonianMixin):
    t: float
    U: float
    mu: float
    periodic: bool

    def get_qubit_hamiltonian(self) -> QubitHamiltonianInput:
        return super().get_qubit_hamiltonian()

    def get_fermionic_hamiltonian(self) -> FermionicHamiltonianInput:
        return super().get_fermionic_hamiltonian()
