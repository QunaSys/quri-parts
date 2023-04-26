from pytket import Circuit  # type: ignore
from pytket.backends.backendresult import BackendResult
from pytket.extensions.qiskit import AerBackend  # type: ignore


def get_aer_sampler_result(circuit: Circuit, n_shots: int) -> BackendResult:
    backend = AerBackend()
    compiled_circ = backend.get_compiled_circuit(circuit)

    handle = backend.process_circuit(compiled_circ, n_shots=n_shots)
    result = backend.get_result(handle)

    return result
