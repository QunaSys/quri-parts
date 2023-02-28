# type: ignore
import numpy as np
from quri_parts.algo.ansatz import HardwareEfficient
from quri_parts.algo.optimizer import Adam, OptimizerStatus
from quri_parts.circuit import CONST, LinearMappedUnboundParametricQuantumCircuit
from quri_parts.core.estimator.gradient import create_numerical_gradient_estimator
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label
from quri_parts.core.state import ComputationalBasisState, ParametricCircuitQuantumState

from quri_parts.itensor.estimator import (
    create_itensor_mps_concurrent_parametric_estimator,
    create_itensor_mps_parametric_estimator,
)

hw_ansatz = HardwareEfficient(qubit_count=4, reps=3)

# Preparation of the parametric state is a bit complicated for now
def prepare_parametric_state(initial_state, ansatz):
    circuit = LinearMappedUnboundParametricQuantumCircuit(initial_state.qubit_count)
    circuit += initial_state.circuit
    circuit += ansatz
    return ParametricCircuitQuantumState(initial_state.qubit_count, circuit)


cb_state = ComputationalBasisState(4, bits=0b0011)
parametric_state = prepare_parametric_state(cb_state, hw_ansatz)

# You can pass optional parameters. See the reference for details
adam_optimizer = Adam()

# This is Jordan-Wigner transformed Hamiltonian of a hydrogen molecule
hamiltonian = Operator(
    {
        PAULI_IDENTITY: 0.03775110394645542,
        pauli_label("Z0"): 0.18601648886230593,
        pauli_label("Z1"): 0.18601648886230593,
        pauli_label("Z2"): -0.2694169314163197,
        pauli_label("Z3"): -0.2694169314163197,
        pauli_label("Z0 Z1"): 0.172976101307451,
        pauli_label("Z0 Z2"): 0.12584136558006326,
        pauli_label("Z0 Z3"): 0.16992097848261506,
        pauli_label("Z1 Z2"): 0.16992097848261506,
        pauli_label("Z1 Z3"): 0.12584136558006326,
        pauli_label("Z2 Z3"): 0.17866777775953396,
        pauli_label("X0 X1 Y2 Y3"): -0.044079612902551774,
        pauli_label("X0 Y1 Y2 X3"): 0.044079612902551774,
        pauli_label("Y0 X1 X2 Y3"): 0.044079612902551774,
        pauli_label("Y0 Y1 X2 X3"): -0.044079612902551774,
    }
)


estimator = create_itensor_mps_parametric_estimator()


def cost_fn(param_values):
    estimate = estimator(hamiltonian, parametric_state, param_values)
    return estimate.value.real


itensor_concurrent_parametric_estimator = (
    create_itensor_mps_concurrent_parametric_estimator(concurrency=4)
)
gradient_estimator = create_numerical_gradient_estimator(
    itensor_concurrent_parametric_estimator,
    delta=1e-4,
)


def grad_fn(param_values):
    estimate = gradient_estimator(hamiltonian, parametric_state, param_values)
    return np.asarray([g.real for g in estimate.values])


def vqe(operator, init_params, cost_fn, grad_fn, optimizer):
    opt_state = optimizer.get_init_state(init_params)
    while True:
        opt_state = optimizer.step(opt_state, cost_fn, grad_fn)
        if opt_state.status == OptimizerStatus.FAILED:
            print("Optimizer failed")
            break
        if opt_state.status == OptimizerStatus.CONVERGED:
            print("Optimizer converged")
            break
    return opt_state


init_params = [0.1] * hw_ansatz.parameter_count
result = vqe(hamiltonian, init_params, cost_fn, grad_fn, adam_optimizer)
print("Optimized value:", result.cost)
print("Optimized parameter:", result.params)
print("Iterations:", result.niter)
print("Cost function calls:", result.funcalls)
print("Gradient function calls:", result.gradcalls)
