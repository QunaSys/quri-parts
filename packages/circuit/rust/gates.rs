use crate::gate::{GenericGateProperty, ParametricQuantumGate, QuantumGate};
use crate::parameter::Wrapper;
use crate::MaybeUnbound;
use num_complex::{Complex64, ComplexFloat};
use pyo3::prelude::*;

#[pyfunction(
    name = "Identity",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn identity(target_index: usize) -> QuantumGate {
    QuantumGate::Identity(target_index)
}

#[pyfunction(
    name = "X",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn x(target_index: usize) -> QuantumGate {
    QuantumGate::X(target_index)
}

#[pyfunction(
    name = "Y",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn y(target_index: usize) -> QuantumGate {
    QuantumGate::Y(target_index)
}

#[pyfunction(
    name = "Z",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn z(target_index: usize) -> QuantumGate {
    QuantumGate::Z(target_index)
}

#[pyfunction(
    name = "H",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn h(target_index: usize) -> QuantumGate {
    QuantumGate::H(target_index)
}

#[pyfunction(
    name = "S",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn s(target_index: usize) -> QuantumGate {
    QuantumGate::S(target_index)
}

#[pyfunction(
    name = "Sdag",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn sdag(target_index: usize) -> QuantumGate {
    QuantumGate::Sdag(target_index)
}

#[pyfunction(
    name = "SqrtX",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn sqrtx(target_index: usize) -> QuantumGate {
    QuantumGate::SqrtX(target_index)
}

#[pyfunction(
    name = "SqrtXdag",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn sqrtxdag(target_index: usize) -> QuantumGate {
    QuantumGate::SqrtXdag(target_index)
}

#[pyfunction(
    name = "SqrtY",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn sqrty(target_index: usize) -> QuantumGate {
    QuantumGate::SqrtY(target_index)
}

#[pyfunction(
    name = "SqrtYdag",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn sqrtydag(target_index: usize) -> QuantumGate {
    QuantumGate::SqrtYdag(target_index)
}

#[pyfunction(
    name = "T",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn t(target_index: usize) -> QuantumGate {
    QuantumGate::T(target_index)
}

#[pyfunction(
    name = "Tdag",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn tdag(target_index: usize) -> QuantumGate {
    QuantumGate::Tdag(target_index)
}

#[pyfunction(
    name = "RX",
    signature = (target_index, angle),
    text_signature = "(target_index: int, angle: float)",
)]
pub fn rx<'py>(target_index: usize, angle: f64) -> QuantumGate {
    QuantumGate::RX(target_index, angle)
}

#[pyfunction(
    name = "ParametricRX",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn parametric_rx(target_index: usize) -> ParametricQuantumGate {
    ParametricQuantumGate(GenericGateProperty {
        name: "ParametricRX".to_owned(),
        target_indices: vec![target_index],
        control_indices: vec![],
        classical_indices: vec![],
        params: vec![],
        pauli_ids: vec![],
        unitary_matrix: None,
    })
}

#[pyfunction(
    name = "UnitaryMatrix",
    signature = (target_indices, unitary_matrix),
    text_signature = "(target_indices: Sequence[int], unitary_matrix: Sequence[Sequence[complex]])",
)]
pub fn unitary_matrix(
    target_indices: Vec<usize>,
    unitary_matrix: Vec<Vec<Complex64>>,
) -> PyResult<QuantumGate> {
    let dim = 2usize.pow(target_indices.len() as u32);
    if unitary_matrix.len() == dim && unitary_matrix.iter().all(|v| v.len() == dim) {
        if (0..dim).all(|i| {
            (0..dim).all(|j| {
                ((0..dim)
                    .map(|k| unitary_matrix[i][k] * unitary_matrix[j][k].conj())
                    .sum::<Complex64>()
                    - Complex64::new(Into::<u8>::into(i == j) as f64, 0f64))
                .abs()
                    < 1e-8
            })
        }) {
            Ok(QuantumGate::UnitaryMatrix(target_indices, unitary_matrix))
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "The given matrix is not unitary.",
            ))
        }
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            "The number of qubits does not match the size of the unitary matrix.",
        ))
    }
}

#[pyfunction(
    name = "SingleQubitUnitaryMatrix",
    signature = (target_index, mat),
    text_signature = "(target_index: int, unitary_matrix: Sequence[Sequence[complex]])",
)]
fn single_qubit_unitary_matrix(
    target_index: usize,
    mat: Vec<Vec<Complex64>>,
) -> PyResult<QuantumGate> {
    unitary_matrix(vec![target_index], mat)
}

#[pyfunction(
    name = "TwoQubitUnitaryMatrix",
    signature = (target_index1, target_index2, mat),
    text_signature = "(target_index1: int, target_index2: int, unitary_matrix: Sequence[Sequence[complex]])",
)]
fn two_qubit_unitary_matrix(
    target_index1: usize,
    target_index2: usize,
    mat: Vec<Vec<Complex64>>,
) -> PyResult<QuantumGate> {
    unitary_matrix(vec![target_index1, target_index2], mat)
}

#[pyfunction(
    name = "Pauli",
    signature = (target_indices, pauli_ids),
    text_signature = "(target_indices: Sequence[int], pauli_ids: Sequence[int])",
)]
pub fn pauli(target_indices: Vec<usize>, pauli_ids: Vec<u8>) -> QuantumGate {
    QuantumGate::Pauli(target_indices, pauli_ids)
}

#[pyfunction(
    name = "PauliRotation",
    signature = (target_indices, pauli_ids, angle),
    text_signature = "(target_indices: Sequence[int], pauli_ids: Sequence[int], angle: float)",
)]
pub fn pauli_rotation(target_indices: Vec<usize>, pauli_ids: Vec<u8>, angle: f64) -> QuantumGate {
    QuantumGate::PauliRotation(target_indices, pauli_ids, angle)
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "gates")?;
    m.add_wrapped(wrap_pyfunction!(identity))?;
    m.add_wrapped(wrap_pyfunction!(x))?;
    m.add_wrapped(wrap_pyfunction!(y))?;
    m.add_wrapped(wrap_pyfunction!(z))?;
    m.add_wrapped(wrap_pyfunction!(h))?;
    m.add_wrapped(wrap_pyfunction!(s))?;
    m.add_wrapped(wrap_pyfunction!(sdag))?;
    m.add_wrapped(wrap_pyfunction!(sqrtx))?;
    m.add_wrapped(wrap_pyfunction!(sqrtxdag))?;
    m.add_wrapped(wrap_pyfunction!(sqrty))?;
    m.add_wrapped(wrap_pyfunction!(sqrtydag))?;
    m.add_wrapped(wrap_pyfunction!(t))?;
    m.add_wrapped(wrap_pyfunction!(tdag))?;
    m.add_wrapped(wrap_pyfunction!(rx))?;
    m.add_wrapped(wrap_pyfunction!(parametric_rx))?;
    m.add_wrapped(wrap_pyfunction!(unitary_matrix))?;
    m.add_wrapped(wrap_pyfunction!(single_qubit_unitary_matrix))?;
    m.add_wrapped(wrap_pyfunction!(two_qubit_unitary_matrix))?;
    m.add_wrapped(wrap_pyfunction!(pauli))?;
    m.add_wrapped(wrap_pyfunction!(pauli_rotation))?;
    Ok(m)
}

impl QuantumGate<MaybeUnbound> {
    pub fn instantiate<'py>(
        self,
    ) -> PyResult<Result<QuantumGate, (ParametricQuantumGate, Wrapper)>> {
        match self {
            Self::Identity(q) => Ok(Ok(identity(q))),
            Self::X(q) => Ok(Ok(x(q))),
            Self::Y(q) => Ok(Ok(y(q))),
            Self::Z(q) => Ok(Ok(z(q))),
            Self::H(q) => Ok(Ok(h(q))),
            Self::S(q) => Ok(Ok(s(q))),
            Self::Sdag(q) => Ok(Ok(sdag(q))),
            Self::SqrtX(q) => Ok(Ok(sqrtx(q))),
            Self::SqrtXdag(q) => Ok(Ok(sqrtxdag(q))),
            Self::SqrtY(q) => Ok(Ok(sqrty(q))),
            Self::SqrtYdag(q) => Ok(Ok(sqrtydag(q))),
            Self::T(q) => Ok(Ok(t(q))),
            Self::Tdag(q) => Ok(Ok(tdag(q))),
            Self::UnitaryMatrix(qs, mat) => Ok(Ok(unitary_matrix(qs, mat)?)),
            Self::RX(q, p) => match p {
                MaybeUnbound::Bound(p) => Ok(Ok(rx(q, p))),
                MaybeUnbound::Unbound(pid) => Ok(Err((parametric_rx(q), pid))),
            },
            Self::Pauli(qs, ps) => Ok(Ok(pauli(qs, ps))),
            Self::PauliRotation(qs, ps, a) => match a {
                MaybeUnbound::Bound(a) => Ok(Ok(pauli_rotation(qs, ps, a))),
                MaybeUnbound::Unbound(_pid) => todo!("ParametricPauliRotation not implemented"),
            },
            Self::Other(o) => Ok(Ok(QuantumGate::Other(o))),
        }
    }
}
