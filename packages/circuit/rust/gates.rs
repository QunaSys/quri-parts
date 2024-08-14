use crate::gate::{ParametricQuantumGate, QuantumGate};
use crate::parameter::Parameter;
use crate::{GenericGateProperty, MaybeUnbound, QuriPartsGate};
use pyo3::prelude::*;

#[pyfunction(
    name = "Identity",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn identity(target_index: usize) -> QuantumGate {
    QuantumGate(QuriPartsGate::Identity(target_index))
}

#[pyfunction(
    name = "X",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn x(target_index: usize) -> QuantumGate {
    QuantumGate(QuriPartsGate::X(target_index))
}

#[pyfunction(
    name = "Y",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn y(target_index: usize) -> QuantumGate {
    QuantumGate(QuriPartsGate::Y(target_index))
}

#[pyfunction(
    name = "Z",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn z(target_index: usize) -> QuantumGate {
    QuantumGate(QuriPartsGate::Z(target_index))
}

#[pyfunction(
    name = "H",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn h(target_index: usize) -> QuantumGate {
    QuantumGate(QuriPartsGate::H(target_index))
}

#[pyfunction(
    name = "S",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn s(target_index: usize) -> QuantumGate {
    QuantumGate(QuriPartsGate::S(target_index))
}

#[pyfunction(
    name = "Sdag",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn sdag(target_index: usize) -> QuantumGate {
    QuantumGate(QuriPartsGate::Sdag(target_index))
}

#[pyfunction(
    name = "SqrtX",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn sqrtx(target_index: usize) -> QuantumGate {
    QuantumGate(QuriPartsGate::SqrtX(target_index))
}

#[pyfunction(
    name = "SqrtXdag",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn sqrtxdag(target_index: usize) -> QuantumGate {
    QuantumGate(QuriPartsGate::SqrtXdag(target_index))
}

#[pyfunction(
    name = "SqrtY",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn sqrty(target_index: usize) -> QuantumGate {
    QuantumGate(QuriPartsGate::SqrtY(target_index))
}

#[pyfunction(
    name = "SqrtYdag",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn sqrtydag(target_index: usize) -> QuantumGate {
    QuantumGate(QuriPartsGate::SqrtYdag(target_index))
}

#[pyfunction(
    name = "T",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn t(target_index: usize) -> QuantumGate {
    QuantumGate(QuriPartsGate::T(target_index))
}

#[pyfunction(
    name = "Tdag",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn tdag(target_index: usize) -> QuantumGate {
    QuantumGate(QuriPartsGate::Tdag(target_index))
}

#[pyfunction(
    name = "RX",
    signature = (target_index, angle),
    text_signature = "(target_index: int, angle: float)",
)]
fn rx<'py>(target_index: usize, angle: f64) -> QuantumGate {
    QuantumGate(QuriPartsGate::RX(target_index, angle))
}

#[pyfunction(
    name = "ParametricRX",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn parametric_rx(target_index: usize) -> ParametricQuantumGate {
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
    Ok(m)
}

impl QuriPartsGate<MaybeUnbound> {
    pub fn instantiate<'py>(self) -> Result<QuantumGate, (ParametricQuantumGate, Py<Parameter>)> {
        match self {
            Self::Identity(q) => Ok(identity(q)),
            Self::X(q) => Ok(x(q)),
            Self::Y(q) => Ok(y(q)),
            Self::Z(q) => Ok(z(q)),
            Self::H(q) => Ok(h(q)),
            Self::S(q) => Ok(s(q)),
            Self::Sdag(q) => Ok(sdag(q)),
            Self::SqrtX(q) => Ok(sqrtx(q)),
            Self::SqrtXdag(q) => Ok(sqrtxdag(q)),
            Self::SqrtY(q) => Ok(sqrty(q)),
            Self::SqrtYdag(q) => Ok(sqrtydag(q)),
            Self::T(q) => Ok(t(q)),
            Self::Tdag(q) => Ok(tdag(q)),
            Self::RX(q, p) => match p {
                MaybeUnbound::Bound(p) => Ok(rx(q, p)),
                MaybeUnbound::Unbound(pid) => Err((parametric_rx(q), pid)),
            },
            Self::Other(o) => Ok(QuantumGate(QuriPartsGate::Other(o))),
        }
    }
}
