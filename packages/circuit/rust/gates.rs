use crate::gate::{ParametricQuantumGate, QuantumGate};
use crate::parameter::Parameter;
use crate::{GenericGateProperty, MaybeUnbound, QuriPartsGate};
use pyo3::prelude::*;

#[pyfunction(
    name = "X",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn x<'py>(target_index: usize) -> QuantumGate {
    QuantumGate(QuriPartsGate::X(target_index))
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
    m.add_wrapped(wrap_pyfunction!(x))?;
    m.add_wrapped(wrap_pyfunction!(rx))?;
    m.add_wrapped(wrap_pyfunction!(parametric_rx))?;
    Ok(m)
}

impl QuriPartsGate<MaybeUnbound> {
    pub fn instantiate<'py>(self) -> Result<QuantumGate, (ParametricQuantumGate, Py<Parameter>)> {
        match self {
            Self::X(q) => Ok(x(q)),
            Self::RX(q, p) => match p {
                MaybeUnbound::Bound(p) => Ok(rx(q, p)),
                MaybeUnbound::Unbound(pid) => Err((parametric_rx(q), pid)),
            },
            Self::Other(o) => Ok(QuantumGate(QuriPartsGate::Other(o))),
        }
    }
}
