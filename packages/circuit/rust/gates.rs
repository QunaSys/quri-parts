use crate::gate::{OtherData, QuantumGate};
use pyo3::prelude::*;

#[derive(Clone, Debug, PartialEq)]
#[pyclass(
    frozen,
    eq,
    extends=QuantumGate,
    module = "quri_parts.circuit.rust.gates"
)]
struct XGate();

#[pymethods]
impl XGate {
    #[pyo3(name = "__hash__")]
    fn py_hash(slf: PyRef<'_, Self>) -> u64 {
        let sup: &QuantumGate = slf.as_ref();
        sup.py_hash()
    }
}

#[pyfunction(
    name = "X",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
fn py_x<'py>(py: Python<'py>, target_index: usize) -> PyResult<Py<XGate>> {
    Ok(Py::new(
        py,
        (XGate(), QuantumGate(QuriPartsGates::X(target_index))),
    )?)
}

#[pyclass(
    frozen,
    get_all,
    eq,
    extends=QuantumGate,
    module = "quri_parts.circuit.rust.gates"
)]
#[derive(PartialEq, Debug)]
struct OtherGate();

#[pymethods]
impl OtherGate {
    #[pyo3(name = "__hash__")]
    fn py_hash(slf: PyRef<'_, Self>) -> u64 {
        let sup: &QuantumGate = slf.as_ref();
        sup.py_hash()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum QuriPartsGates {
    X(usize),
    Other(Box<OtherData>),
}

impl QuriPartsGates {
    pub fn get_qubits(&self) -> Vec<usize> {
        match self {
            Self::X(q) => vec![*q],
            Self::Other(o) => {
                let mut ret = o.control_indices.clone();
                ret.extend(o.target_indices.clone());
                ret
            }
        }
    }

    pub fn get_cbits(&self) -> Vec<usize> {
        vec![]
    }

    pub fn from_other(other: OtherData) -> Self {
        match other.name.as_str() {
            "X" if other.target_indices.len() == 1
                && other.control_indices.is_empty()
                && other.classical_indices.is_empty()
                && other.params.is_empty()
                && other.pauli_ids.is_empty()
                && other.unitary_matrix.is_none() =>
            {
                Self::X(other.target_indices[0])
            }
            _ => Self::Other(Box::new(other)),
        }
    }

    pub fn into_other(self) -> OtherData {
        match self {
            Self::X(q) => OtherData {
                name: "X".to_owned(),
                target_indices: vec![q],
                control_indices: vec![],
                classical_indices: vec![],
                params: vec![],
                pauli_ids: vec![],
                unitary_matrix: None,
            },
            Self::Other(o) => *o,
        }
    }

    pub fn instanciate<'py>(self, py: Python<'py>) -> PyResult<Py<QuantumGate>> {
        match &self {
            Self::X(_) => Ok(Py::new(py, (XGate(), QuantumGate(self)))?
                .into_any()
                .downcast_bound::<QuantumGate>(py)
                .unwrap()
                .clone()
                .unbind()),
            Self::Other(_) => Ok(Py::new(py, (OtherGate(), QuantumGate(self)))?
                .into_any()
                .downcast_bound::<QuantumGate>(py)
                .unwrap()
                .clone()
                .unbind()),
        }
    }
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "gates")?;
    m.add_wrapped(wrap_pyfunction!(py_x))?;
    m.add_class::<OtherGate>()?;
    m.add_class::<XGate>()?;
    Ok(m)
}
