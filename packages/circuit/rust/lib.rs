use num_complex::Complex64;
use pyo3::prelude::*;

mod circuit;
mod circuit_parametric;
mod gate;
mod gates;
mod parameter;

#[pymodule]
fn quri_parts_circuit_rs(py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_submodule(&gate::py_module(py)?)?;
    m.add_submodule(&gates::py_module(py)?)?;
    m.add_submodule(&circuit::py_module(py)?)?;
    m.add_submodule(&circuit_parametric::py_module(py)?)?;
    m.add_submodule(&parameter::py_module(py)?)?;
    Ok(())
}

#[derive(Clone, Debug)]
pub enum MaybeUnbound {
    Bound(f64),
    Unbound(Py<parameter::Parameter>),
}

impl core::cmp::PartialEq for MaybeUnbound {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MaybeUnbound::Bound(lhs), MaybeUnbound::Bound(rhs)) => lhs == rhs,
            (MaybeUnbound::Unbound(lhs), MaybeUnbound::Unbound(rhs)) => {
                lhs.as_ptr() as usize == rhs.as_ptr() as usize
            }
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum QuriPartsGate<P> {
    X(usize),
    RX(usize, P),
    Other(Box<GenericGateProperty>),
}

impl<P: Clone> QuriPartsGate<P> {
    pub fn get_qubits(&self) -> Vec<usize> {
        match self {
            Self::X(q) => vec![*q],
            Self::RX(q, _) => vec![*q],
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

    pub fn get_params(&self) -> Vec<P> {
        match self {
            Self::RX(_, p) => vec![p.clone()],
            _ => vec![],
        }
    }

    pub fn map_param<Q>(self, mut f: impl FnMut(P) -> Q) -> QuriPartsGate<Q> {
        match self {
            QuriPartsGate::X(q) => QuriPartsGate::X(q),
            QuriPartsGate::RX(q, p) => QuriPartsGate::RX(q, f(p)),
            QuriPartsGate::Other(o) => QuriPartsGate::Other(o),
        }
    }
}

impl QuriPartsGate<f64> {
    pub fn from_property(prop: GenericGateProperty) -> Option<Self> {
        match prop.name.as_str() {
            "X" if prop.target_indices.len() == 1
                && prop.control_indices.is_empty()
                && prop.classical_indices.is_empty()
                && prop.params.is_empty()
                && prop.pauli_ids.is_empty()
                && prop.unitary_matrix.is_none() =>
            {
                Some(Self::X(prop.target_indices[0]))
            }
            "RX" if prop.target_indices.len() == 1
                && prop.control_indices.is_empty()
                && prop.classical_indices.is_empty()
                && prop.params.len() == 1
                && prop.pauli_ids.is_empty()
                && prop.unitary_matrix.is_none() =>
            {
                Some(Self::RX(prop.target_indices[0], prop.params[0]))
            }
            "ParametricRX" => None,
            _ => Some(Self::Other(Box::new(prop))),
        }
    }

    pub fn into_property(self) -> GenericGateProperty {
        match self {
            Self::X(q) => GenericGateProperty {
                name: "X".to_owned(),
                target_indices: vec![q],
                control_indices: vec![],
                classical_indices: vec![],
                params: vec![],
                pauli_ids: vec![],
                unitary_matrix: None,
            },
            Self::RX(q, p) => GenericGateProperty {
                name: "RX".to_owned(),
                target_indices: vec![q],
                control_indices: vec![],
                classical_indices: vec![],
                params: vec![p],
                pauli_ids: vec![],
                unitary_matrix: None,
            },
            Self::Other(o) => *o,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenericGateProperty {
    pub name: String,
    pub target_indices: Vec<usize>,
    pub control_indices: Vec<usize>,
    pub classical_indices: Vec<usize>,
    pub params: Vec<f64>,
    pub pauli_ids: Vec<u8>,
    pub unitary_matrix: Option<Vec<Vec<Complex64>>>,
}
