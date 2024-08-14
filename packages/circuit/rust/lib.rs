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
    Identity(usize),
    X(usize),
    Y(usize),
    Z(usize),
    H(usize),
    S(usize),
    Sdag(usize),
    SqrtX(usize),
    SqrtXdag(usize),
    SqrtY(usize),
    SqrtYdag(usize),
    T(usize),
    Tdag(usize),
    RX(usize, P),
    Other(Box<GenericGateProperty>),
}

impl<P: Clone> QuriPartsGate<P> {
    pub fn get_qubits(&self) -> Vec<usize> {
        match self {
            Self::Identity(q)
            | Self::X(q)
            | Self::Y(q)
            | Self::Z(q)
            | Self::H(q)
            | Self::S(q)
            | Self::Sdag(q)
            | Self::SqrtX(q)
            | Self::SqrtXdag(q)
            | Self::SqrtY(q)
            | Self::SqrtYdag(q)
            | Self::T(q)
            | Self::Tdag(q)
            | Self::RX(q, _) => vec![*q],
            Self::Other(o) => {
                let mut ret = o.control_indices.clone();
                ret.extend(o.target_indices.clone());
                ret
            }
        }
    }

    pub fn get_cbits(&self) -> Vec<usize> {
        match self {
            Self::Other(o) => o.classical_indices.clone(),
            _ => vec![],
        }
    }

    pub fn get_params(&self) -> Vec<P> {
        match self {
            Self::RX(_, p) => vec![p.clone()],
            _ => vec![],
        }
    }

    pub fn map_param<Q>(self, mut f: impl FnMut(P) -> Q) -> QuriPartsGate<Q> {
        match self {
            QuriPartsGate::Identity(q) => QuriPartsGate::Identity(q),
            QuriPartsGate::X(q) => QuriPartsGate::X(q),
            QuriPartsGate::Y(q) => QuriPartsGate::Y(q),
            QuriPartsGate::Z(q) => QuriPartsGate::Z(q),
            QuriPartsGate::H(q) => QuriPartsGate::H(q),
            QuriPartsGate::S(q) => QuriPartsGate::S(q),
            QuriPartsGate::Sdag(q) => QuriPartsGate::Sdag(q),
            QuriPartsGate::SqrtX(q) => QuriPartsGate::SqrtX(q),
            QuriPartsGate::SqrtXdag(q) => QuriPartsGate::SqrtXdag(q),
            QuriPartsGate::SqrtY(q) => QuriPartsGate::SqrtY(q),
            QuriPartsGate::SqrtYdag(q) => QuriPartsGate::SqrtYdag(q),
            QuriPartsGate::T(q) => QuriPartsGate::T(q),
            QuriPartsGate::Tdag(q) => QuriPartsGate::Tdag(q),
            QuriPartsGate::RX(q, p) => QuriPartsGate::RX(q, f(p)),
            QuriPartsGate::Other(o) => QuriPartsGate::Other(o),
        }
    }
}

impl QuriPartsGate<f64> {
    pub fn from_property(prop: GenericGateProperty) -> Option<Self> {
        let is_nonpara_named_1q_gate = || {
            prop.target_indices.len() == 1
                && prop.control_indices.is_empty()
                && prop.classical_indices.is_empty()
                && prop.params.is_empty()
                && prop.pauli_ids.is_empty()
                && prop.unitary_matrix.is_none()
        };

        let is_para_1q_gate = || {
            prop.target_indices.len() == 1
                && prop.control_indices.is_empty()
                && prop.classical_indices.is_empty()
                && prop.params.len() == 1
                && prop.pauli_ids.is_empty()
                && prop.unitary_matrix.is_none()
        };

        match prop.name.as_str() {
            "Identity" if is_nonpara_named_1q_gate() => {
                Some(Self::Identity(prop.target_indices[0]))
            }
            "X" if is_nonpara_named_1q_gate() => Some(Self::X(prop.target_indices[0])),
            "Y" if is_nonpara_named_1q_gate() => Some(Self::Y(prop.target_indices[0])),
            "Z" if is_nonpara_named_1q_gate() => Some(Self::Z(prop.target_indices[0])),
            "H" if is_nonpara_named_1q_gate() => Some(Self::H(prop.target_indices[0])),
            "S" if is_nonpara_named_1q_gate() => Some(Self::S(prop.target_indices[0])),
            "Sdag" if is_nonpara_named_1q_gate() => Some(Self::Sdag(prop.target_indices[0])),
            "SqrtX" if is_nonpara_named_1q_gate() => Some(Self::SqrtX(prop.target_indices[0])),
            "SqrtXdag" if is_nonpara_named_1q_gate() => {
                Some(Self::SqrtXdag(prop.target_indices[0]))
            }
            "SqrtY" if is_nonpara_named_1q_gate() => Some(Self::SqrtY(prop.target_indices[0])),
            "SqrtYdag" if is_nonpara_named_1q_gate() => {
                Some(Self::SqrtYdag(prop.target_indices[0]))
            }
            "T" if is_nonpara_named_1q_gate() => Some(Self::T(prop.target_indices[0])),
            "Tdag" if is_nonpara_named_1q_gate() => Some(Self::Tdag(prop.target_indices[0])),
            "RX" if is_para_1q_gate() => Some(Self::RX(prop.target_indices[0], prop.params[0])),
            "ParametricRX" => None,
            _ => Some(Self::Other(Box::new(prop))),
        }
    }

    pub fn into_property(self) -> GenericGateProperty {
        fn nonpara_named_1q_gate(name: &str, q: usize) -> GenericGateProperty {
            GenericGateProperty {
                name: name.to_owned(),
                target_indices: vec![q],
                control_indices: vec![],
                classical_indices: vec![],
                params: vec![],
                pauli_ids: vec![],
                unitary_matrix: None,
            }
        }

        fn para_1q_gate(name: &str, q: usize, p: f64) -> GenericGateProperty {
            GenericGateProperty {
                name: name.to_owned(),
                target_indices: vec![q],
                control_indices: vec![],
                classical_indices: vec![],
                params: vec![p],
                pauli_ids: vec![],
                unitary_matrix: None,
            }
        }

        match self {
            Self::Identity(q) => nonpara_named_1q_gate("Identity", q),
            Self::X(q) => nonpara_named_1q_gate("X", q),
            Self::Y(q) => nonpara_named_1q_gate("Y", q),
            Self::Z(q) => nonpara_named_1q_gate("Z", q),
            Self::H(q) => nonpara_named_1q_gate("H", q),
            Self::S(q) => nonpara_named_1q_gate("S", q),
            Self::Sdag(q) => nonpara_named_1q_gate("Sdag", q),
            Self::SqrtX(q) => nonpara_named_1q_gate("SqrtX", q),
            Self::SqrtXdag(q) => nonpara_named_1q_gate("SqrtXdag", q),
            Self::SqrtY(q) => nonpara_named_1q_gate("SqrtY", q),
            Self::SqrtYdag(q) => nonpara_named_1q_gate("SqrtYdag", q),
            Self::T(q) => nonpara_named_1q_gate("T", q),
            Self::Tdag(q) => nonpara_named_1q_gate("Tdag", q),
            Self::RX(q, p) => para_1q_gate("RX", q, p),
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
