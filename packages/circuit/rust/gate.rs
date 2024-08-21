use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[derive(Clone, Debug, PartialEq)]
pub enum QuantumGate<P = f64> {
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
    RY(usize, P),
    RZ(usize, P),
    U1(usize, P),
    U2(usize, P, P),
    U3(usize, P, P, P),
    CNOT(usize, usize),
    CZ(usize, usize),
    SWAP(usize, usize),
    TOFFOLI(usize, usize, usize),
    UnitaryMatrix(Vec<usize>, Vec<Vec<Complex64>>),
    Pauli(Vec<usize>, Vec<u8>),
    PauliRotation(Vec<usize>, Vec<u8>, P),
    Other(Box<GenericGateProperty>),
}

impl<P: Clone> QuantumGate<P> {
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
            | Self::RX(q, _)
            | Self::RY(q, _)
            | Self::RZ(q, _)
            | Self::U1(q, _)
            | Self::U2(q, _, _)
            | Self::U3(q, _, _, _) => vec![*q],
            Self::CNOT(q0, q1) | Self::CZ(q0, q1) | Self::SWAP(q0, q1) => vec![*q0, *q1],
            Self::TOFFOLI(q0, q1, q2) => vec![*q0, *q1, *q2],
            Self::UnitaryMatrix(qs, _) | Self::Pauli(qs, _) | Self::PauliRotation(qs, _, _) => {
                qs.clone()
            }
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
            Self::RX(_, p) | Self::RY(_, p) | Self::RZ(_, p) | Self::U1(_, p) => vec![p.clone()],
            Self::U2(_, p0, p1) => vec![p0.clone(), p1.clone()],
            Self::U3(_, p0, p1, p2) => vec![p0.clone(), p1.clone(), p2.clone()],
            Self::PauliRotation(_, _, p) => vec![p.clone()],
            _ => vec![],
        }
    }

    pub fn map_param<Q>(self, mut f: impl FnMut(P) -> Q) -> QuantumGate<Q> {
        match self {
            QuantumGate::Identity(q) => QuantumGate::Identity(q),
            QuantumGate::X(q) => QuantumGate::X(q),
            QuantumGate::Y(q) => QuantumGate::Y(q),
            QuantumGate::Z(q) => QuantumGate::Z(q),
            QuantumGate::H(q) => QuantumGate::H(q),
            QuantumGate::S(q) => QuantumGate::S(q),
            QuantumGate::Sdag(q) => QuantumGate::Sdag(q),
            QuantumGate::SqrtX(q) => QuantumGate::SqrtX(q),
            QuantumGate::SqrtXdag(q) => QuantumGate::SqrtXdag(q),
            QuantumGate::SqrtY(q) => QuantumGate::SqrtY(q),
            QuantumGate::SqrtYdag(q) => QuantumGate::SqrtYdag(q),
            QuantumGate::T(q) => QuantumGate::T(q),
            QuantumGate::Tdag(q) => QuantumGate::Tdag(q),
            QuantumGate::RX(q, p) => QuantumGate::RX(q, f(p)),
            QuantumGate::RY(q, p) => QuantumGate::RY(q, f(p)),
            QuantumGate::RZ(q, p) => QuantumGate::RZ(q, f(p)),
            QuantumGate::U1(q, p) => QuantumGate::U1(q, f(p)),
            QuantumGate::U2(q, p0, p1) => QuantumGate::U2(q, f(p0), f(p1)),
            QuantumGate::U3(q, p0, p1, p2) => QuantumGate::U3(q, f(p0), f(p1), f(p2)),
            QuantumGate::CNOT(q0, q1) => QuantumGate::CNOT(q0, q1),
            QuantumGate::CZ(q0, q1) => QuantumGate::CZ(q0, q1),
            QuantumGate::SWAP(q0, q1) => QuantumGate::SWAP(q0, q1),
            QuantumGate::TOFFOLI(q0, q1, q2) => QuantumGate::TOFFOLI(q0, q1, q2),
            QuantumGate::UnitaryMatrix(qs, mat) => QuantumGate::UnitaryMatrix(qs, mat),
            QuantumGate::Pauli(qs, ps) => QuantumGate::Pauli(qs, ps),
            QuantumGate::PauliRotation(qs, ps, a) => QuantumGate::PauliRotation(qs, ps, f(a)),
            QuantumGate::Other(o) => QuantumGate::Other(o),
        }
    }
}

impl QuantumGate<f64> {
    pub fn from_property(prop: GenericGateProperty) -> PyResult<Option<Self>> {
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
                Ok(Some(Self::Identity(prop.target_indices[0])))
            }
            "X" if is_nonpara_named_1q_gate() => Ok(Some(Self::X(prop.target_indices[0]))),
            "Y" if is_nonpara_named_1q_gate() => Ok(Some(Self::Y(prop.target_indices[0]))),
            "Z" if is_nonpara_named_1q_gate() => Ok(Some(Self::Z(prop.target_indices[0]))),
            "H" if is_nonpara_named_1q_gate() => Ok(Some(Self::H(prop.target_indices[0]))),
            "S" if is_nonpara_named_1q_gate() => Ok(Some(Self::S(prop.target_indices[0]))),
            "Sdag" if is_nonpara_named_1q_gate() => Ok(Some(Self::Sdag(prop.target_indices[0]))),
            "SqrtX" if is_nonpara_named_1q_gate() => Ok(Some(Self::SqrtX(prop.target_indices[0]))),
            "SqrtXdag" if is_nonpara_named_1q_gate() => {
                Ok(Some(Self::SqrtXdag(prop.target_indices[0])))
            }
            "SqrtY" if is_nonpara_named_1q_gate() => Ok(Some(Self::SqrtY(prop.target_indices[0]))),
            "SqrtYdag" if is_nonpara_named_1q_gate() => {
                Ok(Some(Self::SqrtYdag(prop.target_indices[0])))
            }
            "T" if is_nonpara_named_1q_gate() => Ok(Some(Self::T(prop.target_indices[0]))),
            "Tdag" if is_nonpara_named_1q_gate() => Ok(Some(Self::Tdag(prop.target_indices[0]))),
            "RX" if is_para_1q_gate() => Ok(Some(Self::RX(prop.target_indices[0], prop.params[0]))),
            "RY" if is_para_1q_gate() => Ok(Some(Self::RY(prop.target_indices[0], prop.params[0]))),
            "RZ" if is_para_1q_gate() => Ok(Some(Self::RZ(prop.target_indices[0], prop.params[0]))),
            "U1" if is_para_1q_gate() => Ok(Some(Self::U1(prop.target_indices[0], prop.params[0]))),
            "U2" if prop.control_indices.is_empty()
                && prop.target_indices.len() == 1
                && prop.classical_indices.is_empty()
                && prop.params.len() == 2
                && prop.pauli_ids.is_empty()
                && prop.unitary_matrix.is_none() =>
            {
                Ok(Some(Self::U2(
                    prop.target_indices[0],
                    prop.params[0],
                    prop.params[1],
                )))
            }
            "U3" if prop.control_indices.is_empty()
                && prop.target_indices.len() == 1
                && prop.classical_indices.is_empty()
                && prop.params.len() == 3
                && prop.pauli_ids.is_empty()
                && prop.unitary_matrix.is_none() =>
            {
                Ok(Some(Self::U3(
                    prop.target_indices[0],
                    prop.params[0],
                    prop.params[1],
                    prop.params[2],
                )))
            }
            "CNOT"
                if prop.control_indices.len() == 1
                    && prop.target_indices.len() == 1
                    && prop.classical_indices.is_empty()
                    && prop.params.is_empty()
                    && prop.pauli_ids.is_empty()
                    && prop.unitary_matrix.is_none() =>
            {
                Ok(Some(Self::CNOT(
                    prop.control_indices[0],
                    prop.target_indices[0],
                )))
            }
            "CZ" if prop.control_indices.len() == 1
                && prop.target_indices.len() == 1
                && prop.classical_indices.is_empty()
                && prop.params.is_empty()
                && prop.pauli_ids.is_empty()
                && prop.unitary_matrix.is_none() =>
            {
                Ok(Some(Self::CZ(
                    prop.control_indices[0],
                    prop.target_indices[0],
                )))
            }
            "SWAP"
                if prop.control_indices.is_empty()
                    && prop.target_indices.len() == 2
                    && prop.classical_indices.is_empty()
                    && prop.params.is_empty()
                    && prop.pauli_ids.is_empty()
                    && prop.unitary_matrix.is_none() =>
            {
                Ok(Some(Self::SWAP(
                    prop.target_indices[0],
                    prop.target_indices[1],
                )))
            }
            "TOFFOLI"
                if prop.control_indices.len() == 2
                    && prop.target_indices.len() == 1
                    && prop.classical_indices.is_empty()
                    && prop.params.is_empty()
                    && prop.pauli_ids.is_empty()
                    && prop.unitary_matrix.is_none() =>
            {
                Ok(Some(Self::TOFFOLI(
                    prop.control_indices[0],
                    prop.control_indices[1],
                    prop.target_indices[0],
                )))
            }
            "UnitaryMatrix" | "SingleQubitUnitaryMatrix" | "TwoQubitUnitaryMatrix"
                if prop.control_indices.is_empty()
                    && prop.classical_indices.is_empty()
                    && prop.params.is_empty()
                    && prop.pauli_ids.is_empty()
                    && prop.unitary_matrix.is_some() =>
            {
                Ok(Some(crate::gates::unitary_matrix(
                    prop.target_indices,
                    prop.unitary_matrix.unwrap(),
                )?))
            }
            "Pauli"
                if prop.control_indices.is_empty()
                    && prop.classical_indices.is_empty()
                    && prop.params.is_empty()
                    && prop.pauli_ids.len() == prop.target_indices.len()
                    && prop.unitary_matrix.is_none() =>
            {
                Ok(Some(Self::Pauli(prop.target_indices, prop.pauli_ids)))
            }
            "PauliRotation"
                if prop.control_indices.is_empty()
                    && prop.classical_indices.is_empty()
                    && prop.params.len() == 1
                    && prop.pauli_ids.len() == prop.target_indices.len()
                    && prop.unitary_matrix.is_none() =>
            {
                Ok(Some(Self::PauliRotation(
                    prop.target_indices,
                    prop.pauli_ids,
                    prop.params[0],
                )))
            }
            "ParametricRX" | "ParametricRY" | "ParametricRZ" => Ok(None),
            _ => Ok(Some(Self::Other(Box::new(prop)))),
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
            Self::RY(q, p) => para_1q_gate("RY", q, p),
            Self::RZ(q, p) => para_1q_gate("RZ", q, p),
            Self::U1(q, p) => para_1q_gate("U1", q, p),
            Self::U2(q, p0, p1) => GenericGateProperty {
                name: "U2".to_owned(),
                target_indices: vec![q],
                params: vec![p0, p1],
                ..Default::default()
            },
            Self::U3(q, p0, p1, p2) => GenericGateProperty {
                name: "U3".to_owned(),
                target_indices: vec![q],
                params: vec![p0, p1, p2],
                ..Default::default()
            },
            Self::CNOT(q0, q1) => GenericGateProperty {
                name: "CNOT".to_owned(),
                control_indices: vec![q0],
                target_indices: vec![q1],
                ..Default::default()
            },
            Self::CZ(q0, q1) => GenericGateProperty {
                name: "CZ".to_owned(),
                control_indices: vec![q0],
                target_indices: vec![q1],
                ..Default::default()
            },
            Self::SWAP(q0, q1) => GenericGateProperty {
                name: "SWAP".to_owned(),
                target_indices: vec![q0, q1],
                ..Default::default()
            },
            Self::TOFFOLI(q0, q1, q2) => GenericGateProperty {
                name: "CNOT".to_owned(),
                control_indices: vec![q0, q1],
                target_indices: vec![q2],
                ..Default::default()
            },
            Self::UnitaryMatrix(target_indices, mat) => GenericGateProperty {
                name: match target_indices.len() {
                    1 => "SingleQubitUnitaryMatrix",
                    2 => "TwoQubitUnitaryMatrix",
                    _ => "UnitaryMatrix",
                }
                .to_owned(),
                target_indices,
                unitary_matrix: Some(mat),
                ..Default::default()
            },
            Self::Pauli(target_indices, pauli_ids) => GenericGateProperty {
                name: "Pauli".to_owned(),
                target_indices,
                pauli_ids,
                ..Default::default()
            },
            Self::PauliRotation(target_indices, pauli_ids, angle) => GenericGateProperty {
                name: "PauliRotation".to_owned(),
                target_indices,
                params: vec![angle],
                pauli_ids,
                ..Default::default()
            },
            Self::Other(o) => *o,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct GenericGateProperty {
    pub name: String,
    pub target_indices: Vec<usize>,
    pub control_indices: Vec<usize>,
    pub classical_indices: Vec<usize>,
    pub params: Vec<f64>,
    pub pauli_ids: Vec<u8>,
    pub unitary_matrix: Option<Vec<Vec<Complex64>>>,
}
#[pyclass(subclass, frozen, eq, module = "quri_parts.circuit.rust.gate")]
#[derive(Clone, Debug, PartialEq)]
pub struct ParametricQuantumGate(pub(crate) GenericGateProperty);

#[pymethods]
impl ParametricQuantumGate {
    #[new]
    #[pyo3(
        signature = (name, target_indices, control_indices=Vec::new(), pauli_ids=Vec::new()),
        text_signature = "(
            name: str,
            target_indices: Sequence[int],
            control_indices: Sequence[int] = [],
            pauli_ids: Sequence[int] = [],
        )"
    )]
    fn py_new<'py>(
        name: String,
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
        pauli_ids: Vec<u8>,
    ) -> ParametricQuantumGate {
        let prop = GenericGateProperty {
            name: name.clone(),
            target_indices,
            control_indices,
            pauli_ids,
            classical_indices: vec![],
            params: vec![],
            unitary_matrix: None,
        };
        ParametricQuantumGate(prop)
    }

    #[pyo3(name = "__repr__")]
    fn py_repr(&self) -> String {
        self.0.get_compat_string_parametric()
    }

    #[pyo3(name = "__reduce__")]
    fn py_reduce(
        slf: &Bound<'_, Self>,
    ) -> PyResult<(PyObject, (String, Vec<usize>, Vec<usize>, Vec<u8>))> {
        let data = &slf.get().0;
        Ok((
            Bound::new(slf.py(), ParametricQuantumGate(slf.get().0.clone()))
                .unwrap()
                .getattr("__class__")
                .unwrap()
                .unbind(),
            (
                data.name.clone(),
                data.target_indices.clone(),
                data.control_indices.clone(),
                data.pauli_ids.clone(),
            ),
        ))
    }

    #[pyo3(name = "__hash__")]
    pub(crate) fn py_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        let data = self.0.clone();
        data.name.hash(&mut hasher);
        data.target_indices.hash(&mut hasher);
        data.control_indices.hash(&mut hasher);
        for p in data.params.iter() {
            p.to_le_bytes().hash(&mut hasher);
        }
        data.pauli_ids.hash(&mut hasher);
        hasher.finish()
    }

    #[getter]
    fn get_name(&self) -> String {
        self.0.name.clone()
    }

    #[getter]
    fn get_target_indices<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        PyTuple::new_bound(slf.py(), slf.get().0.target_indices.clone())
    }

    #[getter]
    fn get_control_indices<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        PyTuple::new_bound(slf.py(), slf.get().0.control_indices.clone())
    }

    #[getter]
    fn get_params<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        PyTuple::new_bound(slf.py(), slf.get().0.params.clone())
    }

    #[getter]
    fn get_pauli_ids<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        PyTuple::new_bound(slf.py(), slf.get().0.pauli_ids.clone())
    }
}

fn format_tuple<T: core::fmt::Display>(input: &[T]) -> String {
    let out = input
        .iter()
        .map(|i| format!("{}", i))
        .collect::<Vec<_>>()
        .join(", ");
    if input.len() == 1 {
        out + ","
    } else {
        out
    }
}

impl GenericGateProperty {
    pub fn get_compat_string(&self) -> String {
        format!("QuantumGate(name='{}', target_indices=({}), control_indices=({}), classical_indices=({}), params=({}), pauli_ids=({}), unitary_matrix={})",
            &self.name,
            format_tuple(self.target_indices.as_slice()),
            format_tuple(self.control_indices.as_slice()),
            format_tuple(self.classical_indices.as_slice()),
            format_tuple(self.params.as_slice()),
            format_tuple(self.pauli_ids.as_slice()),
            {
                if let Some(matrix) = &self.unitary_matrix {
                    format_tuple(matrix.iter().map(|row| format!("({})", format_tuple(row))).collect::<Vec<_>>().as_slice())
                }else {"()".to_owned()}
            }
        )
    }
    pub fn get_compat_string_parametric(&self) -> String {
        format!("ParametricQuantumGate(name='{}', target_indices=({}), control_indices=({}), pauli_ids=({}))",
            &self.name,
            format_tuple(self.target_indices.as_slice()),
            format_tuple(self.control_indices.as_slice()),
            format_tuple(self.pauli_ids.as_slice()),
        )
    }
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    fn add_quantum_gate(m: &Bound<'_, PyModule>) -> PyResult<()> {
        #[pyclass(
            frozen,
            eq,
            module = "quri_parts.circuit.rust.gate",
            name = "QuantumGate"
        )]
        #[derive(Clone, Debug, PartialEq)]
        struct QuantumGateWrapper(QuantumGate<f64>);

        impl<T> IntoPy<T> for QuantumGate<f64>
        where
            QuantumGateWrapper: IntoPy<T>,
        {
            fn into_py(self, py: Python<'_>) -> T {
                QuantumGateWrapper(self).into_py(py)
            }
        }

        impl<'py> pyo3::conversion::FromPyObject<'py> for QuantumGate<f64> {
            fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
                Ok(
                    <QuantumGateWrapper as pyo3::conversion::FromPyObject<'py>>::extract_bound(ob)?
                        .0,
                )
            }
        }

        #[pymethods]
        impl QuantumGateWrapper {
            #[new]
            #[pyo3(
            signature = (name, target_indices, control_indices=Vec::new(), classical_indices=Vec::new(), params=Vec::new(), pauli_ids=Vec::new(), unitary_matrix=None),
            text_signature = "(
                name: str,
                target_indices: Sequence[int],
                control_indices: Sequence[int] = [],
                classical_indices: Sequence[int] = [],
                params: Sequence[float] = [],
                pauli_ids: Sequence[int] = [],
                unitary_matrix: Optional[Sequence[Sequence[complex]]] = None
            )"
        )]
            fn py_new(
                name: String,
                target_indices: Vec<usize>,
                control_indices: Vec<usize>,
                classical_indices: Vec<usize>,
                params: Vec<f64>,
                pauli_ids: Vec<u8>,
                mut unitary_matrix: Option<Vec<Vec<Complex64>>>,
            ) -> PyResult<Self> {
                if let Some(matrix) = &unitary_matrix {
                    if matrix.len() == 0 {
                        unitary_matrix = None;
                    }
                }
                let prop = GenericGateProperty {
                    name: name.clone(),
                    target_indices,
                    control_indices,
                    classical_indices,
                    params,
                    pauli_ids,
                    unitary_matrix,
                };
                Ok(Self(QuantumGate::from_property(prop)?.ok_or(
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Cannot initialize QuantumGate with {}",
                        &name
                    )),
                )?))
            }

            #[pyo3(name = "__repr__")]
            fn py_repr(&self) -> String {
                self.0.clone().into_property().get_compat_string()
            }

            #[pyo3(name = "__reduce__")]
            fn py_reduce(
                slf: &Bound<'_, Self>,
            ) -> PyResult<(
                PyObject,
                (
                    String,
                    Vec<usize>,
                    Vec<usize>,
                    Vec<usize>,
                    Vec<f64>,
                    Vec<u8>,
                    Option<Vec<Vec<Complex64>>>,
                ),
            )> {
                let data = &slf.get().0.clone().into_property();
                Ok((
                    slf.getattr("__class__").unwrap().unbind(),
                    (
                        data.name.clone(),
                        data.target_indices.clone(),
                        data.control_indices.clone(),
                        data.classical_indices.clone(),
                        data.params.clone(),
                        data.pauli_ids.clone(),
                        data.unitary_matrix.clone(),
                    ),
                ))
            }

            #[pyo3(name = "__hash__")]
            pub(crate) fn py_hash(&self) -> u64 {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                let data = self.0.clone().into_property();
                data.name.hash(&mut hasher);
                data.target_indices.hash(&mut hasher);
                data.control_indices.hash(&mut hasher);
                data.classical_indices.hash(&mut hasher);
                for p in data.params.iter() {
                    p.to_le_bytes().hash(&mut hasher);
                }
                data.pauli_ids.hash(&mut hasher);
                if let Some(matrix) = &data.unitary_matrix {
                    for p in matrix.iter() {
                        for q in p.iter() {
                            q.re.to_le_bytes().hash(&mut hasher);
                            q.im.to_le_bytes().hash(&mut hasher);
                        }
                    }
                }
                hasher.finish()
            }

            #[getter]
            fn get_name(&self) -> String {
                self.0.clone().into_property().name
            }

            #[getter]
            fn get_target_indices<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
                PyTuple::new_bound(slf.py(), slf.get().0.clone().into_property().target_indices)
            }

            #[getter]
            fn get_control_indices<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
                PyTuple::new_bound(
                    slf.py(),
                    slf.get().0.clone().into_property().control_indices,
                )
            }

            #[getter]
            fn get_classical_indices<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
                PyTuple::new_bound(
                    slf.py(),
                    slf.get().0.clone().into_property().classical_indices,
                )
            }

            #[getter]
            fn get_params<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
                PyTuple::new_bound(slf.py(), slf.get().0.clone().into_property().params)
            }

            #[getter]
            fn get_pauli_ids<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
                PyTuple::new_bound(slf.py(), slf.get().0.clone().into_property().pauli_ids)
            }

            #[getter]
            fn get_unitary_matrix<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
                if let Some(mat) = slf.get().0.clone().into_property().unitary_matrix {
                    PyTuple::new_bound(slf.py(), mat)
                } else {
                    PyTuple::new_bound(slf.py(), None as Option<usize>)
                }
            }
        }

        m.add_class::<QuantumGateWrapper>()?;
        Ok(())
    }

    let m = PyModule::new_bound(py, "gate")?;
    m.add_class::<ParametricQuantumGate>()?;
    add_quantum_gate(&m)?;
    Ok(m)
}
