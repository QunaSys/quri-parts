use crate::gates;
use gates::QuriPartsGate;
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyclass(subclass, frozen, eq, module = "quri_parts.circuit.rust.gate")]
#[derive(Clone, Debug, PartialEq)]
pub struct QuantumGate(pub(crate) QuriPartsGate);

#[pymethods]
impl QuantumGate {
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
    fn py_new<'py>(
        py: Python<'py>,
        name: String,
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
        classical_indices: Vec<usize>,
        params: Vec<f64>,
        pauli_ids: Vec<u8>,
        mut unitary_matrix: Option<Vec<Vec<Complex64>>>,
    ) -> PyResult<Py<QuantumGate>> {
        if let Some(matrix) = &unitary_matrix {
            if matrix.len() == 0 {
                unitary_matrix = None;
            }
        }
        let other = OtherData {
            name,
            target_indices,
            control_indices,
            classical_indices,
            params,
            pauli_ids,
            unitary_matrix,
        };
        QuriPartsGate::from_other(other).instantiate(py)
    }

    #[pyo3(name = "__repr__")]
    fn py_repr(&self) -> String {
        self.0.clone().into_other().get_compat_string()
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
        let data = &slf.get().0.clone().into_other();
        Ok((
            Bound::new(slf.py(), QuantumGate(slf.get().0.clone()))
                .unwrap()
                .getattr("__class__")
                .unwrap()
                .unbind(),
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
        let data = self.0.clone().into_other();
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
        self.0.clone().into_other().name
    }

    #[getter]
    fn get_target_indices<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        PyTuple::new_bound(slf.py(), slf.get().0.clone().into_other().target_indices)
    }

    #[getter]
    fn get_control_indices<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        PyTuple::new_bound(slf.py(), slf.get().0.clone().into_other().control_indices)
    }

    #[getter]
    fn get_classical_indices<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        PyTuple::new_bound(slf.py(), slf.get().0.clone().into_other().classical_indices)
    }

    #[getter]
    fn get_params<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        PyTuple::new_bound(slf.py(), slf.get().0.clone().into_other().params)
    }

    #[getter]
    fn get_pauli_ids<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        PyTuple::new_bound(slf.py(), slf.get().0.clone().into_other().pauli_ids)
    }

    #[getter]
    fn get_unitary_matrix<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        if let Some(mat) = slf.get().0.clone().into_other().unitary_matrix {
            PyTuple::new_bound(slf.py(), mat)
        } else {
            PyTuple::new_bound(slf.py(), None as Option<usize>)
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OtherData {
    pub name: String,
    pub target_indices: Vec<usize>,
    pub control_indices: Vec<usize>,
    pub classical_indices: Vec<usize>,
    pub params: Vec<f64>,
    pub pauli_ids: Vec<u8>,
    pub unitary_matrix: Option<Vec<Vec<Complex64>>>,
}

impl OtherData {
    pub fn get_compat_string(&self) -> String {
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
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "gate")?;
    m.add_class::<QuantumGate>()?;
    Ok(m)
}
