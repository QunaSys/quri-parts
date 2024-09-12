use crate::gate::QuantumGate;
use abi_stable::external_types::RRwLock;
use abi_stable::std_types::{RBox, ROption, RSome};
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::{PySequence, PyString, PyTuple};
use pyo3_commonize::Commonized;
use quri_parts::BasicBlock;
use std::collections::HashMap;

const PICKLE_STUB_ARG: &'static str = "__QURI_PARTS_STUB_ARG_FOR_UNPICKLING";

#[pyclass(subclass, frozen, eq, module = "quri_parts.circuit.rust.circuit")]
#[derive(Commonized)]
#[repr(C)]
pub struct ImmutableQuantumCircuit {
    #[pyo3(get)]
    pub qubit_count: usize,
    #[pyo3(get)]
    pub cbit_count: usize,
    pub gates: RBox<RRwLock<BasicBlock<QuantumGate<f64>>>>,
    pub depth_cache: RBox<RRwLock<ROption<usize>>>,
}

impl PartialEq for ImmutableQuantumCircuit {
    fn eq(&self, other: &Self) -> bool {
        if self.qubit_count != other.qubit_count || self.cbit_count != other.cbit_count {
            return false;
        }
        let lhs = self.gates.read();
        let rhs = other.gates.read();
        *lhs == *rhs
    }
}

impl Clone for ImmutableQuantumCircuit {
    fn clone(&self) -> Self {
        let gates = self.gates.read().clone();
        let depth_cache = self.depth_cache.read().clone();
        ImmutableQuantumCircuit {
            qubit_count: self.qubit_count,
            cbit_count: self.cbit_count,
            gates: RBox::new(RRwLock::new(gates)),
            depth_cache: RBox::new(RRwLock::new(depth_cache)),
        }
    }
}

#[pymethods]
impl ImmutableQuantumCircuit {
    #[new]
    #[pyo3(signature = (*args))]
    #[pyo3(text_signature = "(circuit: ImmutableQuantumCircuit)")]
    fn py_new<'py>(args: &Bound<'py, PyAny>) -> PyResult<Py<Self>> {
        if let Ok((stub, qubit_count, cbit_count, gates)) =
            args.extract::<(Bound<'py, PyString>, usize, usize, Vec<QuantumGate>)>()
        {
            // Called from pickle.load()
            if stub.to_str()? == PICKLE_STUB_ARG {
                return Py::new(
                    args.py(),
                    Self {
                        qubit_count,
                        cbit_count,
                        gates: RBox::new(RRwLock::new(BasicBlock(gates.into()))),
                        depth_cache: RBox::new(RRwLock::new(ROption::RNone)),
                    },
                );
            }
        }
        Ok(args.extract::<(Py<ImmutableQuantumCircuit>,)>()?.0)
    }

    #[pyo3(name = "__reduce__")]
    fn py_reduce(
        slf: &Bound<'_, Self>,
    ) -> PyResult<(PyObject, (Py<PyString>, usize, usize, Vec<QuantumGate>))> {
        let gates = slf.get().gates.read();
        Ok((
            slf.getattr("__class__")?.unbind(),
            (
                PyString::new_bound(slf.py(), PICKLE_STUB_ARG).unbind(),
                slf.get().qubit_count,
                slf.get().cbit_count,
                gates.0.clone().into(),
            ),
        ))
    }

    #[getter]
    fn get_gates<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let gates = slf.get().gates.read();
        Ok(PyTuple::new_bound(
            slf.py(),
            gates.0.iter().map(|g| g.clone().into_py(slf.py())),
        ))
    }

    #[getter]
    fn get_depth(&self) -> usize {
        let mut depth = self.depth_cache.write();
        if let RSome(depth) = *depth {
            depth
        } else {
            let mut ds = HashMap::<usize, usize>::new();
            let gates = self.gates.read();
            for gate in gates.0.iter() {
                let qubits = gate.get_qubits();
                let d = 1 + qubits
                    .iter()
                    .map(|q| ds.get(&q).unwrap_or(&0))
                    .max()
                    .unwrap();
                qubits.iter().for_each(|q| {
                    ds.insert(*q, d);
                });
            }
            let d = ds.into_values().max().unwrap_or(0);
            *depth = ROption::RSome(d);
            d
        }
    }

    #[pyo3(text_signature = "(gates: ImmutableQuantumCircuit | Sequence[QuantumGate])")]
    fn combine(slf: &Bound<'_, Self>, gates: Bound<'_, PyAny>) -> PyResult<Py<QuantumCircuit>> {
        let ret = Self::get_mutable_copy(slf)?;
        QuantumCircuit::extend(ret.bind(slf.py()), gates)?;
        Ok(ret)
    }

    #[pyo3(name = "__add__")]
    fn py_add(slf: &Bound<'_, Self>, gates: Bound<'_, PyAny>) -> PyObject {
        Self::combine(slf, gates)
            .map(|c| c.into_py(slf.py()))
            .unwrap_or(slf.py().NotImplemented())
    }

    fn freeze(&self) -> Self {
        self.clone()
    }

    fn get_mutable_copy(slf: &Bound<'_, Self>) -> PyResult<Py<QuantumCircuit>> {
        Py::new(slf.py(), (QuantumCircuit(), slf.get().clone()))
    }
}

#[pyclass(
    extends=ImmutableQuantumCircuit,
    subclass,
    module = "quri_parts.circuit.rust.circuit"
)]
#[derive(Clone, Debug, PartialEq, Commonized)]
pub struct QuantumCircuit();

#[pymethods]
impl QuantumCircuit {
    #[new]
    #[pyo3(signature = (qubit_count, cbit_count=None, gates=Vec::new()))]
    fn py_new(
        qubit_count: usize,
        cbit_count: Option<Bound<'_, PyAny>>,
        gates: Vec<QuantumGate>,
    ) -> PyResult<(Self, ImmutableQuantumCircuit)> {
        let (cbit_count, gates) = if let Some(cbit_count) = cbit_count.as_ref() {
            if let Ok(seq) = cbit_count.downcast::<PySequence>() {
                cbit_count
                    .py()
                    .run_bound(
                        r#"
import warnings
warnings.warn(
    "QuantumCircuit initialization takes cbit_count before gates.",
    DeprecationWarning,
)
                        "#,
                        None,
                        None,
                    )
                    .unwrap();
                (
                    0,
                    (0..(seq.len()?))
                        .map(|i| FromPyObject::extract_bound(&seq.get_item(i)?))
                        .collect::<PyResult<Vec<_>>>()?,
                )
            } else {
                (FromPyObject::extract_bound(&cbit_count)?, gates)
            }
        } else {
            (0, gates)
        };

        let base = ImmutableQuantumCircuit {
            qubit_count,
            cbit_count,
            gates: RBox::new(RRwLock::new(BasicBlock(gates.into()))),
            depth_cache: RBox::new(RRwLock::new(ROption::RNone)),
        };
        Ok((QuantumCircuit(), base))
    }

    #[pyo3(name = "__reduce__")]
    fn py_reduce(slf: &Bound<'_, Self>) -> PyResult<(PyObject, (usize, usize, Vec<QuantumGate>))> {
        let borrowed = slf.borrow();
        let sup = borrowed.as_super();
        let gates = sup.gates.read();
        Ok((
            slf.getattr("__class__")?.unbind(),
            (sup.qubit_count, sup.cbit_count, gates.0.clone().into()),
        ))
    }

    #[pyo3(signature = (gate, gate_index = None))]
    #[pyo3(text_signature = "(gate: QuantumGate, gate_index: Optional[int])")]
    fn add_gate(
        slf: PyRef<'_, Self>,
        gate: QuantumGate,
        gate_index: Option<usize>,
    ) -> PyResult<()> {
        slf.as_super().depth_cache.write().take();
        if gate
            .get_qubits()
            .iter()
            .max()
            .map(|n| n >= &slf.as_super().qubit_count)
            .unwrap_or(false)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "The indices of the gate applied must be smaller than qubit_count",
            ));
        }
        if gate
            .get_cbits()
            .iter()
            .max()
            .map(|n| n >= &slf.as_super().cbit_count)
            .unwrap_or(false)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "The classical indices of the gate applied must be smaller than cbit_count",
            ));
        }
        let gates = &mut slf.as_super().gates.write().0;
        if let Some(gate_index) = gate_index {
            if gate_index <= gates.len() {
                gates.insert(gate_index, gate.clone());
                Ok(())
            } else {
                Err(pyo3::exceptions::PyIndexError::new_err(""))
            }
        } else {
            gates.push(gate.clone());
            Ok(())
        }
    }

    #[pyo3(text_signature = "(gates: ImmutableQuantumCircuit | Sequence[QuantumGate])")]
    fn extend(slf: &Bound<'_, Self>, gates: Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(other) = gates.downcast::<ImmutableQuantumCircuit>() {
            for gate in other.get().gates.read().0.iter() {
                Self::add_gate(slf.borrow(), gate.clone(), None)?;
            }
            Ok(())
        } else if let Ok(other) = gates.downcast::<PySequence>() {
            for i in 0..other.len()? {
                if let Ok(gate) = QuantumGate::extract_bound(&other.get_item(i)?) {
                    Self::add_gate(slf.borrow(), gate, None)?;
                }
            }
            Ok(())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Bad argument `gates`",
            ))
        }
    }

    #[pyo3(name = "__iadd__")]
    fn py_iadd(slf: &Bound<'_, Self>, gates: Bound<'_, PyAny>) -> PyResult<()> {
        Self::extend(slf, gates)
            .map_err(|_| pyo3::exceptions::PyNotImplementedError::new_err("Not implemented"))
    }

    #[allow(non_snake_case)]
    fn add_Identity_gate(slf: PyRef<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::Identity(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_X_gate(slf: PyRef<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::X(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_Y_gate(slf: PyRef<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::Y(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_Z_gate(slf: PyRef<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::Z(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_H_gate(slf: PyRef<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::H(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_S_gate(slf: PyRef<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::S(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_Sdag_gate(slf: PyRef<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::Sdag(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_SqrtX_gate(slf: PyRef<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::SqrtX(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_SqrtXdag_gate(slf: PyRef<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::SqrtXdag(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_SqrtY_gate(slf: PyRef<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::SqrtY(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_SqrtYdag_gate(slf: PyRef<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::SqrtYdag(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_T_gate(slf: PyRef<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::T(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_Tdag_gate(slf: PyRef<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::Tdag(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_U1_gate(slf: PyRef<'_, Self>, qubit_index: usize, lmd: f64) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::U1(qubit_index, lmd), None)
    }

    #[allow(non_snake_case)]
    fn add_U2_gate(slf: PyRef<'_, Self>, qubit_index: usize, phi: f64, lmd: f64) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::U2(qubit_index, phi, lmd), None)
    }

    #[allow(non_snake_case)]
    fn add_U3_gate(
        slf: PyRef<'_, Self>,
        qubit_index: usize,
        theta: f64,
        phi: f64,
        lmd: f64,
    ) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::U3(qubit_index, theta, phi, lmd), None)
    }

    #[allow(non_snake_case)]
    fn add_RX_gate(slf: PyRef<'_, Self>, qubit_index: usize, angle: f64) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::RX(qubit_index, angle), None)
    }

    #[allow(non_snake_case)]
    fn add_RY_gate(slf: PyRef<'_, Self>, qubit_index: usize, angle: f64) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::RY(qubit_index, angle), None)
    }

    #[allow(non_snake_case)]
    fn add_RZ_gate(slf: PyRef<'_, Self>, qubit_index: usize, angle: f64) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::RZ(qubit_index, angle), None)
    }

    #[allow(non_snake_case)]
    fn add_CNOT_gate(
        slf: PyRef<'_, Self>,
        control_index: usize,
        target_index: usize,
    ) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::CNOT(control_index, target_index), None)
    }

    #[allow(non_snake_case)]
    fn add_CZ_gate(
        slf: PyRef<'_, Self>,
        control_index: usize,
        target_index: usize,
    ) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::CZ(control_index, target_index), None)
    }

    #[allow(non_snake_case)]
    fn add_SWAP_gate(
        slf: PyRef<'_, Self>,
        target_index1: usize,
        target_index2: usize,
    ) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::SWAP(target_index1, target_index2), None)
    }

    #[allow(non_snake_case)]
    fn add_TOFFOLI_gate(
        slf: PyRef<'_, Self>,
        control_index1: usize,
        control_index2: usize,
        target_index: usize,
    ) -> PyResult<()> {
        Self::add_gate(
            slf,
            QuantumGate::TOFFOLI(control_index1, control_index2, target_index),
            None,
        )
    }

    #[allow(non_snake_case)]
    fn add_UnitaryMatrix_gate(
        slf: PyRef<'_, Self>,
        target_indices: Vec<usize>,
        unitary_matrix: Vec<Vec<Complex64>>,
    ) -> PyResult<()> {
        Self::add_gate(
            slf,
            crate::gates::unitary_matrix(target_indices, unitary_matrix)?,
            None,
        )
    }

    #[allow(non_snake_case)]
    fn add_SingleQubitUnitaryMatrix_gate(
        slf: PyRef<'_, Self>,
        target_index: usize,
        unitary_matrix: Vec<Vec<Complex64>>,
    ) -> PyResult<()> {
        Self::add_gate(
            slf,
            crate::gates::single_qubit_unitary_matrix(target_index, unitary_matrix)?,
            None,
        )
    }

    #[allow(non_snake_case)]
    fn add_TwoQubitUnitaryMatrix_gate(
        slf: PyRef<'_, Self>,
        target_index1: usize,
        target_index2: usize,
        unitary_matrix: Vec<Vec<Complex64>>,
    ) -> PyResult<()> {
        Self::add_gate(
            slf,
            crate::gates::two_qubit_unitary_matrix(target_index1, target_index2, unitary_matrix)?,
            None,
        )
    }

    #[allow(non_snake_case)]
    fn add_Pauli_gate(
        slf: PyRef<'_, Self>,
        target_indices: Vec<usize>,
        pauli_ids: Vec<u8>,
    ) -> PyResult<()> {
        Self::add_gate(slf, crate::gates::pauli(target_indices, pauli_ids), None)
    }

    #[allow(non_snake_case)]
    fn add_PauliRotation_gate(
        slf: PyRef<'_, Self>,
        target_qubits: Vec<usize>,
        pauli_id_list: Vec<u8>,
        angle: f64,
    ) -> PyResult<()> {
        Self::add_gate(
            slf,
            crate::gates::pauli_rotation(target_qubits, pauli_id_list, angle),
            None,
        )
    }

    fn measure(
        slf: PyRef<'_, Self>,
        qubit_indices: &Bound<'_, PyAny>,
        classical_indices: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let qubit_indices = if let Ok(seq) = qubit_indices.downcast::<PySequence>() {
            (0..(seq.len()?))
                .map(|i| FromPyObject::extract_bound(&seq.get_item(i)?))
                .collect::<PyResult<Vec<_>>>()?
        } else {
            vec![FromPyObject::extract_bound(qubit_indices)?]
        };
        let classical_indices: Vec<usize> =
            if let Ok(seq) = classical_indices.downcast::<PySequence>() {
                (0..(seq.len()?))
                    .map(|i| FromPyObject::extract_bound(&seq.get_item(i)?))
                    .collect::<PyResult<Vec<_>>>()?
            } else {
                vec![FromPyObject::extract_bound(classical_indices)?]
            };

        Self::add_gate(
            slf,
            crate::gates::measurement(qubit_indices, classical_indices)?,
            None,
        )
    }
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "circuit")?;
    m.add_class::<QuantumCircuit>()?;
    m.add_class::<ImmutableQuantumCircuit>()?;
    Ok(m)
}
