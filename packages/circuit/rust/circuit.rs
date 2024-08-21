use crate::gate::QuantumGate;
use pyo3::prelude::*;
use pyo3::types::{PySequence, PyString};
use quri_parts::BasicBlock;
use std::collections::HashMap;
use std::sync::RwLock;

const PICKLE_STUB_ARG: &'static str = "__QURI_PARTS_STUB_ARG_FOR_UNPICKLING";

#[pyclass(subclass, frozen, eq, module = "quri_parts.circuit.rust.circuit")]
#[derive(Debug)]
pub struct ImmutableQuantumCircuit {
    #[pyo3(get)]
    qubit_count: usize,
    #[pyo3(get)]
    cbit_count: usize,
    pub gates: Box<RwLock<BasicBlock<QuantumGate<f64>>>>,
    depth_cache: Box<RwLock<Option<usize>>>,
}

impl PartialEq for ImmutableQuantumCircuit {
    fn eq(&self, other: &Self) -> bool {
        if self.qubit_count != other.qubit_count || self.cbit_count != other.cbit_count {
            return false;
        }
        let lhs = self.gates.read().unwrap();
        let rhs = other.gates.read().unwrap();
        *lhs == *rhs
    }
}

impl Clone for ImmutableQuantumCircuit {
    fn clone(&self) -> Self {
        let gates = self.gates.read().unwrap().clone();
        let depth_cache = self.depth_cache.read().unwrap().clone();
        ImmutableQuantumCircuit {
            qubit_count: self.qubit_count,
            cbit_count: self.cbit_count,
            gates: Box::new(RwLock::new(gates)),
            depth_cache: Box::new(RwLock::new(depth_cache)),
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
                        gates: Box::new(RwLock::new(BasicBlock(gates))),
                        depth_cache: Box::new(RwLock::new(None)),
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
        let gates = slf.get().gates.read().unwrap();
        Ok((
            slf.getattr("__class__")?.unbind(),
            (
                PyString::new_bound(slf.py(), PICKLE_STUB_ARG).unbind(),
                slf.get().qubit_count,
                slf.get().cbit_count,
                gates.0.clone(),
            ),
        ))
    }

    #[getter]
    fn get_gates(slf: &Bound<'_, Self>) -> Vec<QuantumGate> {
        let gates = slf.get().gates.read().unwrap();
        gates.0.iter().cloned().collect()
    }

    #[getter]
    fn get_depth(&self) -> usize {
        if let Some(depth) = *self.depth_cache.read().unwrap() {
            depth
        } else {
            let mut ds = HashMap::<usize, usize>::new();
            let gates = self.gates.read().unwrap();
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
            let depth = ds.into_values().max().unwrap_or(0);
            *self.depth_cache.write().unwrap() = Some(depth);
            depth
        }
    }

    #[pyo3(text_signature = "(gates: ImmutableQuantumCircuit | Sequence[QuantumGate])")]
    fn combine(slf: &Bound<'_, Self>, gates: Bound<'_, PyAny>) -> PyResult<Py<QuantumCircuit>> {
        let ret = Self::get_mutable_copy(slf)?;
        QuantumCircuit::extend(ret.bind(slf.py()), gates)?;
        Ok(ret)
    }

    #[pyo3(name = "__add__")]
    fn py_add(slf: &Bound<'_, Self>, gates: Bound<'_, PyAny>) -> PyResult<Py<QuantumCircuit>> {
        Self::combine(slf, gates)
    }

    fn freeze(&self) -> ImmutableQuantumCircuit {
        self.clone()
    }

    fn get_mutable_copy(slf: &Bound<'_, Self>) -> PyResult<Py<QuantumCircuit>> {
        Py::new(slf.py(), (QuantumCircuit(), slf.get().clone()))
    }
}

#[pyclass(
    extends=ImmutableQuantumCircuit,
    subclass,
    eq,
    module = "quri_parts.circuit.rust.circuit"
)]
#[derive(Clone, Debug, PartialEq)]
pub struct QuantumCircuit();

#[pymethods]
impl QuantumCircuit {
    #[new]
    #[pyo3(signature = (qubit_count, cbit_count=0, gates=Vec::new()))]
    #[pyo3(
        text_signature = "(qubit_count: int, cbit_count: int = 0, gates: Sequence[QuantumGate] = [])"
    )]
    fn py_new(
        qubit_count: usize,
        cbit_count: usize,
        gates: Vec<QuantumGate>,
    ) -> (Self, ImmutableQuantumCircuit) {
        let base = ImmutableQuantumCircuit {
            qubit_count,
            cbit_count,
            gates: Box::new(RwLock::new(BasicBlock(gates))),
            depth_cache: Box::new(RwLock::new(None)),
        };
        (QuantumCircuit(), base)
    }

    #[pyo3(name = "__reduce__")]
    fn py_reduce(slf: &Bound<'_, Self>) -> PyResult<(PyObject, (usize, usize, Vec<QuantumGate>))> {
        let borrowed = slf.borrow();
        let sup = borrowed.as_super();
        let gates = sup.gates.read().unwrap();
        Ok((
            slf.getattr("__class__")?.unbind(),
            (sup.qubit_count, sup.cbit_count, gates.0.clone()),
        ))
    }

    #[pyo3(signature = (gate, gate_index = None))]
    #[pyo3(text_signature = "(gate: QuantumGate, gate_index: Optional[int])")]
    fn add_gate(
        slf: PyRef<'_, Self>,
        gate: QuantumGate,
        gate_index: Option<usize>,
    ) -> PyResult<()> {
        slf.as_super().depth_cache.write().unwrap().take();
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
        let gates = &mut slf.as_super().gates.write().unwrap().0;
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
            for gate in other.get().gates.read().unwrap().0.iter() {
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
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "circuit")?;
    m.add_class::<QuantumCircuit>()?;
    m.add_class::<ImmutableQuantumCircuit>()?;
    Ok(m)
}
