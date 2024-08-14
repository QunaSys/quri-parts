use crate::gate::QuantumGate;
use pyo3::prelude::*;
use pyo3::types::{PySequence, PyString};
use quri_parts::BasicBlock;
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
            todo!()
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
                if let Ok(gate) = QuantumGate::downcast_from(&other.get_item(i)?) {
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
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "circuit")?;
    m.add_class::<QuantumCircuit>()?;
    m.add_class::<ImmutableQuantumCircuit>()?;
    Ok(m)
}
