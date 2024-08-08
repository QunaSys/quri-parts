use crate::gate::QuantumGate;
use crate::gates::QuriPartsGate;
use pyo3::prelude::*;
use pyo3::types::PySequence;
use quri_parts::BasicBlock;
use std::sync::RwLock;

#[pyclass(subclass, frozen, eq, module = "quri_parts.circuit.rust.circuit")]
#[derive(Debug)]
pub struct ImmutableQuantumCircuit {
    #[pyo3(get)]
    qubit_count: usize,
    #[pyo3(get)]
    cbit_count: usize,
    gates: Box<RwLock<BasicBlock<QuriPartsGate>>>,
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
    #[pyo3(signature = (circuit))]
    #[pyo3(text_signature = "(circuit: ImmutableQuantumCircuit)")]
    fn py_new(circuit: &Self) -> Self {
        let gates = circuit.gates.read().unwrap().clone();
        ImmutableQuantumCircuit {
            qubit_count: circuit.qubit_count,
            cbit_count: circuit.cbit_count,
            gates: Box::new(RwLock::new(gates)),
            depth_cache: Box::new(RwLock::new(None)),
        }
    }

    #[pyo3(name = "__reduce__")]
    fn py_reduce(slf: &Bound<'_, Self>) -> PyResult<(PyObject, (usize, usize, Vec<QuantumGate>))> {
        todo!()
    }

    #[getter]
    fn get_gates(slf: &Bound<'_, Self>) -> PyResult<Vec<Py<QuantumGate>>> {
        let gates = slf.get().gates.read().unwrap();
        gates
            .0
            .iter()
            .map(|g| g.clone().instantiate(slf.py()))
            .collect()
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
            gates: Box::new(RwLock::new(BasicBlock(
                gates.into_iter().map(|a| a.0).collect(),
            ))),
            depth_cache: Box::new(RwLock::new(None)),
        };
        (QuantumCircuit(), base)
    }

    #[pyo3(signature = (gate, gate_index = None))]
    #[pyo3(text_signature = "(gate: QuantumGate, gate_index: Optional[int])")]
    fn add_gate(
        slf: PyRef<'_, Self>,
        gate: &QuantumGate,
        gate_index: Option<usize>,
    ) -> PyResult<()> {
        slf.as_super().depth_cache.write().unwrap().take();
        if gate.0.get_qubits().iter().max().unwrap_or(&0) >= &slf.as_super().qubit_count {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "The indices of the gate applied must be smaller than qubit_count",
            ));
        }
        if gate.0.get_cbits().iter().max().unwrap_or(&0) >= &slf.as_super().cbit_count {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "The classical indices of the gate applied must be smaller than cbit_count",
            ));
        }
        let gates = &mut slf.as_super().gates.write().unwrap().0;
        if let Some(gate_index) = gate_index {
            if gate_index <= gates.len() {
                gates.insert(gate_index, gate.0.clone());
                Ok(())
            } else {
                Err(pyo3::exceptions::PyIndexError::new_err(""))
            }
        } else {
            gates.push(gate.0.clone());
            Ok(())
        }
    }

    #[pyo3(text_signature = "(gates: ImmutableQuantumCircuit | Sequence[QuantumGate])")]
    fn extend(slf: &Bound<'_, Self>, gates: Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(other) = gates.downcast::<ImmutableQuantumCircuit>() {
            for gate in other.get().gates.read().unwrap().0.iter() {
                Self::add_gate(slf.borrow(), &QuantumGate(gate.clone()), None)?;
            }
            Ok(())
        } else if let Ok(other) = gates.downcast::<PySequence>() {
            for i in 0..other.len()? {
                if let Ok(gate) = other.get_item(i)?.downcast::<QuantumGate>() {
                    Self::add_gate(slf.borrow(), &*gate.borrow(), None)?;
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
        Self::add_gate(slf, &QuantumGate(QuriPartsGate::X(qubit_index)), None)
    }
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "circuit")?;
    m.add_class::<QuantumCircuit>()?;
    m.add_class::<ImmutableQuantumCircuit>()?;
    Ok(m)
}
