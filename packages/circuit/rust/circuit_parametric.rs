use crate::circuit::ImmutableQuantumCircuit;
use crate::gate::QuantumGate;
use crate::parameter::{Parameter, Wrapper};
use crate::MaybeUnbound;
use pyo3::prelude::*;
use pyo3::types::PySequence;
use quri_parts::BasicBlock;
use std::collections::HashMap;
use std::sync::RwLock;

#[pyclass(
    subclass,
    frozen,
    eq,
    module = "quri_parts.circuit.rust.circuit_parametric"
)]
#[derive(Debug)]
pub struct ImmutableParametricQuantumCircuit {
    #[pyo3(get)]
    qubit_count: usize,
    #[pyo3(get)]
    cbit_count: usize,
    gates: Box<RwLock<BasicBlock<QuantumGate<MaybeUnbound>>>>,
    depth_cache: Box<RwLock<Option<usize>>>,
}

impl PartialEq for ImmutableParametricQuantumCircuit {
    fn eq(&self, other: &Self) -> bool {
        if self.qubit_count != other.qubit_count || self.cbit_count != other.cbit_count {
            return false;
        }
        let lhs = self.gates.read().unwrap();
        let rhs = other.gates.read().unwrap();
        *lhs == *rhs
    }
}

impl Clone for ImmutableParametricQuantumCircuit {
    fn clone(&self) -> Self {
        let gates = self.gates.read().unwrap().clone();
        let depth_cache = self.depth_cache.read().unwrap().clone();
        ImmutableParametricQuantumCircuit {
            qubit_count: self.qubit_count,
            cbit_count: self.cbit_count,
            gates: Box::new(RwLock::new(gates)),
            depth_cache: Box::new(RwLock::new(depth_cache)),
        }
    }
}

#[pymethods]
impl ImmutableParametricQuantumCircuit {
    #[new]
    #[pyo3(signature = (circuit))]
    #[pyo3(text_signature = "(circuit: ImmutableParametricQuantumCircuit)")]
    fn py_new(circuit: &Self) -> Self {
        let gates = circuit.gates.read().unwrap().clone();
        ImmutableParametricQuantumCircuit {
            qubit_count: circuit.qubit_count,
            cbit_count: circuit.cbit_count,
            gates: Box::new(RwLock::new(gates)),
            depth_cache: Box::new(RwLock::new(None)),
        }
    }

    #[getter]
    fn get_gates(slf: &Bound<'_, Self>) -> PyResult<Vec<Py<PyAny>>> {
        let gates = slf.get().gates.read().unwrap();
        gates
            .0
            .iter()
            .map(|g| match g.clone().instantiate()? {
                Ok(g) => Ok(QuantumGate::into_py(g, slf.py())),
                Err(g) => Ok(Py::new(slf.py(), g.0)?.into_any()),
            })
            .collect()
    }

    #[getter]
    fn get_gates_and_params(slf: &Bound<'_, Self>) -> PyResult<Vec<(Py<PyAny>, Option<Wrapper>)>> {
        let gates = slf.get().gates.read().unwrap();
        gates
            .0
            .iter()
            .map(|g| match g.clone().instantiate()? {
                Ok(g) => Ok((QuantumGate::into_py(g, slf.py()), None)),
                Err(g) => Ok((Py::new(slf.py(), g.0)?.into_any(), Some(g.1))),
            })
            .collect()
    }

    /// For compatibility with
    /// quri_parts.circuit.circuit_parametric.ImmutableBoundParametricQuantumCircuit
    #[allow(non_snake_case)]
    #[getter]
    fn get__gates(slf: &Bound<'_, Self>) -> PyResult<Vec<(Py<PyAny>, Option<Wrapper>)>> {
        Self::get_gates_and_params(slf)
    }

    #[getter]
    fn get_has_trivial_parameter_mapping(&self) -> bool {
        true
    }

    #[getter]
    fn get_depth(&self) -> usize {
        if let Some(depth) = *self.depth_cache.read().unwrap() {
            depth
        } else {
            todo!()
        }
    }

    #[getter]
    fn get_param_mapping<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = Self::get__params(slf);
        slf.py()
            .import_bound("quri_parts.circuit.parameter_mapping")?
            .getattr("LinearParameterMapping")?
            .call1((
                params.clone(),
                params.clone(),
                params
                    .iter()
                    .map(|p| (p.clone(), p.clone()))
                    .collect::<HashMap<_, _>>(),
            ))
    }

    fn primitive_circuit(&self) -> ImmutableParametricQuantumCircuit {
        self.freeze()
    }

    fn get_mutable_copy(slf: &Bound<'_, Self>) -> PyResult<Py<ParametricQuantumCircuit>> {
        Py::new(slf.py(), (ParametricQuantumCircuit(), slf.get().clone()))
    }

    #[pyo3(
        text_signature = "(gates: ImmutableQuantumCircuit | ImmutableParametricQuantumCircuit | Sequence[QuantumGate])"
    )]
    fn combine(
        slf: &Bound<'_, Self>,
        gates: &Bound<'_, PyAny>,
    ) -> PyResult<Py<ParametricQuantumCircuit>> {
        let ret = Self::get_mutable_copy(slf)?;
        ParametricQuantumCircuit::extend(ret.bind(slf.py()), gates)?;
        Ok(ret)
    }

    fn freeze(&self) -> ImmutableParametricQuantumCircuit {
        self.clone()
    }

    fn bind_parameters(slf: &Bound<'_, Self>, params: Vec<f64>) -> PyResult<PyObject> {
        let param_list = Self::get__params(slf);
        if param_list.len() != params.len() {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Passed value count ({}) does not match parameter count ({}).",
                params.len(),
                param_list.len()
            )))
        } else {
            Ok(slf
                .py()
                .import_bound("quri_parts.circuit.circuit_parametric")?
                .getattr("ImmutableBoundParametricQuantumCircuit")?
                .call1((
                    slf.as_any(),
                    param_list
                        .into_iter()
                        .zip(params)
                        .collect::<HashMap<_, _>>(),
                ))?
                .unbind())
        }
    }

    #[allow(non_snake_case)]
    #[getter]
    fn get__params(slf: &Bound<'_, Self>) -> Vec<Wrapper> {
        let gates = slf.get().gates.read().unwrap();
        gates
            .0
            .iter()
            .map(|g| g.get_params())
            .flatten()
            .filter_map(|g| {
                if let MaybeUnbound::Unbound(p) = g {
                    Some(p)
                } else {
                    None
                }
            })
            .collect()
    }

    #[getter]
    fn parameter_count(slf: &Bound<'_, Self>) -> usize {
        Self::get__params(slf).len()
    }

    #[pyo3(name = "__add__")]
    fn py_add(
        slf: &Bound<'_, Self>,
        gates: &Bound<'_, PyAny>,
    ) -> PyResult<Py<ParametricQuantumCircuit>> {
        Self::combine(slf, gates)
    }

    #[pyo3(name = "__radd__")]
    fn py_radd(
        slf: &Bound<'_, Self>,
        gates: &Bound<'_, PyAny>,
    ) -> PyResult<Py<ParametricQuantumCircuit>> {
        let ret = Py::new(
            slf.py(),
            (
                ParametricQuantumCircuit(),
                Self {
                    qubit_count: slf.get().qubit_count,
                    cbit_count: 0,
                    gates: Box::new(RwLock::new(BasicBlock(Vec::new()))),
                    depth_cache: Box::new(RwLock::new(None)),
                },
            ),
        )?;
        ParametricQuantumCircuit::extend(ret.bind(slf.py()), gates)?;
        ParametricQuantumCircuit::extend(ret.bind(slf.py()), slf.as_any())?;
        Ok(ret)
    }
}

#[pyclass(
    extends=ImmutableParametricQuantumCircuit,
    subclass,
    eq,
    module = "quri_parts.circuit.rust.circuit_parametric"
)]
#[derive(Clone, Debug, PartialEq)]
pub struct ParametricQuantumCircuit();

impl ParametricQuantumCircuit {
    fn add_gate_inner(
        slf: PyRef<'_, Self>,
        gate: QuantumGate<MaybeUnbound>,
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
                gates.insert(gate_index, gate);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyIndexError::new_err(""))
            }
        } else {
            gates.push(gate);
            Ok(())
        }
    }
}

#[pymethods]
impl ParametricQuantumCircuit {
    #[new]
    #[pyo3(signature = (qubit_count, cbit_count=0))]
    #[pyo3(text_signature = "(qubit_count: int, cbit_count: int = 0)")]
    fn py_new(qubit_count: usize, cbit_count: usize) -> (Self, ImmutableParametricQuantumCircuit) {
        let base = ImmutableParametricQuantumCircuit {
            qubit_count,
            cbit_count,
            gates: Box::new(RwLock::new(BasicBlock(vec![]))),
            depth_cache: Box::new(RwLock::new(None)),
        };
        (ParametricQuantumCircuit(), base)
    }

    #[pyo3(signature = (gate, gate_index = None))]
    #[pyo3(text_signature = "(gate: QuantumGate, gate_index: Optional[int])")]
    fn add_gate(
        slf: PyRef<'_, Self>,
        gate: QuantumGate,
        gate_index: Option<usize>,
    ) -> PyResult<()> {
        let gate = gate.map_param(|p| MaybeUnbound::Bound(p));
        Self::add_gate_inner(slf, gate, gate_index)
    }

    #[pyo3(
        text_signature = "(gates: ImmutableParametricQuantumCircuit | ImmutableQuantumCircuit | Sequence[QuantumGate])"
    )]
    fn extend(slf: &Bound<'_, Self>, gates: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(other) = gates.downcast::<ImmutableQuantumCircuit>() {
            for gate in other.get().gates.read().unwrap().0.iter() {
                Self::add_gate(slf.borrow(), gate.clone(), None)?;
            }
            Ok(())
        } else if let Ok(other) = gates.downcast::<ImmutableParametricQuantumCircuit>() {
            for gate in other.get().gates.read().unwrap().0.iter() {
                Self::add_gate_inner(slf.borrow(), gate.clone(), None)?;
            }
            Ok(())
        } else if let Ok(other) = gates.downcast::<PySequence>() {
            for i in 0..other.len()? {
                let item = other.get_item(i)?;
                let gate = QuantumGate::extract_bound(&item)?;
                Self::add_gate(slf.borrow(), gate, None)?;
            }
            Ok(())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Bad argument `gates`",
            ))
        }
    }

    #[pyo3(name = "__iadd__")]
    fn py_iadd(slf: &Bound<'_, Self>, gates: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::extend(slf, gates)
    }

    #[allow(non_snake_case)]
    fn add_ParametricRX_gate(slf: Bound<'_, Self>, qubit_index: usize) -> PyResult<Wrapper> {
        let param = Py::new(slf.py(), Parameter::new(String::new()))?;
        let pw = Wrapper(param);
        Self::add_gate_inner(
            slf.borrow(),
            QuantumGate::RX(qubit_index, MaybeUnbound::Unbound(pw.clone())),
            None,
        )?;
        Ok(pw)
    }

    #[allow(non_snake_case)]
    fn add_ParametricRY_gate(slf: Bound<'_, Self>, qubit_index: usize) -> PyResult<Wrapper> {
        let param = Py::new(slf.py(), Parameter::new(String::new()))?;
        let pw = Wrapper(param);
        Self::add_gate_inner(
            slf.borrow(),
            QuantumGate::RY(qubit_index, MaybeUnbound::Unbound(pw.clone())),
            None,
        )?;
        Ok(pw)
    }

    #[allow(non_snake_case)]
    fn add_ParametricRZ_gate(slf: Bound<'_, Self>, qubit_index: usize) -> PyResult<Wrapper> {
        let param = Py::new(slf.py(), Parameter::new(String::new()))?;
        let pw = Wrapper(param);
        Self::add_gate_inner(
            slf.borrow(),
            QuantumGate::RZ(qubit_index, MaybeUnbound::Unbound(pw.clone())),
            None,
        )?;
        Ok(pw)
    }
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "circuit_parametric")?;
    m.add_class::<ImmutableParametricQuantumCircuit>()?;
    m.add_class::<ParametricQuantumCircuit>()?;
    Ok(m)
}
