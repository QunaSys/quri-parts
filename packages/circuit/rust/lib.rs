use pyo3::prelude::*;
use pyo3_commonize::commonize;

pub mod circuit;
pub mod circuit_parametric;
pub mod gate;
pub mod gates;
pub mod noise;
pub mod parameter;

#[pymodule]
fn quri_parts_circuit_rs(py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    gate::commonize_quantum_gate(py)?;
    commonize::<circuit::ImmutableQuantumCircuit>(py)?;
    commonize::<circuit::QuantumCircuit>(py)?;
    commonize::<gate::ParametricQuantumGate>(py)?;
    commonize::<circuit_parametric::ImmutableParametricQuantumCircuit>(py)?;
    commonize::<circuit_parametric::ImmutableBoundParametricQuantumCircuit>(py)?;
    commonize::<circuit_parametric::ParametricQuantumCircuit>(py)?;
    commonize::<parameter::Parameter>(py)?;
    commonize::<noise::noise_model::NoiseModel>(py)?;
    m.add_submodule(&gate::py_module(py)?)?;
    m.add_submodule(&gates::py_module(py)?)?;
    m.add_submodule(&circuit::py_module(py)?)?;
    m.add_submodule(&circuit_parametric::py_module(py)?)?;
    m.add_submodule(&parameter::py_module(py)?)?;
    m.add_submodule(&noise::py_module(py)?)?;
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub enum MaybeUnbound {
    Bound(f64),
    Unbound(parameter::Wrapper),
}
