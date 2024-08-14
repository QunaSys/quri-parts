use pyo3::prelude::*;

mod circuit;
mod circuit_parametric;
mod gate;
pub mod gates;
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

#[derive(Clone, Debug, PartialEq)]
pub enum MaybeUnbound {
    Bound(f64),
    Unbound(parameter::Wrapper),
}
