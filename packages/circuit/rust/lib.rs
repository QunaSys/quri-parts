use pyo3::prelude::*;

mod circuit;
mod gate;
mod gates;

#[pymodule]
fn quri_parts_circuit_rs(py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_submodule(&gate::py_module(py)?)?;
    m.add_submodule(&gates::py_module(py)?)?;
    m.add_submodule(&circuit::py_module(py)?)?;
    Ok(())
}
