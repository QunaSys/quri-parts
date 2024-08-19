use pyo3::prelude::*;

#[derive(Clone, Debug)]
pub struct Wrapper(pub Py<Parameter>);

impl core::cmp::PartialEq for Wrapper {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() as usize == other.0.as_ptr() as usize
    }
}

impl core::cmp::Eq for Wrapper {}

impl core::hash::Hash for Wrapper {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.as_ptr() as u64)
    }
}

impl pyo3::conversion::IntoPy<PyObject> for Wrapper {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.0.into_py(py)
    }
}

#[pyclass]
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Parameter {
    #[pyo3(get, set)]
    name: String,
}

impl core::fmt::Display for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

#[pymethods]
impl Parameter {
    #[new]
    #[pyo3(signature = (name = String::new()))]
    pub fn new(name: String) -> Self {
        Parameter { name }
    }

    #[pyo3(name = "__eq__")]
    fn py_eq(slf: Py<Self>, other: PyObject) -> bool {
        slf.as_ptr() == other.as_ptr()
    }

    #[pyo3(name = "__repr__")]
    fn py_repr(&self) -> String {
        format!("Parameter(name={})", &self.name)
    }

    #[pyo3(name = "__hash__")]
    fn py_hash(slf: Py<Self>) -> usize {
        slf.as_ptr() as usize
    }

    #[pyo3(name = "__reduce__")]
    fn py_reduce(slf: Bound<'_, Self>) -> (PyObject, (String,)) {
        (
            Bound::new(
                slf.py(),
                Self {
                    name: String::new(),
                },
            )
            .unwrap()
            .getattr("__class__")
            .unwrap()
            .unbind(),
            (slf.borrow().name.clone(),),
        )
    }
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "parameter")?;
    m.add_class::<Parameter>()?;
    Ok(m)
}