use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Parameter {
    #[pyo3(get, set)]
    name: String,
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
