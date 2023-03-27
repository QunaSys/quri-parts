import os

from juliacall import Main as jl


def ensure_itensor_loaded(calling_file_path: str) -> None:
    abs_dir = os.path.dirname(os.path.abspath(calling_file_path))
    library_path = os.path.join(abs_dir, "library.jl")
    include_statement = 'include("' + library_path + '")'
    jl.seval(include_statement)
