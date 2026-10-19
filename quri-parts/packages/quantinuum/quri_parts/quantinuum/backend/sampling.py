# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A :class:`~quri_parts.backend.SamplingBackend` that submits circuits to
Quantinuum systems (H-Series, Reimei), their emulators and Nexus-hosted
simulators through the Quantinuum Nexus client (``qnexus``).

Quantinuum Helios is not supported yet: its Nexus endpoints accept only QIR or
HUGR programs, not pytket circuits. Helios device names and
:class:`qnexus.HeliosConfig` are rejected with a :class:`ValueError`.

Job flow of :meth:`QuantinuumSamplingBackend.sample`:

1. The QURI Parts circuit is transpiled (optional), converted to a pytket
   :class:`~pytket.circuit.Circuit` and all qubits are measured.
2. The circuit is uploaded to a Nexus project.
3. (Optional, default) The circuit is compiled on Nexus for the target backend
   with Quantinuum's TKET compiler. This step blocks until the compile job
   finishes, which usually takes seconds.
4. An execute job is submitted. :meth:`QuantinuumSamplingJob.result` waits for
   the job and converts the measurement results.

Authentication is handled by ``qnexus``: run ``qnexus.login()`` (browser) or
``qnexus.auth.login_no_interaction(email, password)`` (headless) once; tokens
are stored under ``~/.qnx/auth``. The backend itself never logs in. If no
valid login is found, every call that talks to Nexus raises a
:class:`~quri_parts.backend.BackendError` that says how to log in.
"""

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Optional, Union
from uuid import uuid4

import qnexus as qnx
from pytket.backends.backendresult import BackendResult
from pytket.circuit import Circuit as TketCircuit
from qnexus.exceptions import AuthenticationError
from qnexus.models.job_status import JobStatusEnum
from qnexus.models.references import CircuitRef, ExecuteJobRef, ProjectRef
from typing_extensions import TypeAlias

from quri_parts.backend import (
    BackendError,
    SamplingBackend,
    SamplingCounts,
    SamplingJob,
    SamplingResult,
)
from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspiler
from quri_parts.tket.circuit import convert_circuit

#: A function converting a QURI Parts circuit to a pytket circuit.
TketCircuitConverter: TypeAlias = Callable[[ImmutableQuantumCircuit], TketCircuit]

#: Maximum number of shots per job item accepted by Quantinuum-hosted devices
#: (H-Series, Reimei). Used as the default ``max_shots`` of
#: :class:`QuantinuumSamplingBackend` for :class:`qnexus.QuantinuumConfig`.
QUANTINUUM_MAX_SHOTS_PER_ITEM = 10000

# The reason (Helios accepts only QIR or HUGR programs, not pytket circuits)
# is documented in the module docstring; users only need to know what to use.
_HELIOS_UNSUPPORTED = (
    "Quantinuum Helios is not supported by QuantinuumSamplingBackend yet. "
    'Use an H-Series device (e.g. "H2-1", "H2-1E", "H2-1SC") or a '
    "Nexus-hosted simulator (e.g. qnexus.AerConfig()) instead."
)


def _is_helios(backend_config: qnx.BackendConfig) -> bool:
    """Return True if the config targets a Helios system."""
    if isinstance(backend_config, qnx.HeliosConfig):
        return True
    device_name = getattr(backend_config, "device_name", "")
    return isinstance(device_name, str) and device_name.lower().startswith("helios")


_LOGIN_HINT = (
    "not logged in to Quantinuum Nexus. Log in once with qnexus.login() "
    "(opens a browser) or qnexus.auth.login_no_interaction(email, password) "
    "(headless); the tokens are stored under ~/.qnx/auth."
)


@contextmanager
def _nexus_errors(what: str) -> Iterator[None]:
    """Turn exceptions raised by ``qnexus`` inside the block into
    :class:`~quri_parts.backend.BackendError`.

    An :class:`qnexus.exceptions.AuthenticationError` becomes a message that
    says how to log in. Any other exception keeps its type and message in the
    :class:`~quri_parts.backend.BackendError` message so that the real cause
    (wrong device name, network failure, ...) stays visible. A
    :class:`~quri_parts.backend.BackendError` raised inside the block is
    re-raised unchanged.

    Args:
        what: Prefix of the error message, e.g. "Failed to submit ...".
    """
    try:
        yield
    except BackendError:
        raise
    except AuthenticationError as e:
        raise BackendError(f"{what}: {_LOGIN_HINT}") from e
    except Exception as e:
        raise BackendError(f"{what}: {type(e).__name__}: {e}") from e


def backend_result_to_sampling_counts(result: BackendResult) -> SamplingCounts:
    """Convert a pytket :class:`~pytket.backends.backendresult.BackendResult`
    into :class:`~quri_parts.backend.SamplingCounts`.

    Classical bits are ordered by register name and index. The ``i``-th
    bit in that order becomes bit ``i`` of the integer key, so for a
    circuit measured with :meth:`pytket.circuit.Circuit.measure_all` the
    key encodes qubit ``i`` at bit ``i`` (the QURI Parts convention).
    """
    bits = sorted(result.c_bits)
    counts = result.get_counts(cbits=bits)
    converted: Counter[int] = Counter()
    for readout, n in counts.items():
        key = 0
        for i, b in enumerate(readout):
            if b:
                key |= 1 << i
        converted[key] += int(n)
    return converted


def distribute_shots(shots: int, max_shots: Optional[int]) -> list[int]:
    """Split ``shots`` into chunks of at most ``max_shots`` shots.

    Returns ``[shots]`` when ``max_shots`` is ``None`` or ``shots`` does
    not exceed it.
    """
    if shots < 1:
        raise ValueError("shots should be a positive integer.")
    if max_shots is None or shots <= max_shots:
        return [shots]
    if max_shots < 1:
        raise ValueError("max_shots should be a positive integer.")
    dist = [max_shots] * (shots // max_shots)
    remaining = shots % max_shots
    if remaining > 0:
        dist.append(remaining)
    return dist


def default_backend_config(device_name: str) -> qnx.QuantinuumConfig:
    """Create the Nexus backend config for a Quantinuum device name.

    Device names (e.g. ``H2-1``, ``H2-1E``, ``H2-1SC``, ``reimei``) map to
    :class:`qnexus.QuantinuumConfig`.

    Raises:
        ValueError: If the name refers to Helios (``Helios-1``, ``Helios-1E``,
            ...), which this backend does not support yet.
    """
    if device_name.lower().startswith("helios"):
        raise ValueError(f"{_HELIOS_UNSUPPORTED} (got device_name={device_name!r})")
    return qnx.QuantinuumConfig(device_name=device_name)


class QuantinuumSamplingResult(SamplingResult):
    """A result of a sampling job executed through Quantinuum Nexus.

    Args:
        backend_results: pytket results of the job items. Counts of all items
            are merged.
    """

    def __init__(self, backend_results: Sequence[BackendResult]):
        """Merge the counts of the given pytket results."""
        self._backend_results = tuple(backend_results)
        total: Counter[int] = Counter()
        for r in self._backend_results:
            total.update(backend_result_to_sampling_counts(r))
        self._counts: SamplingCounts = total

    @property
    def counts(self) -> SamplingCounts:
        """Measurement counts merged over all job items."""
        return self._counts

    @property
    def backend_results(self) -> Sequence[BackendResult]:
        """Raw pytket results, one per job item."""
        return self._backend_results


class QuantinuumSamplingJob(SamplingJob):
    """A sampling job submitted to Quantinuum Nexus.

    Args:
        job_ref: The Nexus execute job reference.
        timeout: Seconds to wait for the job in :meth:`result`. ``None`` falls
            back to the ``QNEXUS_DEFAULT_JOB_TIMEOUT_SECONDS`` environment
            variable if set, otherwise waits without limit.
    """

    def __init__(self, job_ref: ExecuteJobRef, timeout: Optional[float] = None):
        """Wrap a Nexus execute job reference."""
        self._job_ref = job_ref
        self._timeout = timeout

    @classmethod
    def from_id(
        cls, job_id: str, timeout: Optional[float] = None
    ) -> "QuantinuumSamplingJob":
        """Recreate a job object from a Nexus execute job id.

        This allows retrieving results in a different process, e.g.
        after an HPC batch job ends.

        Raises:
            BackendError: If the job cannot be fetched from Nexus (including
                when not logged in) or is not an execute job.
        """
        with _nexus_errors(f"Failed to get Nexus job {job_id}"):
            job_ref = qnx.jobs.get(id=job_id)
        if not isinstance(job_ref, ExecuteJobRef):
            raise BackendError(f"Nexus job {job_id} is not an execute job.")
        return cls(job_ref, timeout)

    @property
    def job_ref(self) -> ExecuteJobRef:
        """The Nexus execute job reference."""
        return self._job_ref

    @property
    def id(self) -> str:
        """The Nexus job id, usable with :meth:`from_id`."""
        return str(self._job_ref.id)

    def status(self) -> JobStatusEnum:
        """Query the current status of the job from Nexus.

        Raises:
            BackendError: If the status cannot be fetched from Nexus
                (including when not logged in).
        """
        with _nexus_errors(f"Failed to get the status of Nexus job {self.id}"):
            return qnx.jobs.status(self._job_ref).status

    def cancel(self) -> None:
        """Request cancellation of the job.

        Raises:
            BackendError: If the request fails (including when not logged in).
        """
        with _nexus_errors(f"Failed to cancel Nexus job {self.id}"):
            qnx.jobs.cancel(self._job_ref)

    def result(self) -> QuantinuumSamplingResult:
        """Wait for the job to finish and return the merged sampling result.

        Raises:
            BackendError: If the job fails, is cancelled, times out, cannot be
                reached (including when not logged in) or returns results
                other than pytket circuit results.
        """
        with _nexus_errors(f"Failed to get the result of Nexus job {self.id}"):
            qnx.jobs.wait_for(self._job_ref, timeout=self._timeout)
            result_refs = qnx.jobs.results(self._job_ref)
            results = []
            for ref in result_refs:
                download = getattr(ref, "download_result", None)
                if download is None:
                    raise BackendError(f"Nexus job {self.id} has incomplete items.")
                results.append(download())
        backend_results: list[BackendResult] = []
        for r in results:
            if not isinstance(r, BackendResult):
                raise BackendError(
                    "Only pytket circuit results are supported, "
                    f"got {type(r).__name__}."
                )
            backend_results.append(r)
        return QuantinuumSamplingResult(backend_results)


class QuantinuumSamplingBackend(SamplingBackend):
    """A sampling backend for Quantinuum devices accessed through Nexus.

    Examples:
        >>> import qnexus as qnx
        >>> qnx.login()  # once per machine; tokens are stored in ~/.qnx/auth
        >>> backend = QuantinuumSamplingBackend("H2-1E")
        >>> job = backend.sample(circuit, shots=1000)
        >>> job.result().counts

        A Nexus-hosted simulator (no HQC needed) can be used by passing the
        config explicitly:

        >>> backend = QuantinuumSamplingBackend(backend_config=qnx.AerConfig())

    Args:
        device_name: Name of a Quantinuum device, emulator or syntax checker,
            e.g. ``"H2-1"``, ``"H2-1E"``, ``"H2-1SC"`` or ``"reimei"``. Helios
            names are rejected (not supported yet). See
            :func:`default_backend_config`.
        backend_config: A ``qnexus`` backend config. Overrides ``device_name``
            and allows any Nexus backend (``QuantinuumConfig``, ``AerConfig``,
            ...) except ``HeliosConfig``.
        project: The Nexus project (name or ``ProjectRef``) that holds the
            uploaded circuits and jobs. A project with the given name is
            created if it does not exist.
        circuit_converter: A function converting a QURI Parts circuit to a
            pytket circuit.
        circuit_transpiler: A QURI Parts transpiler applied to the circuit
            before conversion.
        compile_on_nexus: If True, the uploaded circuit is compiled on Nexus
            for the target backend before execution. If False, the circuit is
            executed as uploaded; it must then already satisfy the backend's
            gate set (e.g. via ``circuit_transpiler``).
        optimisation_level: TKET optimisation level (0-3) used by the Nexus
            compile job.
        max_shots: A request with more shots is split into several job items
            of at most ``max_shots`` shots inside a single Nexus job, and the
            counts are merged in the result. Defaults to
            :data:`QUANTINUUM_MAX_SHOTS_PER_ITEM` (10000, the per-item limit of
            Quantinuum-hosted devices) when ``backend_config`` is a
            ``QuantinuumConfig``, and to ``None`` (no split) otherwise.
        compile_timeout: Seconds to wait for the compile job. ``None`` means
            no limit (or ``QNEXUS_DEFAULT_JOB_TIMEOUT_SECONDS`` if set).
        result_timeout: Seconds to wait for the execute job in
            :meth:`QuantinuumSamplingJob.result`.
        job_name_prefix: Prefix of the circuit and job names shown on Nexus.
        compile_kwargs: Additional keyword arguments for
            :func:`qnexus.start_compile_job`.
        execute_kwargs: Additional keyword arguments for
            :func:`qnexus.start_execute_job`, e.g. ``max_cost`` (HQC limit),
            ``credential_name`` or ``user_group``.
    """

    def __init__(
        self,
        device_name: Optional[str] = None,
        *,
        backend_config: Optional[qnx.BackendConfig] = None,
        project: Union[str, ProjectRef] = "quri-parts",
        circuit_converter: TketCircuitConverter = convert_circuit,
        circuit_transpiler: Optional[CircuitTranspiler] = None,
        compile_on_nexus: bool = True,
        optimisation_level: int = 2,
        max_shots: Optional[int] = None,
        compile_timeout: Optional[float] = None,
        result_timeout: Optional[float] = None,
        job_name_prefix: str = "quri-parts",
        compile_kwargs: Mapping[str, Any] = {},
        execute_kwargs: Mapping[str, Any] = {},
    ):
        """Create the backend.

        No request is sent to Nexus until
        :meth:`sample` or :attr:`project` is used.

        Raises:
            ValueError: If neither ``device_name`` nor ``backend_config`` is
                given, or if the target is Helios (not supported yet).
        """
        if backend_config is None:
            if device_name is None:
                raise ValueError("Either device_name or backend_config is required.")
            backend_config = default_backend_config(device_name)
        if _is_helios(backend_config):
            target = getattr(backend_config, "system_name", None) or getattr(
                backend_config, "device_name", None
            )
            raise ValueError(
                f"{_HELIOS_UNSUPPORTED} "
                f"(got {type(backend_config).__name__} for {target!r})"
            )
        self._backend_config = backend_config
        if max_shots is None and isinstance(backend_config, qnx.QuantinuumConfig):
            max_shots = QUANTINUUM_MAX_SHOTS_PER_ITEM

        self._project_ref: Optional[ProjectRef] = (
            project if isinstance(project, ProjectRef) else None
        )
        self._project_name: Optional[str] = (
            project if isinstance(project, str) else None
        )

        self._circuit_converter = circuit_converter
        self._circuit_transpiler = circuit_transpiler
        self._compile_on_nexus = compile_on_nexus
        self._optimisation_level = optimisation_level
        self._max_shots = max_shots
        self._compile_timeout = compile_timeout
        self._result_timeout = result_timeout
        self._job_name_prefix = job_name_prefix
        self._compile_kwargs = dict(compile_kwargs)
        self._execute_kwargs = dict(execute_kwargs)

    @property
    def backend_config(self) -> qnx.BackendConfig:
        """The Nexus backend config used for compile and execute jobs."""
        return self._backend_config

    @property
    def max_shots(self) -> Optional[int]:
        """Maximum shots per job item (``None`` means no split)."""
        return self._max_shots

    @property
    def project(self) -> ProjectRef:
        """The Nexus project.

        Resolved (and created if needed) on first access.

        Raises:
            BackendError: If the project cannot be fetched or created on Nexus
                (including when not logged in).
        """
        if self._project_ref is None:
            assert self._project_name is not None
            name = self._project_name
            with _nexus_errors(f"Failed to get or create Nexus project {name!r}"):
                self._project_ref = qnx.projects.get_or_create(name=name)
        return self._project_ref

    def _to_tket_circuit(self, circuit: ImmutableQuantumCircuit) -> TketCircuit:
        if self._circuit_transpiler is not None:
            circuit = self._circuit_transpiler(circuit)
        tket_circuit = self._circuit_converter(circuit)
        tket_circuit.measure_all()
        return tket_circuit

    def _compile(
        self, circuit_ref: CircuitRef, job_name: str, project: ProjectRef
    ) -> CircuitRef:
        compile_ref = qnx.start_compile_job(
            programs=[circuit_ref],
            backend_config=self._backend_config,
            name=f"{job_name}-compile",
            project=project,
            optimisation_level=self._optimisation_level,
            **self._compile_kwargs,
        )
        qnx.jobs.wait_for(compile_ref, timeout=self._compile_timeout)
        compile_results = qnx.jobs.results(compile_ref)
        if len(compile_results) != 1:
            raise BackendError(
                f"Expected 1 compile result, got {len(compile_results)}."
            )
        first = compile_results[0]
        if not hasattr(first, "get_output"):
            raise BackendError("Nexus compile job did not complete.")
        output: CircuitRef = first.get_output()
        return output

    def sample(self, circuit: ImmutableQuantumCircuit, shots: int) -> SamplingJob:
        """Upload, compile (optional) and submit the circuit as a Nexus execute
        job. Returns as soon as the execute job is submitted.

        Raises:
            ValueError: If ``shots`` is not positive.
            BackendError: If any step on Nexus fails, including when not
                logged in.
        """
        shot_dist = distribute_shots(shots, self._max_shots)
        tket_circuit = self._to_tket_circuit(circuit)
        job_name = f"{self._job_name_prefix}-{uuid4().hex[:8]}"

        with _nexus_errors("Failed to submit a sampling job to Quantinuum Nexus"):
            project = self.project
            circuit_ref = qnx.circuits.upload(
                circuit=tket_circuit, name=f"{job_name}-circuit", project=project
            )
            program = (
                self._compile(circuit_ref, job_name, project)
                if self._compile_on_nexus
                else circuit_ref
            )
            execute_ref = qnx.start_execute_job(
                programs=[program] * len(shot_dist),
                n_shots=shot_dist,
                backend_config=self._backend_config,
                name=f"{job_name}-execute",
                project=project,
                **self._execute_kwargs,
            )

        return QuantinuumSamplingJob(execute_ref, timeout=self._result_timeout)
