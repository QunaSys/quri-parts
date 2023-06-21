from collections.abc import Callable, Hashable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from functools import update_wrapper
import logging
import threading
from typing import Any, Generic, NamedTuple, Optional, TypeVar

from typing_extensions import Concatenate, ParamSpec, TypeAlias

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


class RecordableFunctionId(NamedTuple):
    module: str
    qualname: str
    param: Hashable

    def to_str(self, full: bool = True) -> str:
        if full:
            base = f"{self.module}.{self.qualname}"
        else:
            base = self.qualname
        if self.param:
            return f"{base}<{str(self.param)}>"
        else:
            return base

    def __str__(self) -> str:
        return self.to_str()


class RecordableFunction(Generic[P, R]):
    def __init__(self, f: Callable[P, R], id: RecordableFunctionId):
        self._f = f
        self._id = id

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._f(*args, **kwargs)

    @property
    def id(self) -> RecordableFunctionId:
        return self._id


class RecordLevel(IntEnum):
    INFO = 20
    DEBUG = 10

    def __str__(self) -> str:
        return self.name


INFO = RecordLevel.INFO
DEBUG = RecordLevel.DEBUG

_DEFAULT_LOGGER_NAME = f"{logging.Logger.root.name}.quri_parts_recording"

_RecKey: TypeAlias = Hashable
_RecValue: TypeAlias = Any
_RecData: TypeAlias = tuple[_RecKey, _RecValue]


class Recorder:
    def __init__(self, fid: RecordableFunctionId) -> None:
        self._func_id = fid

    @contextmanager
    def start_func(self) -> Iterator[None]:
        for session in _active_sessions:
            session.enter_func(self._func_id)
        yield
        for session in _active_sessions:
            session.exit_func(self._func_id)

    def record(self, level: RecordLevel, key: _RecKey, value: _RecValue) -> None:
        for session in _active_sessions:
            if session.is_enabled_for(level, self._func_id):
                session.handler(level, self._func_id, key, value)

    def debug(self, key: _RecKey, value: _RecValue) -> None:
        self.record(DEBUG, key, value)

    def info(self, key: _RecKey, value: _RecValue) -> None:
        self.record(INFO, key, value)

    def is_enabled_for(self, level: RecordLevel) -> bool:
        return any(
            session.is_enabled_for(level, self._func_id) for session in _active_sessions
        )


_recorders: dict[RecordableFunctionId, Recorder] = {}


def _get_recorder(fid: RecordableFunctionId) -> Recorder:
    if fid in _recorders:
        return _recorders[fid]
    else:
        recorder = Recorder(fid)
        _recorders[fid] = recorder
        return recorder


def recordable(f: Callable[Concatenate[Recorder, P], R]) -> RecordableFunction[P, R]:
    param = ()  # TODO
    f_id = RecordableFunctionId(f.__module__, f.__qualname__, param)

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        recorder = _get_recorder(f_id)
        with recorder.start_func():
            return f(recorder, *args, **kwargs)

    rf = RecordableFunction(wrapper, f_id)

    return update_wrapper(rf, f)


@dataclass
class RecordEntry:
    level: RecordLevel
    func_id: RecordableFunctionId
    data: _RecData

    def __str__(self) -> str:
        return f"{self.level}:{self.func_id}:{self.data}"


_group_id = threading.local()
_group_id.current = 0


def _next_group_id() -> int:
    id: int = _group_id.current
    _group_id.current += 1
    return id


@dataclass
class RecordGroup:
    func_id: RecordableFunctionId
    entries: list[RecordEntry]
    id: int = field(default_factory=_next_group_id)

    def add_entry(self, entry: RecordEntry) -> None:
        self.entries.append(entry)

    def __str__(self) -> str:
        return (
            f"""RecordGroup(
  func_id: {self.func_id},
  entries: [
"""
            + "\n".join(f"    {entry}," for entry in self.entries)
            + """
  ]
)"""
        )


class RecordSet:
    def __init__(self) -> None:
        self._history: list[RecordGroup] = []

    def add_group(self, fid: RecordableFunctionId) -> RecordGroup:
        group = RecordGroup(fid, [])
        self._history.append(group)
        return group

    def remove_last_group(self) -> None:
        self._history.pop()

    def get_history(self, func: RecordableFunction[P, R]) -> Iterable[RecordGroup]:
        return filter(lambda g: g.func_id == func.id, self._history)


def _to_logging_level(level: RecordLevel) -> int:
    # Each RecordLevel has the same value as a logging level at least at the moment
    return level.value


class RecordSession:
    def __init__(self) -> None:
        self._levels: dict[RecordableFunctionId, RecordLevel] = {}
        self._record_set = RecordSet()
        self._group_stack: list[RecordGroup] = []
        self._loggers: set[logging.Logger] = set()

    def set_level(self, level: RecordLevel, func: RecordableFunction[P, R]) -> None:
        self._levels[func.id] = level

    def is_enabled_for(self, level: RecordLevel, fid: RecordableFunctionId) -> bool:
        return fid in self._levels and level >= self._levels[fid]

    def handler(
        self,
        level: RecordLevel,
        fid: RecordableFunctionId,
        key: _RecKey,
        value: _RecValue,
    ) -> None:
        entry = RecordEntry(level, fid, (key, value))
        group = self._group_stack[-1]
        group.add_entry(entry)
        self._log(entry, group)

    def _log(self, entry: RecordEntry, group: RecordGroup) -> None:
        log_level = _to_logging_level(entry.level)
        msg = ""
        for logger in self._loggers:
            if not logger.isEnabledFor(log_level):
                continue
            if not msg:
                k, v = entry.data
                msg = f"{entry.func_id.to_str(False)}: {k}={v}"
            logger.getChild(entry.func_id.module).log(
                log_level, msg, extra={"record_group": group.id}
            )

    @contextmanager
    def start(self) -> Iterator[None]:
        _active_sessions.append(self)
        yield
        _active_sessions.pop()

    def enter_func(self, fid: RecordableFunctionId) -> None:
        group = self._record_set.add_group(fid)
        self._group_stack.append(group)

    def exit_func(self, fid: RecordableFunctionId) -> None:
        group = self._group_stack.pop()
        if not group.entries:
            self._record_set.remove_last_group()

    def get_records(self) -> RecordSet:
        return self._record_set

    def add_logger(self, logger: Optional[logging.Logger] = None) -> None:
        if logger is None:
            logger = logging.getLogger(_DEFAULT_LOGGER_NAME)
        self._loggers.add(logger)


_active_sessions: list[RecordSession] = []
