from quri_parts.core.utils.recording import (
    INFO,
    RecordEntry,
    Recorder,
    RecordSession,
    recordable,
)


@recordable
def func_to_record(recorder: Recorder, x: int) -> int:
    recorder.info("x", x)
    recorder.info("2x", 2 * x)
    return 2 * x


def test_recordable() -> None:
    fid = func_to_record.id
    session = RecordSession()
    session.set_level(INFO, func_to_record)

    with session.start():
        assert func_to_record(3) == 6
        assert func_to_record(4) == 8

    records = session.get_records()
    history = list(records.get_history(func_to_record))
    assert len(history) == 2
    group0, group1 = history

    assert group0.func_id == fid
    assert group0.entries == [
        RecordEntry(fid, ("x", 3)),
        RecordEntry(fid, ("2x", 6)),
    ]

    assert group1.func_id == fid
    assert group1.entries == [
        RecordEntry(fid, ("x", 4)),
        RecordEntry(fid, ("2x", 8)),
    ]


@recordable
def nested_func(recorder: Recorder, x: int) -> int:
    recorder.info("x", x)
    if x // 2 == 1:
        y = nested_func(2 * x)
        recorder.info("y", y)
    else:
        y = x
    recorder.info("2y", 2 * y)
    return 2 * y


def test_nested_func_record() -> None:
    fid = nested_func.id
    session = RecordSession()
    session.set_level(INFO, nested_func)

    with session.start():
        assert nested_func(3) == 24
        assert nested_func(4) == 8

    records = session.get_records()
    history = list(records.get_history(nested_func))
    assert len(history) == 3
    group0, group1, group2 = history

    assert group0.func_id == fid
    assert group0.entries == [
        RecordEntry(fid, ("x", 3)),
        RecordEntry(fid, ("y", 12)),
        RecordEntry(fid, ("2y", 24)),
    ]

    assert group1.func_id == fid
    assert group1.entries == [
        RecordEntry(fid, ("x", 6)),
        RecordEntry(fid, ("2y", 12)),
    ]

    assert group2.func_id == fid
    assert group2.entries == [
        RecordEntry(fid, ("x", 4)),
        RecordEntry(fid, ("2y", 8)),
    ]
