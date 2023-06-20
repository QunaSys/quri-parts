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
