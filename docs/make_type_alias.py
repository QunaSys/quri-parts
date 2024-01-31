import json
import os
from typing import Callable

from typing_extensions import TypeAlias

DIR = r"../packages/"
TYPE_ALIAS_RECORDER = {}

Action: TypeAlias = Callable[[str], None]


def iterate_files(dir: str, action: Action) -> None:
    for root, _, files in os.walk(dir):
        if ("test" in root) or ("__pycache__" in root):
            continue
        for name in files:
            filepath = root + os.sep + name
            if filepath.endswith(".py"):
                action(filepath)


def iterate_line_to_find_type_alias(file_name: str) -> None:
    with open(file_name, "r") as f:
        for line in f:
            if "TypeAlias" in line and "import" not in line:
                alias = line.split(":")[0]
                TYPE_ALIAS_RECORDER[alias] = alias


def insert_future_annotation(file_name: str) -> None:
    """Insert "from __future__ import annotations"."""
    lines = []
    import_text = "from __future__ import annotations   # isort: skip\n"

    with open(file_name, "r") as f:
        for line in f:
            lines.append(line)

    lines.insert(0, import_text)

    new_line_text = "".join(lines)
    with open(file_name, "w") as f:
        f.write(new_line_text)


if __name__ == "__main__":
    iterate_files(DIR, iterate_line_to_find_type_alias)
    with open("qp_type_aliases.json", "w") as f:
        json.dump(TYPE_ALIAS_RECORDER, fp=f)

    iterate_files(DIR, insert_future_annotation)