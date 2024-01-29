import json
import os
from typing import Callable

from typing_extensions import TypeAlias

DIR = r"../packages/"
type_alias_recorder = {}

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
                type_alias_recorder[alias] = alias


def insert_future_annotation(file_name: str) -> None:
    """insert "from __future__ import annotations" """
    print(file_name)
    lines = []
    import_text = "from __future__ import annotations"
    license_text_last_line = "# limitations under the License."

    with open(file_name, "r") as f:
        for l in f:
            lines.append(l)
    
    if len(lines) == 0:
        return

    if import_text in [l.strip() for l in lines]:
        return

    insert_line = 0
    while insert_line < len(lines):
        line_text = lines[insert_line]
        if (
            ("from" in line_text)
            or ("import" in line_text)
            or ("def" in line_text)
            or ("class" in line_text)
        ):
            break
        elif "=" in line_text:
            while ("#" in lines[insert_line - 1]) and (
                lines[insert_line - 1].strip() != license_text_last_line
            ):
                insert_line -= 1
            break

        insert_line += 1
    else:
        return

    lines.insert(insert_line, import_text+"\n")
    new_line_text = "".join(lines)

    with open(file_name, "w") as f:
        f.write(new_line_text)


if __name__ == "__main__":
    iterate_files(DIR, iterate_line_to_find_type_alias)    
    with open("qp_type_aliases.json", "w") as f:
        json.dump(type_alias_recorder, fp=f)

    iterate_files(DIR, insert_future_annotation)