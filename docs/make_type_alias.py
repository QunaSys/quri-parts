import json
import os

DIR = r"../packages/"
type_alias_recorder = {}


def iterate_files(dir: str) -> None:
    for root, _, files in os.walk(dir):
        if ("test" in root) or ("__pycache__" in root):
            continue
        for name in files:
            filepath = root + os.sep + name
            if filepath.endswith(".py"):
                itetate_line_to_find_type_alias(filepath)


def itetate_line_to_find_type_alias(file_name: str) -> None:
    with open(file_name, "r") as f:
        for line in f:
            if "TypeAlias" in line and "import" not in line:
                alias = line.split(":")[0]
                type_alias_recorder[alias] = alias


if __name__ == "__main__":
    iterate_files(DIR)
    # print(type_alias_recorder)
    with open("qp_type_aliases.json", "w") as f:
        json.dump(type_alias_recorder, fp=f)

    with open("../packages/qiskit/quri_parts/qiskit/backend/sampling.py", "a") as f:
        print(any([l.strip() == "from __future__ import annotations" for l in f]))
        # f.write()
