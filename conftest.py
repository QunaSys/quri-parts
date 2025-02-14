import sys

collect_ignore = []
if sys.version_info < (3, 10):
    collect_ignore.extend([
        "packages/tket"
    ])
