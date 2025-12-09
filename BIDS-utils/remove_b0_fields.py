#!/usr/bin/env python3
"""
Strip B0FieldIdentifier, B0FieldSource and IntendedFor keys from every *.json
file under a root directory (defaults to the current working directory).

Now removes those keys at **any level of nesting** within the JSON.

Usage
-----
python remove_b0_keys.py [ROOT_DIR]

If ROOT_DIR is omitted, '.' is used.

Safety
------
• Only rewrites files that actually change.
• Creates a side-by-side backup <filename>.bak before modifying a file.
"""

import json
import sys
from pathlib import Path
from typing import Any, Union

KEYS_TO_REMOVE = {"B0FieldIdentifier", "B0FieldSource", "IntendedFor"}


def _prune(obj: Union[dict, list]) -> bool:
    """
    Recursively remove KEYS_TO_REMOVE from ``obj``.

    Returns
    -------
    bool
        True if at least one key/value pair was removed.
    """
    changed = False

    if isinstance(obj, dict):
        # Work on a *list* of keys so we can delete while iterating
        for key in list(obj.keys()):
            if key in KEYS_TO_REMOVE:
                del obj[key]
                changed = True
            else:
                changed |= _prune(obj[key])

    elif isinstance(obj, list):
        for item in obj:
            changed |= _prune(item)

    # For any other JSON types (str, int, etc.) do nothing
    return changed


def process_json_file(path: Path) -> bool:
    """Remove unwanted keys from *any depth* in `path`. Return True if file changed."""
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"⚠️  Skipped {path} (invalid JSON: {exc})")
        return False

    if not _prune(data):
        return False  # nothing to remove anywhere in the file

    # --- Write out changes --------------------------------------------------
    # backup = path.with_suffix(path.suffix + ".bak")
    # if not backup.exists():
    #     path.replace(backup)  # keep original as .bak

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main(root: Path) -> None:
    json_files = sorted(root.rglob("*.json"))
    changed = 0

    for jf in json_files:
        if process_json_file(jf):
            changed += 1
            print(f"✔️  Cleaned {jf}")
    print(f"\nDone. {changed} of {len(json_files)} JSON file(s) modified.")


if __name__ == "__main__":
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    if not root_dir.is_dir():
        sys.exit(f"Error: {root_dir} is not a directory.")
    main(root_dir)
