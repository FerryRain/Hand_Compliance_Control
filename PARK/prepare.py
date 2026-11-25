"""Restore PARK's published, checksum-locked experiment archives into a checkout."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent


def checksum(stream):
    result = sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.digest()


def restore(bundle, root):
    for entry in bundle.infolist():
        name = PurePosixPath(entry.filename)
        if (name.is_absolute() or ".." in name.parts or ".git" in name.parts
                or "\\" in entry.filename or ":" in entry.filename or entry.is_dir()
                or (entry.external_attr >> 16) & 0o170000 == 0o120000):
            raise ValueError(f"Unexpected archive member {entry.filename}")
        path = root.joinpath(*name.parts)
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError(f"Archive target outside checkout: {path}")
        with bundle.open(entry) as source:
            if path.is_file():
                with path.open("rb") as existing:
                    if checksum(existing) != checksum(source):
                        raise FileExistsError(f"Conflicting existing file: {path}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                partial = Path(tmp.name)
            try:
                with partial.open("wb") as tmp:
                    copyfileobj(source, tmp, 1024 * 1024)
                try:
                    if os.name == "nt":
                        partial.rename(path)
                    else:
                        os.link(partial, path)
                except FileExistsError:
                    with path.open("rb") as current, partial.open("rb") as proposed:
                        if checksum(current) != checksum(proposed):
                            raise
            finally:
                partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Downloaded park directory containing inputs/ and formal/.")
    parser.add_argument("--comparisons", action="store_true", help="Also restore both sets of matched baseline records.")
    args = parser.parse_args()
    root = HERE.parent.resolve()
    index = json.loads((HERE / "archives.json").read_bytes())
    records = [r for r in index["archives"] if r["group"] != "comparison" or args.comparisons]
    sources = [(args.data / r["path"], r) for r in records]
    for archive, record in sources:
        with archive.open("rb") as stream:
            if archive.stat().st_size != record["bytes"] or checksum(stream).hex() != record["sha256"]:
                raise ValueError(f"PARK archive checksum mismatch: {archive}")
    for archive, record in sources:
        with ZipFile(archive) as bundle:
            if len(bundle.infolist()) != record["members"]:
                raise ValueError(f"Unexpected member count: {archive}")
            restore(bundle, root)
    print(f"Restored {len(records)} verified PARK archives into {root}")


if __name__ == "__main__":
    main()
