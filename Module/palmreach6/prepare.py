"""Verify the published controller and restore the paired runtime inputs."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(path):
    h = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(archive):
    inputs = json.loads((HERE / "inputs.json").read_bytes())
    sources = json.loads((HERE / "source-sha256.json").read_bytes())
    for name, expected in sources.items():
        if digest(REPO / name) != expected:
            raise ValueError(f"Controller differs from the release: {name}")
    if digest(archive) != inputs["archive_sha256"]:
        raise ValueError("Runtime archive SHA-256 differs from the release")

    pending = []
    with ZipFile(archive) as bundle:
        if sorted(bundle.namelist()) != sorted(inputs["files"]):
            raise ValueError("Unexpected runtime archive members")
        try:
            for name, expected in inputs["files"].items():
                target = REPO / name
                if target.exists():
                    if digest(target) != expected:
                        raise FileExistsError(f"Conflicting existing input: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                    path = Path(tmp.name)
                    pending.append((path, target, expected))
                    with bundle.open(name) as source:
                        shutil.copyfileobj(source, tmp)
                if digest(path) != expected:
                    raise ValueError(f"Input checksum mismatch: {name}")
            for path, target, expected in pending:
                try:
                    if os.name == "nt":
                        path.rename(target)  # Windows refuses to replace an existing destination.
                    else:
                        os.link(path, target)
                except FileExistsError:
                    if digest(target) != expected:
                        raise FileExistsError(f"Conflicting existing input: {target}")
        finally:
            for path, _, _ in pending:
                path.unlink(missing_ok=True)
    print(f"Verified {len(sources)} controller files and {len(inputs['files'])} runtime inputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    prepare(parser.parse_args().archive)
