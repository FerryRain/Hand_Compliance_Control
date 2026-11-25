"""Verify and install the experiment's published inputs into this checkout."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def prepare(fixtures, inputs):
    spec = json.loads((HERE / "inputs.json").read_bytes())
    sources = json.loads((HERE / "source-sha256.json").read_bytes())
    for name, expected in sources.items():
        path = REPO / name
        saved = HERE / "facing4" / path.name
        if path.parent == REPO / "Module/wrist_active" and saved.is_file():
            path = saved
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source differs: {name}")
    if sha256(fixtures.read_bytes()).hexdigest() != spec["fixtures_zip_sha256"]:
        raise ValueError("Fixture archive differs from the published revision")
    # Check the complete batch before installing anything. The package maps
    # exact members of the shared fixtures ZIP to the native experiment layout.
    pending = []
    with ZipFile(fixtures) as archive:
        for entry in [*spec["fixtures"], *spec["inputs"]]:
            data = archive.read(entry["member"]) if "member" in entry else (inputs / entry["input"]).read_bytes()
            target = REPO / entry["destination"]
            expected = entry["sha256"]
            if sha256(data).hexdigest() != expected:
                raise ValueError(f"Input checksum mismatch: {target.relative_to(REPO)}")
            if target.exists():
                if sha256(target.read_bytes()).hexdigest() != expected:
                    raise FileExistsError(f"Conflicting local input: {target}")
            else:
                pending.append((target, data))
    for target, data in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = NamedTemporaryFile(dir=target.parent, delete=False)
        path = Path(tmp.name)
        try:
            with tmp:
                tmp.write(data)
            if target.exists():
                if target.read_bytes() != data:
                    raise FileExistsError(f"Conflicting local input: {target}")
            else:
                path.replace(target)
        finally:
            path.unlink(missing_ok=True)
    print(f"Verified {len(sources)} source files and {len(spec['fixtures']) + len(spec['inputs'])} inputs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True, help="Published active-2/fixtures.zip")
    parser.add_argument("--inputs", type=Path, required=True, help="Downloaded palmguide/inputs directory")
    args = parser.parse_args()
    prepare(args.fixtures, args.inputs)


if __name__ == "__main__":
    main()
