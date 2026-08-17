"""Download simulation-relevant YCB meshes from the official YCB S3 site.

The downloader intentionally excludes raw RGB and RGB-D recordings.  It can
download the official Google 16k, 64k, or 512k textured mesh and optionally
falls back to the Berkeley processed mesh archive when that Google model is
unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


BASE_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com"
OBJECTS_URL = f"{BASE_URL}/data/objects.json"
DEFAULT_OUTPUT = Path("assets_external/ycb")
OFFICIAL_FILE_ID_ALIASES = {
    # The official objects.json uses a dash, while the Google archive uses an
    # underscore and extracts into an underscore-named directory.
    "027-skillet": "027_skillet",
}


def _fetch_object_ids() -> list[str]:
    with urlopen(OBJECTS_URL, timeout=60) as response:  # noqa: S310 - fixed URL
        payload = json.load(response)
    objects = payload.get("objects")
    if not isinstance(objects, list) or not all(isinstance(x, str) for x in objects):
        raise ValueError(f"Unexpected YCB object index at {OBJECTS_URL}")
    return objects


GOOGLE_RESOLUTIONS = ("16k", "64k", "512k")


def _google_url(object_id: str, resolution: str) -> str:
    file_id = OFFICIAL_FILE_ID_ALIASES.get(object_id, object_id)
    return f"{BASE_URL}/data/google/{file_id}_google_{resolution}.tgz"


def _berkeley_url(object_id: str) -> str:
    return (
        f"{BASE_URL}/data/berkeley/{object_id}/"
        f"{object_id}_berkeley_meshes.tgz"
    )


def _archive_path(archive_dir: Path, object_id: str, source: str) -> Path:
    suffix = source if source.startswith("google_") else "berkeley_meshes"
    return archive_dir / f"{object_id}_{suffix}.tgz"


def _aria2_download(
    jobs: list[tuple[str, Path]],
    *,
    workers: int,
) -> int:
    if not jobs:
        return 0
    aria2 = shutil.which("aria2c")
    if aria2 is None:
        raise RuntimeError("aria2c is required for resumable parallel downloads")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8") as stream:
        for url, output in jobs:
            output.parent.mkdir(parents=True, exist_ok=True)
            stream.write(f"{url}\n")
            stream.write(f"  dir={output.parent.resolve()}\n")
            stream.write(f"  out={output.name}\n")
        stream.flush()
        environment = os.environ.copy()
        # aria2 does not accept the curl-style ``socks://`` ALL_PROXY value
        # used by this workstation. HTTP_PROXY/HTTPS_PROXY remain available.
        for name in ("ALL_PROXY", "all_proxy"):
            if environment.get(name, "").lower().startswith("socks://"):
                environment.pop(name)
        result = subprocess.run(
            [
                aria2,
                f"--input-file={stream.name}",
                f"--max-concurrent-downloads={workers}",
                "--split=1",
                "--min-split-size=20M",
                "--continue=true",
                "--max-tries=5",
                "--retry-wait=3",
                "--connect-timeout=30",
                "--timeout=60",
                "--auto-file-renaming=false",
                "--allow-overwrite=false",
                "--summary-interval=10",
                "--download-result=full",
            ],
            check=False,
            env=environment,
        )
    return result.returncode


def _archive_is_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            return bool(members) and any(
                member.isfile()
                and member.name.lower().endswith((".obj", ".ply", ".stl"))
                for member in members
            )
    except (OSError, tarfile.TarError):
        return False


def _safe_extract(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe member {member.name!r} in {path}")
        archive.extractall(destination, filter="data")


def _has_extracted_mesh(
    models_dir: Path,
    object_id: str,
    source: str | None = None,
) -> bool:
    object_dir = models_dir / object_id
    if not object_dir.is_dir():
        return False
    search_root = object_dir / source if source is not None else object_dir
    if not search_root.is_dir():
        return False
    return any(search_root.rglob("*.obj")) or any(search_root.rglob("*.ply"))


def _infer_existing_source(
    models_dir: Path,
    object_id: str,
    preferred_google_source: str,
) -> str:
    object_dir = models_dir / object_id
    if _has_extracted_mesh(models_dir, object_id, preferred_google_source):
        return preferred_google_source
    if object_dir.is_dir():
        return "berkeley_processed"
    return "missing"


def _normalize_extracted_object_dir(models_dir: Path, object_id: str) -> None:
    file_id = OFFICIAL_FILE_ID_ALIASES.get(object_id, object_id)
    if file_id == object_id:
        return
    extracted = models_dir / file_id
    canonical = models_dir / object_id
    if extracted.is_dir() and not canonical.exists():
        extracted.rename(canonical)
    elif extracted.is_dir():
        # A second Google resolution may be extracted after another tier has
        # already created the canonical directory. Merge the distinct
        # ``google_16k``/``google_64k``/``google_512k`` children instead of
        # leaving an alias directory outside the catalog.
        for child in extracted.iterdir():
            destination = canonical / child.name
            if destination.exists():
                raise FileExistsError(
                    f"Cannot merge extracted YCB directory: {destination} exists"
                )
            shutil.move(str(child), destination)
        extracted.rmdir()


def _convert_surface_meshes_to_obj(models_dir: Path, object_id: str) -> list[Path]:
    """Create an OBJ only when a valid triangle surface exists.

    Berkeley archives for a few small objects contain only point clouds.  A
    point cloud is deliberately not reconstructed here because inventing a
    surface would make collision geometry unreliable.
    """

    object_dir = models_dir / object_id
    existing = sorted(object_dir.rglob("*.obj")) if object_dir.is_dir() else []
    if existing:
        return existing
    candidates = (
        sorted(object_dir.rglob("*.stl")) + sorted(object_dir.rglob("*.ply"))
        if object_dir.is_dir()
        else []
    )
    if not candidates:
        return []
    import trimesh

    for source in candidates:
        loaded = trimesh.load(source, force=None, process=False)
        if isinstance(loaded, trimesh.Scene):
            if not loaded.geometry:
                continue
            loaded = loaded.to_geometry()
        if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
            continue
        destination = object_dir / "converted" / "surface.obj"
        destination.parent.mkdir(parents=True, exist_ok=True)
        loaded.export(destination)
        return [destination]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 downloads all objects")
    parser.add_argument(
        "--object-ids",
        nargs="+",
        default=None,
        help="Optional exact YCB object IDs; otherwise use the official full index.",
    )
    parser.add_argument(
        "--google-resolution",
        choices=GOOGLE_RESOLUTIONS,
        default="16k",
        help=(
            "Official Google mesh polygon tier. 64k is the recommended "
            "high-resolution collision source; 512k is mainly useful as a "
            "reference surface before collision decomposition."
        ),
    )
    parser.add_argument(
        "--no-berkeley-fallback",
        action="store_true",
        help="Do not use Berkeley processed meshes when the Google tier is absent.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep verified .tgz files after extraction.",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")

    output = args.output.resolve()
    google_source = f"google_{args.google_resolution}"
    archive_dir = output / "archives"
    models_dir = output / "models"
    archive_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    previous_manifest = output / "manifest.json"
    known_no_surface: set[str] = set()
    if previous_manifest.is_file():
        try:
            previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
            known_no_surface = {
                record["object_id"]
                for record in previous.get("records", [])
                if record.get("status") == "no_surface_mesh"
                and record.get("requested_google_variant") == google_source
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            known_no_surface = set()

    object_ids = _fetch_object_ids()
    if args.object_ids:
        unknown = sorted(set(args.object_ids) - set(object_ids))
        if unknown:
            raise ValueError(f"Unknown YCB object IDs: {', '.join(unknown)}")
        requested = set(args.object_ids)
        object_ids = [object_id for object_id in object_ids if object_id in requested]
    if args.limit:
        object_ids = object_ids[: args.limit]
    print(f"[YCB] official index: {len(object_ids)} object directories")

    google_jobs = [
        (
            _google_url(object_id, args.google_resolution),
            _archive_path(archive_dir, object_id, google_source),
        )
        for object_id in object_ids
        if object_id not in known_no_surface
        if not _has_extracted_mesh(models_dir, object_id, google_source)
        and not _archive_is_valid(
            _archive_path(archive_dir, object_id, google_source)
        )
    ]
    print(
        f"[YCB] Google {args.google_resolution} downloads pending: "
        f"{len(google_jobs)}"
    )
    google_result = _aria2_download(google_jobs, workers=args.workers)
    if google_result:
        print(
            f"[YCB] Some Google {args.google_resolution} URLs were unavailable; "
            "checking fallback."
        )

    source_by_object: dict[str, str] = {}
    missing: list[str] = []
    for object_id in object_ids:
        if object_id in known_no_surface and not _has_extracted_mesh(
            models_dir, object_id, google_source
        ):
            source_by_object[object_id] = "missing"
            continue
        if _has_extracted_mesh(models_dir, object_id, google_source):
            source_by_object[object_id] = google_source
            continue
        google_path = _archive_path(archive_dir, object_id, google_source)
        if _archive_is_valid(google_path):
            source_by_object[object_id] = google_source
        else:
            google_path.unlink(missing_ok=True)
            Path(f"{google_path}.aria2").unlink(missing_ok=True)
            missing.append(object_id)

    if missing and not args.no_berkeley_fallback:
        berkeley_jobs = [
            (
                _berkeley_url(object_id),
                _archive_path(archive_dir, object_id, "berkeley_processed"),
            )
            for object_id in missing
            if not _archive_is_valid(
                _archive_path(archive_dir, object_id, "berkeley_processed")
            )
        ]
        print(f"[YCB] Berkeley fallback downloads pending: {len(berkeley_jobs)}")
        berkeley_result = _aria2_download(berkeley_jobs, workers=args.workers)
        still_missing: list[str] = []
        for object_id in missing:
            path = _archive_path(archive_dir, object_id, "berkeley_processed")
            if _archive_is_valid(path):
                source_by_object[object_id] = "berkeley_processed"
            else:
                path.unlink(missing_ok=True)
                Path(f"{path}.aria2").unlink(missing_ok=True)
                still_missing.append(object_id)
        missing = still_missing
        if berkeley_result and still_missing:
            print("[YCB] Berkeley download completed with unavailable objects.")

    for index, object_id in enumerate(object_ids, start=1):
        source = source_by_object.get(object_id)
        if source not in (google_source, "berkeley_processed"):
            continue
        if _has_extracted_mesh(models_dir, object_id, source):
            continue
        archive_path = _archive_path(archive_dir, object_id, source)
        print(f"[YCB] extract {index:03d}/{len(object_ids)} {object_id} ({source})")
        _safe_extract(archive_path, models_dir)
        _normalize_extracted_object_dir(models_dir, object_id)
        if not args.keep_archives:
            archive_path.unlink()

    records = []
    for object_id in object_ids:
        _normalize_extracted_object_dir(models_dir, object_id)
        object_meshes = _convert_surface_meshes_to_obj(models_dir, object_id)
        mesh_files = sorted(str(path.relative_to(output)) for path in object_meshes)
        records.append(
            {
                "object_id": object_id,
                "source_variant": source_by_object.get(
                    object_id,
                    _infer_existing_source(
                        models_dir, object_id, google_source
                    ),
                ),
                "requested_google_variant": google_source,
                "mesh_files": mesh_files,
                "available": bool(mesh_files),
                "status": "ready" if mesh_files else "no_surface_mesh",
            }
        )
    manifest = {
        "dataset": "YCB Object and Model Set",
        "official_index": OBJECTS_URL,
        "license": "CC-BY-4.0",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "raw_rgb_included": False,
        "raw_rgbd_included": False,
        "requested_google_variant": google_source,
        "records": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Local YCB simulation meshes\n\n"
        "Downloaded from the official YCB S3 object index. Only simulation "
        "meshes and textures are included; raw RGB/RGB-D captures are excluded.\n\n"
        "Dataset license: CC BY 4.0. See "
        "https://www.ycbbenchmarks.com/ and cite the YCB Object and Model Set.\n",
        encoding="utf-8",
    )
    available = sum(record["available"] for record in records)
    unavailable = [record["object_id"] for record in records if not record["available"]]
    print(
        f"[YCB] complete: {available}/{len(records)} object directories available; "
        f"unavailable={len(unavailable)}; output={output}"
    )
    if unavailable:
        print("[YCB] no usable triangle surface: " + ", ".join(unavailable))
    if available == 0:
        raise RuntimeError("No YCB simulation meshes were downloaded")


if __name__ == "__main__":
    main()
