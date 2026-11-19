"""Restore the original I04 MP4 files from an extracted Google Drive download."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile


HERE = Path(__file__).resolve().parent
BLOCK = 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def write_chapters(output, video, find_file):
    layout_file = find_file(video["layout"]["name"])
    if sha256(layout_file) != video["layout"]["sha256"]:
        raise ValueError(f"Invalid layout file: {layout_file}")
    with zipfile.ZipFile(layout_file) as archive:
        layout = json.loads(archive.read("layout.json"))
        if (layout["format_version"] != 1 or layout["sha256"] != video["sha256"]
                or layout["bytes"] != video["bytes"]):
            raise ValueError(f"Layout does not match {video['name']}")
        output.write(archive.read("prefix.bin"))
        for chapter in layout["chapters"]:
            path = find_file(chapter["name"])
            if path.stat().st_size != chapter["bytes"] or sha256(path) != chapter["sha256"]:
                raise ValueError(f"Invalid or incomplete chapter: {path}")
            offset = chapter["media_offset"]
            remaining = chapter["media_bytes"]
            if offset < 0 or remaining <= 0 or offset + remaining > chapter["bytes"]:
                raise ValueError(f"Invalid media range for {path}")
            with path.open("rb") as source:
                source.seek(offset)
                while remaining:
                    block = source.read(min(remaining, BLOCK))
                    if not block:
                        raise ValueError(f"Truncated chapter: {path}")
                    output.write(block)
                    remaining -= len(block)
        output.write(archive.read("suffix.bin"))


def restore(video, files, output_root):
    target = (output_root / video["repository_path"]).resolve()
    target.relative_to(output_root)
    if target.exists():
        if target.is_file() and sha256(target) == video["sha256"]:
            return f"Already verified: {target}"
        raise FileExistsError(f"Existing file has different contents: {target}")

    def find_file(name):
        matches = files.get(name.casefold(), [])
        if len(matches) != 1:
            raise ValueError(f"Expected one extracted {name}, found {len(matches)}")
        return matches[0]

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, suffix=".partial", delete=False)
    partial_path = Path(partial.name)
    try:
        with partial:
            if files.get(video["name"].casefold()) or not video.get("layout"):
                with find_file(video["name"]).open("rb") as source:
                    shutil.copyfileobj(source, partial, BLOCK)
            else:
                write_chapters(partial, video, find_file)
        if partial_path.stat().st_size != video["bytes"] or sha256(partial_path) != video["sha256"]:
            raise ValueError(f"Restored file failed the original SHA-256 check: {video['name']}")
        # Publish only complete output, and preserve any file created by another process.
        if os.name == "nt":
            partial_path.rename(target)
        else:
            os.link(partial_path, target)
    finally:
        partial_path.unlink(missing_ok=True)
    return f"Restored and verified: {target}"


def main():
    manifest = json.loads((HERE / "downloads.json").read_text(encoding="utf-8"))
    videos = {Path(v["name"]).stem: v for v in manifest["videos"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("downloads", type=Path, help="Extracted Drive download directory")
    parser.add_argument("--video", choices=sorted(videos), action="append",
                        help="Restore only this video (repeatable). Default: all six.")
    parser.add_argument("--output-root", type=Path, default=HERE.parents[1],
                        help="Destination repository root. Defaults to this checkout.")
    args = parser.parse_args()
    if not args.downloads.is_dir():
        parser.error(f"Directory does not exist: {args.downloads}")
    files = {}
    for path in args.downloads.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".mp4", ".zip"}:
            files.setdefault(path.name.casefold(), []).append(path)
    try:
        for label in dict.fromkeys(args.video or videos):
            print(restore(videos[label], files, args.output_root.resolve()))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
