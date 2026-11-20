#!/usr/bin/env python3
"""Fetch pinned data-only checkpoints for native audio tests (stdlib only).

No repository Python/model code is imported or executed. Existing files must
already match their declared checksum; a corrupt cache is never silently reused.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def checked_relative(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'unsafe relative artifact path: {name!r}')
    return path


def fetch(root, directory, artifact, verify_only=False):
    target = directory / checked_relative(artifact['destination'])
    expected = artifact['sha256']
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError(f'invalid checksum for {target}')
    if target.exists():
        if target.stat().st_size != artifact['size'] or digest(target) != expected:
            raise ValueError(f'cached artifact does not match manifest: {target}')
        print(f'verified {target}', flush=True)
        return
    if verify_only:
        raise FileNotFoundError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            if 'local_source' in artifact:
                source = root / checked_relative(artifact['local_source'])
                with source.open('rb') as reader:
                    shutil.copyfileobj(reader, output, 8 * 1024 * 1024)
            else:
                url = artifact.get('url')
                if url is None:
                    repository = artifact['repo']
                    revision = artifact['revision']
                    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
                        raise ValueError('checkpoint revision must be a complete Git SHA')
                    source = checked_relative(artifact['source']).as_posix()
                    url = f'https://huggingface.co/{repository}/resolve/{revision}/{source}'
                if not url.startswith('https://'):
                    raise ValueError('checkpoint downloads require HTTPS')
                request = urllib.request.Request(url, headers={'User-Agent': 'candle-native-audio-tests'})
                with urllib.request.urlopen(request, timeout=120) as reader:
                    shutil.copyfileobj(reader, output, 8 * 1024 * 1024)
        if temporary.stat().st_size != artifact['size'] or digest(temporary) != expected:
            raise ValueError(f'downloaded artifact does not match pinned checksum: {target}')
        # Data is validated before publication; do not overwrite a concurrent file.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.stat().st_size != artifact['size'] or digest(target) != expected:
                raise ValueError(f'concurrent artifact does not match manifest: {target}')
        print(f'fetched {target}', flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('audio-test-models.json'))
    parser.add_argument('--family', action='append', help='Repeat to select families; default: all manifest families')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest['schema_version'] != 1:
        parser.error('unsupported manifest version')
    families = args.family or list(manifest['families'])
    unknown = set(families) - manifest['families'].keys()
    if unknown:
        parser.error(f'unknown checkpoint families: {sorted(unknown)}')
    root = Path(__file__).resolve().parents[1]
    for family in families:
        for artifact in manifest['families'][family]:
            fetch(root, args.destination / checked_relative(family), artifact, args.verify_only)


if __name__ == '__main__':
    main()
