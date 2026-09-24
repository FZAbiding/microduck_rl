"""Local provenance and atomic status files for jump experiments."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import tarfile


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')
    os.replace(temp,path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():
        return str(output/'manifest.json')
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','src','scripts','tests','pyproject.toml','uv.lock','AGENTS.md'],text=True).splitlines()
    paths=sorted(set(p for p in paths if Path(p).is_file()))
    hashes={p:sha256(p) for p in paths}
    (output/'working.diff').write_bytes(subprocess.check_output(['git','diff','--binary']))
    with tarfile.open(output/'source.tar.gz','w:gz') as archive:
        for path in paths: archive.add(path)
    atomic_json(output/'manifest.json',{'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'created_at':time.time(),'files':hashes,'archive_sha256':sha256(output/'source.tar.gz'),
        'lock_sha256':sha256('uv.lock')})
    return str(output/'manifest.json')
