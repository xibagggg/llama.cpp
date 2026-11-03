"""Bind an OCR build to a commit, complete source diff, flags, and binary hashes."""
from pathlib import Path
import argparse,hashlib,json,subprocess,datetime

p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True);a=p.parse_args()
def git(*args):return subprocess.check_output(['git','-C',str(a.repo),*args])
changed=git('status','--porcelain','--untracked-files=normal').decode()
diff=git('diff','HEAD','--binary')
sources={}
for raw in git('ls-files','-z').split(b'\0'):
    if raw:
        name=raw.decode();f=a.repo/name
        if f.is_file():sources[name]=hashlib.sha256(f.read_bytes()).hexdigest()
record={'schema':1,'created_at':datetime.datetime.now().astimezone().isoformat(),
        'commit':git('rev-parse','HEAD').decode().strip(),
        'branch':git('branch','--show-current').decode().strip(),
        'source_clean':not changed,'status':changed,'diff_sha256':hashlib.sha256(diff).hexdigest(),
        'tracked_source_sha256':sources,
        'cmake_cache_sha256':hashlib.sha256((a.build_dir/'CMakeCache.txt').read_bytes()).hexdigest(),
        'binaries':{str(f.relative_to(a.build_dir)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.build_dir/'bin').iterdir() if f.is_file()}}
(a.build_dir/'OCR_SOURCE_MANIFEST.json').write_text(json.dumps(record,indent=2))
print('OCR build',record['commit'],'clean=',record['source_clean'])
