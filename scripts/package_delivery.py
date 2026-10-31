"""Package only the allowlisted deliverables, then verify every ZIP member."""
import hashlib
import json
from pathlib import Path
import zipfile

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    root=Path(__file__).resolve().parents[1]
    verification=json.loads((root/'results/feature_selection/verification.json').read_text(encoding='utf-8'))
    if verification['status']!='PASS':
        raise RuntimeError('Results must pass verification before packaging.')
    files=[root/'README_FEATURE_SELECTION.md',root/'requirements-feature-selection.txt',root/'.gitignore']
    for directory in ['src','scripts','examples','tests','docs','results']:
        files.extend(p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in {'.py','.md','.csv','.json','.png'} and '__pycache__' not in p.parts)
    files=sorted(files,key=lambda p:p.relative_to(root).as_posix())
    hashes={p.relative_to(root).as_posix():digest(p) for p in files}
    manifest='\n'.join(f'{hashes[name]}  {name}' for name in hashes)+'\n'
    delivery=root/'delivery'
    delivery.mkdir(exist_ok=True)
    archive=delivery/'TCGA-BRCA-feature-selection-issue5-20261031.zip'
    prefix='TCGA-BRCA-feature-selection-issue5'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as out:
        for p in files:
            out.write(p,prefix+'/'+p.relative_to(root).as_posix())
        out.writestr(prefix+'/DELIVERY_SHA256.txt',manifest)
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
        assert len(check.infolist())==len(files)+1
        for name,expected in hashes.items():
            assert hashlib.sha256(check.read(prefix+'/'+name)).hexdigest()==expected
        assert not any('/data/' in n or '/.venv/' in n or n.endswith('.npz') for n in check.namelist())
    package_hash=digest(archive)
    (delivery/(archive.name+'.sha256')).write_text(f'{package_hash}  {archive.name}\n',encoding='utf-8')
    print(json.dumps(dict(path=str(archive),files=len(files)+1,bytes=archive.stat().st_size,sha256=package_hash,zip_verified=True),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
