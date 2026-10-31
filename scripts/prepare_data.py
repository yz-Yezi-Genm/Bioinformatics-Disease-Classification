"""Extract the six known input files, without trusting paths inside the ZIP."""
import argparse
import hashlib
from pathlib import Path
import zipfile

EXPECTED = {'audit.json','feature_ids.tsv','README_preprocessed.md','sample_manifest.tsv','train_data.npz','test_data.npz'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output-dir', default='data/processed/tcga_brca_20261029_v2')
    args=parser.parse_args()
    destination=Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        files=[item for item in archive.infolist() if not item.is_dir()]
        if len(files)!=len(EXPECTED) or {Path(i.filename).name for i in files}!=EXPECTED:
            raise ValueError('Unexpected files in C preprocessed archive.')
        for item in files:
            # Only fixed allowlisted basenames are ever used for extraction.
            target=destination / Path(item.filename).name
            contents=archive.read(item)
            if target.exists() and target.read_bytes()!=contents:
                raise ValueError(f'Refusing to overwrite a different file: {target}')
            target.write_bytes(contents)
    with Path(args.archive).open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    print(f'Extracted {len(EXPECTED)} files to {destination}\nArchive SHA256: {digest}')

if __name__=='__main__':
    main()
