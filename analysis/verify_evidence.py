"""Verify the numerical input archive and every retained entry."""
from pathlib import Path
import hashlib
import json
import zipfile

def run():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'data/input_evidence_manifest.json').read_text())
    evidence=root/'data/original_evidence.zip'
    evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest()
    if evidence_sha256!=manifest['evidence_sha256']:
        raise AssertionError('Evidence archive digest differs from manifest')
    records=[]
    with zipfile.ZipFile(evidence) as archive:
        expected={r['path'] for r in manifest['files']}
        if len(expected)!=len(manifest['files']):
            raise AssertionError('Duplicate evidence paths in manifest')
        if len(archive.namelist())!=len(expected) or set(archive.namelist())!=expected:
            raise AssertionError('Evidence entry set differs from manifest')
        if archive.testzip() is not None:
            raise AssertionError('Evidence archive CRC verification failed')
        for item in manifest['files']:
            payload=archive.read(item['path'])
            if len(payload)!=item['bytes'] or hashlib.sha256(payload).hexdigest()!=item['sha256']:
                raise AssertionError('Evidence entry verification failed: '+item['path'])
            records.append(item['path'])
    result=dict(verified_entries=len(records),all_passed=True,
                evidence_sha256=evidence_sha256,
                source_parent_sha256=manifest['parent_sha256'],
                filtered_from_sha256=manifest['filtered_from_sha256'],
                scope='Every retained numerical input, configuration and environment entry verified; omitted parent ZIP entries are not asserted to have been rechecked.')
    out=root/'results/validation';out.mkdir(parents=True,exist_ok=True)
    (out/'evidence_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
