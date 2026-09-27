#!/usr/bin/env python3
"""Materialize the conflict-resolved two-stream + two-field research candidate."""

from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "candidate"
CONTROL = Path(__file__).resolve().parent / "two_field_control"
MERGED_TREE = "fbbe28593b330c86d261a01daa1cbd8ccca0bef4"
TWO_FIELD = "554fa24cd8d3672e38d249ee7700b4fc40d18cb1"
FILES = ("pinning.cu", "GPUMath.h", "GPUHash.h", "cofactor_checkpoint.h", "COPYING")
AUDITS = (
    "audit_deferred_chain.py",
    "audit_external_pipeline.py",
    "audit_fast_tail_contract.py",
    "audit_field_final_carry.py",
    "audit_shared_tree.py",
    "audit_stream_recode.py",
    "audit_superbatch_representation.py",
    "audit_superbatch_roots.py",
    "audit_vector_state_layout.py",
    "check_tail_words.py",
)


def git_show(path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{MERGED_TREE}:candidates/pinning/{path}"], cwd=ROOT
    )


def git_show_at(revision: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{revision}:candidates/pinning/{path}"], cwd=ROOT
    )


source = git_show("pinning.cu").decode()
start = source.index("<<<<<<< a067fdadb2099eb5d6304ac560fe0d424a9e1022")
end_marker = ">>>>>>> 554fa24cd8d3672e38d249ee7700b4fc40d18cb1\n"
end = source.index(end_marker, start) + len(end_marker)
resolution = """    for (int s = 0; s < QSB_SLOTS; s++) {
        d_pipeline_state[s]=NULL; d_pipeline_roots[s]=NULL; d_pipeline_tree[s]=NULL;
        d_super_roots[s]=NULL; d_root_checkpoint[s]=NULL;
        cudaError_t pipeline_err=cudaMalloc(&d_pipeline_state[s],pipeline_state_bytes);
        if(pipeline_err==cudaSuccess)
            pipeline_err=cudaMalloc(&d_pipeline_roots[s],pipeline_root_bytes);
        // d_pipeline_tree remains null; neither candidate stage dereferences it.
        if(pipeline_err==cudaSuccess)
            pipeline_err=cudaMalloc(&d_super_roots[s],super_root_bytes);
        if(pipeline_err==cudaSuccess)
            pipeline_err=cudaMalloc(&d_root_checkpoint[s],root_checkpoint_bytes);
        if(pipeline_err!=cudaSuccess){
            fprintf(stderr,"Pipeline allocation failed: %s\\n",cudaGetErrorString(pipeline_err));
            return 1;
        }
        if(((uintptr_t)d_pipeline_state[s] & (alignof(ulonglong2)-1u)) != 0){
            fprintf(stderr,"Pipeline state allocation is not 16-byte aligned\\n");
            return 1;
        }
"""
source = source[:start] + resolution + source[end:]

OUT.mkdir(parents=True, exist_ok=True)
for name in FILES:
    data = source.encode() if name == "pinning.cu" else git_show(name)
    (OUT / name).write_bytes(data)
for name in AUDITS:
    (OUT / name).write_bytes(git_show(name))

CONTROL.mkdir(parents=True, exist_ok=True)
for name in FILES + AUDITS:
    (CONTROL / name).write_bytes(git_show_at(TWO_FIELD, name))

provenance = {
    "purpose": "research-only composition; not a qualified submission",
    "two_stream_submission": "11ba7e4",
    "two_stream_commit": "a067fdadb2099eb5d6304ac560fe0d424a9e1022",
    "two_field_submission": "31e98e4",
    "two_field_commit": "554fa24cd8d3672e38d249ee7700b4fc40d18cb1",
    "merge_base": "df2fb8b8f5f3e47ff7a6a1848ccebf40e902f634",
    "merge_tree": MERGED_TREE,
    "resolution": "retain per-stream-slot allocations and omit the zero-byte cross-kernel tree allocation",
}
(Path(__file__).resolve().parent / "provenance.json").write_text(
    json.dumps(provenance, indent=2) + "\n"
)

assert not any(
    line.startswith(("<<<<<<< ", "=======", ">>>>>>> "))
    for line in source.splitlines()
)
print(OUT)
