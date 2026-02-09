This directory contains LLM-authored test case manifests organized by subfield.

Subfield layout mirrors `tests/`:
- `tests/llm/mapper`
- `tests/llm/model`
- `tests/llm/toll`
- `tests/llm/plotting`
- `tests/llm/mapping_viz`
- `tests/llm/notebooks`
- `tests/llm/isl/mapper/*`
- `tests/llm/isl/distributed/*`
- `tests/llm/vibe_see_readme_in_this_dir/*`

Each subfield directory has a `cases.yaml` manifest with LLM test cases.

`tests/llm/mapper/ground_truth.json` is generated from live runs in the `pyenv`
`accelforge` environment and is used by
`tests/llm/mapper/test_mapper_ground_truth.py` as golden output for:
- access counts
- energy
- latency
- area
- per-component and per-einsum energy/latency checks
