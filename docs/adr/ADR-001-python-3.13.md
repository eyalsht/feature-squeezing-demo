# ADR-001: Python 3.13

**Status:** Accepted — 2026-05-13

## Context

Eyal's initial scaffold targeted Python 3.10. imree prefers Python 3.13 (latest stable as of project start) for new work, both as a learning exercise and to stay current.

Most of the original pinned dependencies were released before 3.13 had broad wheel support:
- `numpy==1.26.4`, `scipy==1.13.1`, `Pillow==10.3.0`, `matplotlib==3.9.0`, `onnxruntime==1.18.0`.

A fresh 3.13 venv would fail to install any of these without source compilation.

## Decision

Target **Python 3.13** for development and Hugging Face Spaces deployment. Bump the affected dependencies to versions with `cp313` wheels (see `requirements.txt`):

| Package | Was | Now |
|---|---|---|
| `numpy` | `==1.26.4` | `>=2.1.0` |
| `scipy` | `==1.13.1` | `>=1.14.1` |
| `Pillow` | `==10.3.0` | `>=10.4.0` |
| `matplotlib` | `==3.9.0` | `>=3.9.2` |
| `onnxruntime` | `==1.18.0` | `>=1.20.0` |

`gradio`, `insightface`, `pytest`, and `pytest-mock` stay pinned.

## Consequences

- ✓ Modern Python features available; aligns with imree's other projects.
- ✓ All deps install cleanly on a fresh `py -3.13 -m venv` venv.
- ⚠ HF Spaces must be configured for Python 3.13 (set `python_version: "3.13"` in the README YAML header or use a `runtime.txt`).
- ⚠ `numpy 2.x` is a major-version bump from 1.26; any future code using deprecated 1.x-only APIs would need updating. The current scaffold uses only basic array ops, so no migration needed.
- ⚠ `insightface==0.7.3` is **source-only on PyPI** for all Python versions — no prebuilt wheel exists. Its `thirdparty/face3d/mesh/cython/mesh_core_cython` extension must be compiled at install time. This requires **Microsoft Visual C++ Build Tools** on Windows (install "Desktop development with C++" workload from <https://visualstudio.microsoft.com/visual-cpp-build-tools/>). One-time per machine; not specific to Python 3.13.
- ⚠ Two extra requirements pins are 3.13-specific:
  - `audioop-lts>=0.2.1` — the stdlib `audioop` module was **removed in Python 3.13** (PEP 594). `pydub` (pulled in transitively by `gradio`'s audio components) still imports `audioop` unconditionally. `audioop-lts` is the canonical PyPI backport that re-registers the name.
  - `huggingface-hub<1.0` — `gradio==4.44.0` imports `HfFolder` from `huggingface_hub`, which was deprecated in 0.x and **removed in 1.0**. Pin keeps us on the compatible 0.36.x line until/unless we bump gradio.

## See also

- `requirements.txt`
- `.python-version`
