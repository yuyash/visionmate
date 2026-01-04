# Copilot Instructions

- Scope: Cross-platform app that streams UVC video/audio and screen captures into Qwen3-omni (local or cloud) to surface contextual answers; current codebase is a scaffold to grow that workflow [README.md](../README.md).
- Layout: Library code lives under [src/deskmate](../src/deskmate/__init__.py); tests live in [tests/test_smoke.py](../tests/test_smoke.py). Add new features as modules in [src/deskmate](../src/deskmate) and cover them in [tests](../tests).
- Python/tooling: Uses hatchling build backend and uv-managed Python 3.14 env [pyproject.toml](../pyproject.toml). Ruff targets py313 with 100-col limit and E4/E7/E9/F/I/B rules [pyproject.toml](../pyproject.toml). `ty` static checks are configured on src and tests with errors treated as fatal [pyproject.toml](../pyproject.toml).
- Tasks:
  - Lint: `uv run ruff check src tests`.
  - Static checks: `uv run ty check` (honors `tool.ty` config).
  - Tests: `uv run pytest -ra` (pytest already set to look at tests and respect python_files globs) [pyproject.toml](../pyproject.toml).
- Conventions: Keep lines ≤100 chars to satisfy Ruff. Prefer modern Python 3.13+ features but stay compatible with Python 3.14 runtime declared in project metadata. Maintain src-layout imports (package name deskmate).
- Packaging: Project metadata defined in [pyproject.toml](../pyproject.toml); update version/description there when shipping features. License is GPL-3.0 per [LICENSE](../LICENSE) (pyproject currently points to the same file).
- Testing expectations: Preserve the smoke import test; add fast, deterministic pytest cases for new modules. Favor unit-level coverage; avoid heavy I/O in tests unless mocked.
- UI/streaming plans: Pending implementation—document assumptions in README when adding video/audio/screen capture or Qwen integrations so future agents understand dependencies (e.g., platform-specific capture backends, local vs cloud Qwen routing).
- Keep instructions updated: when adding new workflows (CLI, web UI, inference loops), append commands and file paths here so agents have a single source of truth.
