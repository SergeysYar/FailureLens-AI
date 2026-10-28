# Contributing

1. Create a focused branch.
2. Keep changes compatible with the existing Python package layout.
3. Run `ruff check packages tests` and `pytest -q`.
4. Do not commit secrets, raw datasets, model binaries, local registries, or generated reports.
5. Document any experiment with its dataset fingerprint, split protocol, seed, and limitations.
6. Never present synthetic or unexecuted metrics as production evidence.

