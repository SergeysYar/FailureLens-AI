# Dataset and data contract

## Sources

The intended real source is [Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data). Large archives are not downloaded automatically. Real ingestion should be an explicit, reproducible command that records source URL, archive checksum, date, selected period, and normalization version.

The local `make demo` command creates deterministic synthetic telemetry. It is useful for API/UI smoke tests and pipeline development only. It is not historical industrial failure data.

## Normalized schema

The model layer expects `date`, `device_id`, `model`, `smart_5`, `smart_187`, `temperature`, `read_errors`, and optionally `failed`. Source-specific SMART columns must be mapped into this contract before feature generation. Unknown SMART attributes should remain available for future adapters rather than being silently assigned to an unrelated field.

## Quality rules

`make data-validate` checks required columns and duplicate `(device_id, date)` pairs. A real pipeline should also check timestamp monotonicity per device, non-negative counters, plausible temperatures, missingness by attribute, schema changes, unexpected models, and input volume.

## Failure labels

Only an explicit registered failure is an event. A device disappearing from the source is not automatically a failure. If the observation window ends before the chosen horizon, the row is right-censored and excluded from supervised evaluation.

## Governance

Do not commit raw archives, sensitive telemetry, credentials, local databases, generated model binaries, or experiment outputs. Keep a manifest for each real-data run so that metrics can be traced to a dataset fingerprint and preprocessing version.
