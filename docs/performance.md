# Performance benchmark

Run `make benchmark`. The command measures feature extraction on a deterministic 240-row demo sample and writes `artifacts/benchmark.json`. It does not claim production throughput, memory, concurrency, or p95 API performance. Those require a pinned target environment and load generator.
