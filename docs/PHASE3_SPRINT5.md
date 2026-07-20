# MAYA — Phase 3 Sprint 5

## AI Performance & Reliability Benchmark Suite

**Status:** Complete  
**Scope:** Offline benchmarking only — **no Grad-CAM, Flask, frontend, SHA-256, face verification, or DB**

---

## Benchmark objectives

- Quantify inference latency on the target Windows 11 / 8 GB / CPU host
- Measure process RSS and estimated model memory footprint
- Estimate throughput (images/second) for 1 / 5 / 10 / 20 synthetic batch sizes
- Score prediction consistency and confidence stability
- Capture a reproducible system + model fingerprint

## Benchmark methodology

1. Resolve a sample image (CLI `--image` or first processed test JPEG)
2. Load the Sprint 4 checkpoint once via `CoreInferencer` / `ModelLoader`
3. Warm up, then time N core inferences (preprocess → softmax → confidence)
4. Sample RSS around repeated inferences (`psutil`)
5. Time 1/5/10/20 sequential inferences for throughput
6. Repeat the same image N times for consistency / reliability
7. Emit JSON + Markdown + matplotlib plots under `artifacts/phase3/benchmark/`

Core inference **does not** allocate investigation IDs or write Sprint 4 artefacts on every timed run (keeps the bench lightweight and non-destructive).

## Hardware profile

| Item | Target |
|------|--------|
| OS | Windows 11 |
| RAM | 8 GB |
| Device | CPU-first (`--device cpu`) |
| Model | EfficientNet-B0 (frozen backbone) |

## Metrics explained

| Metric | Meaning |
|--------|---------|
| Latency avg/min/max/median/σ | Wall-clock ms per full core inference |
| Peak RSS | Highest process resident set during bench |
| Estimated model MiB | Parameter + buffer tensor bytes |
| Images/sec | Throughput for N sequential inferences |
| Consistency % | Share of runs agreeing with the dominant label |
| Reliability score | Consistency penalized by confidence volatility |

## Interpretation guidelines

- On CPU / 8 GB, expect latency in hundreds of ms depending on load
- Consistency should be ~100% for a deterministic eval-mode model
- Rising peak RSS without returning toward baseline may indicate a leak (investigate)
- Throughput for n=20 is not multi-threaded batching — it is sequential stress

## Current limitations

- Single-image repeated N times (not a true multi-file folder throughput test)
- No GPU/CUDA-specific kernel timers
- Reliability score is a simple heuristic, not a forensic standard
- Smoke training checkpoints may yield low absolute accuracy (bench still valid)

## Future improvements

- True multi-image folder throughput
- Optional CUDA event timers
- Compare multiple checkpoints side-by-side
- Continuous integration smoke job with fixed `--runs 3`

## How to run

```bash
python scripts/benchmark.py
python scripts/benchmark.py --runs 20
python scripts/benchmark.py --image path\to\image.jpg --device cpu
python -m pytest tests/test_benchmark.py -q
```

## Module map

| Module | Role |
|--------|------|
| `core.py` | Side-effect-free inferencer |
| `latency.py` | Latency stats |
| `memory.py` | RSS + model footprint |
| `throughput.py` | Images/sec |
| `consistency.py` | Stability / reliability |
| `system_info.py` | Host fingerprint |
| `model_stats.py` | Params / checkpoint size |
| `plots.py` | Matplotlib figures |
| `report.py` | JSON + Markdown writers |
| `benchmark.py` | Coordinator |
| `cli.py` | CLI |

## Explicit non-goals

Grad-CAM, Flask/FastAPI, frontend, SHA-256, face verification, database integration.
