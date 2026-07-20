# MAYA Benchmark Report

## Hardware / Software

- OS: Windows (`Windows-10-10.0.26200-SP0`)
- CPU: AMD64 Family 23 Model 160 Stepping 0, AuthenticAMD
- RAM: 0.44 / 7.28 GB available
- Python: 3.11.9
- PyTorch: 2.13.0+cpu
- Torchvision: 0.28.0+cpu
- Device: cpu

## Latency (ms)

- Runs: 5
- Average: 101.65187999991758
- Median: 100.6865999997899
- Min / Max: 96.40239999953337 / 109.83440000018163
- Std Dev: 4.556934809607607

## Memory (MiB)

- Baseline RSS: 388.91
- Peak RSS: 388.98
- Delta: 0.07
- Estimated model: 15.46

## Throughput

| Images | img/s | Seconds |
|--------|-------|---------|
| 1 | 10.9418 | 0.0914 |
| 5 | 11.3383 | 0.441 |
| 10 | 11.7576 | 0.8505 |
| 20 | 8.4048 | 2.3796 |

## Consistency

- Dominant prediction: `REAL`
- Consistency %: 100.0
- Mean confidence: 64.2574
- Confidence σ: 0.0
- Reliability score: 100.0

## Model

- Name / version: `efficientnet_b0` / `sprint3.2-best`
- Dataset version: `v1`
- Checkpoint: `C:\Users\Asus\Desktop\MAYA\artifacts\checkpoints\best.pt` (15.582 MiB)
- Params total / trainable / frozen: 4010110 / 2562 / 4007548

## Figures

- `latency_distribution.png`
- `memory_usage.png`
- `throughput.png`
- `consistency_chart.png`
- `system_summary.png`
