# MAYA Explainability Benchmark Report

- Image: `2793.jpg`
- Prediction: **REAL**
- Device: `CPU`
- Timestamp (UTC): `2026-08-03T13:33:11.416183+00:00`
- Explainers: `gradcam, gradcam_plus_plus, layercam, scorecam, eigencam`

## Recommendation

**`gradcam`** (rank #1, score 67.1)

- `gradcam` ranked #1 with overall score 67.1/100 under the configured weights.
- Explanation quality 68.4/100; focus 49.6; localization 91.5.
- Coverage 58.4%; mean agreement 68.8%; generation 517.2 ms; peak RAM 458.1 MiB.
- Leads `eigencam` by 2.3 overall points.

### Alternatives

- `eigencam` (rank 2, score 64.82): Faster generation — useful for quick triage.
- `layercam` (rank 3, score 61.63): Faster generation — useful for quick triage.

## Leaderboard

1. `gradcam` — overall **67.1** | quality 68.4 | focus 49.6 | loc 91.5 | coverage 58.4% | agreement 68.8% | 517.2 ms | 458.1 MiB
2. `eigencam` — overall **64.8** | quality 74.3 | focus 67.8 | loc 82.3 | coverage 7.6% | agreement 8.5% | 101.5 ms | 542.3 MiB
3. `layercam` — overall **61.6** | quality 62.2 | focus 41.7 | loc 87.2 | coverage 65.1% | agreement 74.2% | 215.3 ms | 465.5 MiB
4. `gradcam_plus_plus` — overall **61.5** | quality 62.1 | focus 41.6 | loc 87.1 | coverage 65.2% | agreement 74.2% | 558.7 ms | 461.7 MiB
5. `scorecam` — overall **54.9** | quality 61.8 | focus 41.2 | loc 86.9 | coverage 65.5% | agreement 73.9% | 50533.8 ms | 536.7 MiB

## Suite statistics

- Mean quality: `65.77`
- Mean time (ms): `10385.28`
- Mean peak RAM (MiB): `492.84`
- Fastest: `eigencam`
- Leanest: `gradcam`
- Highest quality: `eigencam`

## Weights

- quality: `0.3500`
- focus: `0.1500`
- localization: `0.1500`
- coverage: `0.1000`
- agreement: `0.1500`
- performance: `0.1000`

## Generated files

- `leaderboard.png`, `performance_chart.png`, `quality_chart.png`
- `agreement_matrix.png`, `memory_usage.png`, `comparison_dashboard.png`
- `benchmark.json`, `leaderboard.json`, `recommendation.json`
- `summary.md`, `benchmark_report.md`
