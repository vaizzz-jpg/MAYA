# MAYA — Phase 4 Sprint 4.4: Explainability Validation & Benchmark Suite

**Status:** Complete  
**Scope:** Benchmark and rank all registered explainers — **no new explainers**, **no algorithm changes**, **no Phase 5**

---

## Goal

Provide an objective **Explainability Validation & Benchmark Suite** that:

- Runs every registered explainer on the same evidence image  
- Measures performance, memory, quality, focus, localization, coverage, agreement  
- Produces automatic rankings and investigator recommendations  

Reuse Sprint 4.3 analytics — do **not** duplicate quality/consistency formulas.

---

## Architecture

```
ExplainabilityBenchmarkSuite.run(image)
        │
        ├── ModelLoader + preprocess          (Sprint 3.4 reuse)
        ├── evaluator.evaluate_explainer()    (registry plugins as-is)
        │         ├── performance.measure_*
        │         └── quality.evaluate_*  → analytics (4.3)
        ├── agreement.compute_agreement()     → analytics consistency
        ├── ranking.rank_explainers()
        ├── recommendation.build_recommendation()
        ├── visualization + report
        ▼
artifacts/phase4/sprint4/
```

| Module | Responsibility |
|--------|----------------|
| `benchmark.py` | Suite coordinator |
| `evaluator.py` | Per-explainer generate + metrics |
| `performance.py` | Time / RAM / CPU |
| `quality.py` | Thin wrapper over Sprint 4.3 |
| `agreement.py` | Consistency + matrix |
| `ranking.py` | Configurable weighted leaderboard |
| `recommendation.py` | Data-driven investigator advice |
| `statistics.py` | Suite aggregates |
| `visualization.py` / `report.py` | Charts + JSON/MD |
| `config.py` / `utils.py` | Weights, paths, RSS helpers |

Explainer algorithm modules under `explainers/` are **not modified**.

---

## Metrics

### Performance
Generation time (ms), peak/average RSS (MiB), CPU percent.

### Quality (via Sprint 4.3)
Focus score, localization score, coverage, explanation quality.

### Agreement
Pairwise cosine / Pearson (analytics) + per-explainer mean agreement.

### Overall score
Configurable weights (`BenchmarkWeights`): quality, focus, localization, coverage, agreement, performance.

---

## Ranking & recommendation

- Ranking sorts by overall score — **no hardcoded preferred explainer**  
- Recommendation selects rank #1 and explains strengths vs alternatives from the same numbers  

---

## Artefacts

`artifacts/phase4/sprint4/`:

- `leaderboard.png`, `performance_chart.png`, `quality_chart.png`  
- `agreement_matrix.png`, `comparison_dashboard.png`, `memory_usage.png`  
- `benchmark.json`, `leaderboard.json`, `recommendation.json`  
- `benchmark_report.md`, `summary.md`  

---

## Usage

```python
from ai.explainability.benchmark import (
    ExplainabilityBenchmarkConfig,
    ExplainabilityBenchmarkSuite,
)

cfg = ExplainabilityBenchmarkConfig(device_preference="cpu")
result = ExplainabilityBenchmarkSuite(cfg).run("path/to/image.jpg")
print(result.recommendation["recommended_explainer"])
```

---

## Tests

`tests/test_explainability_benchmark.py` (Phase 3.5 inference benchmarks remain in `tests/test_benchmark.py`).

Covers: performance, agreement, ranking, recommendation, JSON/MD/plots, end-to-end suite (ScoreCAM excluded in unit runtime; included in full artefact runs).

---

## Definition of Done

- [x] All configured registry explainers benchmarked  
- [x] Automatic rankings  
- [x] Investigator recommendations  
- [x] Reports + visualizations  
- [x] Explainability algorithms unchanged  
- [x] Tests pass  

**STOP after Sprint 4.4. Do not begin Phase 5.**
