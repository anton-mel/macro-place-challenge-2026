# yale_am

**Team:** yale_am (Anton Melnychuk)
**Entry point:** `placer.py`
**Approach:** adapted from a public leaderboard submission (partcleda/macro-place-challenge-2026).

## Pipeline

1. Adam GD on a differentiable smooth proxy (`smooth_proxy.py`): LSE wirelength, overlap-area density, sigmoid-softened L/T-route congestion. Wirelength sharpness annealed soft → sharp.
2. Legalize hard macros (`legalize.py`): pairwise push-apart + jitter escape.
3. Cycle: coordinate-descent polish (`local_search.py`) → warm-restart GD → legalize.
4. Cold simulated annealing (`local_search.py`): also perturbs soft macros, fills remaining time budget.
5. Shelf-pack fallback (`placer.py`) guarantees zero overlaps if GD/legalize never converges.

## Files

- `smooth_proxy.py` — differentiable proxy relaxation
- `fast_proxy.py` — Numba fast evaluator for search
- `legalize.py` — overlap resolution
- `local_search.py` — coordinate descent + simulated annealing
- `placer.py` — orchestrator (`YaleAMPlacer`)
- `make_gif.py` — GD trajectory animation

## Run

```bash
uv run evaluate submissions/reimplemented_hybrid/placer.py -b ibm01
uv run evaluate submissions/reimplemented_hybrid/placer.py --all
```

Budgets via env vars: `HGS_GD1_STEPS`, `HGS_GD1_MAX_SEC`, `HGS_GD2_STEPS`, `HGS_GD2_MAX_SEC`, `HGS_CYCLES`, `HGS_CD_SEC`, `HGS_FINAL_CD_SEC`, `HGS_SA_CAP_SEC`, `HGS_SA_MIN_SEC`.

## Results

AVG proxy **1.1915** across all 17 IBM benchmarks (reduced ~6-10 min/benchmark budget), 0 overlaps on every benchmark. vs RePlAce baseline 1.4578 (+18.3%), vs SA baseline 2.1251 (+43.9%).

| Benchmark | Ours | RePlAce |
|---|---:|---:|
| ibm01 | 0.8927 | 0.9976 |
| ibm02 | 1.2988 | 1.8370 |
| ibm03 | 1.1175 | 1.3222 |
| ibm04 | 1.0901 | 1.3024 |
| ibm06 | 1.3094 | 1.6187 |
| ibm07 | 1.1530 | 1.4633 |
| ibm08 | 1.1904 | 1.4285 |
| ibm09 | 0.9271 | 1.1194 |
| ibm10 | 1.6039 | 1.5009 |
| ibm11 | 0.9287 | 1.1774 |
| ibm12 | 1.3080 | 1.7261 |
| ibm13 | 1.0295 | 1.3355 |
| ibm14 | 1.2375 | 1.5436 |
| ibm15 | 1.2539 | 1.5159 |
| ibm16 | 1.1832 | 1.4780 |
| ibm17 | 1.3536 | 1.6446 |
| ibm18 | 1.3788 | 1.7722 |
| **AVG** | **1.1915** | 1.4578 |
