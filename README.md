# Problem 6: Turing Machine Simulation and Bounded Halting Prediction

A configurable Turing machine simulator plus an ML triage component that predicts whether a
`(machine, input)` pair **halts within a step limit L**, with multi-seed experiments analyzing
step limits, machine complexity, probe budgets, feature importance, and out-of-distribution machines.

See [REPORT.md](REPORT.md) for the full experimental report.

## Quickstart

Requires Python 3.10+.

**Linux / macOS**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -v       # run unit tests (simulator, features, models, app)
.venv/bin/python experiment.py            # datasets, multi-seed ML evaluation, plots, models
.venv/bin/streamlit run app.py            # interactive simulator & predictor
```

**Windows**

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m pytest tests -v
.venv\Scripts\python experiment.py
.venv\Scripts\streamlit run app.py
```

`experiment.py --n 2000` runs a quicker, smaller version of the full pipeline.

Pre-computed datasets, models, and plots are already included in `results/`, so `app.py` runs immediately.

## Files

| File | What it does |
|---|---|
| `tm.py` | Simulator: sparse tape, transitions (`L`/`R`/`S`), halting, exact & Zobrist loop detection, DOT graph renderer, parser, 6 presets |
| `features.py` | 19 static features (transition table + graph reachability + generalized input stats) and 8 probe features (10-step trial run) |
| `ml_models.py` | Decision Tree, Random Forest, and `BalancedMLP` (with training-fold minority oversampling) |
| `experiment.py` | Dataset generation (with rules, inputs & traces), 3-seed evaluation, complexity & probe-sweep studies, OOD benchmarks, plots |
| `app.py` | Streamlit app: step-through tape & Graphviz simulator, side-by-side 3-model halting predictor, experiment dashboard, theory notes |
| `tests/` | Unit tests for simulator semantics, Busy Beavers (3 & 4 states), custom symbols, features, `BalancedMLP`, and headless Streamlit app smoke tests |
| `REPORT.md` | Formal experimental report covering TOC background, methodology, results, error analysis, and limitations |

## Key Design Point: Three Outcomes, Not Two

The simulator never answers just "halts / doesn't halt":

1. **halted**: reached the halt state `H` (or encountered a missing rule).
2. **looping**: the full configuration `(state, head, tape)` repeated, proving it never halts.
3. **limit**: budget exhausted. This means *unknown*, never "does not halt".

The ML label is `"halts within L steps"` (`1`) vs `"has not halted within L steps"` (`0`). Class `0`
deliberately does **not** mean "never halts", so the system never claims to solve the general halting problem.
