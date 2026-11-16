"""Turing Machine Lab: simulator + bounded halting prediction.

    streamlit run app.py
"""
from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from features import PROBE_STEPS, all_features
from tm import (
    PRESETS,
    format_machine,
    parse_machine,
    random_input,
    random_machine,
    run,
    sample_machine,
    to_dot,
)

RESULTS = Path(__file__).parent / "results"
LIMITS = [10, 25, 50, 100, 250, 500, 1000]
MODEL_NAMES = ["Decision Tree", "Random Forest", "Neural Net (MLP)"]

st.set_page_config(page_title="Turing Machine Lab", page_icon="🧮", layout="wide")


@st.cache_resource
def load_models():
    p = RESULTS / "models.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_data
def load_csv(name: str):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def prob_pos(model, X: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.shape[1] == 2:
            return float(probs[0][1])
        return float(int(model.classes_[0]))
    return float(model.predict(X)[0])


def describe(res, limit: int) -> tuple[str, str]:
    """(kind, message) for a run result. kind in success|info|warning."""
    if res.status == "halted":
        return "success", f"**Halted** after {res.steps} steps."
    if res.status == "looping":
        return "info", (
            f"**Provably never halts.** The exact same configuration (state, head, tape) "
            f"occurred twice by step {res.steps}, so the machine is stuck in a cycle."
        )
    return "warning", (
        f"**Not halted within {limit} steps.** This is *not* the same as 'never halts' - "
        f"it might halt at step {limit + 1} or never. The halting problem is undecidable, "
        f"so no program can settle this in general."
    )


def tape_html(tape: dict, head: int, state: str, blank: str = "_", radius: int = 12) -> str:
    cells = []
    for i in range(head - radius, head + radius + 1):
        sym = tape.get(i, blank)
        style = (
            "display:inline-block;width:30px;height:34px;line-height:34px;text-align:center;"
            "font-family:monospace;font-size:18px;border:1px solid rgba(128,128,128,.5);"
            "margin-right:-1px;"
        )
        if i == head:
            style += "background:#f4b400;color:#000;border:2px solid #b37e00;font-weight:700;"
        cells.append(f'<span style="{style}">{sym}</span>')
    return (
        f'<div style="overflow-x:auto;white-space:nowrap;padding:6px 0">{"".join(cells)}</div>'
        f'<div style="font-family:monospace;margin-top:2px">head at cell {head} &nbsp;|&nbsp; '
        f"state <b>{state}</b> &nbsp;|&nbsp; blank <b>{blank}</b></div>"
    )


# ----------------------------------------------------------------- sidebar
st.sidebar.title("Machine")
choice = st.sidebar.selectbox("Choose a machine", list(PRESETS) + ["Random machine"])
n_states, seed = 4, 1
if choice == "Random machine":
    n_states = st.sidebar.slider("Number of states", 2, 10, 4)
    seed = st.sidebar.number_input("Random seed", 0, 10_000_000, 1)
    default_text = format_machine(random_machine(n_states, random.Random(seed)))
    default_inp = random_input(random.Random(seed + 7))
else:
    default_text, default_inp = PRESETS[choice]

c_cfg1, c_cfg2 = st.sidebar.columns(2)
start_state = c_cfg1.text_input(
    "Start state", value="q0", key=f"start::{choice}::{n_states}::{seed}",
    help="Leave as q0 or enter any state name present in your rules.",
)
blank_sym = c_cfg2.text_input(
    "Blank symbol", value="_", max_chars=1, key=f"blank::{choice}::{n_states}::{seed}",
    help="Single character representing an empty tape cell.",
)

rules_text = st.sidebar.text_area(
    "Transition rules (editable)",
    default_text,
    height=300,
    key=f"rules::{choice}::{n_states}::{seed}",
    help="One rule per line: `state read write move next`. Moves: L, R, S. "
    "Halt state is `H`. A missing rule also halts the machine.",
)
input_str = st.sidebar.text_input(
    "Input tape", default_inp, key=f"inp::{choice}::{n_states}::{seed}",
    help="Head starts on the first character. Supports any single-character symbols.",
)
max_steps = st.sidebar.slider("Step limit for simulation", 10, 5000, 1000)

tm, errors = parse_machine(
    rules_text,
    start=start_state.strip() or None,
    blank=(blank_sym or "_"),
)
st.title("🧮 Turing Machine Lab")
st.caption("Simulate Turing machines, then see how well ML can predict whether they halt within a step limit.")
if errors:
    st.error("Could not parse the rules:\n\n" + "\n".join(f"- {e}" for e in errors))
    st.stop()

tab_sim, tab_pred, tab_exp, tab_theory = st.tabs(
    ["▶ Simulator", "🔮 Halting prediction", "📊 Experiments", "📚 Theory notes"]
)

# --------------------------------------------------------------- simulator
with tab_sim:
    res = run(tm, input_str, max_steps, record=True, exact_loop_check=True)
    kind, msg = describe(res, max_steps)
    getattr(st, kind)(msg)

    trace = res.trace
    if len(trace) > 1:
        step = st.slider("Step through the run", 0, len(trace) - 1, 0)
    else:
        step = 0
    s_no, s_state, s_head, s_tape = trace[step]

    left_col, right_col = st.columns([1.3, 1.0])
    with left_col:
        st.markdown(f"**Step {s_no}**", help="Tape snapshot at this step")
        st.markdown(tape_html(s_tape, s_head, s_state, blank=tm.blank), unsafe_allow_html=True)
        if len(trace) >= 3000:
            st.caption("Trace recording is capped at 3000 snapshots; the verdict above covers the full run.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Steps executed", res.steps)
        c2.metric("Non-blank cells at end", len(res.tape))
        c3.metric("Final state", res.state)
        if res.status == "halted" and res.tape:
            st.code("Final tape: " + res.tape_string(tm.blank), language=None)
    with right_col:
        st.markdown("**State transition diagram**")
        if len(tm.transitions) <= 45:
            st.graphviz_chart(to_dot(tm, active_state=s_state), width="stretch")
        else:
            st.info(f"Diagram omitted for large transition tables ({len(tm.transitions)} rules).")

# -------------------------------------------------------------- prediction
def model_verdict(clf, X: np.ndarray) -> tuple[float, bool]:
    """(probability of 'halts within L', yes/no decision).

    The decision uses the model's own predict(): the MLP has a tuned threshold, not 0.5.
    """
    return prob_pos(clf, X), bool(clf.predict(X)[0])


with tab_pred:
    bundle = load_models()
    if bundle is None:
        st.warning("No trained models found. Run `python experiment.py` first.")
    else:
        ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.0, 1.0])
        L = ctrl1.select_slider("Step limit L", options=LIMITS, value=100)
        pop_choice = ctrl2.radio(
            "Training population",
            ["hard", "random"],
            horizontal=True,
            help="'hard' = trained on machines still unresolved after 10 steps; "
                 "'random' = trained on general 2-6 state random machines.",
        )
        primary_model = ctrl3.selectbox("Primary model (for batch test)", MODEL_NAMES, index=1)

        feats = all_features(tm, input_str)
        X = np.array([[feats[c] for c in bundle["cols"]]])
        truth = run(tm, input_str, L, exact_loop_check=True)
        actual = truth.status == "halted"

        st.subheader("Model predictions vs. simulation ground truth")
        cols_ui = st.columns(4)
        for idx, m_name in enumerate(MODEL_NAMES):
            p_m, guess_m = model_verdict(bundle["models"][pop_choice][m_name][L], X)
            with cols_ui[idx]:
                st.metric(m_name, f"{p_m:.0%}", delta="halts within L" if guess_m else "not within L")
                st.caption("Matches simulation" if guess_m == actual else "Disagrees with simulation")
        with cols_ui[3]:
            st.markdown("**Ground truth (simulation)**")
            k, m = describe(truth, L)
            getattr(st, k)(m)

        if pop_choice == "hard" and L == 10:
            st.info("By construction no 'hard' training machine halts within 10 steps, "
                    "so the hard-trained models always answer 'not within 10' here.")

        st.caption(
            f"Models see transition-table statistics + a {PROBE_STEPS}-step probe run. "
            "ML is a heuristic triage signal, never a proof; simulation stays the ground truth."
        )

        # Same machine, every step limit: where do the models agree with simulation?
        st.markdown("**This machine at every step limit**")
        full = run(tm, input_str, max(LIMITS), exact_loop_check=True)
        table = []
        for lim in LIMITS:
            sim = run(tm, input_str, lim, exact_loop_check=True)
            row = {"L": lim, "simulation": {"halted": "halts within L", "looping": "provably never halts",
                                            "limit": "not halted within L"}[sim.status]}
            for m_name in MODEL_NAMES:
                p_m, g_m = model_verdict(bundle["models"][pop_choice][m_name][lim], X)
                ok = "✓" if g_m == (sim.status == "halted") else "✗"
                row[m_name] = f"{'halts' if g_m else 'not within L'} ({p_m:.0%}) {ok}"
            table.append(row)
        st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
        if full.status == "halted":
            st.caption(f"Simulation: this machine halts at step {full.steps}.")

        with st.expander("Features fed to the models"):
            feat_df = pd.DataFrame(X.T, index=bundle["cols"], columns=["value"])
            if "feature_importance" in bundle:
                fi = bundle["feature_importance"]
                fi_sub = fi[fi["population"] == pop_choice][["feature", "importance"]].set_index("feature")
                feat_df = feat_df.join(fi_sub, how="left")
            st.dataframe(feat_df, width="stretch")

        st.divider()
        st.subheader("Batch test on fresh unseen machines")
        b1, b2 = st.columns(2)
        n_batch = b1.slider("Number of machines", 50, 1000, 300, step=50)
        batch_pop = b2.radio(
            "Test generator",
            ["hard", "random"],
            index=0 if pop_choice == "hard" else 1,
            horizontal=True,
            key=f"batch_pop::{pop_choice}",
            help="Defaults to the same distribution the selected models were trained on.",
        )
        if batch_pop != pop_choice:
            st.warning(
                f"Distribution shift: models trained on **{pop_choice}** machines are being "
                f"tested on **{batch_pop}** machines. Expect degraded accuracy "
                "(see the out-of-distribution discussion in the report)."
            )
        if st.button("Run batch test"):
            rng = random.Random(987654)  # differs from the training seeds -> unseen machines
            clf = bundle["models"][pop_choice][primary_model][L]
            rows = []
            while len(rows) < n_batch:
                m = sample_machine(rng, batch_pop)
                i = random_input(rng)
                if batch_pop == "hard" and run(m, i, PROBE_STEPS).status != "limit":
                    continue  # hard population = unresolved after the probe
                f = all_features(m, i)
                xp = np.array([[f[c] for c in bundle["cols"]]])
                truth_b = int(run(m, i, L).status == "halted")
                rows.append((truth_b, int(clf.predict(xp)[0]), int(f["probe_halted"])))
            d = pd.DataFrame(rows, columns=["actual", "ml", "rule"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{primary_model} accuracy", f"{accuracy_score(d.actual, d.ml):.1%}")
            m2.metric(f"{primary_model} F1", f"{f1_score(d.actual, d.ml, zero_division=0):.3f}")
            m3.metric(f"Rule ({PROBE_STEPS} steps) accuracy", f"{accuracy_score(d.actual, d.rule):.1%}")
            m4.metric(f"Rule ({PROBE_STEPS} steps) F1", f"{f1_score(d.actual, d.rule, zero_division=0):.3f}")
            st.caption(
                f"{primary_model}: precision {precision_score(d.actual, d.ml, zero_division=0):.3f}, "
                f"recall {recall_score(d.actual, d.ml, zero_division=0):.3f}; "
                f"{int(d.actual.sum())} of {len(d)} machines actually halt within {L} steps."
            )

            st.write(f"Confusion matrix ({primary_model}):")
            st.dataframe(
                pd.crosstab(
                    d.actual.map({0: "actual: not within L", 1: "actual: halts"}),
                    d.ml.map({0: "predicted: not within L", 1: "predicted: halts"}),
                ),
                width="stretch",
            )

# ------------------------------------------------------------- experiments
FIGURES = [
    ("accuracy_vs_limit.png", "1. Prediction quality vs step limit L (mean ± std over 5 seeds)"),
    ("hard_cases.png", "2. Hard cases (unresolved after the probe): F1, PR-AUC and number of slow halters"),
    ("accuracy_vs_complexity.png", "3. Generalising to unseen, bigger machines"),
    ("feature_importance.png", "4. Random Forest feature importance"),
    ("probe_sweep.png", "5. Effect of the probe budget k"),
    ("mlp_ablation.png", "6. Fixing the neural network on imbalanced data"),
    ("ood_heatmap.png", "7. Out-of-distribution: busy beavers and hand-written machines"),
]

with tab_exp:
    res_df = load_csv("results_summary.csv")
    if res_df is None:
        st.warning("No results yet. Run `python experiment.py`.")
    else:
        for fname, title in FIGURES:
            if (RESULTS / fname).exists():
                st.subheader(title)
                st.image(str(RESULTS / fname))

        st.subheader("Results table (mean ± std over seeds)")
        e1, e2 = st.columns(2)
        pop = e1.radio("Population", ["random", "hard"], horizontal=True, key="exp_pop")
        lim = e2.select_slider("Step limit", options=LIMITS, value=250, key="exp_L")
        show = res_df[(res_df["population"] == pop) & (res_df["limit"] == lim)]
        metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        table = show[["features", "model"]].copy()
        for mcol in metric_cols:
            table[mcol] = [f"{a:.3f} ± {0 if pd.isna(b) else b:.3f}" if not pd.isna(a) else "n/a"
                           for a, b in zip(show[f"{mcol}_mean"], show[f"{mcol}_std"])]
        table["test positives"] = show["n_pos_test"].round(0).astype(int).astype(str) + " / " + show["n_test"].round(0).astype(int).astype(str)
        st.dataframe(table, width="stretch", hide_index=True)

        ood_df = load_csv("ood_benchmarks.csv")
        if ood_df is not None:
            with st.expander("Out-of-distribution table (all step limits)"):
                st.dataframe(ood_df.round(3), width="stretch", hide_index=True)
        cx_df = load_csv("complexity.csv")
        if cx_df is not None:
            with st.expander("Complexity generalisation table (per seed)"):
                st.dataframe(cx_df.round(3), width="stretch", hide_index=True)
        if (RESULTS / "summary.md").exists():
            with st.expander("All key numbers (results/summary.md)"):
                st.markdown((RESULTS / "summary.md").read_text(encoding="utf-8"))

# ------------------------------------------------------------------ theory
with tab_theory:
    st.markdown(
        """
**Why 'bounded' halting?** The general halting problem (*does machine M halt on input w?*) is
**undecidable** — no algorithm can solve it for every machine and input. Asking the bounded
question (*does M halt within L steps?*) *is* decidable, and simulation answers it exactly in O(L) steps.

**The three outcomes the simulator reports**
- **Halted** — reached the halt state `H` (or encountered a missing transition rule) within the step budget.
- **Provably looping** — a full configuration `(state, head position, tape contents)` repeated, so
  the deterministic machine is trapped in a cycle and will never halt.
- **Limit reached** — unknown. Either it halts after step `L`, or it diverges forever without repeating
  (for example by writing `1`s endlessly to the right).

**Where ML fits.** Static features (transition table graph, halting reachability, runaway writers)
plus a short probe run (`10` steps) let Decision Trees, Random Forests, and Neural Networks triage
which unresolved machines are most likely to halt later — without ever claiming to solve general halting.
        """
    )
