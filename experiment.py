"""Bounded-halting prediction experiments.

    python experiment.py            # full run (~60-90 s)
    python experiment.py --n 2000   # quicker run

Label for a step limit L:  1 = "halts within L steps", 0 = "has not halted within L steps".
(0 deliberately does NOT mean "never halts".)

Two populations are studied:
  random : small random machines (2-6 states) + random binary inputs.
           Almost everything that halts does so within a few steps.
  hard   : bigger machines (4-10 states, low halting bias), keeping ONLY pairs that are
           still unresolved after the cheap probe (PROBE_STEPS steps). This is where a
           learned predictor adds real value as a triage tool.

Outputs (in results/):
  dataset_random.csv, dataset_hard.csv (with machine rules, inputs, and probe traces)
  results.csv, results_summary.csv, complexity.csv, feature_importance.csv,
  probe_sweep.csv, ood_benchmarks.csv, models.joblib, and *.png plots.
"""
from __future__ import annotations

import argparse
import random
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from features import PROBE_COLS, PROBE_STEPS, STATIC_COLS, all_features, static_features
from ml_models import make_models
from tm import (
    PRESETS,
    format_rules_compact,
    parse_machine,
    random_input,
    random_machine,
    run,
)

OUT = Path(__file__).parent / "results"
LIMITS = [10, 25, 50, 100, 250, 500, 1000]
MAX_LIMIT = max(LIMITS)
SEEDS = [42, 43, 44]
BASE_SEED = 42
warnings.filterwarnings("ignore")

FEATURE_SETS = {
    "static only": STATIC_COLS,
    f"static + {PROBE_STEPS}-step probe": STATIC_COLS + PROBE_COLS,
}
MODELS = ["Decision Tree", "Random Forest", "Neural Net (MLP)"]


# ------------------------------------------------------------------ dataset
def _format_trace_preview(trace: list) -> str:
    return " -> ".join(f"{step}:{state}@{head}" for step, state, head, _ in trace)


def build_dataset(n: int, hard: bool = False, seed: int = BASE_SEED) -> pd.DataFrame:
    """Sample n unique (machine, input) pairs and record ground truth up to MAX_LIMIT steps."""
    rng = random.Random(seed + (1000 if hard else 0))
    rows = []
    seen_pairs: set[tuple[str, str]] = set()
    while len(rows) < n:
        if hard:
            tm = random_machine(
                rng.randint(4, 10),
                rng,
                p_halt_range=(0.001, 0.04),
                allow_stay=rng.random() < 0.25,
            )
        else:
            tm = random_machine(rng.randint(2, 6), rng)
        inp = random_input(rng)
        rules_str = format_rules_compact(tm)
        pair_key = (rules_str, inp)
        if pair_key in seen_pairs:
            continue

        probe_res = run(tm, inp, PROBE_STEPS, record=True)
        if hard and probe_res.status != "limit":
            continue  # already resolved by the cheap probe -> not interesting
        seen_pairs.add(pair_key)

        res = run(tm, inp, MAX_LIMIT)
        row = {
            "rules": rules_str,
            "input_str": inp,
            "probe_trace": _format_trace_preview(probe_res.trace),
        }
        row.update(all_features(tm, inp))
        row["status"] = res.status
        row["halt_step"] = res.steps if res.status == "halted" else -1
        rows.append(row)
    return pd.DataFrame(rows)


def label(df: pd.DataFrame, limit: int) -> np.ndarray:
    return ((df["halt_step"] >= 0) & (df["halt_step"] <= limit)).astype(int).values


# ------------------------------------------------------------------ metrics
def scores(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    has_both = len(np.unique(y_true)) == 2
    if y_prob is not None and has_both:
        roc = float(roc_auc_score(y_true, y_prob))
        pr = float(average_precision_score(y_true, y_prob))
    else:
        roc = np.nan
        pr = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc,
        "pr_auc": pr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _prob_pos(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        cls = int(model.classes_[0])
        return np.full(len(X), float(cls))
    return model.predict(X).astype(float)


# ------------------------------------------------------------------ studies
def run_study(df: pd.DataFrame, name: str, seeds: list[int] = SEEDS) -> pd.DataFrame:
    """Train/evaluate every model across step limits and multiple split seeds."""
    rows = []
    indices = np.arange(len(df))
    strat_key = (df["halt_step"] >= 0).astype(int)
    for seed in seeds:
        idx_train, idx_test = train_test_split(
            indices, test_size=0.3, random_state=seed, stratify=strat_key
        )
        for L in LIMITS:
            y = label(df, L)
            ytr, yte = y[idx_train], y[idx_test]
            base = dict(
                population=name,
                seed=seed,
                limit=L,
                n_pos_test=int(yte.sum()),
                n_test=len(yte),
            )
            majority = int(ytr.mean() >= 0.5)
            maj_pred = np.full_like(yte, majority)
            rows.append(
                dict(
                    base,
                    features="-",
                    model="Majority baseline",
                    **scores(yte, maj_pred, maj_pred.astype(float)),
                    train_s=0.0,
                    infer_ms=0.0,
                )
            )
            rule_pred = df["probe_halted"].values[idx_test]
            rows.append(
                dict(
                    base,
                    features="-",
                    model=f"Rule: simulate {PROBE_STEPS} steps",
                    **scores(yte, rule_pred, rule_pred.astype(float)),
                    train_s=0.0,
                    infer_ms=0.0,
                )
            )
            if ytr.sum() < 5:
                continue
            for fs_name, cols in FEATURE_SETS.items():
                Xtr, Xte = df[cols].values[idx_train], df[cols].values[idx_test]
                for m_name, model in make_models(seed).items():
                    t1 = time.time()
                    model.fit(Xtr, ytr)
                    train_s = time.time() - t1
                    t2 = time.time()
                    pred = model.predict(Xte)
                    prob = _prob_pos(model, Xte)
                    infer_ms = (time.time() - t2) / len(Xte) * 1000
                    rows.append(
                        dict(
                            base,
                            features=fs_name,
                            model=m_name,
                            **scores(yte, pred, prob),
                            train_s=train_s,
                            infer_ms=infer_ms,
                        )
                    )
    return pd.DataFrame(rows)


def summarize_results(res: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "train_s", "infer_ms"]
    grp = res.groupby(["population", "limit", "features", "model"], sort=False)
    agg = {m: ["mean", "std"] for m in metrics}
    agg["n_pos_test"] = "mean"
    agg["n_test"] = "mean"
    summary = grp.agg(agg)
    summary.columns = [
        f"{col}_{stat}" if stat else col for col, stat in summary.columns.to_flat_index()
    ]
    return summary.reset_index()


def complexity_study(
    df_rnd: pd.DataFrame, df_hard: pd.DataFrame, L: int = 250
) -> pd.DataFrame:
    """Train on smaller machines and test generalization to unseen larger state counts."""
    rows = []
    configs = [
        ("random", df_rnd, 4),
        ("hard", df_hard, 6),
    ]
    for pop_name, df, max_train_states in configs:
        y = label(df, L)
        indices = np.arange(len(df))
        idx_train, idx_test = train_test_split(
            indices, test_size=0.3, random_state=BASE_SEED, stratify=(df["halt_step"] >= 0)
        )
        is_train = np.isin(indices, idx_train)
        is_test = ~is_train
        ns_col = df["n_states"].values
        for fs_name, cols in FEATURE_SETS.items():
            tr = is_train & (ns_col <= max_train_states)
            if y[tr].sum() < 5:
                continue
            for m_name, model in make_models(BASE_SEED).items():
                model.fit(df[cols].values[tr], y[tr])
                for ns in sorted(df["n_states"].unique()):
                    te = (ns_col == ns) & (is_test if ns <= max_train_states else True)
                    if te.sum() == 0:
                        continue
                    Xte = df[cols].values[te]
                    pred = model.predict(Xte)
                    prob = _prob_pos(model, Xte)
                    rows.append(
                        dict(
                            population=pop_name,
                            train_max_states=max_train_states,
                            features=fs_name,
                            model=m_name,
                            n_states=int(ns),
                            n_test=int(te.sum()),
                            n_pos=int(y[te].sum()),
                            **scores(y[te], pred, prob),
                        )
                    )
    return pd.DataFrame(rows)


def feature_importance_study(
    df_rnd: pd.DataFrame, df_hard: pd.DataFrame, L: int = 250
) -> pd.DataFrame:
    """Extract Random Forest feature importances for both populations."""
    cols = STATIC_COLS + PROBE_COLS
    rows = []
    for pop_name, df in [("random", df_rnd), ("hard", df_hard)]:
        rf = make_models(BASE_SEED)["Random Forest"]
        rf.fit(df[cols].values, label(df, L))
        for feat, imp in zip(cols, rf.feature_importances_):
            rows.append(
                {
                    "population": pop_name,
                    "feature": feat,
                    "group": "static" if feat in STATIC_COLS else "probe",
                    "importance": float(imp),
                }
            )
    return pd.DataFrame(rows).sort_values(["population", "importance"], ascending=[True, False])


def probe_sweep_study(n: int = 2500, L: int = 250) -> pd.DataFrame:
    """Measure how the probe budget k affects hard-case prediction quality."""
    rng = random.Random(BASE_SEED + 777)
    samples = []
    while len(samples) < n:
        tm = random_machine(
            rng.randint(4, 10), rng, p_halt_range=(0.001, 0.04), allow_stay=rng.random() < 0.25
        )
        inp = random_input(rng)
        if run(tm, inp, PROBE_STEPS).status != "limit":
            continue
        res = run(tm, inp, L)
        samples.append((tm, inp, int(res.status == "halted")))

    y_all = np.array([lbl for _, _, lbl in samples])
    idx_tr, idx_te = train_test_split(
        np.arange(len(samples)), test_size=0.3, random_state=BASE_SEED, stratify=y_all
    )

    rows = []
    for k in [0, 5, 10, 25, 50]:
        feat_rows = []
        for tm, inp, _ in samples:
            if k == 0:
                feat_rows.append(static_features(tm, inp))
            else:
                feat_rows.append(all_features(tm, inp, k=k))
        X = pd.DataFrame(feat_rows).values
        Xtr, Xte = X[idx_tr], X[idx_te]
        ytr, yte = y_all[idx_tr], y_all[idx_te]
        for m_name, model in make_models(BASE_SEED).items():
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            prob = _prob_pos(model, Xte)
            rows.append(dict(probe_k=k, limit=L, model=m_name, **scores(yte, pred, prob)))
    return pd.DataFrame(rows)


def ood_benchmark_study(app_models: dict, cols: list[str]) -> pd.DataFrame:
    """Evaluate trained models on classic out-of-distribution preset machines."""
    rows = []
    for name, (rules_txt, inp) in PRESETS.items():
        tm, err = parse_machine(rules_txt)
        assert tm is not None and not err
        feats = all_features(tm, inp)
        X = pd.DataFrame([[feats[c] for c in cols]], columns=cols).values
        for L in LIMITS:
            res = run(tm, inp, L)
            actual = int(res.status == "halted")
            for pop_name in ["random", "hard"]:
                for m_name in MODELS:
                    clf = app_models[pop_name][m_name][L]
                    prob = float(_prob_pos(clf, X)[0])
                    pred = int(prob >= 0.5)
                    rows.append(
                        {
                            "preset": name,
                            "limit": L,
                            "actual_status": res.status,
                            "actual_halts": actual,
                            "train_pop": pop_name,
                            "model": m_name,
                            "prob_halt": prob,
                            "pred_halts": pred,
                            "correct": int(pred == actual),
                        }
                    )
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- plots
def make_plots(
    summary: pd.DataFrame,
    cx: pd.DataFrame,
    fi: pd.DataFrame,
    sweep: pd.DataFrame,
) -> None:
    # 1. Random population accuracy vs limit (with +/- 1 std bands)
    rnd = summary[summary["population"] == "random"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, fs in zip(axes, FEATURE_SETS):
        for m in MODELS:
            d = rnd[(rnd["features"] == fs) & (rnd["model"] == m)].sort_values("limit")
            ax.plot(d["limit"], d["accuracy_mean"], marker="o", label=m)
            std = d["accuracy_std"].fillna(0)
            ax.fill_between(
                d["limit"], d["accuracy_mean"] - std, d["accuracy_mean"] + std, alpha=0.15
            )
        for m, st in [("Majority baseline", ":"), (f"Rule: simulate {PROBE_STEPS} steps", "--")]:
            d = rnd[rnd["model"] == m].sort_values("limit")
            ax.plot(d["limit"], d["accuracy_mean"], st, color="gray", label=m)
        ax.set_xscale("log")
        ax.set_title(f"Features: {fs}")
        ax.set_xlabel("step limit L")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy (mean ± std, 3 seeds)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Random population: predicting 'halts within L steps'")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_limit.png", dpi=130)
    plt.close(fig)

    # 2. Hard cases: F1 & PR-AUC across all 3 models + positive count bar chart
    hard = summary[summary["population"] == "hard"]
    probe_fs = f"static + {PROBE_STEPS}-step probe"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for m in MODELS:
        d = hard[(hard["features"] == probe_fs) & (hard["model"] == m)].sort_values("limit")
        axes[0].errorbar(
            d["limit"], d["f1_mean"], yerr=d["f1_std"].fillna(0), marker="o", capsize=3, label=m
        )
        axes[1].errorbar(
            d["limit"],
            d["pr_auc_mean"],
            yerr=d["pr_auc_std"].fillna(0),
            marker="s",
            capsize=3,
            label=m,
        )
    d_maj = hard[hard["model"] == "Majority baseline"].sort_values("limit")
    axes[0].plot(d_maj["limit"], d_maj["f1_mean"], ":", color="gray", label="Baselines (F1=0)")
    for ax, ylabel, title in [
        (axes[0], "F1 on 'halts within L'", f"Hard cases F1 ({probe_fs})"),
        (axes[1], "PR-AUC (Average Precision)", f"Hard cases PR-AUC ({probe_fs})"),
    ]:
        ax.set_xscale("log")
        ax.set_xlabel("step limit L")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[2].bar(range(len(d_maj)), d_maj["n_pos_test_mean"], tick_label=d_maj["limit"].astype(str))
    axes[2].set_xlabel("step limit L")
    axes[2].set_ylabel("# positive examples in test set")
    axes[2].set_title("Slow halters resolving by step L")
    axes[2].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "hard_cases.png", dpi=130)
    plt.close(fig)

    # 3. Complexity generalization (random 2-6 states, hard 3-10 states)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, pop_name, metric, cutoff, title in [
        (axes[0], "random", "accuracy", 4.5, "Random: trained on <=4 states (L=250)"),
        (axes[1], "hard", "f1", 6.5, "Hard: trained on <=6 states (L=250)"),
    ]:
        sub = cx[(cx["population"] == pop_name) & (cx["features"] == probe_fs)]
        for m in MODELS:
            d = sub[sub["model"] == m].sort_values("n_states")
            ax.plot(d["n_states"], d[metric], marker="o", label=f"{m} (probe)")
        d_st = cx[
            (cx["population"] == pop_name)
            & (cx["features"] == "static only")
            & (cx["model"] == "Random Forest")
        ].sort_values("n_states")
        ax.plot(d_st["n_states"], d_st[metric], "--", marker="^", color="gray", label="RF (static only)")
        ax.axvline(cutoff, color="black", ls=":", lw=1, label="unseen sizes ->")
        ax.set_xlabel("number of states in machine")
        ax.set_ylabel(metric.upper() if metric == "f1" else metric)
        ax.set_title(title)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_complexity.png", dpi=130)
    plt.close(fig)

    # 4. Feature importance (top 12 per population)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, pop_name in zip(axes, ["random", "hard"]):
        d = fi[fi["population"] == pop_name].head(12).iloc[::-1]
        colors = ["#1f77b4" if g == "static" else "#ff7f0e" for g in d["group"]]
        ax.barh(d["feature"], d["importance"], color=colors)
        ax.set_title(f"Top 12 RF features ({pop_name}, L=250)")
        ax.set_xlabel("Gini importance")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=130)
    plt.close(fig)

    # 5. Probe budget sweep
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in MODELS:
        d = sweep[sweep["model"] == m].sort_values("probe_k")
        axes[0].plot(d["probe_k"], d["f1"], marker="o", label=m)
        axes[1].plot(d["probe_k"], d["pr_auc"], marker="s", label=m)
    for ax, ylabel, title in [
        (axes[0], "F1 score", "Hard cases F1 vs probe steps k (L=250)"),
        (axes[1], "PR-AUC", "Hard cases PR-AUC vs probe steps k (L=250)"),
    ]:
        ax.set_xlabel("probe budget k (0 = static only)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "probe_sweep.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main(n: int) -> None:
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    print(f"[1/6] random population ({n} pairs)")
    rnd = build_dataset(n)
    rnd.to_csv(OUT / "dataset_random.csv", index=False)
    print(rnd["status"].value_counts().to_string(), f"   ({time.time()-t0:.0f}s)")
    res_rnd = run_study(rnd, "random")

    print(f"[2/6] hard-case population ({n} pairs unresolved after {PROBE_STEPS} steps)")
    hard = build_dataset(n, hard=True)
    hard.to_csv(OUT / "dataset_hard.csv", index=False)
    print(hard["status"].value_counts().to_string(), f"   ({time.time()-t0:.0f}s)")
    res_hard = run_study(hard, "hard")

    res = pd.concat([res_rnd, res_hard], ignore_index=True)
    res.to_csv(OUT / "results.csv", index=False)
    summary = summarize_results(res)
    summary.to_csv(OUT / "results_summary.csv", index=False)

    print("[3/6] complexity generalization study")
    cx = complexity_study(rnd, hard)
    cx.to_csv(OUT / "complexity.csv", index=False)

    print("[4/6] feature importance & probe budget sweep")
    fi = feature_importance_study(rnd, hard)
    fi.to_csv(OUT / "feature_importance.csv", index=False)
    sweep = probe_sweep_study(n=min(n, 2500))
    sweep.to_csv(OUT / "probe_sweep.csv", index=False)

    print("[5/6] training all app models (DT, RF, Balanced MLP) on random & hard")
    full_cols = STATIC_COLS + PROBE_COLS
    app_models: dict[str, dict[str, dict[int, object]]] = {"random": {}, "hard": {}}
    for pop_name, df_pop in [("random", rnd), ("hard", hard)]:
        for m_name in MODELS:
            app_models[pop_name][m_name] = {}
        X_full = df_pop[full_cols].values
        for L in LIMITS:
            y_full = label(df_pop, L)
            if y_full.sum() < 5:
                from sklearn.dummy import DummyClassifier

                for m_name in MODELS:
                    dc = DummyClassifier(strategy="constant", constant=0)
                    dc.fit(X_full, np.zeros(len(X_full), dtype=int))
                    app_models[pop_name][m_name][L] = dc
            else:
                for m_name, clf in make_models(BASE_SEED).items():
                    clf.fit(X_full, y_full)
                    app_models[pop_name][m_name][L] = clf

    joblib.dump(
        {"models": app_models, "cols": full_cols, "feature_importance": fi},
        OUT / "models.joblib",
        compress=3,
    )

    print("[6/6] out-of-distribution preset benchmarks & plots")
    ood = ood_benchmark_study(app_models, full_cols)
    ood.to_csv(OUT / "ood_benchmarks.csv", index=False)
    make_plots(summary, cx, fi, sweep)
    print(f"\nAll done in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="pairs per population")
    main(ap.parse_args().n)
