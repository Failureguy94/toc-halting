"""Feature extraction for (machine, input) pairs.

Two feature groups, so the experiments can compare them:
  static  - read straight off the transition table + input (no execution)
  probe   - behaviour observed during a SHORT trial run (PROBE_STEPS steps)
"""
from __future__ import annotations

from collections import deque

from tm import HALT, TuringMachine, run

PROBE_STEPS = 10


def _state_graph_distance_to_halt(tm: TuringMachine) -> tuple[int, int, int]:
    """BFS over the state graph (ignoring the tape).

    Returns (reachable_states, min_distance_to_halt or 99, halting_rules_reachable).
    """
    adj: dict[str, set[str]] = {}
    for (s, _), (_, _, n) in tm.transitions.items():
        adj.setdefault(s, set()).add(n)
    dist = {tm.start: 0}
    q = deque([tm.start])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    reachable = {s for s in dist if s != HALT}
    halting_rules = sum(
        1 for (s, _), (_, _, n) in tm.transitions.items() if s in reachable and n == HALT
    )
    return len(reachable), dist.get(HALT, 99), halting_rules


def static_features(tm: TuringMachine, input_str: str) -> dict:
    rules = list(tm.transitions.items())
    n = len(rules)
    n_states = len(tm.states)
    reach, d_halt, halt_rules = _state_graph_distance_to_halt(tm)
    start_rules = [(k, v) for k, v in rules if k[0] == tm.start]

    self_loops = sum(1 for (s, _), (_, _, nx) in rules if s == nx)
    # "sweepers": rule that keeps its state, rewrites the same symbol and keeps moving
    sweep_r = sum(1 for (s, a), (w, m, nx) in rules if s == nx and w == a and m == "R")
    sweep_l = sum(1 for (s, a), (w, m, nx) in rules if s == nx and w == a and m == "L")
    # "run-away writers": same state, on blank, writes non-blank, moves on
    runaway = sum(
        1 for (s, a), (w, m, nx) in rules
        if s == nx and a == tm.blank and w != tm.blank and m in ("L", "R")
    )
    rule_symbols = {a for _, a in tm.transitions} | {w for w, _, _ in tm.transitions.values()} | {tm.blank}
    nonblank_in = [c for c in input_str if c != tm.blank]
    return {
        "n_states": n_states,
        "n_rules": n,
        "alphabet_size": len(rule_symbols),
        "frac_to_halt": sum(1 for _, (_, _, nx) in rules if nx == HALT) / max(n, 1),
        "frac_move_right": sum(1 for _, (_, m, _) in rules if m == "R") / max(n, 1),
        "frac_move_stay": sum(1 for _, (_, m, _) in rules if m == "S") / max(n, 1),
        "frac_write_blank": sum(1 for _, (w, _, _) in rules if w == tm.blank) / max(n, 1),
        "frac_self_loop": self_loops / max(n, 1),
        "sweepers_right": sweep_r,
        "sweepers_left": sweep_l,
        "runaway_writers": runaway,
        "reachable_states": reach,
        "dist_to_halt": d_halt,
        "halting_rules_reachable": halt_rules,
        "start_halting_rules": sum(1 for _, (_, _, nx) in start_rules if nx == HALT),
        "input_len": len(input_str),
        "input_nonblank": len(nonblank_in),
        "input_unique_symbols": len(set(nonblank_in)),
        "input_ones": input_str.count("1"),
    }


def probe_features(tm: TuringMachine, input_str: str, k: int = PROBE_STEPS) -> dict:
    res = run(tm, input_str, k, record=True)
    heads = [h for _, _, h, _ in res.trace]
    states = {s for _, s, _, _ in res.trace}
    init_nonblank = sum(1 for c in input_str if c != tm.blank)
    return {
        "probe_halted": int(res.status == "halted"),
        "probe_looping": int(res.status == "looping"),
        "probe_steps": res.steps,
        "probe_distinct_states": len(states),
        "probe_head_span": max(heads) - min(heads) + 1,
        "probe_net_move": heads[-1] - heads[0],
        "probe_nonblank_cells": len(res.tape),
        "probe_tape_growth": len(res.tape) - init_nonblank,
    }


def all_features(tm: TuringMachine, input_str: str, k: int = PROBE_STEPS) -> dict:
    f = static_features(tm, input_str)
    f.update(probe_features(tm, input_str, k=k))
    return f


STATIC_COLS = list(static_features(TuringMachine({("q0", "_"): ("_", "R", "H")}), "").keys())
PROBE_COLS = list(
    probe_features(TuringMachine({("q0", "_"): ("_", "R", "H")}), "").keys()
)
