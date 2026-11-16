"""Unit tests for the Turing machine simulator and feature extractor."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import PROBE_COLS, STATIC_COLS, all_features, probe_features, static_features
from tm import (
    HALT,
    PRESETS,
    TuringMachine,
    format_machine,
    format_rules_compact,
    parse_machine,
    run,
    to_dot,
)


def test_binary_incrementer():
    rules, inp = PRESETS["Binary incrementer (halts)"]
    tm, err = parse_machine(rules)
    assert err == []
    assert tm is not None
    res = run(tm, inp, 100, record=True)
    assert res.status == "halted"
    assert res.tape_string() == "1100"
    assert res.trace[0][0] == 0


def test_busy_beaver_3_and_4():
    rules3, inp3 = PRESETS["3-state busy beaver (halts)"]
    tm3, _ = parse_machine(rules3)
    res3 = run(tm3, inp3, 100, exact_loop_check=True)
    assert res3.status == "halted"
    assert res3.steps == 14
    assert res3.tape_string().count("1") == 6

    rules4, inp4 = PRESETS["4-state busy beaver (halts in 107 steps)"]
    tm4, _ = parse_machine(rules4)
    res4 = run(tm4, inp4, 200)
    assert res4.status == "halted"
    assert res4.steps == 107
    assert res4.tape_string().count("1") == 13


def test_looping_vs_limit():
    rules_loop, inp_loop = PRESETS["Ping-pong (provably loops)"]
    tm_loop, _ = parse_machine(rules_loop)
    assert run(tm_loop, inp_loop, 100, exact_loop_check=False).status == "looping"
    assert run(tm_loop, inp_loop, 100, exact_loop_check=True).status == "looping"

    rules_run, inp_run = PRESETS["Run-away right (never halts, but not provable by repetition)"]
    tm_run, _ = parse_machine(rules_run)
    res_run = run(tm_run, inp_run, 200, exact_loop_check=True)
    assert res_run.status == "limit"
    assert res_run.steps == 200


def test_missing_rule_halts():
    tm = TuringMachine({("q0", "1"): ("0", "R", "q1")}, start="q0")
    res = run(tm, "1", 50)
    assert res.status == "halted"
    assert res.state == HALT
    assert res.steps == 1


def test_custom_start_and_blank_and_dot():
    rules = "s0 # a R s1\ns1 a b S H"
    tm, err = parse_machine(rules, start="s0", blank="#")
    assert err == []
    assert tm is not None
    assert tm.start == "s0"
    assert tm.blank == "#"
    res = run(tm, "", 20)
    assert res.status == "halted"
    assert res.tape_string("#") == "a"
    dot = to_dot(tm, active_state="s1")
    assert "digraph TM" in dot
    assert '"s0" -> "s1"' in dot
    assert "# start: s0" in format_machine(tm)
    assert "; " in format_rules_compact(tm)


def test_parser_errors():
    assert parse_machine("")[0] is None
    assert parse_machine("q0 0 1 X q1")[0] is None
    assert parse_machine("q0 00 1 R q1")[0] is None
    assert parse_machine("q0 0 1 R q1\nq0 0 0 L q1")[0] is None
    assert parse_machine("q0 0 1 R H", blank="__")[0] is None


def test_features_static_and_probe():
    rules, inp = PRESETS["Custom symbols a/b: flip & halt"]
    tm, _ = parse_machine(rules)
    sf = static_features(tm, inp)
    pf = probe_features(tm, inp, k=10)
    af = all_features(tm, inp)
    assert list(sf.keys()) == STATIC_COLS
    assert list(pf.keys()) == PROBE_COLS
    assert set(af.keys()) == set(STATIC_COLS) | set(PROBE_COLS)
    assert sf["alphabet_size"] == 3  # _, a, b
    assert sf["input_nonblank"] == 5
    assert sf["input_unique_symbols"] == 2
    assert pf["probe_halted"] == 1

    # Unreachable halt state -> dist_to_halt == 99
    tm_no_halt, _ = parse_machine("q0 _ _ R q0")
    assert static_features(tm_no_halt, "")["dist_to_halt"] == 99

