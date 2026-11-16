"""Turing machine simulator with honest halting outcomes.

A run ends in exactly one of three ways:
  halted  - machine reached the halt state (or had no applicable rule)
  looping - a full configuration (state, head, tape) repeated, so it PROVABLY never halts
  limit   - step budget ran out; we know NOTHING about whether it halts later

Distinguishing "limit" from "looping" is the key point of the project: the
general halting problem is undecidable, so "not halted yet" is not "never halts".
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

HALT = "H"
BLANK = "_"
SYMBOLS = ("_", "0", "1")

_MASK = (1 << 64) - 1


def _mix(x: int) -> int:
    """splitmix64-style hash, used for an incremental (Zobrist) tape hash."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


@dataclass
class TuringMachine:
    # (state, read_symbol) -> (write_symbol, move 'L'|'R'|'S', next_state)
    transitions: dict
    start: str = "q0"
    blank: str = BLANK

    @property
    def states(self) -> list[str]:
        seen = {self.start}
        for (s, _), (_, _, n) in self.transitions.items():
            seen.add(s)
            seen.add(n)
        seen.discard(HALT)
        return sorted(seen)


@dataclass
class RunResult:
    status: str  # "halted" | "looping" | "limit"
    steps: int  # steps executed
    state: str
    head: int
    tape: dict
    trace: list = field(default_factory=list)  # (step, state, head, tape_copy)

    def tape_string(self, blank: str = BLANK) -> str:
        if not self.tape:
            return ""
        lo, hi = min(self.tape), max(self.tape)
        return "".join(self.tape.get(i, blank) for i in range(lo, hi + 1))


def run(
    tm: TuringMachine,
    input_str: str,
    max_steps: int,
    record: bool = False,
    detect_loops: bool = True,
    exact_loop_check: bool = False,
) -> RunResult:
    """Run `tm` on `input_str` for at most `max_steps` steps."""
    blank = tm.blank
    tape = {i: c for i, c in enumerate(input_str) if c != blank}
    state, head = tm.start, 0

    h = 0
    for pos, sym in tape.items():
        h ^= _mix(pos * 256 + ord(sym))

    seen: set = set()
    seen_exact: dict = {}
    trace = []
    if record:
        trace.append((0, state, head, dict(tape)))

    steps = 0
    while steps < max_steps:
        if state == HALT:
            return RunResult("halted", steps, state, head, tape, trace)
        if detect_loops:
            if exact_loop_check:
                cfg = (state, head, frozenset(tape.items()))
                if cfg in seen_exact:
                    return RunResult("looping", steps, state, head, tape, trace)
                seen_exact[cfg] = True
            else:
                key = (state, head, h)
                if key in seen:
                    return RunResult("looping", steps, state, head, tape, trace)
                seen.add(key)

        read = tape.get(head, blank)
        rule = tm.transitions.get((state, read))
        if rule is None:  # no rule -> machine halts
            return RunResult("halted", steps, HALT, head, tape, trace)

        write, move, nxt = rule
        if read != blank:
            h ^= _mix(head * 256 + ord(read))
        if write != blank:
            h ^= _mix(head * 256 + ord(write))
            tape[head] = write
        else:
            tape.pop(head, None)

        head += 1 if move == "R" else -1 if move == "L" else 0
        state = nxt
        steps += 1
        if record and len(trace) < 3000:
            trace.append((steps, state, head, dict(tape)))

    if state == HALT:
        return RunResult("halted", steps, state, head, tape, trace)
    return RunResult("limit", steps, state, head, tape, trace)


# ---------------------------------------------------------------- text format
def format_machine(tm: TuringMachine) -> str:
    """One rule per line:  state read write move next   e.g.  q0 1 0 R q1"""
    lines = [f"# start: {tm.start}   halt state: {HALT}   blank: {tm.blank}"]
    for (s, a), (w, m, n) in sorted(tm.transitions.items()):
        lines.append(f"{s} {a} {w} {m} {n}")
    return "\n".join(lines)


def format_rules_compact(tm: TuringMachine) -> str:
    """Single-line semicolon-separated representation for dataset storage."""
    return "; ".join(
        f"{s} {a} {w} {m} {n}" for (s, a), (w, m, n) in sorted(tm.transitions.items())
    )


def parse_machine(
    text: str,
    start: str | None = None,
    blank: str = BLANK,
) -> tuple[TuringMachine | None, list[str]]:
    """Parse the text format. Returns (machine, errors)."""
    if len(blank) != 1:
        return None, ["blank symbol must be a single character"]
    transitions, errors, first_state = {}, [], None
    for ln, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if blank != "#":
            line = raw.split("#")[0].strip()
        else:
            parts_tmp = stripped.split()
            if len(parts_tmp) > 5 and parts_tmp[5].startswith("#"):
                line = " ".join(parts_tmp[:5])
            else:
                line = stripped
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {ln}: expected 'state read write move next'")
            continue
        s, a, w, m, n = parts
        if len(a) != 1 or len(w) != 1:
            errors.append(f"line {ln}: symbols must be single characters")
            continue
        if m not in ("L", "R", "S"):
            errors.append(f"line {ln}: move must be L, R or S")
            continue
        if (s, a) in transitions:
            errors.append(f"line {ln}: duplicate rule for ({s}, {a})")
            continue
        transitions[(s, a)] = (w, m, n)
        first_state = first_state or s
    if not transitions and not errors:
        errors.append("no rules found")
    if errors:
        return None, errors
    if start:
        chosen_start = start.strip()
    else:
        # Start state: explicit 'q0' if present, else the first rule's state
        chosen_start = "q0" if any(s == "q0" for s, _ in transitions) else first_state
    return TuringMachine(transitions, start=chosen_start, blank=blank), []


def to_dot(tm: TuringMachine, active_state: str | None = None) -> str:
    """Render the Turing machine's state transition diagram in Graphviz DOT."""
    grouped: dict[tuple[str, str], list[str]] = {}
    all_states = set(tm.states)
    has_halt = False
    for (s, a), (w, m, n) in sorted(tm.transitions.items()):
        all_states.add(s)
        if n == HALT:
            has_halt = True
        else:
            all_states.add(n)
        grouped.setdefault((s, n), []).append(f"{a}/{w},{m}")

    lines = [
        "digraph TM {",
        '  rankdir=LR;',
        '  node [shape=circle, fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9];',
        '  __entry [shape=point, width=0.1];',
        f'  __entry -> "{tm.start}";',
    ]
    for st in sorted(all_states):
        if st == active_state:
            lines.append(f'  "{st}" [style=filled, fillcolor="#f4b400", fontcolor="#000000", penwidth=2];')
        else:
            lines.append(f'  "{st}";')
    if has_halt or active_state == HALT:
        if active_state == HALT:
            lines.append(f'  "{HALT}" [shape=doublecircle, style=filled, fillcolor="#34a853", fontcolor="#ffffff", penwidth=2];')
        else:
            lines.append(f'  "{HALT}" [shape=doublecircle, style=filled, fillcolor="#e6f4ea"];')
    for (s, n), labels in sorted(grouped.items()):
        lbl = "\\n".join(labels).replace('"', '\\"')
        lines.append(f'  "{s}" -> "{n}" [label="{lbl}"];')
    lines.append("}")
    return "\n".join(lines)


# -------------------------------------------------------------------- presets
def _t(txt: str) -> TuringMachine:
    tm, err = parse_machine(txt)
    assert tm is not None, err
    return tm


PRESETS = {
    "Binary incrementer (halts)": (
        "# Head starts on the leftmost bit. Adds 1 to a binary number.\n"
        "q0 0 0 R q0\nq0 1 1 R q0\nq0 _ _ L q1\n"
        "q1 1 0 L q1\nq1 0 1 S H\nq1 _ 1 S H",
        "1011",
    ),
    "3-state busy beaver (halts)": (
        "# Classic 3-state 2-symbol champion (writes six 1s in 14 steps).\n"
        "q0 _ 1 R q1\nq0 1 1 S H\n"
        "q1 _ _ R q2\nq1 1 1 R q1\n"
        "q2 _ 1 L q2\nq2 1 1 L q0",
        "",
    ),
    "4-state busy beaver (halts in 107 steps)": (
        "# Classic 4-state 2-symbol champion (writes thirteen 1s in 107 steps).\n"
        "q0 _ 1 R q1\nq0 1 1 L q1\n"
        "q1 _ 1 L q0\nq1 1 _ L q2\n"
        "q2 _ 1 R H\nq2 1 1 L q3\n"
        "q3 _ 1 R q3\nq3 1 _ R q0",
        "",
    ),
    "2-state busy beaver (halts)": (
        "# Classic 2-state 2-symbol champion (writes four 1s in 6 steps).\n"
        "q0 _ 1 R q1\nq0 1 1 L q1\n"
        "q1 _ 1 L q0\nq1 1 1 R H",
        "",
    ),
    "Unary addition 111+11 (halts)": (
        "# Adds two unary numbers: 111+11 -> 11111.\n"
        "# Replace '+' by 1, run to the end, then erase one trailing 1.\n"
        "q0 1 1 R q0\nq0 + 1 R q1\n"
        "q1 1 1 R q1\nq1 _ _ L q2\n"
        "q2 1 _ S H",
        "111+11",
    ),
    "Binary palindrome checker (halts, writes Y/N)": (
        "# Repeatedly erases the first and last symbol and checks that they match.\n"
        "# Halts writing Y (palindrome) or N (not a palindrome). O(n^2) steps.\n"
        "q0 0 _ R q1\nq0 1 _ R q2\nq0 _ Y S H\n"
        "q1 0 0 R q1\nq1 1 1 R q1\nq1 _ _ L q3\n"
        "q2 0 0 R q2\nq2 1 1 R q2\nq2 _ _ L q4\n"
        "q3 0 _ L q5\nq3 1 N S H\nq3 _ Y S H\n"
        "q4 1 _ L q5\nq4 0 N S H\nq4 _ Y S H\n"
        "q5 0 0 L q5\nq5 1 1 L q5\nq5 _ _ R q0",
        "1011101",
    ),
    "Binary countdown to zero (slow halter)": (
        "# Decrements a binary number until it reaches 0, then halts.\n"
        "# Running time grows with the VALUE of the input, so it halts late.\n"
        "q0 0 0 R q0\nq0 1 1 R q1\nq0 _ _ S H\n"
        "q1 0 0 R q1\nq1 1 1 R q1\nq1 _ _ L q2\n"
        "q2 0 1 L q2\nq2 1 0 L q3\n"
        "q3 0 0 L q3\nq3 1 1 L q3\nq3 _ _ R q0",
        "1100",
    ),
    "Custom symbols a/b: flip & halt": (
        "# Demonstrates custom tape symbols beyond 0/1: swaps a <-> b and halts.\n"
        "q0 a b R q0\nq0 b a R q0\nq0 _ _ S H",
        "aabba",
    ),
    "Ping-pong (provably loops)": (
        "# Repeats a configuration -> provably never halts.\n"
        "q0 _ _ R q1\nq1 _ _ L q0",
        "",
    ),
    "Run-away right (never halts, but not provable by repetition)": (
        "# Writes 1s forever, moving right. Tape keeps growing, so no\n"
        "# configuration repeats: the simulator can only say 'limit reached'.\n"
        "q0 _ 1 R q0",
        "",
    ),
    "Binary counter (counts forever, never halts)": (
        "# Increments a binary number forever. The number keeps growing, so no\n"
        "# configuration ever repeats: a structured non-halter the simulator\n"
        "# can only report as 'limit reached'.\n"
        "q0 0 0 R q0\nq0 1 1 R q0\nq0 _ _ L q1\n"
        "q1 1 0 L q1\nq1 0 1 R q0\nq1 _ 1 R q0",
        "0",
    ),
}

# ------------------------------------------------------------ random machines
def random_machine(
    n_states: int,
    rng: random.Random,
    symbols=SYMBOLS,
    p_halt_range: tuple[float, float] = (0.03, 0.30),
    allow_stay: bool = False,
    p_missing: float = 0.0,
) -> TuringMachine:
    states = [f"q{i}" for i in range(n_states)]
    p_halt = rng.uniform(*p_halt_range)  # vary halting bias so labels stay mixed
    moves = "LRS" if allow_stay else "LR"
    trans = {}
    for s in states:
        for a in symbols:
            if p_missing > 0.0 and rng.random() < p_missing:
                continue
            w = rng.choice(symbols)
            m = rng.choice(moves)
            n = HALT if rng.random() < p_halt else rng.choice(states)
            trans[(s, a)] = (w, m, n)
    return TuringMachine(trans, start="q0")


def random_input(rng: random.Random, max_len: int = 8) -> str:
    return "".join(rng.choice("01") for _ in range(rng.randint(0, max_len)))


# The two machine populations studied in experiment.py (also used by the app's
# batch test, so the app always tests on the same distributions the models saw).
#   random : small machines with a fairly high chance of halting rules
#   hard   : bigger machines with fewer halting rules; callers additionally keep only
#            pairs that are still unresolved after the cheap probe run
RANDOM_STATES, RANDOM_P_HALT = (2, 6), (0.03, 0.30)
HARD_STATES, HARD_P_HALT = (4, 10), (0.01, 0.12)


def sample_machine(rng: random.Random, population: str) -> TuringMachine:
    """Draw one random machine from the 'random' or 'hard' population."""
    if population == "hard":
        return random_machine(
            rng.randint(*HARD_STATES), rng, p_halt_range=HARD_P_HALT, allow_stay=rng.random() < 0.25
        )
    return random_machine(rng.randint(*RANDOM_STATES), rng, p_halt_range=RANDOM_P_HALT)
