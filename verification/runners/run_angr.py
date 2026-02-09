#!/usr/bin/env python3
"""
Minimal angr runner (baseline)

Usage:
  python run_angr.py /path/to/request.json /path/to/response.json

Contract:
  - Reads request JSON (see schemas/request.schema.json)
  - Loads binary at /bin_in/<binary_name> by default
  - Symbolic-exec from entry function using call_state
  - Applies constraints (mini DSL)
  - Searches for "find" addr_reached; optionally avoids "avoid" addr_avoided
  - Enforces budgets: timeout, max path length, loop bound, state cap
  - Writes response JSON (see schemas/response.schema.json)
"""

import json, sys, time, re
from typing import Dict, Any

import angr
import claripy


def _parse_addr(proj: angr.Project, value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s.lower().startswith("0x"):
            return int(s, 16)
        sym = proj.loader.find_symbol(s)
        if sym is not None:
            return sym.rebased_addr
        # fallback CFG-based function resolution
        try:
            cfg = proj.analyses.CFGFast(normalize=True)
            f = cfg.kb.functions.function(name=s)
            if f is not None:
                return int(f.addr)
        except Exception:
            pass
        raise ValueError(f"Could not resolve address/symbol: {value!r}")
    raise TypeError(f"Unsupported address type: {type(value)}")


_cmp_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|<=|>=|<|>)\s*([A-Za-z_][A-Za-z0-9_]*|0x[0-9a-fA-F]+|-?\d+)\s*$")

def _parse_atom(atom: str, symmap: Dict[str, claripy.ast.BV]) -> claripy.ast.BV:
    atom = atom.strip()
    if atom in symmap:
        return symmap[atom]
    if atom.lower().startswith("0x"):
        return claripy.BVV(int(atom, 16), 64)
    if re.fullmatch(r"-?\d+", atom):
        return claripy.BVV(int(atom), 64)
    raise ValueError(f"Unknown atom in constraint: {atom!r}")

def _apply_constraints(state: angr.SimState, constraints, symmap: Dict[str, claripy.ast.BV]):
    for c in constraints:
        m = _cmp_re.match(c)
        if not m:
            raise ValueError(f"Constraint parse error: {c!r}")
        lhs, op, rhs = m.group(1), m.group(2), m.group(3)
        a = _parse_atom(lhs, symmap)
        b = _parse_atom(rhs, symmap)
        w = max(a.size(), b.size())
        if a.size() != w: a = a.zero_extend(w - a.size())
        if b.size() != w: b = b.zero_extend(w - b.size())
        if op == "==":
            state.solver.add(a == b)
        elif op == "!=":
            state.solver.add(a != b)
        elif op == "<=":
            state.solver.add(a <= b)
        elif op == ">=":
            state.solver.add(a >= b)
        elif op == "<":
            state.solver.add(a < b)
        elif op == ">":
            state.solver.add(a > b)
        else:
            raise ValueError(f"Unknown operator: {op}")

def _mk_symbol(spec: dict) -> claripy.ast.BV:
    name = spec["name"]
    kind = spec["kind"]
    bits = int(spec["bits"])
    if kind == "bool":
        return claripy.BVS(name, 1)
    if kind == "int":
        return claripy.BVS(name, bits)
    if kind == "bytes":
        ln = spec.get("length")
        if ln is None:
            raise ValueError("bytes symbol requires length")
        return claripy.BVS(name, int(ln) * 8)
    raise ValueError(f"Unsupported kind: {kind!r}")

def _write_error(resp_path: str, msg: str, t0: float, extra: dict):
    out = {
        "status": "error",
        "counterexample": None,
        "stats": {"elapsed_s": time.time() - t0, **extra},
        "logs": msg
    }
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

def main(req_path: str, resp_path: str):
    t0 = time.time()
    with open(req_path, "r", encoding="utf-8") as f:
        req = json.load(f)

    target = req["target"]
    binary_name = target["binary_name"]
    entry = target["entry"]
    budget = req["budget"]
    constraints = req.get("constraints", [])
    find = req["find"]
    avoid = req.get("avoid")

    bin_path = f"/bin_in/{binary_name}"
    proj = angr.Project(bin_path, auto_load_libs=False)

    try:
        entry_addr = _parse_addr(proj, entry)
    except Exception as e:
        _write_error(resp_path, f"entry resolve error: {e}", t0, {})
        return

    sym_specs = req.get("symbols", [])
    syms = []
    symmap: Dict[str, claripy.ast.BV] = {}
    for s in sym_specs:
        sym = _mk_symbol(s)
        symmap[s["name"]] = sym
        syms.append(sym)

    state = proj.factory.call_state(entry_addr, *syms)

    try:
        _apply_constraints(state, constraints, symmap)
    except Exception as e:
        _write_error(resp_path, f"constraint error: {e}", t0, {"constraints": constraints})
        return

    try:
        if find["kind"] != "addr_reached":
            _write_error(resp_path, f"unsupported find kind: {find['kind']}", t0, {})
            return
        find_addr = _parse_addr(proj, find["value"])
        avoid_addr = None
        if avoid is not None:
            if avoid["kind"] != "addr_avoided":
                _write_error(resp_path, f"unsupported avoid kind: {avoid['kind']}", t0, {})
                return
            avoid_addr = _parse_addr(proj, avoid["value"])
    except Exception as e:
        _write_error(resp_path, f"address resolve error: {e}", t0, {})
        return

    simgr = proj.factory.simgr(state)

    # Exploration governors (best-effort; some angr builds may vary)
    try:
        simgr.use_technique(angr.exploration_techniques.Veritesting())
    except Exception:
        pass
    try:
        simgr.use_technique(angr.exploration_techniques.LengthLimiter(int(budget["max_path_len"]), drop=True))
    except Exception:
        pass
    try:
        simgr.use_technique(angr.exploration_techniques.LoopSeer(bound=int(budget["max_loop_iters"])))
    except Exception:
        pass
    try:
        simgr.use_technique(angr.exploration_techniques.Timeout(int(budget["timeout_s"])))
    except Exception:
        pass

    max_states = int(budget["max_states"])
    steps = 0
    found_state = None

    def prune():
        if len(simgr.active) > max_states:
            simgr.active = simgr.active[:max_states]

    try:
        while simgr.active and (time.time() - t0) < int(budget["timeout_s"]):
            simgr.step()
            steps += 1
            prune()

            if avoid_addr is not None:
                simgr.move(from_stash="active", to_stash="avoid", filter_func=lambda s: s.addr == avoid_addr)

            for s in simgr.active:
                if s.addr == find_addr:
                    found_state = s
                    break
            if found_state is not None:
                break
    except Exception as e:
        _write_error(resp_path, f"execution error: {e}", t0, {"steps": steps})
        return

    elapsed = time.time() - t0
    stats = {
        "elapsed_s": elapsed,
        "steps": steps,
        "active": len(simgr.active),
        "deadended": len(simgr.deadended),
        "errored": len(simgr.errored),
    }

    if found_state is None:
        out = {"status": "no_ce_within_budget", "counterexample": None, "stats": stats, "logs": None}
        with open(resp_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return

    model = {}
    for spec in sym_specs:
        name = spec["name"]
        sym = symmap[name]
        model[name] = int(found_state.solver.eval(sym, cast_to=int))

    trace = [{"addr": hex(a)} for a in list(found_state.history.bbl_addrs)[-500:]]

    out = {
        "status": "ce_found",
        "counterexample": {
            "model": model,
            "trace": trace,
            "path_constraints": [],
            "notes": {
                "found_addr": hex(find_addr),
                "avoid_addr": hex(avoid_addr) if avoid_addr is not None else None,
                "binary": bin_path,
                "entry": hex(entry_addr),
            }
        },
        "stats": stats,
        "logs": None
    }
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run_angr.py request.json response.json", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
