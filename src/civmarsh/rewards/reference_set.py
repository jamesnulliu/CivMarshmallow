"""Periodic rebuild of the CivTelescope reference set from the training policy.

The potential's target is non-stationary: as the policy improves, a frozen
reference set saturates.  Every ``civ_value_refresh_every`` training steps the
set is rebuilt from the freshest policy episodes, which the value functions
harvest during the training rollouts themselves (``civ_value_harvest``).

Trigger: ``train.reward_post.post_process`` calls ``maybe_refresh(args)`` at
the end of each step.  Rebuild and validation are synchronous, so the set
never changes mid-step and Phi stays consistent within a batch.

Recipe, per bucket turn of the AI reference pool: from the pool episodes'
renderings at that turn, ``N_POLICY`` positions evenly spaced by final score,
plus the ``N_AI`` strongest positions of the AI pool.  A bucket the pool
episodes never reach (an episode that starts mid-game) keeps its current AI
positions.  The newest ``POOL_EPS`` harvested episodes build the candidate;
the ``VAL_EPS`` after them validate it with the RUNTIME value configuration:
Spearman(Phi, final score) at every bucket turn divisible by 20 and at the
last decision.  A candidate below ``STOP_BELOW`` (or NaN) at turn 40 or at the
last decision is discarded with a warning and the in-use set is kept; a
passing candidate atomically replaces the reference-set file (``os.replace``),
which ``potential.load_reference_set`` picks up by mtime.

Config keys (read from the slime args):
  civ_value_refresh_every  rebuild every N steps (default 4)
  civ_value_ai_refset      AI reference pool jsonl; the refresh runs only when
                           this is set
  civ_value_harvest        harvest jsonl written by the value functions
  civ_value_refset         the live reference-set file that is replaced
  civ_value_refresh_log    optional jsonl, one report line per refresh
"""

from __future__ import annotations

import json
import math
import os
import statistics
from types import SimpleNamespace

REFRESH_EVERY = 4
N_POLICY, N_AI = 6, 2
POOL_EPS = 40  # newest harvested episodes that build the candidate
VAL_EPS = 40  # the held-out episodes after them that validate it
MIN_EPISODES = POOL_EPS + 20  # no rebuild on thinner evidence
STOP_BELOW = 0.2  # promotion bar on the "40" and "last" probes
GATE_PROBES = ("40", "last")
_step_counter = {"n": 0}


def _spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    den = (sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry)) ** 0.5
    return num / den if den else float("nan")


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_reference_set(pool, ai_refset_path):
    """Per AI-pool bucket: N_POLICY evenly score-spaced policy positions plus
    the N_AI strongest AI positions.  Returns (rows, rebuilt bucket turns);
    rows is None when no bucket had enough policy candidates."""
    ai = {}
    for r in _read_jsonl(ai_refset_path):
        ai.setdefault(r["turn"], []).append(r)

    rows, rebuilt = [], []
    for t in sorted(ai):
        cands = []
        for e in pool:
            d = next((d for d in e["decisions"] if d["turn"] == t), None)
            if d is not None:
                cands.append(
                    {
                        "turn": t,
                        "focal": d["focal"],
                        "rendering": d["rendering"],
                        "end_score": e["score_end"],
                        "position_id": f"policy_T{t}",
                    }
                )
        if len(cands) < N_POLICY:
            rows.extend(ai[t])  # bucket not reached by the pool: keep as is
            continue
        cands.sort(key=lambda r: r["end_score"])
        idx = sorted(
            {round(i * (len(cands) - 1) / (N_POLICY - 1)) for i in range(N_POLICY)}
        )
        picked = [cands[i] for i in idx]
        strongest = sorted(ai[t], key=lambda r: r["end_score"])[-N_AI:]
        rows.extend(picked + strongest)
        rebuilt.append(t)
    if not rebuilt:
        return None, []
    return rows, rebuilt


def validate_reference_set(candidate_path, val_eps, args):
    """Held-out Spearman(Phi, final score) per probe, computed with the
    runtime value configuration but the candidate reference-set file."""
    from .potential import DEFAULT_CONCURRENCY, telescope_value

    val_args = SimpleNamespace(
        civ_value_url=args.civ_value_url,
        civ_value_refset=candidate_path,
        civ_value_concurrency=getattr(
            args, "civ_value_concurrency", DEFAULT_CONCURRENCY
        ),
        civ_value_refset_idx=getattr(args, "civ_value_refset_idx", None),
        civ_value_single_order=getattr(args, "civ_value_single_order", True),
    )
    buckets = {r["turn"] for r in _read_jsonl(candidate_path)}
    probe_turns = sorted(t for t in buckets if t % 20 == 0) or [20, 40, 60]
    probes = {t: [] for t in probe_turns}
    probes["last"] = []
    for e in val_eps:
        decs, tags = [], []
        for t in probe_turns:
            d = next((d for d in e["decisions"] if d["turn"] == t), None)
            if d:
                decs.append(
                    {"turn": t, "focal": d["focal"], "value_input": d["rendering"]}
                )
                tags.append(t)
        d = e["decisions"][-1]
        decs.append(
            {"turn": d["turn"], "focal": d["focal"], "value_input": d["rendering"]}
        )
        tags.append("last")
        vals = telescope_value(decs, {"score_end": e["score_end"]}, val_args)
        for tag, v in zip(tags, vals):
            probes[tag].append((v, e["score_end"]))

    report = {}
    for tag, pairs in probes.items():
        if len(pairs) > 3:
            report[str(tag)] = {
                "spearman": round(
                    _spearman([p for p, _ in pairs], [s for _, s in pairs]), 3
                ),
                "phi_sd": round(statistics.stdev([p for p, _ in pairs]), 2),
                "n": len(pairs),
            }
    return report


def refresh(args, step):
    """One rebuild + validate + promote cycle; returns the report (or a skip
    record).  A failing candidate is removed and the in-use set kept."""
    harvest_path = args.civ_value_harvest
    refset_path = args.civ_value_refset
    ai_path = args.civ_value_ai_refset

    eps = _read_jsonl(harvest_path)
    if len(eps) < MIN_EPISODES:
        report = {
            "step": step,
            "skipped": f"only {len(eps)} harvested episodes (< {MIN_EPISODES})",
        }
        _write_report(args, report)
        return report
    window = eps[-(POOL_EPS + VAL_EPS) :]
    pool, val = window[:POOL_EPS], window[POOL_EPS:]

    rows, rebuilt = build_reference_set(pool, ai_path)
    if rows is None:
        report = {"step": step, "skipped": "no bucket had enough policy candidates"}
        _write_report(args, report)
        return report

    candidate = refset_path + ".candidate"
    with open(candidate, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    report = validate_reference_set(candidate, val, args)
    report["step"] = step
    report["n_pool"] = len(pool)
    report["n_val"] = len(val)
    report["rebuilt_buckets"] = rebuilt

    bad = []
    for tag in GATE_PROBES:
        s = report.get(tag, {}).get("spearman", 1.0)
        # A zero-variance Phi (e.g. harvested from a collapsed policy) gives
        # Spearman NaN, and NaN < x is False: test it explicitly.
        if math.isnan(s) or s < STOP_BELOW:
            bad.append(tag)
    if bad:
        report["promoted"] = False
        report["warning"] = (
            f"validation below {STOP_BELOW} on {bad}; keeping the in-use reference set"
        )
        os.remove(candidate)
        _write_report(args, report)
        return report

    os.replace(candidate, refset_path)
    report["promoted"] = True
    _write_report(args, report)
    return report


def _write_report(args, report):
    log_path = getattr(args, "civ_value_refresh_log", None)
    if log_path:
        with open(log_path, "a") as f:
            f.write(json.dumps(report) + "\n")
    print(f"[refset-refresh] {json.dumps(report)}", flush=True)


def maybe_refresh(args):
    """Per-step hook (call once per training step).

    Active only when an AI reference pool is configured.  Steps are counted
    in-process, so a restarted run restarts the cadence; that only delays the
    next rebuild and never skips validation of a promoted set.
    """
    if not getattr(args, "civ_value_ai_refset", None):
        return None
    every = getattr(args, "civ_value_refresh_every", REFRESH_EVERY)
    if not every:
        return None
    _step_counter["n"] += 1
    step = _step_counter["n"]
    if step % every:
        return None
    return refresh(args, step)
