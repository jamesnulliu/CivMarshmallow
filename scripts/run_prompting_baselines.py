#!/usr/bin/env python3
"""Play the prompting-scaffold baselines on a set of starts.

Five arms on the same base model, the same starts and the same engine
interface as the RL policy:

  direct     no scaffold: one call, the RL task contract, no reasoning ask.
             The untrained-base floor.
  baselang   one unstructured chain-of-thought call per decision.
  mastaba    three department advisors + one president call per decision.
  saga       mid-term goal refreshed every N turns + a decide call.
  reflexion  lessons written at the end of each game, injected into the next
             game on the same start.

Each arm writes ``<out-dir>/<arm>.jsonl`` (one record per episode) and the run
writes ``<out-dir>/summary.json``.  Reflexion plays the samples of one start in
order (sample k reads the lessons of samples < k), so that arm runs starts in
parallel and samples serially; every other arm parallelizes both.

The policy server is a pure-inference sglang server serving the base model
under the name ``policy`` (see scripts/run_prompting_baselines.sh).  The
freeciv server binary comes from ``CIVHARNESS_SERVER``.

Usage:
  python scripts/run_prompting_baselines.py --url http://127.0.0.1:30900 \\
      --starts starts_test_rem120.jsonl --data-root /path/to/data \\
      --out-dir runs/prompting_baselines
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from civmarsh.baselines import driver, scaffolds

ARMS = ["direct", "baselang", "mastaba", "saga", "reflexion"]
SERIAL_ARMS = {"reflexion"}  # cross-episode memory => ordered samples


class Cfg:
    """Stand-in for slime's args namespace: the civ_* keys the engine thread
    (``civmarsh.train.episode``, which also fixes the harness timeouts) and
    the driver read."""

    def __init__(self, workdir, endturn, url):
        self.civ_workdir = str(workdir)
        self.civ_endturn = endturn
        self.civ_engine_seed = 1
        self.civ_digest_window = 5
        self.civ_baseline_llm_url = url


def load_starts(path, data_root, n=None):
    """Starts-bank rows with ``save_path`` resolved against ``data_root``."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = rows[:n] if n else rows
    root = Path(data_root)
    for meta in rows:
        meta["save_path"] = str(root / meta["save_path"])
    return rows


def _done_pairs(path):
    """(position_id, sample_idx) already finished without an engine error in
    an existing arm log: a failed row is retried, a finished one is never
    replayed."""
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # truncated tail of a killed run
                continue
            if not r.get("error"):
                done.add((r.get("position_id"), r.get("sample_idx")))
    return done


def _append_line(path, line):
    with open(path, "a") as f:
        f.write(line + "\n")


def _read_records(path):
    recs = []
    with open(path) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


async def _one(
    arm, scaffold, meta, cfg, llm, sample_idx, memory, sem, out_path, out_lock, timeout
):
    async with sem:
        t0 = time.time()
        try:
            rec = await asyncio.wait_for(
                driver.run_episode(scaffold, meta, cfg, llm=llm, memory=memory),
                timeout=timeout,
            )
        except BaseException as exc:  # noqa: BLE001 -- one bad game, not a stop
            rec = {
                "position_id": meta.get("position_id"),
                "scaffold": arm,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rec["arm"] = arm
        rec["sample_idx"] = sample_idx
        rec["wall_s"] = round(time.time() - t0, 1)
        async with out_lock:
            await asyncio.to_thread(_append_line, out_path, json.dumps(rec))
        tokens = rec.get("tokens") or {}
        bad = sum(1 for d in rec.get("decisions", []) if d.get("error"))
        print(
            f"[{arm}] {rec.get('position_id')} s{sample_idx} "
            f"score={rec.get('score_end')} elim={rec.get('eliminated')} "
            f"lost={rec.get('control_lost')} err={rec.get('error')} "
            f"dec={rec.get('n_decisions')} invalid={bad} "
            f"tok={tokens.get('total_tokens')} calls={tokens.get('n_calls')} "
            f"wall={rec['wall_s']}s",
            flush=True,
        )
        return rec


async def run_arm(arm, starts, args, out_dir):
    scaffold = scaffolds.SCAFFOLDS[arm]()
    workdir = Path(args.workdir) / arm
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = Cfg(workdir, args.endturn, args.url)
    # the floor arm keeps the RL rollout's decode budget per decision, so its
    # invalid rate is measured under the same limit as the trained policy's
    max_tokens = args.direct_max_tokens if arm == "direct" else args.max_tokens
    llm = driver.make_sglang_llm(
        args.url,
        max_tokens=max_tokens,
        temperature=args.temperature,
        extra_body=args.extra_body,
    )

    out_path = out_dir / f"{arm}.jsonl"
    done = _done_pairs(out_path) if args.resume else set()
    if not args.resume:
        out_path.write_text("")
    memory = {}
    mem_path = out_dir / f"{arm}_lessons.json"
    if args.resume and arm in SERIAL_ARMS and mem_path.exists():
        memory = json.loads(mem_path.read_text())

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    def episode(meta, i):
        return _one(
            arm,
            scaffold,
            meta,
            cfg,
            llm,
            i,
            memory,
            sem,
            out_path,
            lock,
            args.episode_timeout,
        )

    async def start_job(meta):
        out = []
        todo = [
            i for i in range(args.n_samples) if (meta["position_id"], i) not in done
        ]
        if arm in SERIAL_ARMS:
            for i in todo:  # sample k must see the lessons of samples < k
                out.append(await episode(meta, i))
                mem_path.write_text(json.dumps(memory, indent=1))
        else:
            out.extend(await asyncio.gather(*[episode(meta, i) for i in todo]))
        return out

    n_skip = sum(
        1
        for m in starts
        for i in range(args.n_samples)
        if (m["position_id"], i) in done
    )
    mode = (
        "starts in parallel, samples serial"
        if arm in SERIAL_ARMS
        else f"concurrency {args.concurrency}"
    )
    print(
        f"\n=== ARM {arm}: {len(starts)} starts x {args.n_samples} samples "
        f"({n_skip} already done, skipped), {mode}, max_tokens={max_tokens} ===",
        flush=True,
    )
    t0 = time.time()
    await asyncio.gather(*[start_job(meta) for meta in starts])
    print(f"=== ARM {arm} done in {(time.time() - t0) / 60:.1f} min ===", flush=True)

    # summarize from the log, so a resumed arm summarizes all of its episodes
    summary = driver.summarize_arm(driver.final_records(_read_records(out_path)))
    print(json.dumps(summary, indent=1), flush=True)
    return summary


async def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default="http://127.0.0.1:30900", help="policy server")
    ap.add_argument("--starts", required=True, help="starts-bank jsonl")
    ap.add_argument(
        "--data-root",
        required=True,
        help="directory the starts' relative save_path values resolve against",
    )
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--n-starts", type=int, default=None, help="first N starts only")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--endturn", type=int, default=120)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--workdir", default=None, help="engine scratch (default: <out-dir>/work)"
    )
    ap.add_argument("--concurrency", type=int, default=28)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument(
        "--direct-max-tokens",
        type=int,
        default=192,
        help="decode budget of the direct arm (the RL rollout's per-decision budget)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="sampling temperature (the RL rollout's)",
    )
    ap.add_argument("--episode-timeout", type=float, default=2400)
    ap.add_argument(
        "--chat-template-kwargs",
        default='{"enable_thinking": false}',
        help="JSON object sent as chat_template_kwargs with every request; "
        "'' sends none",
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.extra_body = (
        {"chat_template_kwargs": json.loads(args.chat_template_kwargs)}
        if args.chat_template_kwargs
        else None
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.workdir = args.workdir or str(out_dir / "work")
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    starts = load_starts(args.starts, args.data_root, args.n_starts)
    missing = [m["position_id"] for m in starts if not Path(m["save_path"]).is_file()]
    if missing:
        print(f"missing start saves under {args.data_root}: {missing}", flush=True)
        return 3
    arms = [a for a in args.arms.split(",") if a]
    unknown = [a for a in arms if a not in scaffolds.SCAFFOLDS]
    if unknown:
        print(f"unknown arms: {unknown}; choose from {sorted(scaffolds.SCAFFOLDS)}")
        return 2

    print(
        f"arms={arms} starts={[s['position_id'] for s in starts]} "
        f"x{args.n_samples} endturn={args.endturn} url={args.url} "
        f"out={out_dir} workdir={args.workdir}",
        flush=True,
    )

    config = {
        "starts": args.starts,
        "n_samples": args.n_samples,
        "endturn": args.endturn,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "direct_max_tokens": args.direct_max_tokens,
    }
    t0 = time.time()
    summaries = {}
    for arm in arms:  # one arm at a time: a whole arm's log lands intact
        summaries[arm] = await run_arm(arm, starts, args, out_dir)
        (out_dir / "summary.json").write_text(
            json.dumps(
                {
                    "arms": summaries,
                    "wall_min": round((time.time() - t0) / 60, 1),
                    "config": config,
                },
                indent=1,
            )
        )
    print(f"total wall {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
