"""Position pools, decidable pairs, and CivTelescope's training and evaluation
sets.

A position row is ``{position_id, turn, group, score_now, value, se,
rendering, focal_player}``: `group` is the game or rollout episode the position
comes from, `score_now` the focal player's visible score (read from the
rendering), `value` its long-run outcome and `se` the standard error of that
outcome.

Three pools feed the training set:

- ``replay``: replay-bank positions; `value` is the replay-oracle mean end score
  and `se` its standard error.
- ``rollout_early`` / ``rollout_late``: positions visited by policies during RL
  training (an early and a late window of training steps); `value` is the score
  the episode actually ended with and `se` is 0.

A pair joins two positions from different groups at most `MAX_TURN_GAP` turns
apart. It is decidable when the outcomes are separated: for replay positions
the gap exceeds ``MIN_GAP_Z`` pooled standard errors (or both standard errors
are 0); for rollout positions the gap is at least ``ROLLOUT_GAP_MIN`` points.
The oracle winner has the higher outcome; a pair is a *trap* when the visible
score points the other way (ties count for side A on both sides).

Pair and dataset files use the fields ``a, b, pool, turn_a, turn_b,
oracle_winner, margin, pooled_se, margin_z, trap``.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from civmarsh.utils.freeciv import score_from_rendering
from civmarsh.utils.io import iter_jsonl, read_jsonl, write_jsonl
from civmarsh.utils.stats import episode_cluster

MAX_TURN_GAP = 10
MIN_GAP_Z = 2.0
ROLLOUT_GAP_MIN = 6.0
PAIR_SEED = 97

POOLS = ("replay", "rollout_early", "rollout_late")
GAP_RULE = {"replay": "z", "rollout_early": "abs", "rollout_late": "abs"}
TRAIN_QUOTA = {"replay": 3600, "rollout_early": 4800, "rollout_late": 2400}
TRAP_CAP = 0.40
EVAL_NATURAL_TOTAL = 600
EVAL_TRAP_TOTAL = 600

# Dataset directory layout written by `write_dataset`.
TRAIN_PAIRS = "train_pairs.jsonl"
EVAL_NATURAL = "eval_natural.jsonl"
EVAL_TRAP = "eval_trap.jsonl"
POSITIONS = "positions.jsonl"
STATS = "stats.json"


# ------------------------------------------------------------------ records


def _as_bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).lower() == "true"


def normalize_pair(p: dict) -> dict:
    """A pair record with a boolean ``trap`` field."""
    q = dict(p)
    if "trap" in q:
        q["trap"] = _as_bool(q["trap"])
    return q


def read_pairs(path) -> list[dict]:
    return [normalize_pair(p) for p in iter_jsonl(path)]


def focal_of(rec: dict) -> str | None:
    """The evaluated player of a position record."""
    return rec.get("focal_player") or rec.get("focal")


def load_renderings(path, ids=None) -> dict[str, dict]:
    """Rendered positions by id, each with a ``focal_player`` field; with
    `ids`, only those positions."""
    out = {}
    for r in iter_jsonl(path):
        if ids is not None and r["position_id"] not in ids:
            continue
        out[r["position_id"]] = {**r, "focal_player": focal_of(r)}
    return out


# -------------------------------------------------------------------- pools


def replay_pool(rendered, labels: dict[str, dict], *, split=None):
    """Replay-bank position rows from fog renderings and oracle labels.

    `rendered` rows carry position_id, game_id, turn, focal player and
    rendering (and optionally a split); `labels` maps position id to a label
    with ``mean_end`` and ``se_end`` (or ``se``). With `split`, rows whose split
    differs are skipped. Returns (rows, drop counts)."""
    rows, dropped = [], Counter()
    for p in rendered:
        if split is not None and p.get("split", split) != split:
            continue
        sc = score_from_rendering(p.get("rendering") or "")
        lab = labels.get(p["position_id"])
        if sc is None:
            dropped["no_score_now"] += 1
            continue
        if lab is None:
            dropped["no_label"] += 1
            continue
        if not p.get("rendering") or not focal_of(p):
            dropped["no_rendering_or_focal"] += 1
            continue
        rows.append(
            {
                "position_id": p["position_id"],
                "turn": p["turn"],
                "group": p["game_id"],
                "score_now": sc,
                "value": lab["mean_end"],
                "se": lab["se_end"] if "se_end" in lab else lab["se"],
                "rendering": p["rendering"],
                "focal_player": focal_of(p),
            }
        )
    return rows, dict(dropped)


def rollout_pool(slice_rows):
    """Rollout position rows from a slice written by `rollout_positions`
    (position_id, turn, focal player, rendering, outcome). The group is the
    rollout episode. Returns (rows, drop counts)."""
    rows, dropped = [], Counter()
    for r in slice_rows:
        sc = score_from_rendering(r.get("rendering") or "")
        if sc is None:
            dropped["no_score_now"] += 1
            continue
        if r["outcome"] is None:
            dropped["no_label"] += 1
            continue
        if not r.get("rendering") or not focal_of(r):
            dropped["no_rendering_or_focal"] += 1
            continue
        rows.append(
            {
                "position_id": r["position_id"],
                "turn": r["turn"],
                "group": episode_cluster(r["position_id"]),
                "score_now": sc,
                "value": float(r["outcome"]),
                "se": 0.0,
                "rendering": r["rendering"],
                "focal_player": focal_of(r),
            }
        )
    return rows, dict(dropped)


def rollout_positions(
    value_log,
    *,
    run: str,
    windows: dict[str, range],
    episodes_per_step: int = 64,
    heldout_every: int = 5,
) -> dict[tuple[str, str], list[dict]]:
    """Split one RL run's per-episode decision log into rollout slices.

    `value_log` is a JSONL file with one line per episode, in rollout order,
    each ``{"decisions": [{"turn", "focal_player", "rendering"}, ...],
    "score_end": <final score>}``; line i belongs to training step
    ``i // episodes_per_step``. `windows` maps a slice name to the training
    steps it covers. Every `heldout_every`-th episode is held out. Returns
    {(slice, "train" | "eval"): position rows}."""
    out: dict[tuple[str, str], list[dict]] = {}
    for i, row in enumerate(iter_jsonl(value_log)):
        step = i // episodes_per_step
        slice_name = next((s for s, w in windows.items() if step in w), None)
        if slice_name is None:
            continue
        split = "eval" if i % heldout_every == 0 else "train"
        for d in row["decisions"]:
            out.setdefault((slice_name, split), []).append(
                {
                    "position_id": f"{run}_e{i}_t{d['turn']}",
                    "run": run,
                    "step": step,
                    "turn": d["turn"],
                    "focal_player": focal_of(d),
                    "rendering": d["rendering"],
                    "outcome": row["score_end"],
                }
            )
    return out


# ------------------------------------------------------------- enumeration


def enumerate_pairs(rows: list[dict], gap_rule: str, *, max_turn_gap=MAX_TURN_GAP):
    """All decidable cross-group pairs of `rows`, split into trap and non-trap.

    Returns (trap_idx, agree_idx), each an N x 2 int32 array of row indices
    (a, b) with turn(a) <= turn(b). The enumeration order is deterministic
    (blocks of ascending turn pairs, row-major inside a block), so seeded
    subsampling of the index arrays is reproducible."""
    import numpy as np

    turn = np.array([r["turn"] for r in rows])
    gmap = {g: i for i, g in enumerate(sorted({r["group"] for r in rows}))}
    grp = np.array([gmap[r["group"]] for r in rows])
    value = np.array([r["value"] for r in rows], dtype=np.float64)
    se = np.array([r["se"] for r in rows], dtype=np.float64)
    score = np.array([r["score_now"] for r in rows], dtype=np.float64)

    trap_blocks, agree_blocks = [], []
    tvals = sorted(set(turn.tolist()))
    idx_by_t = {t: np.nonzero(turn == t)[0] for t in tvals}
    for ii, ta in enumerate(tvals):
        for tb in tvals[ii:]:
            if tb - ta > max_turn_gap:
                break
            A, B = idx_by_t[ta], idx_by_t[tb]
            cand = grp[A][:, None] != grp[B][None, :]
            if ta == tb:
                cand &= np.triu(np.ones((len(A), len(B)), dtype=bool), k=1)
            gap = np.abs(value[A][:, None] - value[B][None, :])
            if gap_rule == "z":
                s = np.hypot(se[A][:, None], se[B][None, :])
                dec = cand & ((s == 0) | (gap > MIN_GAP_Z * s))
            elif gap_rule == "abs":
                dec = cand & (gap >= ROLLOUT_GAP_MIN)
            else:
                raise ValueError(f"unknown gap rule {gap_rule!r}")
            oracle_a = value[A][:, None] >= value[B][None, :]
            score_a = score[A][:, None] >= score[B][None, :]
            trap = dec & (oracle_a != score_a)
            agree = dec & ~(oracle_a != score_a)
            for mask, blocks in ((trap, trap_blocks), (agree, agree_blocks)):
                r_i, c_i = np.nonzero(mask)
                if len(r_i):
                    blocks.append(np.stack([A[r_i], B[c_i]], axis=1).astype(np.int32))

    def cat(blocks):
        return np.concatenate(blocks) if blocks else np.zeros((0, 2), dtype=np.int32)

    return cat(trap_blocks), cat(agree_blocks)


def pair_row(rows: list[dict], i: int, j: int, pool: str) -> dict:
    a, b = rows[i], rows[j]
    gap = abs(a["value"] - b["value"])
    s = math.hypot(a["se"], b["se"])
    ow = "A" if a["value"] >= b["value"] else "B"
    sw = "A" if a["score_now"] >= b["score_now"] else "B"
    return {
        "a": a["position_id"],
        "b": b["position_id"],
        "pool": pool,
        "turn_a": a["turn"],
        "turn_b": b["turn"],
        "oracle_winner": ow,
        "margin": round(gap, 3),
        "pooled_se": round(s, 3),
        "margin_z": round(gap / s, 2) if s else None,
        "trap": ow != sw,
    }


def sample_idx(rng: random.Random, n_avail: int, k: int) -> list[int]:
    """k sorted indices from range(n_avail) (all of them when k >= n_avail)."""
    if k >= n_avail:
        return list(range(n_avail))
    return sorted(rng.sample(range(n_avail), k))


def split_quota(total: int, quota: dict[str, int]) -> dict[str, int]:
    """Largest-remainder apportionment of `total` in proportion to `quota`."""
    exact = {p: total * q / sum(quota.values()) for p, q in quota.items()}
    base = {p: int(v) for p, v in exact.items()}
    rem = total - sum(base.values())
    for p in sorted(exact, key=lambda p: exact[p] - base[p], reverse=True)[:rem]:
        base[p] += 1
    return base


# ------------------------------------------------------------- training set


@dataclass
class Dataset:
    """CivTelescope's pairs and the positions they reference."""

    train: list[dict]
    eval_natural: list[dict]
    eval_trap: list[dict]
    positions: dict[str, dict]  # position_id -> {focal_player, rendering}
    stats: dict = field(default_factory=dict)

    def heldout_pairs(self) -> list[dict]:
        """The held-out set as one list, natural slice first. The slice is the
        file a pair comes from (natural pairs are sampled from every decidable
        pair, so some are traps by the score rule; they still count as
        natural here)."""
        return [dict(p, trap=False) for p in self.eval_natural] + [
            dict(p, trap=True) for p in self.eval_trap
        ]


def build_dataset(
    pools: dict[tuple[str, str], list[dict]],
    *,
    seed: int = PAIR_SEED,
    train_quota: dict[str, int] = TRAIN_QUOTA,
    trap_cap: float = TRAP_CAP,
    eval_natural_total: int = EVAL_NATURAL_TOTAL,
    eval_trap_total: int = EVAL_TRAP_TOTAL,
) -> Dataset:
    """Sample the training and held-out pairs from position pools.

    `pools` maps (pool, "train" | "eval") to position rows for every pool in
    `train_quota`. Per pool, training keeps `train_quota[pool]` pairs of which
    at most `trap_cap` are traps (all traps when fewer are available, the rest
    non-trap). The held-out natural slice samples all decidable pairs of the
    eval rows uniformly; the held-out trap slice samples trap pairs only. Both
    split their totals across pools in proportion to `train_quota`. One
    ``random.Random(seed)`` is consumed in a fixed order: training pools
    (traps then non-traps, pool by pool), the natural slice, the trap slice,
    then the training shuffle. Raises if a position is referenced by both a
    training and a held-out pair."""
    rng = random.Random(seed)
    order = list(train_quota)
    stats: dict = {"seed": seed, "train": {}, "eval_natural": {}, "eval_trap": {}}

    train_pairs = []
    for pool in order:
        rows = pools[(pool, "train")]
        trap_idx, agree_idx = enumerate_pairs(rows, GAP_RULE[pool])
        quota = train_quota[pool]
        n_trap = min(len(trap_idx), int(quota * trap_cap))
        n_agree = min(len(agree_idx), quota - n_trap)
        sel_trap = sample_idx(rng, len(trap_idx), n_trap)
        sel_agree = sample_idx(rng, len(agree_idx), n_agree)
        ps = [pair_row(rows, int(i), int(j), pool) for i, j in trap_idx[sel_trap]] + [
            pair_row(rows, int(i), int(j), pool) for i, j in agree_idx[sel_agree]
        ]
        train_pairs += ps
        stats["train"][pool] = {
            "quota": quota,
            "kept": len(ps),
            "trap_available": len(trap_idx),
            "non_trap_available": len(agree_idx),
            "trap_kept": n_trap,
            "non_trap_kept": n_agree,
        }

    nat_quota = split_quota(eval_natural_total, train_quota)
    trap_quota = split_quota(eval_trap_total, train_quota)
    stats["eval_quota"] = {"natural": nat_quota, "trap": trap_quota}
    eval_enum = {p: enumerate_pairs(pools[(p, "eval")], GAP_RULE[p]) for p in order}

    eval_natural, eval_trap = [], []
    for pool in order:
        rows = pools[(pool, "eval")]
        trap_idx, agree_idx = eval_enum[pool]
        n_all = len(trap_idx) + len(agree_idx)
        ranks = sample_idx(rng, n_all, nat_quota[pool])
        ps = [
            pair_row(
                rows,
                *map(
                    int,
                    trap_idx[r] if r < len(trap_idx) else agree_idx[r - len(trap_idx)],
                ),
                pool,
            )
            for r in ranks
        ]
        eval_natural += ps
        stats["eval_natural"][pool] = {"quota": nat_quota[pool], "kept": len(ps)}
    for pool in order:
        rows = pools[(pool, "eval")]
        trap_idx, _ = eval_enum[pool]
        sel = sample_idx(rng, len(trap_idx), trap_quota[pool])
        ps = [pair_row(rows, int(i), int(j), pool) for i, j in trap_idx[sel]]
        eval_trap += ps
        stats["eval_trap"][pool] = {"quota": trap_quota[pool], "kept": len(ps)}

    rng.shuffle(train_pairs)

    byid = {r["position_id"]: r for rows in pools.values() for r in rows}

    def bankable(p):
        ra, rb = byid.get(p["a"]), byid.get(p["b"])
        return bool(
            ra
            and rb
            and ra["rendering"]
            and ra["focal_player"]
            and rb["rendering"]
            and rb["focal_player"]
        )

    stats["pair_drop_counts"] = {}
    kept = {}
    for name, ps in (
        ("train", train_pairs),
        ("eval_natural", eval_natural),
        ("eval_trap", eval_trap),
    ):
        kept[name] = [p for p in ps if bankable(p)]
        stats["pair_drop_counts"][name] = len(ps) - len(kept[name])

    ref_train = {p[k] for p in kept["train"] for k in ("a", "b")}
    ref_eval = {
        p[k] for p in kept["eval_natural"] + kept["eval_trap"] for k in ("a", "b")
    }
    overlap = ref_train & ref_eval
    if overlap:
        raise ValueError(f"train/eval position overlap: {sorted(overlap)[:10]}")
    positions = {
        pid: {
            "position_id": pid,
            "focal_player": byid[pid]["focal_player"],
            "rendering": byid[pid]["rendering"],
        }
        for pid in sorted(ref_train | ref_eval)
    }
    stats["referenced_positions"] = {
        "train": len(ref_train),
        "eval": len(ref_eval),
        "total": len(positions),
    }
    return Dataset(
        kept["train"], kept["eval_natural"], kept["eval_trap"], positions, stats
    )


def write_dataset(out_dir, ds: Dataset) -> None:
    import json

    out = Path(out_dir)
    write_jsonl(out / TRAIN_PAIRS, ds.train)
    write_jsonl(out / EVAL_NATURAL, ds.eval_natural)
    write_jsonl(out / EVAL_TRAP, ds.eval_trap)
    write_jsonl(out / POSITIONS, ds.positions.values())
    (out / STATS).write_text(json.dumps(ds.stats, indent=1) + "\n")


def load_positions(data_dir) -> dict[str, dict]:
    """Positions of a dataset directory: {position_id: {focal_player,
    rendering}}."""
    return {r["position_id"]: r for r in iter_jsonl(Path(data_dir) / POSITIONS)}


def load_dataset(data_dir) -> Dataset:
    """Read a dataset directory written by `write_dataset`."""
    d = Path(data_dir)
    return Dataset(
        read_pairs(d / TRAIN_PAIRS),
        read_pairs(d / EVAL_NATURAL),
        read_pairs(d / EVAL_TRAP),
        load_positions(d),
    )


# -------------------------------------------------------- ruleset transfer

TRANSFER_TRAP_CAP = 1800
TRANSFER_NATURAL_KEEP = 600
TRANSFER_API_TRAP = 200
TRANSFER_API_NATURAL = 100
TRANSFER_API_SEED = 29


def transfer_pairs(
    records: list[dict],
    *,
    ruleset: str,
    max_turn_gap: int = MAX_TURN_GAP,
    min_gap_z: float = MIN_GAP_Z,
) -> list[dict]:
    """Every decidable cross-game pair of labeled bank positions.

    `records` carry position_id, game_id, turn, score_now (the bank's
    visible score), mean_end and se_end (or se). Records are paired in
    position-id order (a before b), under the replay rule of this module."""
    rows = sorted(records, key=lambda r: r["position_id"])
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if (
                a["game_id"] == b["game_id"]
                or abs(a["turn"] - b["turn"]) > max_turn_gap
            ):
                continue
            va, vb = a["mean_end"], b["mean_end"]
            sa = a["se_end"] if "se_end" in a else a["se"]
            sb = b["se_end"] if "se_end" in b else b["se"]
            gap = abs(va - vb)
            s = math.hypot(sa, sb)
            if s > 0 and gap <= min_gap_z * s:
                continue
            ow = "A" if va >= vb else "B"
            sw = "A" if a["score_now"] >= b["score_now"] else "B"
            pairs.append(
                {
                    "a": a["position_id"],
                    "b": b["position_id"],
                    "game_a": a["game_id"],
                    "game_b": b["game_id"],
                    "turn_a": a["turn"],
                    "turn_b": b["turn"],
                    "oracle_winner": ow,
                    "margin": round(gap, 3),
                    "pooled_se": round(s, 3),
                    "margin_z": round(gap / s, 2) if s else None,
                    "trap": ow != sw,
                    "ruleset": ruleset,
                }
            )
    return pairs


def subsample_transfer_pairs(
    pairs: list[dict],
    *,
    trap_cap: int = TRANSFER_TRAP_CAP,
    natural_keep: int = TRANSFER_NATURAL_KEEP,
    seed: int = PAIR_SEED,
    api_trap: int = TRANSFER_API_TRAP,
    api_natural: int = TRANSFER_API_NATURAL,
    api_seed: int = TRANSFER_API_SEED,
) -> tuple[list[dict], list[dict]]:
    """The seeded ruleset-transfer evaluation set and its zero-shot API subset.

    Keeps at most `trap_cap` trap and `natural_keep` non-trap pairs
    (``random.Random(seed)``), then draws `api_trap` + `api_natural` of the
    kept pairs (``random.Random(api_seed)``). Returns (kept, api)."""
    traps = [p for p in pairs if p["trap"]]
    rest = [p for p in pairs if not p["trap"]]
    rng = random.Random(seed)
    rng.shuffle(traps)
    rng.shuffle(rest)
    keep = traps[:trap_cap] + rest[:natural_keep]
    rng.shuffle(keep)
    rng2 = random.Random(api_seed)
    kt = [p for p in keep if p["trap"]]
    kn = [p for p in keep if not p["trap"]]
    rng2.shuffle(kt)
    rng2.shuffle(kn)
    api = kt[:api_trap] + kn[:api_natural]
    rng2.shuffle(api)
    return keep, api


def turn_bucket(pair: dict) -> str:
    """Early / middle / late bucket of a pair by its later turn."""
    t = max(pair["turn_a"], pair["turn_b"])
    return "t10-29" if t < 30 else ("t30-49" if t < 50 else "t50-69")


def read_bank_records(labels_path, positions_path=None) -> list[dict]:
    """Labeled bank records; fields missing from a label record (e.g. the
    visible score) are taken from the matching position record."""
    labels = read_jsonl(labels_path)
    if positions_path is None:
        return labels
    pos = {r["position_id"]: r for r in iter_jsonl(positions_path)}
    return [{**pos.get(r["position_id"], {}), **r} for r in labels]
