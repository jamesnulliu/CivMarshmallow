"""Episode logging + replay — makes an agent-driven run reproducible even when
the agent itself is nondeterministic (an LLM).

`client_branch`/`run_agents` give a *deterministic engine*: same seed + same
orders -> byte-identical save. But `branch.Trajectory` records only the engine
outcome, not what the agent observed or decided, so on its own a stochastic
agent's trajectory could not be replayed or branched.

An **episode log** closes that: a JSONL with a header (seed, horizon, players,
agent identities, env spec) then one `step` record per turn per player
capturing the observation it was read from and the exact orders issued. Because
the engine is deterministic given the orders, replaying the recorded orders at
their turns reproduces the trajectory bit-for-bit — regardless of how the agent
originally chose them. `replay_episode` does exactly that via `run_agents` with
a static `ClientPolicy` per player, so a captured LLM run becomes replayable and
branchable like any other position.
"""

import dataclasses
import json
from pathlib import Path

from civharness import parse
from civharness.envspec import EnvSpec
from civharness.policy import client as _order_mod
from civharness.policy.client import ClientPolicy, SetWorklist, UnitOrders


class ReplayMismatch(RuntimeError):
    """A replay was pointed at a position that isn't the one recorded, or it
    diverged from the recorded trajectory. Raised so the episode's provenance
    (the source-save digest and the per-turn observation digests) is actually
    *verified*, not merely stored."""


# Every order type is a dataclass in policy.client exposing `.apply(client)`.
# Discover them dynamically so order types added later register automatically.
_ORDER_TYPES = {
    name: obj
    for name, obj in vars(_order_mod).items()
    if isinstance(obj, type) and dataclasses.is_dataclass(obj) and hasattr(obj, "apply")
}


def order_as_record(order) -> dict:
    """Serialize an order to a JSON-safe dict: its type name + its fields."""
    return {"type": type(order).__name__, "fields": dataclasses.asdict(order)}


def order_from_record(rec: dict):
    """Rebuild an order from `order_as_record` output. Nested sequences that the
    frozen dataclasses declare as tuples are coerced back so a round-tripped
    order equals the original."""
    cls = _ORDER_TYPES[rec["type"]]
    fields = dict(rec["fields"])
    if cls is SetWorklist and "entries" in fields:
        fields["entries"] = tuple(tuple(e) for e in fields["entries"])
    if cls is UnitOrders and "orders" in fields:
        fields["orders"] = tuple(dict(o) for o in fields["orders"])
    return cls(**fields)


class EpisodeWriter:
    """Append-only JSONL writer: one header line, then one line per step. Flushed
    per record so a crash still leaves a readable partial episode."""

    def __init__(self, path: Path, header: dict):
        self.path = Path(path)
        self._f = open(self.path, "w")  # noqa: SIM115 -- closed by close()
        self._sha_cache: dict[str, str | None] = {}
        self._write({"kind": "header", **header})

    def _write(self, rec: dict) -> None:
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()

    def _obs_sha(self, obs_save) -> str | None:
        """The normalized (wall-clock-stripped) hash of the save the agent
        observed, cached per path since a turn's seats share one save. Recorded
        so the log is a self-contained, verifiable record of what was seen — a
        replay can assert it saw the same state."""
        key = str(obs_save)
        if key not in self._sha_cache:
            try:
                self._sha_cache[key] = parse.normalized_save_sha(obs_save)
            except Exception:  # noqa: BLE001 -- unreadable save: no digest
                self._sha_cache[key] = None
        return self._sha_cache[key]

    def step(
        self, turn: int, player: str, obs_save, orders, audit: dict | None = None
    ) -> None:
        rec = {
            "kind": "step",
            "turn": turn,
            "player": player,
            "obs_save": str(obs_save),
            "obs_sha": self._obs_sha(obs_save),
            "orders": [order_as_record(o) for o in orders],
        }
        if audit:
            # Agent-registered decision audit — e.g. the exact offered
            # candidate set (or its SHA), the visibility mode it decided
            # under, prompt/bundle digests. Free-form but JSON-safe.
            rec["audit"] = audit
        self._write(rec)

    def footer(self, **fields) -> None:
        """Write the closing completion record. It carries the run's final
        normalized save digest, so an exact replay can prove it reproduced the
        WHOLE trajectory — the per-step obs digests miss the last turn (its
        orders' effect is never re-observed) and miss truncation. An episode with
        no footer is incomplete and exact replay must reject it."""
        self._write({"kind": "footer", "complete": True, **fields})

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_episode(path: Path) -> tuple[dict, list[dict]]:
    """Return (header, steps) parsed from an episode JSONL. A completion footer,
    if present, is attached as `header["footer"]`; its absence means the log is
    incomplete (the run crashed or was truncated)."""
    header, steps, footer = {}, [], None
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        kind = rec.get("kind")
        if kind == "header":
            header = rec
        elif kind == "step":
            steps.append(rec)
        elif kind == "footer":
            footer = rec
    if footer is not None:
        header["footer"] = footer
    return header, steps


def episode_policies(steps: list[dict]) -> dict[str, ClientPolicy]:
    """Reconstruct a {player: ClientPolicy(turn -> [order])} from step records —
    the static policies that reissue exactly what the agents did."""
    per_player: dict[str, dict[int, list]] = {}
    for s in steps:
        by_turn = per_player.setdefault(s["player"], {})
        by_turn.setdefault(int(s["turn"]), []).extend(
            order_from_record(r) for r in s["orders"]
        )
    return {p: ClientPolicy(orders=by_turn) for p, by_turn in per_player.items()}


def _require_complete_recording(footer: dict | None, steps: list[dict]) -> str:
    """Validate that a recording is complete and well-formed enough to be
    exact-replayed, returning its recorded final-save digest. Fails **closed**
    (ReplayMismatch) on any gap — a missing/incomplete footer, a missing or
    malformed final digest, or a step with no observation digest — instead of
    silently skipping a check when the metadata is absent.
    `verify=False` is the explicit opt-out."""
    if footer is None:
        raise ReplayMismatch(
            "episode log has no completion footer — it is incomplete (truncated "
            "or the run crashed); cannot exact-replay. Pass until=/seeds= to "
            "branch from it, or verify=False to skip verification."
        )
    if footer.get("complete") is not True:
        raise ReplayMismatch(
            "episode footer is not marked complete — the recording is unfinished."
        )
    final_sha = footer.get("final_save_sha")
    if not isinstance(final_sha, str) or len(final_sha) != 64:
        raise ReplayMismatch(
            f"episode footer has no valid final-save digest ({final_sha!r}); "
            "cannot verify the trajectory reproduced."
        )
    missing = [
        (s.get("turn"), s.get("player")) for s in steps if s.get("obs_sha") is None
    ]
    if missing:
        raise ReplayMismatch(
            f"{len(missing)} recorded step(s) carry no observation digest "
            f"(e.g. {missing[:3]}); the recording is not fully verifiable."
        )
    return final_sha


def _verify_obs_shas(recorded_steps: list[dict], trajs) -> None:
    """Cross-check the replay's per-turn observation digests against the recorded
    ones, raising `ReplayMismatch` at the exact turn/player a divergence first
    appears — so obs_sha is live verification, not dead metadata.
    Recorded digests are guaranteed non-null here (validated up front), so a
    replay step that produced *no* digest is itself a mismatch, not a skip."""
    rec = {(int(s["turn"]), s["player"]): s.get("obs_sha") for s in recorded_steps}
    for tr in trajs:
        path = getattr(tr, "episode_log_path", None)
        if not path:
            continue
        _h, rsteps = read_episode(path)
        for s in rsteps:
            want = rec.get((int(s["turn"]), s["player"]))
            if want is None:
                continue  # a replay step with no recorded counterpart to check
            got = s.get("obs_sha")
            if got != want:
                raise ReplayMismatch(
                    f"replay diverged from the recording at turn {s['turn']} "
                    f"player {s['player']!r}: observation {got!r} != recorded "
                    f"{want[:12]}…"
                )


def replay_episode(
    log_path: Path,
    save,
    workdir: Path,
    *,
    until: int | None = None,
    seeds: list[int] | None = None,
    verify: bool = True,
    **kwargs,
):
    """Replay a recorded episode: reissue each player's exact orders at their
    turns, via `run_agents` with a static ClientPolicy per player. With the
    episode's own seed and horizon this reproduces the original trajectory
    byte-for-byte, even if the agent that produced it was nondeterministic.

    `save` must be the same position the episode was recorded from — which
    `verify=True` (default) enforces by comparing its normalized digest to the
    recorded `from_save_sha`, so a *different same-turn* save is rejected rather
    than silently accepted. When reproducing (neither `until`
    nor `seeds` overridden), the replay's per-turn observation digests are also
    checked against the recorded ones, localizing any divergence.

    The AI configuration (`skill`, `skill_by_player`) and the recorded `EnvSpec`
    are restored from the header so uncontrolled seats and the action gating
    match the original run. `until`/`seeds` default to the header's horizon/seed;
    override them (or pass `skill`/`env` explicitly) to branch instead.
    """
    from civharness.agent import run_agents  # deferred: agent.py imports us

    header, steps = read_episode(log_path)
    footer = header.get("footer")
    agents = episode_policies(steps)
    reproducing = until is None and seeds is None
    if until is None:
        until = header.get("until")
    if seeds is None:
        seeds = [header.get("seed")]
    kwargs.setdefault("skill", header.get("skill"))
    kwargs.setdefault("skill_by_player", header.get("skill_by_player"))
    if "env" not in kwargs and header.get("env"):
        kwargs["env"] = EnvSpec.from_dict(header["env"])

    # An exact reproduce demands a COMPLETE, well-formed recording — validated
    # BEFORE launching a server so a malformed log fails fast, and so the final
    # digest is guaranteed present (no silent skip on absent metadata).
    want_final = None
    if verify:
        if reproducing:
            want_final = _require_complete_recording(footer, steps)
        expected = header.get("from_save_sha")
        if expected is not None:
            got = parse.normalized_save_sha(getattr(save, "path", save))
            if got != expected:
                raise ReplayMismatch(
                    f"replay save does not match the recorded position "
                    f"(sha {got[:12]}… != recorded {expected[:12]}…); replay_episode "
                    f"needs the exact save the episode was recorded from"
                )

    trajs = run_agents(save, workdir, agents=agents, until=until, seeds=seeds, **kwargs)

    if verify and reproducing:
        _verify_obs_shas(steps, trajs)
        # The end-to-end check the per-step digests can't give: the FINAL save
        # (including the last turn's orders, whose effect is never re-observed)
        # must match. `want_final` is non-null here (validated above), so this
        # always runs. Catches last-turn divergence and truncation.
        for tr in trajs:
            got_final = getattr(tr, "save_sha256_normalized", None)
            if got_final != want_final:
                raise ReplayMismatch(
                    f"replay final save {got_final!r} != recorded "
                    f"{want_final[:12]}… — the trajectory did not reproduce "
                    f"(last-turn divergence or a truncated recording)"
                )
    return trajs
