"""Play whole games in env v4.1 with a language-model policy (inference only, no training).

Player
  LLMPlayer  an OpenAI-compatible chat server (e.g. sglang), thinking off, one call per turn; with
             constrained=True the reply is decoded under the menu's JSON schema

One game = one start save played T1 -> T120 by the focal player against the save's other players as Hard bots
(on the CivMarshmallow server build, Hard bots see only what their player sees), under the player-visible
EnvSpec preset "saga_visible" (fog of war, live legality).  Output per game (one JSON record, written once at the
end, so an interrupted run simply replays unfinished games):
  record     player, seed, params, end score, eliminated, per-decision stats (keys, parse error, prompt chars)
  positions  the state text at turns 10, 20, ..., 110 with the score at that turn
"""

from __future__ import annotations

import json
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

from civmarsh.env.v41.menu import (
    BundleValidationError,
    obs_block_v4,
    parse_action_bundle_v4,
    parse_entity_bundle,
)

ENDTURN = 120
BUCKETS = tuple(range(10, ENDTURN, 10))
SYSTEM_V4 = (
    "You are playing Freeciv (civ2civ3 ruleset) as player '{focal}'. You play from turn 1 to turn {endturn}. "
    "Each turn you receive the current state (OVERVIEW, RIVALS, CITIES, UNITS, MAP, EVENTS) and an action menu. "
    "Return exactly one JSON object whose domain values are lists of offered keys (see ACTION_JSON). "
    "Orders such as goto and settle continue over later turns, so only units listed under UNITS need orders. "
    "Your goal is the highest score at turn {endturn}: expand, build cities, grow the economy, defend, and "
    "research. Return JSON only."
)
# v4.1: entity action format, civm41 rules
SYSTEM_V41 = (
    "You are playing Freeciv 3.2 as player '{focal}' from turn 1 to turn {endturn}. Rules: civ2civ3 without great "
    "wonders, ships, diplomats, caravans, Coinage and pillage; there are no treaties, so every civilization you meet "
    "is at war with you. Each turn you receive the state with the actions of every city and unit (TURN "
    "civm-turn-v41). Return exactly one JSON object as described under ANSWER. Your goal is the highest score at "
    "turn {endturn}: expand, build cities, grow the economy, defend, and research. Return JSON only."
)


class LLMPlayer:
    def __init__(
        self,
        url: str,
        *,
        model: str = "policy",
        temperature: float = 1.0,
        max_tokens: int = 320,
        timeout: float = 300,
        constrained: bool = False,
    ):
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.model, self.temperature, self.max_tokens, self.timeout = (
            model,
            temperature,
            max_tokens,
            timeout,
        )
        self.constrained = constrained  # v4.1: sglang json_schema over the offered keys (menu.json_schema())

    def __call__(self, prompt: str, menu, state, rng) -> str:
        req_body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.constrained:
            schema = (
                menu.entity_schema()
                if getattr(menu, "fmt", "domain") == "entity"
                else menu.json_schema()
            )
            req_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "action_bundle", "schema": schema},
            }
        body = json.dumps(req_body).encode()
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        return out["choices"][0]["message"]["content"] or ""


def verb_of(key: str) -> str:
    tail = key.split("/", 1)[1] if "/" in key else key
    return tail.split(":", 1)[0]


def decision_tags(state, menu) -> dict:
    """Decision tags of this turn: settler = idle settlers; threat = an enemy military unit within 3 tiles of an
    own city or next to an own unit; government = a government change is offered; plus each city's (item, stock),
    so finished production shows up when consecutive turns are compared."""
    from civmarsh.env.v41.observation import SETTLER_TYPES

    idle = state.idle_units()
    mil = [u for u in state.foreign_units if state._military(u)]
    threat = any(
        state.map.distance(u["tile"], c["tile"]) <= 3 for u in mil for c in state.cities
    ) or any(
        state.map.distance(u["tile"], v["tile"]) == 1 for u in mil for v in state.units
    )
    b = getattr(menu, "_builder", None)
    return {
        "settler": [u["id"] for u in idle if u.get("type") in SETTLER_TYPES],
        "threat": bool(threat),
        "government": bool(b and b.gov_values),
        "cities": {
            c["id"]: [c.get("building"), c.get("shield_stock")] for c in state.cities
        },
        "score": state.my.get("score"),
    }


def play_game(
    start: dict,
    player_name: str,
    player,
    seed: int,
    workdir: Path,
    *,
    until: int = ENDTURN,
    fmt: str = "entity",
    keep_prompts: bool = False,
    tags: bool = False,
) -> dict:
    """Play one game; never raises (errors are recorded)."""
    from civharness import EnvSpec, FunctionAgent, run_agents

    # seeded from a string, not hash(): str hashes are salted per process, which would make games irreproducible
    rng = random.Random(f"{start['game_id']}|{player_name}|{seed}")
    focal = start["focal_player"]
    system = (SYSTEM_V41 if fmt == "entity" else SYSTEM_V4).format(
        focal=focal, endturn=ENDTURN
    )
    box = {"prev": None}
    decisions, positions = [], []
    decisions_tags = {}

    def act(obs):
        t0 = time.time()
        text, menu, state = obs_block_v4(
            obs, endturn=ENDTURN, prev=box["prev"], fmt=fmt
        )
        box["prev"] = state.summary()
        if obs.turn in BUCKETS:
            positions.append(
                {
                    "turn": obs.turn,
                    "score_now": state.my.get("score"),
                    "cities": len(state.cities),
                    "units": len(state.units),
                    "rendering": state.render(),
                }
            )
        prompt = system + "\n" + text
        if keep_prompts and obs.turn % 10 == 0:
            positions.append({"turn": obs.turn, "prompt": prompt, "kind": "prompt"})
        err, keys, raw, dropped = None, [], "", []
        try:
            raw = player(prompt, menu, state, rng)
            bundle = (
                parse_entity_bundle(raw, menu)
                if fmt == "entity"
                else parse_action_bundle_v4(raw, menu, lenient=True)
            )
            keys, orders = list(bundle.keys), list(bundle.orders)
            dropped = list(bundle.dropped)
            if box["prev"] is not None and getattr(bundle, "notes", ()):
                box["prev"]["notes"] = list(bundle.notes)  # shown in next turn's EVENTS
        except BundleValidationError as e:
            err, orders = f"invalid: {e}"[:200], []
        except Exception as e:  # noqa: BLE001 -- LLM/network failure: pass the turn, record it
            err, orders = f"{type(e).__name__}: {e}"[:200], []
        if tags:
            decisions_tags[obs.turn] = decision_tags(state, menu)
        decisions.append(
            {
                "turn": obs.turn,
                "keys": keys,
                "error": err,
                "prompt_chars": len(prompt),
                "dropped": [f"{k} ({why})" for k, why in dropped][:6],
                "n_dropped": len(dropped),
                "n_cands": len(menu.candidates),
                "idle": len(state.idle_units()),
                "cities": len(state.cities),
                "raw_head": (raw or "")[:160] if err else None,
                "s": round(time.time() - t0, 2),
            }
        )
        return orders

    rec = {
        "game_id": start["game_id"],
        "focal": focal,
        "player": player_name,
        "seed": seed,
        "params": getattr(player, "params", None),
        "start_save": start["save_path"],
    }
    t0 = time.time()
    try:
        tr = run_agents(
            Path(start["save_path"]),
            workdir,
            agents={focal: FunctionAgent(act)},
            until=until,
            seeds=[seed],
            env=EnvSpec.preset("saga_visible"),
            skill="hard",
        )[0]
        sc = tr.scores.get(focal)
        rec.update(
            {
                "score_end": sc["total"] if sc else 0,
                "eliminated": sc is None,
                "end_turn": tr.turns,
            }
        )
    except Exception as e:  # noqa: BLE001 -- ControlLost etc.
        msg = f"{type(e).__name__}: {e}"
        game_over = msg.startswith("ControlLost") and "GAME" in msg[:40]
        rec.update(
            {
                "score_end": 0 if game_over else None,
                "eliminated": game_over,
                "error": None if game_over else msg[:400],
            }
        )
    rec["wall_s"] = round(time.time() - t0, 1)
    n = len(decisions)
    verbs = Counter(verb_of(k) for d in decisions for k in d["keys"])
    rec["stats"] = {
        "decisions": n,
        "invalid_rate": round(sum(1 for d in decisions if d["error"]) / max(1, n), 3),
        "empty_rate": round(
            sum(1 for d in decisions if not d["error"] and not d["keys"]) / max(1, n), 3
        ),
        "dropped_per_turn": round(
            sum(d.get("n_dropped", 0) for d in decisions) / max(1, n), 2
        ),
        "dropped_reasons": dict(
            Counter(
                x.rsplit("(", 1)[1].rstrip(")")
                for d in decisions
                for x in d.get("dropped", [])
            ).most_common()
        ),
        "keys_per_turn": round(sum(len(d["keys"]) for d in decisions) / max(1, n), 2),
        "verbs": dict(verbs.most_common()),
        "cities_t60": next((d["cities"] for d in decisions if d["turn"] == 60), None),
        "cities_end": decisions[-1]["cities"] if decisions else None,
        "max_prompt_chars": max((d["prompt_chars"] for d in decisions), default=0),
        "errors": Counter(
            (d["error"] or "").split(":")[0]
            + ":"
            + (d["error"] or "").split(":")[1][:40]
            for d in decisions
            if d["error"] and ":" in d["error"]
        ).most_common(5),
    }
    rec["decisions"] = decisions
    rec["positions"] = positions
    if tags:
        rec["tags"] = decisions_tags
        rec["workdir"] = str(workdir)
    return rec
