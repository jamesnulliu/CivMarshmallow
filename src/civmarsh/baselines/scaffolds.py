"""Scaffold adapters for the prompting baselines.

Four ``-style`` adapters (BaseLang, Mastaba, SAGA, Reflexion) plus the
zero-scaffold ``DirectStyle`` floor.  The adapters are reimplementations, not
the original systems: each reproduces one published system's core control
structure (single-shot chain of thought / hierarchical advisors / refreshed
mid-term goal / cross-episode self-reflection) on top of this project's
interface, a native freeciv-server session and the six-domain action-key
bundle, rather than freeciv-web and the CivRealm API the originals were built
against.  The prompts are ours, and every adapter runs on the same Qwen3-8B
base model as the RL policy.  Their numbers are comparable with each other and
with the RL policy, not with the numbers published for the original systems.

Shared contract: ``async def reply(ctx) -> str`` returns the decision text the
engine handshake parses as an action bundle; every LLM call goes through
``ctx.ask``, which books its tokens in the episode ledger.  An optional
``async def episode_end(ctx)`` runs after the terminal score is known
(Reflexion uses it; the others do not define it).
"""

import re

from . import prompts


def _last_json_object(text):
    """Last balanced ``{...}`` block in ``text``.

    Scaffolds emit reasoning around their answer, while the engine-side
    decoder is the RL policy's strict JSON parser.  Isolating the answer
    object is the adapter's job; it is not output repair: a malformed object
    still reaches the parser verbatim and still counts as an invalid
    decision.  Returns ``text`` unchanged when there is no balanced block, so
    the parser reports the real error.
    """
    text = text or ""
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for i in range(end, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    return text[i : end + 1]
        end = text.rfind("}", 0, end)
    return text


class DirectStyle:
    """No scaffold: the RL policy's task contract, one call, raw reply.

    The untrained-base floor the other adapters are read against.  It asks
    for no reasoning and does not run answer extraction: the RL rollout hands
    the model's text straight to the strict parser, so this arm does too, and
    its invalid rate is the RL policy's invalid rate measured on the
    untrained model.
    """

    name = "direct"

    async def reply(self, ctx):
        return await ctx.ask(ctx.decision_prompt(), tag="direct")


class BaseLangStyle:
    """Single unstructured chain-of-thought call per decision."""

    name = "baselang"

    async def reply(self, ctx):
        text = await ctx.ask(
            ctx.decision_prompt(prompts.BASELANG_COT.format(endturn=ctx.endturn)),
            tag="cot",
        )
        return _last_json_object(text)


class MastabaStyle:
    """Hierarchical advisors: one call per department, then a president call
    that sees the department reports and commits the bundle."""

    name = "mastaba"
    DEPARTMENTS = ("military", "economy", "expansion")

    def __init__(self, departments=None):
        self.departments = tuple(departments or self.DEPARTMENTS)

    async def reply(self, ctx):
        reports = []
        for dept in self.departments:
            advice = await ctx.ask(
                ctx.decision_prompt(
                    prompts.MASTABA_ADVISOR.format(
                        department=dept, focal=ctx.meta["focal_player"]
                    )
                ),
                tag=f"advisor:{dept}",
            )
            reports.append(f"[{dept}] {(advice or '').strip()}")
        text = await ctx.ask(
            ctx.decision_prompt(
                prompts.MASTABA_PRESIDENT.format(
                    focal=ctx.meta["focal_player"], advice="\n".join(reports)
                )
            ),
            tag="president",
        )
        return _last_json_object(text)


class SagaStyle:
    """Situation-aware mid-term goal, refreshed every N turns and cached in
    the episode scratch; each decision is conditioned on the cached goal."""

    name = "saga"

    def __init__(self, refresh_every=10):
        self.refresh_every = refresh_every

    def _needs_refresh(self, ctx):
        cached = ctx.state.get("saga_goal")
        if cached is None:
            return True
        return ctx.turn - cached["turn"] >= self.refresh_every

    async def reply(self, ctx):
        if self._needs_refresh(ctx):
            goal = await ctx.ask(
                ctx.decision_prompt(
                    prompts.SAGA_GOAL.format(refresh_every=self.refresh_every)
                ),
                tag="goal_refresh",
            )
            ctx.state["saga_goal"] = {"turn": ctx.turn, "text": (goal or "").strip()}
        text = await ctx.ask(
            ctx.decision_prompt(
                prompts.SAGA_DECIDE.format(goal=ctx.state["saga_goal"]["text"])
            ),
            tag="decide",
        )
        return _last_json_object(text)


class ReflexionStyle:
    """Cross-episode self-reflection: one decision call with the lessons from
    previous games on this start injected, plus one end-of-game call that
    writes the next lesson.

    Lessons live in ``ctx.memory`` (caller-owned, survives across episodes)
    keyed by position_id, so lessons from one start never leak into another.
    """

    name = "reflexion"

    def __init__(self, max_lessons=3):
        self.max_lessons = max_lessons

    def _key(self, ctx):
        return ctx.meta.get("position_id", "pos")

    def lessons(self, ctx):
        return ctx.memory.get(self._key(ctx), [])

    async def reply(self, ctx):
        past = self.lessons(ctx)
        block = (
            "\n".join(past[-self.max_lessons :])
            if past
            else "(none yet -- this is your first game on this start.)"
        )
        text = await ctx.ask(
            ctx.decision_prompt(prompts.REFLEXION_DECIDE.format(lessons=block)),
            tag="decide",
        )
        return _last_json_object(text)

    async def episode_end(self, ctx):
        history = (
            "\n".join(
                f"- turn {r['turn']}: "
                + (
                    f"no-op ({r['error'][:60]})"
                    if r.get("error")
                    else (", ".join(r["keys"]) if r["keys"] else "passed")
                )
                for r in ctx.records
            )
            or "(no decisions)"
        )
        text = await ctx.ask(
            prompts.REFLEXION_LESSON.format(
                focal=ctx.meta["focal_player"],
                startturn=ctx.startturn,
                end_turn=ctx.result.get("end_turn"),
                score_end=ctx.result.get("score_end"),
                history=history,
            ),
            tag="reflect",
        )
        lines = [
            ln.strip() for ln in (text or "").splitlines() if ln.strip().startswith("-")
        ]
        if not lines:
            # keep the whole reply rather than silently learning nothing
            flat = re.sub(r"\s+", " ", (text or "").strip())[:300]
            lines = [f"- {flat}"]
        ctx.memory.setdefault(self._key(ctx), []).extend(lines[: self.max_lessons])


SCAFFOLDS = {
    DirectStyle.name: DirectStyle,
    BaseLangStyle.name: BaseLangStyle,
    MastabaStyle.name: MastabaStyle,
    SagaStyle.name: SagaStyle,
    ReflexionStyle.name: ReflexionStyle,
}
