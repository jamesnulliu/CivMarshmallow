"""The pairwise prompt CivTelescope and the zero-shot models answer, and its
answer parser.

The template text is model-visible: CivTelescope was trained on exactly these
bytes, so any edit changes what the model reads. A position record for
`pairwise_prompt` carries ``focal_player`` (the evaluated player's name) and
``rendering`` (the fog-of-war value input of that player).
"""

from __future__ import annotations

import re

PAIRWISE_PROMPT_TMPL = """You will see two positions from two different games of \
Freeciv ({ruleset} ruleset), snapshotted at similar game turns. For each \
position, evaluate the named player's LONG-RUN prospects over the next \
several dozen turns — not just the current score.

=== Position A (evaluate player '{{focal_a}}') ===
{{rendering_a}}

=== Position B (evaluate player '{{focal_b}}') ===
{{rendering_b}}

Question: which evaluated player is better positioned for long-run success \
over the next several dozen turns — player '{{focal_a}}' in Position A, or \
player '{{focal_b}}' in Position B?

Reply with exactly one line: "ANSWER: A" or "ANSWER: B"."""

# The template with the training ruleset filled in; its remaining fields are
# focal_a, rendering_a, focal_b, rendering_b.
PAIRWISE_PROMPT = PAIRWISE_PROMPT_TMPL.format(ruleset="civ2civ3")

# The prefix the answer letter follows. CivTelescope is scored on the logits
# of the token right after it.
ANSWER_PREFIX = "ANSWER:"

# One paragraph telling the model that the visible score can mislead. The
# warned prompt inserts it after the task sentence, before the two positions.
TRAP_WARNING = (
    "Important: the visible score in each position can be misleading. A player who is "
    "ahead on score right now is often behind in long-run terms, and a player who is "
    "behind on score right now is often ahead. Judge the long-run outcome, not the "
    "current score.\n"
)

_POSITION_A = "\n=== Position A"


def fill_pairwise(template: str, rec_a: dict, rec_b: dict) -> str:
    """Fill a ruleset-specific template (``PAIRWISE_PROMPT_TMPL`` with the
    ruleset already formatted in) with two position records."""
    return template.format(
        focal_a=rec_a["focal_player"],
        rendering_a=rec_a["rendering"],
        focal_b=rec_b["focal_player"],
        rendering_b=rec_b["rendering"],
    )


def pairwise_prompt(
    rec_a: dict, rec_b: dict, *, ruleset: str = "civ2civ3", warned: bool = False
) -> str:
    """The prompt comparing `rec_a` (shown as Position A) with `rec_b`.

    With `warned`, ``TRAP_WARNING`` is inserted as its own paragraph between
    the task sentence and Position A; nothing else changes."""
    prompt = fill_pairwise(PAIRWISE_PROMPT_TMPL.format(ruleset=ruleset), rec_a, rec_b)
    if not warned:
        return prompt
    head, sep, tail = prompt.partition(_POSITION_A)
    if not sep:
        raise ValueError("pairwise prompt has no Position A block")
    return head + "\n" + TRAP_WARNING + sep + tail


def parse_pairwise(content: str) -> str | None:
    """The answer letter of a completion: the last ``ANSWER: A|B`` (any case),
    else the last standalone A or B, else None. Models that reason before
    answering are read from their final answer."""
    hits = re.findall(r"\bANSWER:\s*([AB])\b", content, re.IGNORECASE)
    if not hits:
        hits = re.findall(r"\b([AB])\b", content)
    return hits[-1].upper() if hits else None
