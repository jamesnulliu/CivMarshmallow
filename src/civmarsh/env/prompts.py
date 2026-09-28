"""Policy prompt text: the system contract and the recent-actions digest.

Each decision prompt is ``SYSTEM_PROMPT`` (filled per episode) + "\\n" +
``event_digest(...)`` + "\\n" + ``menu.obs_block(...)``.  Both strings are
model-visible and must stay byte-identical to what the trained policies saw.
"""

SYSTEM_PROMPT = """You are playing Freeciv (civ2civ3 ruleset) as player '{focal}'. \
You will play the game from turn {startturn} to turn {endturn}. Each turn you receive a \
bounded SAGA-aligned state digest, a short log of your recent actions, and six \
groups of stable action keys. Return exactly one JSON object whose domain values \
are lists of offered keys. Select at most four total actions and never select two \
actions for the same entity. Use {{}} to pass. Your goal is the highest score at \
turn {endturn}: expand, build cities, grow the economy, defend, and research. \
Return JSON only.
"""


def event_digest(records, window):
    """Deterministic recent-actions digest built from this episode's own
    decision records (``turn``, ``keys``, optional ``error``); it carries no
    model text and no earlier observation."""
    if not records:
        return "Recent actions: none (first decision).\n"
    lines = ["Recent actions:"]
    for r in records[-window:]:
        if r.get("error"):
            lines.append(f"- turn {r['turn']}: no-op ({r['error'][:80]})")
        elif r.get("keys"):
            lines.append(f"- turn {r['turn']}: executed {', '.join(r['keys'])}")
        else:
            lines.append(f"- turn {r['turn']}: passed")
    return "\n".join(lines) + "\n"
