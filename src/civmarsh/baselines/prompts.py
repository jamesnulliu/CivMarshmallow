"""Prompt text for the prompting-scaffold baselines.

TASK_CONTRACT is the RL policy's system prompt: the baselines state the same
task and JSON contract as the trained policy, so their scores are comparable.
"""

from civmarsh.env.prompts import SYSTEM_PROMPT

TASK_CONTRACT = SYSTEM_PROMPT

# --- BaseLang-style: one unstructured chain-of-thought call ----------------

BASELANG_COT = """Think step by step before you act. In a few sentences, \
weigh your situation (growth, production, research, threats) and say which \
actions this turn serve the score at turn {endturn} best. Then, on the LAST \
line, output the JSON action object and nothing else after it.
"""

# --- Mastaba-style: department advisors + a president who decides ----------

MASTABA_ADVISOR = """You are the {department} advisor of player '{focal}'. \
Look only at your own department. In at most four sentences, state what this \
civilization should do this turn from a {department} point of view and name the \
action keys from the menu above that you recommend. Do not output JSON.
"""

MASTABA_PRESIDENT = """You are the leader of player '{focal}'. Your advisors \
report:

{advice}

Weigh the reports against each other and the state above, then commit. \
Output the JSON action object only, no explanation.
"""

# --- SAGA-style: situation-aware mid-term goal, refreshed every N turns ----

SAGA_GOAL = """Assess the position and set a mid-term plan for the next \
{refresh_every} turns. Answer in at most five lines:
SITUATION: <one line>
GOAL: <one line, the single mid-term objective>
STEPS: <up to three concrete steps toward it>
Do not output JSON.
"""

SAGA_DECIDE = """Your current mid-term plan:

{goal}

Choose this turn's actions so that they advance that plan (deviate only if \
the state above makes the plan impossible). Output the JSON action object \
only, no explanation.
"""

# --- Reflexion-style: cross-episode lessons from past games -----------------

REFLEXION_DECIDE = """Lessons you wrote after your previous games on this \
start:

{lessons}

Apply them. Output the JSON action object only, no explanation.
"""

REFLEXION_LESSON = """That game is over. You played player '{focal}' from turn \
{startturn} to turn {end_turn} and finished with score {score_end}.

Your action log:
{history}

Write at most three short lessons, one per line, each starting with "- ", \
that would raise your score if you replayed this start. Name concrete \
mistakes (what you did too late, too early, or not at all). No preamble.
"""
