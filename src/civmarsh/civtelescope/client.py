"""HTTP client for a served CivTelescope model (sglang ``/generate``).

Used during RL to score many pairwise comparisons per episode.  One call sends
the filled pairwise prompt followed by ``"\\nANSWER:"``, decodes a single token
greedily and reads the top-20 next-token log-probabilities; the preference is

    P(A) = p("A") / (p("A") + p("B"))

with a missing letter taken as log-probability -30.  A failed request (or a
response with neither letter in the top 20) returns ``None``; callers decide
how to aggregate failures.

A single process-wide thread pool bounds the number of in-flight requests:
episodes finish concurrently and each scores its decisions from a worker
thread, so they must share one cap instead of each oversubscribing the server.
"""

from __future__ import annotations

import json
import math
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from civmarsh.civtelescope.prompt import PAIRWISE_PROMPT

ANSWER_SUFFIX = "\nANSWER:"
TOP_LOGPROBS = 20
MISSING_LOGPROB = -30.0
REQUEST_TIMEOUT = 300.0
DEFAULT_CONCURRENCY = 64

_POOL = None
_POOL_LOCK = threading.Lock()


def _pool(workers: int) -> ThreadPoolExecutor:
    """The shared pool; its size is fixed by the first caller."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=workers)
        return _POOL


def prompt_preference(url: str, prompt: str) -> float | None:
    """P(A better) for one filled pairwise prompt, or None on failure."""
    payload = {
        "text": prompt + ANSWER_SUFFIX,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_logprob": True,
        "top_logprobs_num": TOP_LOGPROBS,
        "return_text_in_logprobs": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
            out = json.load(r)
    except Exception:  # noqa: BLE001 - any failed request counts as no answer
        return None
    lp = {}
    for logprob, _token_id, token in out["meta_info"]["output_top_logprobs"][0]:
        t = (token or "").strip()
        if t in ("A", "B") and t not in lp:
            lp[t] = logprob
    if not lp:
        return None
    pa = math.exp(lp.get("A", MISSING_LOGPROB))
    pb = math.exp(lp.get("B", MISSING_LOGPROB))
    return pa / (pa + pb)


def preference(
    url: str, focal_a: str, rendering_a: str, focal_b: str, rendering_b: str
) -> float | None:
    """P(position A ends better than position B) from the served model."""
    prompt = PAIRWISE_PROMPT.format(
        focal_a=focal_a,
        rendering_a=rendering_a,
        focal_b=focal_b,
        rendering_b=rendering_b,
    )
    return prompt_preference(url, prompt)


def preferences(
    url: str, prompts: list[str], *, workers: int = DEFAULT_CONCURRENCY
) -> list[float | None]:
    """``prompt_preference`` over many prompts through the shared pool, in order."""
    return list(_pool(workers).map(lambda p: prompt_preference(url, p), prompts))
