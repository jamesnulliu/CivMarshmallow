"""Cached, retrying, rate-limited client for an OpenAI-compatible
``/chat/completions`` endpoint.

Every successful call is appended to a JSONL cache keyed by
sha256(model | reasoning_effort | tag | prompt), so reruns are free and an
interrupted job resumes where it stopped. Transport errors and HTTP 429/5xx are
retried with exponential backoff; an empty completion counts as a failure.

The endpoint and key come from the constructor or from the environment
variables ``CIVMARSH_API_BASE`` and ``CIVMARSH_API_KEY``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class _Throttle:
    """Bounds concurrent requests and spaces request starts.

    Without `lock_dir` the limits hold within this process. With `lock_dir`
    they hold across every process that uses the same directory: each request
    holds one flock'd slot file for its duration, and the start time of the
    last request is kept in a shared file."""

    def __init__(self, max_concurrency, min_interval, lock_dir):
        self.max_concurrency = max_concurrency
        self.min_interval = float(min_interval or 0.0)
        self.lock_dir = Path(lock_dir) if lock_dir else None
        self._sem = (
            threading.BoundedSemaphore(max_concurrency)
            if max_concurrency and not self.lock_dir
            else None
        )
        self._pace_lock = threading.Lock()
        self._last_start = 0.0

    @contextmanager
    def slot(self):
        if self.lock_dir:
            with self._file_slot():
                self._file_pace()
                yield
            return
        if self._sem:
            self._sem.acquire()
        try:
            self._local_pace()
            yield
        finally:
            if self._sem:
                self._sem.release()

    def _local_pace(self):
        if self.min_interval <= 0:
            return
        with self._pace_lock:
            wait = self._last_start + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_start = time.time()

    @contextmanager
    def _file_slot(self):
        import fcntl

        self.lock_dir.mkdir(parents=True, exist_ok=True)
        if not self.max_concurrency:
            yield
            return
        held = None
        while held is None:
            for i in range(self.max_concurrency):
                f = open(self.lock_dir / f"slot{i}", "w")  # noqa: SIM115 -- held until release
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held = f
                    break
                except BlockingIOError:
                    f.close()
            else:
                time.sleep(0.25)
        try:
            yield
        finally:
            fcntl.flock(held, fcntl.LOCK_UN)
            held.close()

    def _file_pace(self):
        import fcntl

        if self.min_interval <= 0:
            return
        with open(self.lock_dir / "last", "a+") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            lf.seek(0)
            s = lf.read().strip()
            last = float(s) if s else 0.0
            wait = last + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            lf.seek(0)
            lf.truncate()
            lf.write(repr(time.time()))
            lf.flush()


class ApiClient:
    """Chat-completions client for one model.

    `reasoning_effort` is sent as the request's ``reasoning_effort`` field when
    set (e.g. ``"none"`` turns off a reasoning model's thinking) and is part of
    the cache key. `max_concurrency` and `min_interval` (seconds between request
    starts) throttle live calls; cache hits are never throttled.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        max_completion_tokens: int = 2000,
        cache_path=None,
        max_concurrency: int | None = None,
        min_interval: float = 0.0,
        lock_dir=None,
        retries: int = 3,
        timeout: float = 240.0,
    ):
        self.model = model
        self.base_url = base_url or os.environ.get("CIVMARSH_API_BASE")
        self.api_key = api_key or os.environ.get("CIVMARSH_API_KEY")
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.retries = retries
        self.timeout = timeout
        self._throttle = _Throttle(max_concurrency, min_interval, lock_dir)
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self.cache_path = Path(cache_path) if cache_path else None
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec

    def call_key(self, prompt: str, tag: str = "") -> str:
        effort = self.reasoning_effort or ""
        h = hashlib.sha256(f"{self.model}|{effort}|{tag}|{prompt}".encode())
        return f"{self.model}|{effort}|{tag}|{h.hexdigest()[:24]}"

    def payload(self, prompt: str) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def chat(self, prompt: str, tag: str = "") -> str:
        """The completion text for `prompt` (cached under `tag`)."""
        return self.call(prompt, tag)["content"]

    def call(self, prompt: str, tag: str = "") -> dict:
        """The cached or live call record: key, model, reasoning_effort, tag,
        prompt_sha, content, usage, latency_s. Raises RuntimeError when every
        attempt fails."""
        key = self.call_key(prompt, tag)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                "API endpoint not configured: pass base_url/api_key or set "
                "CIVMARSH_API_BASE and CIVMARSH_API_KEY"
            )
        body = json.dumps(self.payload(prompt)).encode()
        last_err = None
        for attempt in range(self.retries):
            t0 = time.time()
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with (
                    self._throttle.slot(),
                    urllib.request.urlopen(req, timeout=self.timeout) as r,
                ):
                    d = json.load(r)
                choice = d["choices"][0]
                content = (choice["message"].get("content") or "").strip()
                if not content:
                    raise ValueError(
                        f"empty content (finish_reason={choice.get('finish_reason')})"
                    )
                rec = {
                    "key": key,
                    "model": self.model,
                    "reasoning_effort": self.reasoning_effort,
                    "tag": tag,
                    "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                    "content": content,
                    "usage": d.get("usage") or {},
                    "latency_s": round(time.time() - t0, 2),
                }
                self._store(rec)
                return rec
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode()[:200]
                except Exception:  # noqa: BLE001, S110 -- the body is optional detail
                    pass
                last_err = f"HTTP {e.code}: {detail}"
                if e.code not in RETRY_STATUSES:
                    break
            except Exception as e:  # noqa: BLE001 -- timeouts, empty content, resets
                last_err = repr(e)
            if attempt + 1 < self.retries:
                time.sleep(min(2**attempt * 2, 30))
        raise RuntimeError(f"API call failed ({self.model}/{tag}): {last_err}")

    def _store(self, rec: dict) -> None:
        with self._lock:
            self._cache[rec["key"]] = rec
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")

    def usage_totals(self) -> dict:
        """Calls and token counts per model over every cached record."""
        by_model: dict[str, dict] = {}
        with self._lock:
            for rec in self._cache.values():
                u = rec.get("usage") or {}
                m = by_model.setdefault(
                    rec["model"],
                    {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
                )
                m["calls"] += 1
                m["prompt_tokens"] += u.get("prompt_tokens") or 0
                m["completion_tokens"] += u.get("completion_tokens") or 0
        return by_model


def add_client_args(ap) -> None:
    """Endpoint, reasoning and throttling options shared by API-backed scripts."""
    g = ap.add_argument_group("API client")
    g.add_argument("--api-base", help="endpoint base URL (default: $CIVMARSH_API_BASE)")
    g.add_argument(
        "--reasoning-effort",
        default="none",
        help='request reasoning_effort ("none" disables thinking; "" omits the field)',
    )
    g.add_argument("--max-completion-tokens", type=int, default=2000)
    g.add_argument("--timeout", type=float, default=240.0)
    g.add_argument("--retries", type=int, default=3)
    g.add_argument("--max-concurrency", type=int, help="requests in flight at once")
    g.add_argument("--min-interval", type=float, default=0.0, help="s between starts")
    g.add_argument("--lock-dir", help="share the limits across processes via this dir")


def client_from_args(args, model: str, cache_path) -> ApiClient:
    return ApiClient(
        model,
        base_url=args.api_base,
        reasoning_effort=args.reasoning_effort or None,
        max_completion_tokens=args.max_completion_tokens,
        cache_path=cache_path,
        max_concurrency=args.max_concurrency,
        min_interval=args.min_interval,
        lock_dir=args.lock_dir,
        retries=args.retries,
        timeout=args.timeout,
    )
