#!/usr/bin/env python3
"""Serve a trained regression value head over HTTP for the value_head reward.

One process = one GPU = one replica.  The model, the prompt and the forward
pass are those of ``civmarsh.civtelescope.value_head`` (the checkpoint written
by scripts/train_value_head.py), so the policy is rewarded with the same head
that scripts/eval_value_head.py scores.

  CUDA_VISIBLE_DEVICES=4 python scripts/serve_value_head.py \\
      --base-model /models/Qwen2.5-7B-Instruct --checkpoint ckpts/value_head.pt \\
      --port 30880

  POST /value  {"items": [{"focal": "...", "rendering": "..."}, ...]}
            -> {"scores": [predicted final score, ...]}   (training units undone:
               prediction * target_sd + target_mu)
  GET  /health -> {"ok": true, "steps": ...}

Requests are served one at a time (a lock around the forward pass); the reward
(``civmarsh.rewards.pointwise.value_head_value``) spreads its requests over
the replicas listed in VALUE_HEAD_URLS of scripts/train_policy.sh.
"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from civmarsh.civtelescope.value_head import (
    build_examples,
    load_value_head,
    predict_positions,
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-model", required=True, help="the head's backbone")
    ap.add_argument("--checkpoint", required=True, help="value-head .pt file")
    ap.add_argument("--lora-r", type=int, default=16, help="LoRA rank of the head")
    ap.add_argument("--ruleset", default="civ2civ3", help="ruleset named in the prompt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30880)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=20000)
    a = ap.parse_args()

    model, head, tok, ck = load_value_head(
        a.base_model, a.checkpoint, device=a.device, lora_r=a.lora_r
    )
    mu, sd, steps = float(ck["target_mu"]), float(ck["target_sd"]), ck.get("steps")
    gpu = threading.Lock()

    def score(items):
        rows = [
            {
                "position_id": str(i),
                "focal_player": item["focal"],
                "rendering": item["rendering"],
                "mean_end": 0.0,
            }
            for i, item in enumerate(items)
        ]
        examples = build_examples(rows, tok, ruleset=a.ruleset)
        with gpu:
            preds = predict_positions(
                model,
                head,
                examples,
                tok,
                a.device,
                batch=a.batch,
                token_budget=a.token_budget,
            )
        return [preds[str(i)] * sd + mu for i in range(len(items))]

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "steps": steps})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/value":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers["Content-Length"])
                items = json.loads(self.rfile.read(length))["items"]
                self._send(200, {"scores": score(items)})
            except Exception as exc:  # noqa: BLE001 - the client counts a failure
                self._send(500, {"error": repr(exc)})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((a.host, a.port), Handler)
    log(f"value head ready on {a.host}:{a.port} ({a.device}, {steps} training steps)")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
