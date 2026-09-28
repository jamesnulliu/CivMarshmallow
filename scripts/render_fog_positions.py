#!/usr/bin/env python3
"""Render bank positions from the focal player's fog-of-war view.

Stage ``recapture`` (needs the Freeciv server, located via CIVHARNESS_SERVER)
loads each position's autosave, takes over the focal player and records the
view the server sends at login, one ``<game_id>.jsonl.gz`` per game under
--rec-dir (resumable per game). Stage ``render`` (CPU only) turns every
captured view into the value input CivTelescope reads and writes one JSONL
file sorted by position id.

Positions are bank records (labels.jsonl or positions.jsonl of an oracle bank)
with position_id, game_id, turn, focal_player and save_path; relative save
paths are resolved against --pool-dir. --endturn is the end turn the server is
given for the loaded game and must exceed every position's turn (the bank's
game length works).

Usage:
  python scripts/render_fog_positions.py --positions bank/labels.jsonl \\
      --pool-dir bank/pool --rec-dir work/fog_rec --out bank/rendered_fog.jsonl \\
      --endturn 70 [--stage all|recapture|render] [--workers 24]
"""

import argparse
import json
from pathlib import Path

from civmarsh.civtelescope import fog
from civmarsh.utils.io import read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--positions", help="bank records to render (recapture stage)")
    ap.add_argument("--pool-dir", help="directory relative save paths resolve against")
    ap.add_argument("--rec-dir", required=True, help="recapture files (per game)")
    ap.add_argument("--out", help="rendered positions JSONL (render stage)")
    ap.add_argument("--endturn", type=int, help="server end turn for loaded saves")
    ap.add_argument("--stage", default="all", choices=("recapture", "render", "all"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument(
        "--games", nargs="*", help="restrict the recapture to these game ids"
    )
    ap.add_argument("--stats", help="write the stage summaries here (JSON)")
    a = ap.parse_args()

    summary = {}
    if a.stage in ("recapture", "all"):
        if not a.positions or a.endturn is None:
            ap.error("the recapture stage needs --positions and --endturn")
        positions = read_jsonl(a.positions)
        if a.games:
            positions = [r for r in positions if r["game_id"] in set(a.games)]
        summary["recapture"] = fog.recapture_positions(
            positions,
            a.rec_dir,
            endturn=a.endturn,
            pool_dir=a.pool_dir,
            workers=a.workers,
        )
        print(json.dumps(summary["recapture"]), flush=True)
    if a.stage in ("render", "all"):
        if not a.out:
            ap.error("the render stage needs --out")
        summary["render"] = fog.render_recaptures(a.rec_dir, a.out, workers=a.workers)
        print(json.dumps(summary["render"]), flush=True)
    if a.stats:
        Path(a.stats).write_text(json.dumps(summary, indent=2) + "\n")
    errors = summary.get("recapture", {}).get("errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
