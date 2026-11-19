#!/usr/bin/env python3
"""Build and install the civm41 ruleset of environment v4.1.

civm41 is civ2civ3 (freeciv 3.2.5) with these changes, for every player:

* Great wonders and Coinage cannot be built: each gets the requirement
  Nation = "Animal Kingdom" (Player range). Nation requirements never change during
  a game (common/requirements.c), so the bots also treat these items as never
  buildable and do not plan research for them. Small wonders stay.
* Diplomat, Spy, Caravan, Freight, Migrants and every ship (unit classes Sea and
  Trireme) get the NoBuild flag. With ships banned, maps should keep all players on
  one continent.
* The Pillage action enabler is removed, and so are the embassy and investigate-city
  enablers (civ2civ3 also grants them to Explorers), so nobody gains embassies.
* Server settings in [settings]: no start Diplomat (startunits cwx), no huts,
  animals, disasters or barbarians, a constant two-turn anarchy, diplomacy disabled
  (every civilization met is at war, no treaties), and all players on one continent
  (startpos ALL; check generated maps, the server falls back silently when it
  cannot place everyone).

The ruleset is generated from the civ2civ3 directory of a server installation and
written next to it (<prefix>/share/freeciv/civm41), so the server and the
buildability evaluator load it by name. scripts/install_freeciv_server.sh runs this
script after building the server.

    python scripts/make_civm41.py [--server <prefix>/bin/freeciv-server]
"""

import argparse
import os
import re
import shutil
from pathlib import Path

NOBUILD_UNITS = {"Diplomat", "Spy", "Caravan", "Freight", "Migrants"}
NOBUILD_CLASSES = {"Sea", "Trireme"}
BAN_REQ = (
    '"Nation", "Animal Kingdom", "Player"',
    {"present": "TRUE", "survives": "FALSE", "quiet": "FALSE"},
)
# civ2civ3 also grants the last four to Explorers
REMOVE_ENABLERS = {
    "Pillage",
    "Establish Embassy",
    "Establish Embassy Stay",
    "Investigate City",
    "Investigate City Spend Unit",
}
# name -> value, strings quoted as in the ruleset
SETTINGS = {
    "startunits": '"cwx"',
    "huts": "0",
    "animals": "0",
    "disasters": "0",
    "barbarians": '"DISABLED"',
    "revolentype": '"FIXED"',
    "revolen": "2",
    "diplomacy": '"DISABLED"',
    "startpos": '"ALL"',
}


def sections(text):
    """[(header, body)] with header = section name ('' for the preamble)."""
    parts = re.split(r"(?m)^\[([A-Za-z0-9_]+)\]\s*$", text)
    out = [("", parts[0])]
    for i in range(1, len(parts), 2):
        out.append((parts[i], parts[i + 1]))
    return out


def join(secs):
    return "".join((f"[{h}]" if h else "") + b for h, b in secs)


def field(body, key):
    m = re.search(rf"(?m)^{key}\s*=\s*(.*)$", body)
    return m.group(1).strip() if m else None


def unquote(v):
    v = v.strip()
    m = re.match(r'_\(\s*"(.*)"\s*\)', v)
    if m:
        return m.group(1)
    return v.strip('"')


def add_ban_req(body):
    row, fill = BAN_REQ
    m = re.search(r"(?ms)^reqs\s*=\s*\{(.*?)\}", body)
    if not m:
        anchor = re.search(r"(?m)^genus\s*=.*$", body)
        new = f'\nreqs\t=\n    {{ "type", "name", "range"\n      {row}\n    }}'
        return body[: anchor.end()] + new + body[anchor.end() :]
    table = m.group(1)
    header = [
        c.strip().strip('"')
        for c in table.strip().splitlines()[0].split(",")
        if c.strip()
    ]
    cells = row + "".join(", " + fill[c] for c in header[3:])
    new_table = table.rstrip() + f"\n      {cells}\n    "
    return body[: m.start(1)] + new_table + body[m.end(1) :]


def update_buildings(text):
    secs, banned = [], []
    for h, b in sections(text):
        if h.startswith("building_"):
            genus = unquote(field(b, "genus") or "")
            name = unquote(field(b, "name") or "")
            if genus == "GreatWonder" or name == "Coinage":
                b = add_ban_req(b)
                banned.append(name)
        secs.append((h, b))
    return join(secs), banned


def update_units(text):
    secs, banned = [], []
    for h, b in sections(text):
        if h.startswith("unit_"):
            name = unquote(field(b, "name") or "")
            uclass = unquote(field(b, "class") or "")
            if name in NOBUILD_UNITS or uclass in NOBUILD_CLASSES:
                m = re.search(r"(?m)^flags\s*=\s*", b)
                if m:
                    head = b[m.end() : m.end() + 400].split("\n\n")[0]
                    if '"NoBuild"' not in head:
                        b = b[: m.end()] + '"NoBuild", ' + b[m.end() :]
                else:
                    c = re.search(r"(?m)^class\s*=.*$", b)
                    b = b[: c.end()] + '\nflags         = "NoBuild"' + b[c.end() :]
                banned.append(name)
        secs.append((h, b))
    return join(secs), banned


def update_actions(text):
    secs, removed = [], []
    for h, b in sections(text):
        action = unquote(field(b, "action") or "")
        if h.startswith("actionenabler_") and action in REMOVE_ENABLERS:
            removed.append(h)
            continue
        secs.append((h, b))
    return join(secs), removed


def update_game(text):
    m = re.search(
        r'(?ms)^\[settings\].*?^set\s*=\s*\{\s*"name",\s*"value"\s*\n(.*?)^\s*\}', text
    )
    rows = m.group(1)
    done = set()

    def setting(mm):
        name = mm.group(2)
        if name in SETTINGS:
            done.add(name)
            return f'{mm.group(1)}"{name}", {SETTINGS[name]}'
        return mm.group(0)

    rows2 = re.sub(r'(?m)^(\s*)"([a-z_]+)",\s*(.*)$', setting, rows)
    for k, v in SETTINGS.items():
        if k not in done:
            rows2 += f'      "{k}", {v}\n'
    text = text[: m.start(1)] + rows2 + text[m.end(1) :]
    text = re.sub(
        r'(?m)^name = _\("Civ2Civ3 ruleset"\)',
        'name = _("CivM41 ruleset (civ2civ3 variant)")',
        text,
    )
    return text.replace(
        'description_file = "civ2civ3/README.civ2civ3"',
        'description_file = "civm41/README.civm41"',
    )


def default_share() -> Path:
    server = os.environ.get("CIVHARNESS_SERVER")
    binary = (
        Path(server) if server else Path.home() / "opt/freeciv-3.2.5/bin/freeciv-server"
    )
    return binary.resolve().parent.parent / "share" / "freeciv"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--server",
        default=None,
        help="freeciv-server binary of the installation (default: CIVHARNESS_SERVER, "
        "else ~/opt/freeciv-3.2.5/bin/freeciv-server)",
    )
    a = ap.parse_args()
    share = (
        Path(a.server).resolve().parent.parent / "share" / "freeciv"
        if a.server
        else default_share()
    )
    src, final = share / "civ2civ3", share / "civm41"
    if not (src / "game.ruleset").exists():
        raise SystemExit(f"no civ2civ3 ruleset under {share}")
    build = share / "civm41.build"
    if build.exists():
        shutil.rmtree(build)
    shutil.copytree(src, build)

    def rewrite(name, update):
        p = build / name
        text, items = update(p.read_text())
        p.write_text(text)
        return items

    buildings = rewrite("buildings.ruleset", update_buildings)
    units = rewrite("units.ruleset", update_units)
    enablers = rewrite("actions.ruleset", update_actions)
    (build / "game.ruleset").write_text(
        update_game((build / "game.ruleset").read_text())
    )
    settings = ", ".join(f"{k}={v}" for k, v in SETTINGS.items())
    summary = (
        "CivM41 = civ2civ3 (freeciv 3.2.5) with the environment v4.1 changes, built by "
        "scripts/make_civm41.py.\n\n"
        f"Not buildable (buildings, {len(buildings)}): {', '.join(buildings)}\n\n"
        f"Not buildable (units, NoBuild, {len(units)}): {', '.join(units)}\n\n"
        f"Removed action enablers: {', '.join(enablers)}\n\n"
        f"Server settings: {settings}\n\n"
    )
    (build / "README.civm41").write_text(
        summary + (src / "README.civ2civ3").read_text()
    )
    (build / "README.civ2civ3").unlink()
    # move file by file (os.replace is atomic per file), so a server loading the
    # ruleset meanwhile never sees a half-written directory
    if final.is_symlink():
        final.unlink()
    final.mkdir(exist_ok=True)
    for f in sorted(build.iterdir()):
        os.replace(f, final / f.name)
    stale = final / "README.civ2civ3"
    if stale.exists():
        stale.unlink()
    build.rmdir()
    modpack = (share / "civ2civ3.modpack").read_text()
    (share / "civm41.modpack").write_text(
        modpack.replace("civ2civ3", "civm41").replace("Civ2Civ3", "CivM41")
    )
    print(summary.rstrip())
    print("installed:", final)


if __name__ == "__main__":
    main()
