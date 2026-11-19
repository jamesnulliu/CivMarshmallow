# Environment v4.1

Environment v4.1 is the Freeciv setting of the step-level experiments: a smaller rule set with fewer side
channels, bots that see only what their player sees, and a turn prompt in which every city and unit appears on one
line with its actions. The code is in `src/civmarsh/env/v41/`; the server and the ruleset come from
`scripts/install_freeciv_server.sh`.

## Rules: the civm41 ruleset

`scripts/make_civm41.py` generates `civm41` from the installed `civ2civ3` (freeciv 3.2.5) and writes it to
`<prefix>/share/freeciv/civm41`; the install script runs it after building the server. The changes apply to every
player:

| Change | How |
|---|---|
| No great wonders, no Coinage | each gets the requirement Nation = "Animal Kingdom", which never holds; bots also treat these items as never buildable. Small wonders (Palace, Ecclesiastical Palace) stay |
| No diplomats, spies, caravans, freight, migrants or ships | `NoBuild` flag on these units and on the Sea and Trireme unit classes |
| No pillage, embassies or city investigation | the action enablers are removed |
| No huts, animals, disasters or barbarians | server settings in `game.ruleset` |
| No treaties | `diplomacy DISABLED`: every civilization met is at war |
| Constant two-turn anarchy | `revolentype FIXED`, `revolen 2` |
| One continent | `startpos ALL`; the server silently falls back on some seeds, so start maps are audited (below) |

## Server: freeciv 3.2.5 from jamesnulliu/freeciv

The install script builds a pinned commit of branch `civm-3.2.5` of
[jamesnulliu/freeciv](https://github.com/jamesnulliu/freeciv): the upstream release (tag `R3_2_5`) plus two
commits.

- **Hard bots without sight through fog.** The Hard level gets the `H_FOG` and `H_MAP` handicaps: no knowledge of
  units and cities under fog, no knowledge of unexplored tiles. Bots decide from what their player can see, like
  the agent, so bot games are a fair reference and their decisions are valid imitation targets.
- **No conquest end.** When one player is left, freeciv normally ends the game with a conquest victory and freezes
  the scores. Here the victory is announced once and the game runs to `endturn`, so the score at the turn limit
  stays comparable across games and winning early is never scored below playing on.

A player that is eliminated scores 0.

## Start positions

```bash
python scripts/build_starts_v41.py --seeds 9100-9199 --workers 16 --out data/starts_v41
```

Each seed is a bot-only game (four Hard bots, 200 tiles per player, 120 turns, a save every turn). The turn-1 save
is the start; the rest is the bots' reference game. A start is kept (`ok` in `starts.jsonl`) only if every
player starts on the same continent, no banned item ever appears and no treaty is ever signed.

## Observation and prompt

`StateV4` (`observation.py`) reads a player-visible observation (EnvSpec preset `saga_visible`) and builds
reference tables: own cities `C`, city sites `S`, threats `T`, enemy cities `G`, explore points `F` and huts `H`.
`render_turn_v41` (`menu.py`) merges state and actions into one prompt (schema `civm-turn-v41`); every city and
every unit appears once, identical busy units share a line. An excerpt:

```
[OVERVIEW]
turn 17/120 | score 6 | gold 64 | government Despotism | researching Horseback Riding | goal Invention | techs 1 | cities 1 | units 2
[MAP] refs usable as targets: C city, S site, T threat, G enemy city, F explore point, H hut
SITES (35 valid known sites)
S1 @18,62 E 3 of C1 | area food 16 shield 15 trade 16 | coastal | foreign city at 7
[CITIES] each city takes one action: build:<item> or buy. Items every city can build: barracks, temple, warriors, workers. ...
city:114 C1 Mexihco Tenochtitlan @16,61 | size 2 | building Settlers 40/30 (+5, done next turn) | food 16 | buildings 1 | defenders 0
[UNITS] each unit takes one action from its list. Every unit also takes goto:<ref> ... or goto:@x,y; settlers also take settle:<S ref> or settle:@x,y ...
unit:108 Explorer @25,21 SE 25 of C1 | mv 0: disband, explore, move:n,nw,sw,s
busy or extra units (same rules; no key = keep current orders):
Workers irrigate: unit:107 @15,61 | work:plant, work:irrigation, work:road, disband
[PLAYER] player:research: alphabet, bronze-working, horseback-riding, masonry, mysticism, pottery, warrior-code
ANSWER: one JSON object mapping each entity you command to ONE action, ...
```

## Actions

One turn is one JSON object that maps each commanded entity to one action, at most 20 entries; entities left out
keep their current orders and `{}` passes:

```json
{"unit:40": "settle:S1", "unit:41": "goto:@12,30", "city:12": "build:settlers", "player:research": "bronze-working"}
```

- `goto:<ref>` accepts any reference (`C S T G F H`) for any unit; enemy references move the unit next to the
  target. `goto:@x,y` accepts any known tile; an unreachable target sends the unit as close as possible.
- `settle:<S ref>` and `settle:@x,y` (settlers only) walk to the site and found a city.
- `attack:<direction>` attacks an adjacent enemy; `explore` hands the unit to the engine's auto-explore.
- `menu.entity_schema()` is the JSON schema of the legal answers, for constrained decoding (e.g. sglang
  `response_format`); `parse_entity_bundle` checks an answer strictly and returns the engine orders.

## Playing a game

```python
from pathlib import Path

from civmarsh.env.v41 import LLMPlayer, play_game

start = {"game_id": "v41g9100", "focal_player": "Sundiata Keita",
         "save_path": "data/starts_v41/v41g9100/saves/freeciv-T0001-Y-4000-auto.sav"}
player = LLMPlayer("http://127.0.0.1:30000", constrained=True, max_tokens=512)
record = play_game(start, "my-policy", player, seed=1, workdir=Path("runs/v41g9100"))
print(record["score_end"], record["stats"])
```

`LLMPlayer` calls an OpenAI-compatible chat endpoint with thinking disabled. The other players are Hard bots of
the server build.
