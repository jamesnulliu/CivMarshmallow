# Othello and Catan

The recipe is not specific to Freeciv. Othello and Catan run in-process under the same save / branch / replay
contract (`civharness.games`), and the full pipeline runs on them: replay-labelled banks, a per-game
CivTelescope, a reference set, and RL with the same slime recipe and the same rewards.

| | Othello | Catan |
|---|---|---|
| policy plays | Black | one of four seats, rotating over the starts |
| other seats | a bundled search engine (depth 2, softmax temperature 20, exact endgame from 10 empty squares) | three Catanatron `VictoryPointPlayer` bots |
| visible score (scoreboard) | disc differential now | public victory points |
| final score | final disc differential | final victory points (first to 10 wins) |
| one decision | one square (`d3`) | one menu key (`a3`), each key described in the prompt |
| dependency | none (stdlib) | `catanatron==3.2.1` (the `games` extra) |

## The harness API

```python
from civharness.games import bank_rows, branch, noise_floor, play, snapshot_series

# save: one engine game, saved at the given progress indices
saves = snapshot_series(
    "catan", seed=11, snaps=[20, 60, 100]
)  # Catan: turn = Catanatron num_turns

# branch: K re-seeded engine continuations to the end of the game (the replay oracle)
values = branch(saves[1], k=48, seed=5)  # the focal seat's final victory points
everyone = branch(
    saves[1], k=48, seed=5, all_players=True
)  # one {player: outcome} per continuation
floor = noise_floor(saves[1], k=48, seed=5)  # mean / sd / se / n

# external control: act_fn(prompt) -> reply plays the focal seat, the engine plays the others
decisions, result = play({**saves[0], "focal_seat": 2}, act_fn, seed=3)

# a labelled position bank (one row per position; Catan: one per seat)
rows = bank_rows(
    "othello", seeds=range(100, 160), snaps=range(12, 41, 5), k=24, workers=8
)
```

A game is a `GameEnv` (`civharness/games/base.py`):

- engine mode: `new_game`, `save` / `load`, `selfplay(rng, snaps)`, `rollout(rng)`, `outcomes()`;
- focal mode: `reset(start, rng)`, then per decision `turn()`, `legal()`, `policy_prompt()`, `value_input()`
  (the rendering CivTelescope sees), `score_now()`, `parse(reply, legal)`, `step(key_or_None, rng)`, and
  `outcome()` at the end.

In focal mode the caller only sees real choices: a forced move (an Othello pass, a lone Catan `ROLL`,
`DISCARD` or `END_TURN`) is played by the harness. An invalid reply is recorded as an error and a random legal
action is played instead. New games plug in with `@register("name")`.

**Determinism.** All randomness comes from the `random.Random` the caller passes, so the same save and seed
give the same continuation, in one process and across processes. Catanatron needs two fixes for this, both
inside the adapter: its module-level random draws run under a lock with the caller's state swapped in, and
every action list is sorted before a bot or the menu sees it.

**What the Catan player sees.** The policy prompt and CivTelescope's input show the table as the player sees
it: the board with every settlement and city and the tiles it collects from, ports, the robber, every player's
public counts, the player's own hand by type, and only the sizes of the opponents' hands.

A Catan save is a pickle of the engine state: it is only valid for the catanatron version that wrote it, and
unpickling runs code, so load only saves you produced.

## Pipeline

`G` is `othello` or `catan`; `D=data/games/$G`. The two games are independent and can run side by side.
`$QWEN25` is a Qwen2.5-7B-Instruct directory (CivTelescope's backbone); `BASE_MODEL`, `SLIME_DIR` and
`MEGATRON_DIR` are set as for [Freeciv training](training.md).

```bash
G=othello; D=data/games/$G

# 1. start positions (CPU): writes $D/starts/{train,eval,val}.jsonl
python scripts/games/build_starts.py --game $G

# 2. banks (CPU), three disjoint seed ranges
#    othello: --games 300 / 60 / 60, --replays 16 / 24 / 24;  catan: --games 150 / 40 / 60, --replays 24 / 48 / 48
python scripts/games/build_bank.py --game $G --role train     --games 300 --replays 16 --workers 32 --out $D/bank_train.jsonl
python scripts/games/build_bank.py --game $G --role heldout   --games 60  --replays 24 --workers 32 --out $D/bank_heldout.jsonl
python scripts/games/build_bank.py --game $G --role reference --games 60  --replays 24 --workers 32 --out $D/bank_reference.jsonl

# 3. pairs (CPU): 7,200 training pairs (half traps), 600 trap and 600 non-trap held-out pairs;
#    stats.json lists any slice that came out short (grow the bank then)
python scripts/games/build_pairs.py --train-bank $D/bank_train.jsonl --heldout-bank $D/bank_heldout.jsonl \
    --out-dir $D/civtelescope

# 4. CivTelescope (one 80 GB GPU): pick the snapshot with the best held-out trap accuracy (acc_trap in the monitor)
python scripts/games/train_civtelescope.py --game $G --base-model $QWEN25 --data-dir $D/civtelescope \
    --out ckpts/games/$G/civtelescope --monitor ckpts/games/$G/civtelescope/monitor.jsonl
python scripts/merge_lora.py --base-model $QWEN25 \
    --adapters ckpts/games/$G/civtelescope/lora_step<N>_e<E>.pt --out models/games/$G/civtelescope

# 5. reference set (CPU): 8 positions per turn bucket, spread over the outcomes
python scripts/games/build_reference_set.py --bank $D/bank_reference.jsonl --out $D/refset.jsonl

# 6. policy training: one 8-GPU node per run
for s in 43 44 45; do
  bash scripts/games/train_policy.sh GAME=$G REWARD=sparse       SEED=$s
  bash scripts/games/train_policy.sh GAME=$G REWARD=scoreboard   SEED=$s
  bash scripts/games/train_policy.sh GAME=$G REWARD=civtelescope SEED=$s CIVTELESCOPE_HF=models/games/$G/civtelescope
  bash scripts/games/train_policy.sh GAME=$G REWARD=zero_shot    SEED=$s CIVTELESCOPE_BASE_MODEL=$QWEN25
done
# final checkpoint of a run: runs/games/$G/<reward>/seed<seed>/hf_iter11

# 7. evaluation: one GPU per cell, every evaluation start x 8 games
bash scripts/games/train_policy.sh GAME=$G EVAL=1 LABEL=base
bash scripts/games/train_policy.sh GAME=$G EVAL=1 LABEL=civtelescope_s43 \
    INIT_CHECKPOINT=runs/games/$G/civtelescope/seed43/hf_iter11
bash scripts/games/train_policy.sh GAME=$G EVAL=1 LABEL=scoreboard_s43 \
    INIT_CHECKPOINT=runs/games/$G/scoreboard/seed43/hf_iter11

# 8. readouts
python scripts/games/summarize.py train runs/games/$G/*/seed*          # training curves
python scripts/games/summarize.py eval  runs/games/$G/eval/*           # final score and win rate per cell
python scripts/games/summarize.py gain --entry runs/games/$G/eval/base --grad runs/games/$G/eval/civtelescope_s43 \
    --entry-b runs/games/$G/eval/base --grad-b runs/games/$G/eval/scoreboard_s43
```

`bash scripts/games/train_policy.sh DRY_RUN=1 ...` prints the configuration and the slime command line without
starting anything. The script header lists every variable.

### Rewards

| `REWARD` | Signal |
|---|---|
| `sparse` | final score, whitened over the sibling episodes of each start |
| `scoreboard` | potential Φ = the visible score now; per-decision GAE |
| `civtelescope` | potential Φ = the game's CivTelescope against the reference set |
| `zero_shot` | the `civtelescope` reward served by the untrained backbone: same prompt, same reference set |
| `mix` | Φ = w_c(t) · CivTelescope + w_s(t) · scoreboard; the leftover weight goes to the sparse reward. `MIX_J0` / `MIX_J1`: CivTelescope weight before / after the hand-off (0.5 / 0.5); `MIX_C` / `MIX_S`: hand-off center and width in turns (0 / 1); `MIX_SB`: share of the remaining weight on the scoreboard (1) |

### Calibrating the Othello opponent

Against the default engine an untrained policy almost never wins, so every reward starts from the same floor of
lost games and runs that differ only in the reward are hard to tell apart. `calibrate_opponent.py` picks the
strongest opponent the untrained policy still beats in at least 15% of its games, looking only at the untrained
policy:

```bash
python scripts/games/calibrate_opponent.py ladder      # rungs, strongest first, as launcher arguments
# evaluate the untrained policy once per rung on the validation starts, e.g.
bash scripts/games/train_policy.sh GAME=othello EVAL=1 LABEL=probe_d2t20x0 STARTS=$D/starts/val.jsonl \
    GAME_OPP_DEPTH=2 GAME_OPP_TEMP=20.0 GAME_OPP_EXACT_FROM=0
python scripts/games/calibrate_opponent.py select \
    --probe d2t20x10=runs/games/othello/eval/probe_d2t20x10 --probe d2t20x0=runs/games/othello/eval/probe_d2t20x0 \
    --starts $D/starts/val.jsonl --out $D/opponent.json
```

Then use the selected `GAME_OPP_*` values for every training and evaluation run, and build the banks with the
same replay engine (`build_bank.py --replay-depth --replay-temp --replay-exact-from`), so the oracle labels come
from the opponent the policy meets.

### First-iteration checks

Look at the first iteration of the first run of each reward before launching the rest:

1. **Invalid replies**: `summarize.py train <run>` reports the share of invalid decisions. If the untrained
   model is mostly invalid, the run trains formatting rather than play; `decision_log.jsonl` has each reply.
2. **Score variance**: the eight games of one start must not all end with the same score (`reward_log.jsonl`).
3. **CivTelescope calls**: `value_fails` is 0 on every line of `value_log.jsonl`.
4. **Potential vs outcome**: from the middle of the game on, a decision's value should rank-correlate clearly
   with the episode's final score (`value_log.jsonl`). If it does not, compare with `VALUE_CHATML=1`, which
   serves the prompt inside the chat turn CivTelescope was trained in, and keep one setting per game.
5. **Catan**: if games run to the engine's 1000-turn cap, set `GAME_MAX_TURNS` (e.g. 400) for every run of the
   game. Episode length drives the CivTelescope cost.

## Code

| Path | Content |
|---|---|
| `civharness.games` | the game interface, `snapshot_series` / `branch` / `noise_floor` / `play`, `bank_rows`, Othello, Catan |
| `civmarsh.rewards.games` | scoreboard, CivTelescope and mix potentials for the games |
| `civmarsh.train.game_rollout` | slime `generate` for in-process games; group reward and filters shared with Freeciv |
| `scripts/games/` | the pipeline above |
