# Scoreboard traps

A *scoreboard trap* is a pair of positions in which the one with the higher in-game score leads to the worse
final outcome. Trap rates are measured on position banks: games are played, positions are saved along the
way, and each position is labelled with the mean final score of K re-seeded continuations to the end of the
game (the replay oracle). Two positions from different games at nearby turns form a pair; the pair is
decidable when the gap between their oracle values exceeds twice their pooled standard error, and a decidable
pair is a trap when the visible score orders the two positions the other way. Pairs with tied visible scores
are dropped.

## Freeciv

```bash
# 120-turn games under both rulesets, positions every 5 turns from turn 10 to 115, K = 8 replays each
for p in civ2civ3_120 classic_120; do
  python scripts/build_freeciv_bank.py --preset $p --work-dir runs/banks/$p --out-dir data/banks/$p --workers 32
done
python scripts/freeciv_trap_rates.py \
    --labels freeciv_civ2civ3=data/banks/civ2civ3_120/labels.jsonl \
    --labels freeciv_classic=data/banks/classic_120/labels.jsonl \
    --out results/freeciv_trap_rates.json

# game-length curve: 60-, 70- and 80-turn games
for n in 60 70 80; do
  python scripts/build_freeciv_bank.py --preset game_length_$n \
      --work-dir runs/banks/game_length_$n --out-dir data/banks/game_length_$n
done
python scripts/game_length_curve.py \
    --labels 60=data/banks/game_length_60/labels.jsonl \
    --labels 70=data/banks/game_length_70/labels.jsonl \
    --labels 80=data/banks/game_length_80/labels.jsonl \
    --out results/game_length_curve.json
```

`build_freeciv_bank.py` runs many game servers in parallel on CPU cores; `--preset` selects the game length,
ruleset and snapshot schedule (`--help` lists the presets).

## Other games

Chess, Othello, backgammon, Go 9×9, Hearts, 2048 and Catan, each with its own engine and visible score:

```bash
GAMES="chess othello backgammon go9 hearts 2048 catan"
for g in $GAMES; do                                  # chess needs STOCKFISH_BIN
  python scripts/build_game_bank.py --game $g --out data/games/${g}_bank.jsonl
done
python scripts/measure_game_lengths.py --out data/games/lengths.json
python scripts/cross_game_trap_rates.py $(for g in $GAMES; do echo --bank $g=data/games/${g}_bank.jsonl; done) \
    --lengths data/games/lengths.json --out results/cross_game_trap_rates.json

# zero-shot language models on chess and Othello pairs
for g in chess othello; do
  python scripts/eval_game_pairs.py --game $g --bank data/games/${g}_bank.jsonl \
      --models MODEL_A,MODEL_B --reasoning-effort none --out results/${g}_zero_shot.json
done
```

More snapshots of new games (`build_game_bank.py --game-offset ... --snaps ...`) can be added to the progress
bins with `cross_game_trap_rates.py --progress-bank`.

A small Othello bank runs on a laptop CPU in under a minute:

```bash
python scripts/build_game_bank.py --game othello --n-games 6 --replays 4 --out results/othello_bank.jsonl
python scripts/cross_game_trap_rates.py --bank othello=results/othello_bank.jsonl \
    --min-bin-pairs 1 --out results/othello_trap_rates.json
```

## Traps in the policy's own games

The same-start trap rate compares sibling training episodes that started from one position. It reads the
value harvests that RL runs write (see [training](training.md)):

```bash
python scripts/same_start_traps.py --out results/same_start_traps.json --inputs \
    $(for f in runs/policy/*/seed*/*/value_harvest.jsonl; do echo "$(dirname ${f#runs/policy/} | tr / _)=$f"; done)
```

## Code

| Module | Content |
|---|---|
| `civmarsh.oracle` | replay-oracle banks and decidable position pairs |
| `civmarsh.traps` | trap rates by game progress, same-start traps, game-length curve |
| `civmarsh.traps.games` | position banks of the other games and the zero-shot pairwise prompt |
