# CivTelescope

CivTelescope is a pairwise value model: shown two Freeciv positions, each from its player's fog-of-war view,
it predicts which player ends the game better off. Training pairs join positions from replay banks (labelled
by the replay oracle) and from policy rollouts (labelled by the score the episode ended with); up to 40% of
them are scoreboard traps, so copying the visible score does not fit them. The model is a LoRA adapter on
Qwen2.5-7B-Instruct; a preference is read from the next-token probabilities of `A` and `B` after the prompt.

The released adapter is [`CivMarshmallow/CivTelescope`](https://huggingface.co/CivMarshmallow/CivTelescope)
(`lora.pt`). Merge it into the backbone to serve it with sglang:

```bash
python scripts/merge_lora.py --base-model /models/Qwen2.5-7B-Instruct --adapters lora.pt --out models/civtelescope
bash scripts/serve_civtelescope.sh    # see the script header for the model path, GPUs and port
```

## Training data

```bash
BASE=/models/Qwen2.5-7B-Instruct

# replay bank: 70-turn games, positions at turns 10-65, rendered from the focal player's fog-of-war view
python scripts/build_freeciv_bank.py --preset short_game \
    --work-dir runs/banks/short_game --out-dir data/banks/short_game
python scripts/render_fog_positions.py --positions data/banks/short_game/labels.jsonl \
    --pool-dir runs/banks/short_game/pool --rec-dir runs/fog/short_game \
    --out data/banks/short_game/rendered_fog.jsonl --endturn 70

# policy-rollout positions from the value harvests of RL runs; scoreboard-reward runs
# (see training.md) write one without a CivTelescope server
python scripts/build_rollout_positions.py --out-dir data/rollout \
    --run RUN=runs/policy/RUN/value_harvest.jsonl \
    --window RUN:rollout_early:FIRST-LAST --window RUN:rollout_late:FIRST-LAST

# training pairs and held-out slices (600 natural, 600 trap)
python scripts/build_civtelescope_pairs.py --out data/civtelescope \
    --replay-train data/banks/short_game/rendered_fog.jsonl \
    --replay-eval data/banks/short_game/rendered_fog.jsonl \
    --replay-labels data/banks/short_game/labels.jsonl --rollout-dir data/rollout
```

## Train, merge, evaluate

```bash
python scripts/train_civtelescope.py --base-model $BASE --data-dir data/civtelescope \
    --out ckpts/civtelescope --monitor ckpts/civtelescope/monitor.jsonl
python scripts/merge_lora.py --base-model $BASE --adapters ckpts/civtelescope/lora.pt --out models/civtelescope
python scripts/eval_civtelescope.py --base-model $BASE --adapters ckpts/civtelescope/lora.pt \
    --data-dir data/civtelescope --out results/heldout.raw.jsonl --readout results/heldout.json
```

Training and scoring need one 80 GB GPU. The readout reports accuracy on natural and trap pairs for
CivTelescope and for its untrained backbone on the same pairs.

## Zero-shot language-model baselines

Any OpenAI-compatible model (`CIVMARSH_API_BASE`, `CIVMARSH_API_KEY`) answers the same pairwise question, with
a neutral prompt or one that warns about the scoreboard:

```bash
for m in MODEL_A MODEL_B; do
  for w in "" --warned; do
    python scripts/eval_llm_baselines.py --model $m --data-dir data/civtelescope \
        --n-trap 300 --n-natural 300 --subsample results/llm/subsample_pairs.jsonl \
        --reasoning-effort none --out-dir results/llm $w
  done
done
python scripts/report_llm_baselines.py --data-dir data/civtelescope \
    --subsample results/llm/subsample_pairs.jsonl --raw results/heldout.raw.jsonl \
    --picks MODEL_A=results/llm/MODEL_A_picks.jsonl \
    --picks MODEL_A-warned=results/llm/MODEL_A-warned_picks.jsonl \
    --out results/llm/comparison.json
```

## Transfer to an unseen ruleset

CivTelescope is trained on `civ2civ3`; this scores it on positions from the `classic` ruleset:

```bash
python scripts/build_freeciv_bank.py --preset classic_transfer \
    --work-dir runs/banks/classic_transfer --out-dir data/banks/classic_transfer
python scripts/render_fog_positions.py --positions data/banks/classic_transfer/labels.jsonl \
    --pool-dir runs/banks/classic_transfer/pool --rec-dir runs/fog/classic_transfer \
    --out data/banks/classic_transfer/rendered_fog.jsonl --endturn 120
python scripts/build_transfer_pairs.py --labels data/banks/classic_transfer/labels.jsonl \
    --rendered data/banks/classic_transfer/rendered_fog.jsonl --ruleset classic --out-dir data/transfer
for i in 0 1 2 3; do                                 # one shard per GPU
  CUDA_VISIBLE_DEVICES=$i python scripts/eval_civtelescope.py --base-model $BASE \
      --adapters ckpts/civtelescope/lora.pt --pairs data/transfer/pairs.jsonl \
      --rendered data/banks/classic_transfer/rendered_fog.jsonl --ruleset classic \
      --shard $i/4 --out results/transfer.shard$i.raw.jsonl &
done; wait
python scripts/eval_llm_baselines.py --model MODEL_A --ruleset classic \
    --pairs data/transfer/pairs_api.jsonl --rendered data/banks/classic_transfer/rendered_fog.jsonl \
    --reasoning-effort none --out-dir results/transfer_llm
python scripts/report_civtelescope.py --pairs data/transfer/pairs.jsonl \
    --raw results/transfer.shard*.raw.jsonl --picks MODEL_A=results/transfer_llm/MODEL_A_picks.jsonl \
    --api-pairs data/transfer/pairs_api.jsonl --out results/transfer_report.json
```

## Comparisons and ablations

```bash
# regression value head on the same backbone: predicts the replayed final score of one position
python scripts/train_value_head.py --base-model $BASE --data-dir data/civtelescope \
    --labels data/banks/short_game/labels.jsonl --out ckpts/value_head.pt --monitor ckpts/value_head_monitor.jsonl
python scripts/eval_value_head.py --base-model $BASE --checkpoint ckpts/value_head.pt \
    --data-dir data/civtelescope --labels data/banks/short_game/labels.jsonl \
    --reference-raw results/heldout.raw.jsonl --out results/value_head.raw.jsonl --readout results/value_head.json

# label source: replay labels vs a zero-shot model's preferences on the same pairs
python scripts/label_preferences.py --data-dir data/civtelescope --model MODEL_A --max N \
    --out results/preference_labels.jsonl
python scripts/build_label_source_pairs.py --data-dir data/civtelescope \
    --labels results/preference_labels.jsonl --out-dir data/label_source
for arm in replay preference; do
  python scripts/train_civtelescope.py --base-model $BASE --data-dir data/civtelescope \
      --train-pairs data/label_source/${arm}_labeled.jsonl --eval-interval EVAL_INTERVAL \
      --out ckpts/label_source_$arm --monitor ckpts/label_source_$arm/monitor.jsonl
done

# inputs: score masked, score only
for v in masked score_only; do
  python scripts/eval_civtelescope.py --base-model $BASE --adapters ckpts/civtelescope/lora.pt \
      --data-dir data/civtelescope --input-variant $v \
      --out results/input_$v.raw.jsonl --readout results/input_$v.json
done
```

`EVAL_INTERVAL = ceil(2 * N / 16)` evaluates the label-source arms once per epoch at the default batch size.
The value head and the preference-labelled model can also serve as RL rewards; see
[training](training.md#other-dense-rewards).

## Code

`civmarsh.civtelescope`: fog-of-war rendering (`fog`), pair building (`data`), the prompt (`prompt`), LoRA
training (`lora`, `train`), evaluation (`evaluate`), zero-shot baselines (`llm_baselines`), the value head
(`value_head`), input ablations (`ablations`) and the HTTP client used during RL (`client`).
