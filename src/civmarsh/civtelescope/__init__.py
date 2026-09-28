"""CivTelescope: a pairwise model of long-run position value, trained on
replay-oracle outcomes.

Modules:

- ``prompt``: the pairwise prompt and its answer parser.
- ``fog``: player-visible (fog-of-war) renderings of bank positions.
- ``data``: position pools, decidable pairs, and the training/evaluation sets.
- ``lora``: the LoRA adapters CivTelescope is trained with, and merging them.
- ``train``: supervised training on the answer letter.
- ``evaluate``: scoring pairs in both presentation orders and the readouts.
- ``llm_baselines``: zero-shot models on the same pairs and prompt.
- ``value_head``: a scalar regression value head on the same backbone.
- ``ablations``: input ablations and the preference-label training set.
- ``client``: the HTTP client used to query a served model during RL.

Modules that need torch or transformers import them inside the functions that
use them.
"""
