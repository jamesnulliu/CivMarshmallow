"""CivMarshmallow: replay-grounded reward signals for long-horizon Freeciv agents.

Subpackages:

- ``oracle``: replay-oracle position banks and decidable position pairs.
- ``traps``: scoreboard-trap measurement in Freeciv and other strategy games.
- ``civtelescope``: the pairwise position model (data, training, evaluation).
- ``env``: the player-visible observation and action menu shown to the policy.
- ``rewards``: sparse, scoreboard, CivTelescope, and hybrid reward signals.
- ``train``: rollout and advantage hooks for training with slime.
- ``eval``: policy evaluation and offline diagnostics.
- ``baselines``: prompting-scaffold agents.
- ``utils``: shared I/O, statistics, and API client helpers.
"""

__version__ = "0.1.0"
