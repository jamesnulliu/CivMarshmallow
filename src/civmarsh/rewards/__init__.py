"""Reward and advantage computation for the policy runs.

- ``gae``: per-decision rewards and cross-decision GAE
- ``potential``: CivTelescope potential against a reference set (``civtelescope``,
  ``terminal_civtelescope``)
- ``scoreboard``: current in-game score as the potential (``scoreboard``)
- ``hybrid``: sigmoid hand-offs between scoreboard, CivTelescope and sparse credit
- ``reference_set``: periodic rebuild of the reference set from policy episodes
"""
