"""slime integration: rollout, reward and advantage hooks for the policy runs.

- ``episode``: engine-thread episode driver (no slime dependency)
- ``rollout``: ``generate``, ``group_reward``, ``drop_failed_episode_groups``
- ``reward_post``: ``post_process``
- ``advantage``: ``broadcast_decision_advantages``
"""
