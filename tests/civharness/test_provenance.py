"""Preset configs, episode provenance and same-seed replay SHA.

Verifies: the two SAGA presets load and mean what they say; episode headers
carry schema/config provenance and the visibility mode; the audit callback
lands offered-candidate SHAs in step records; and a recorded run's exact
orders replay to the identical normalized final save.
"""

from pathlib import Path

import pytest

from civharness import (
    EnvSpec,
    FunctionAgent,
    GameConfig,
    candidates,
    position,
    run_agents,
    snapshot_series,
    spatial,
)
from civharness.agent import OBSERVATION_SCHEMA
from civharness.episode import read_episode, replay_episode


def test_saga_presets_load_and_gate():
    pytest.importorskip("yaml")
    vis = EnvSpec.preset("saga_visible")
    assert vis.obs("visibility") == "player_visible"
    assert not vis.full_information(), "saga_visible must not count as full information"
    full = EnvSpec.preset("saga_fullinfo")
    assert full.obs("visibility") == "full_state"


def _mid_save(root: Path):
    cfg = GameConfig(aifill=3, endturn=20, mapseed=5, gameseed=5)
    refs = snapshot_series(cfg, root / "seed", every=10)
    return next(r for r in refs if r.turn >= 10)


@pytest.mark.server
def test_header_provenance_step_audit_and_replay_sha(tmp_path):
    pytest.importorskip("yaml")
    save = _mid_save(tmp_path)
    pos = position(save)
    focal = next(n for n, p in pos.players.items() if p["cities"])

    env = EnvSpec.preset("saga_visible")

    def act(obs):
        # register the decision audit the way a downstream agent would
        if obs.audit is not None:
            offered = [f"noop:{obs.turn}"]
            obs.audit(
                offered_sha=f"sha-{obs.turn}",
                offered=offered,
                visibility=obs.player_view["visibility"],
            )
        return []

    trajs = run_agents(
        save,
        tmp_path / "rec",
        agents={focal: FunctionAgent(act)},
        until=save.turn + 3,
        env=env,
    )
    log = Path(trajs[0].episode_log_path)
    header, steps = read_episode(log)

    # 1. header provenance
    prov = header.get("provenance")
    assert prov, "episode header lost the provenance block"
    assert prov["visibility"] == "player_visible"
    assert prov["observation_schema"] == OBSERVATION_SCHEMA
    assert prov["scene_schema"] == spatial.SCENE_SCHEMA
    assert prov["digest_schema"] == spatial.DIGEST_SCHEMA
    assert prov["candidate_schema"] == candidates.ACTION_CANDIDATE_SCHEMA
    assert prov["graph_config_sha"] == spatial.SceneGraphConfig.saga().sha
    assert len(prov["graph_config_sha"]) == 16  # spatial._content_sha short form

    # 2. per-step audit records landed
    audited = [s for s in steps if s.get("audit")]
    assert audited, "no step carried the offered-candidate audit"
    a = audited[0]["audit"]
    assert a["offered_sha"].startswith("sha-")
    assert a["visibility"] == "player_visible"

    # 3. same-seed replay reproduces the identical normalized final save
    replays = replay_episode(log, save, tmp_path / "rep")
    assert replays[0].save_sha256_normalized == trajs[0].save_sha256_normalized
