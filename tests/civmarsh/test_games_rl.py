"""CPU tests for the policy pipeline on other games (Othello, Catan): the
rollout loop with a fake tokenizer and generate call, the game potentials with
a fake CivTelescope client, and the data builders and readouts under
scripts/games/.  The Catan tests need catanatron (the ``games`` extra); the
sample-assembly test needs slime."""

import argparse
import asyncio
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import pytest

from civharness.games import make_env, play_episode
from civharness.games import othello_engine as oth
from civmarsh.civtelescope import client
from civmarsh.rewards import games as game_values
from civmarsh.train import game_rollout
from civmarsh.traps.games import othello_bank
from civmarsh.traps.games.prompt import cross_game_prompt

HAVE_CATAN = importlib.util.find_spec("catanatron") is not None
needs_catan = pytest.mark.skipif(not HAVE_CATAN, reason="catanatron not installed")

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "games"
STANDARD = {
    "position_id": "std",
    "game": "othello",
    "board_black": oth.START_BLACK,
    "board_white": oth.START_WHITE,
    "black_to_move": True,
    "move_no": 0,
}


def _script(name):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# engine and prompt shared with the trap banks
# ---------------------------------------------------------------------------


def test_harness_engine_plays_like_the_trap_bank_engine():
    rng = random.Random(0)
    for _ in range(20):
        black, white = oth.START_BLACK, oth.START_WHITE
        for _ply in range(rng.randint(0, 30)):
            legal = list(oth.bits(oth.moves(black, white)))
            if not legal:
                break
            black, white = oth.apply_move(black, white, rng.choice(legal))
            black, white = white, black
        assert oth.moves(black, white) == othello_bank.moves(black, white)
        assert oth.render(black, white, True, 7) == othello_bank.render(
            black, white, True, 7
        )
        seed = rng.randrange(10**6)
        assert oth.play_out(black, white, True, 2, random.Random(seed), 20.0) == (
            othello_bank.play_out(black, white, True, 2, random.Random(seed), 20.0)
        )


def test_value_prompt_is_the_cross_game_prompt():
    rec_a = {"focal_player": "Black", "rendering": "RA"}
    rec_b = {"focal_player": "Black", "rendering": "RB"}
    filled = game_values.game_pairwise_prompt("othello").format(
        focal_a="Black", rendering_a="RA", focal_b="Black", rendering_b="RB"
    )
    assert filled == cross_game_prompt("othello", rec_a, rec_b)
    catan = game_values.value_prompt(argparse.Namespace(civ_game="catan"))
    assert "two different games of Settlers of Catan," in catan
    wrapped = game_values.value_prompt(
        argparse.Namespace(civ_game="othello", civ_game_value_chatml=True)
    )
    assert wrapped.startswith("<|im_start|>system\n")
    assert wrapped.endswith("<|im_start|>assistant")


# ---------------------------------------------------------------------------
# synchronous episode driver
# ---------------------------------------------------------------------------


def test_play_episode_records_valid_and_invalid_decisions():
    env = make_env("othello")
    calls = []

    def act(prompt):
        calls.append(prompt)
        if len(calls) % 3 == 0:
            return "I think d3"  # invalid -> random legal fallback
        legal = prompt.split("Your legal moves: ")[1].split("\n")[0].split(", ")
        return legal[0]

    decisions, result = play_episode(env, STANDARD, act, random.Random(11))
    assert len(decisions) == len(calls) > 10
    for i, d in enumerate(decisions):
        assert set(d) >= {
            "turn",
            "keys",
            "error",
            "value_input",
            "focal",
            "score_now",
            "candidate_sha",
            "fallback",
        }
        assert d["focal"] == "Black"
        if (i + 1) % 3 == 0:
            assert d["keys"] == [] and d["error"] and d["fallback"]
        else:
            assert len(d["keys"]) == 1 and d["error"] is None and d["fallback"] is None
    assert [d["turn"] for d in decisions] == sorted({d["turn"] for d in decisions})
    assert result == {
        "score_end": env.outcome(),
        "end_turn": env.turn(),
        "eliminated": False,
        "control_lost": None,
        "elim_class": None,
    }
    # the first decision's value input is the bank rendering of the start position
    assert decisions[0]["value_input"] == oth.render(
        oth.START_BLACK, oth.START_WHITE, True, 0
    )
    assert decisions[0]["score_now"] == 0.0


# ---------------------------------------------------------------------------
# asynchronous rollout loop (no slime: fake tokenizer and fake generate)
# ---------------------------------------------------------------------------


class _FakeTok:
    def __init__(self):
        self.kwargs = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        self.kwargs.append(kwargs)
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 251 for c in text[:64]]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i < 1000)  # ids >= 1000 are special


def _fake_post(replies, abort_at=None):
    state = {"n": 0}

    async def post(payload):
        i = state["n"]
        state["n"] += 1
        assert payload["return_logprob"] is True
        assert payload["sampling_params"]["max_new_tokens"] == 5
        assert payload["sampling_params"]["stop"] is None
        if abort_at is not None and i == abort_at:
            return {
                "text": "",
                "meta_info": {
                    "finish_reason": {"type": "abort"},
                    "output_token_logprobs": [],
                },
            }
        text = replies(i, payload)
        return {
            "text": text + "<|im_end|>",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [(-0.1, ord(c), None) for c in text]
                + [(-0.2, 1001, None)],
            },
        }

    return post


def test_run_episode_builds_decisions_the_assembler_can_read():
    env = make_env("othello")
    tok = _FakeTok()

    def replies(i, payload):
        return "zz" if i == 1 else env.legal()[-1]  # env is at the decision position

    decisions, result, aborted = asyncio.run(
        game_rollout.run_episode(
            env,
            STANDARD,
            random.Random(5),
            tok=tok,
            post_fn=_fake_post(replies),
            sampling_params={"temperature": 1.0},
            choice_tokens=5,
            tpl_kwargs={"enable_thinking": False},
        )
    )
    assert not aborted and env.done() and len(decisions) > 10
    assert all(kw == {"enable_thinking": False} for kw in tok.kwargs)
    needed = {
        "turn",
        "prompt_text",
        "prompt_ids",
        "prompt_sha",
        "response_ids",
        "response_lps",
        "response_text",
        "candidate_sha",
        "keys",
        "error",
        "value_input",
        "focal",
        "score_now",
    }
    for d in decisions:
        assert set(d) >= needed
        assert len(d["response_ids"]) == len(d["response_lps"])
        assert d["response_ids"][-1] == 1001  # the end-of-turn id stays in the tokens
        assert d["response_text"].endswith("<|im_end|>")
    # the router's end-of-turn tail did not turn a clean move into an error
    assert decisions[0]["error"] is None and len(decisions[0]["keys"]) == 1
    assert decisions[1]["error"] and decisions[1]["keys"] == []
    assert result["score_end"] == env.outcome()


def test_run_episode_reports_a_generation_abort():
    env = make_env("othello")
    decisions, result, aborted = asyncio.run(
        game_rollout.run_episode(
            env,
            STANDARD,
            random.Random(2),
            tok=_FakeTok(),
            post_fn=_fake_post(lambda i, p: env.legal()[0], abort_at=3),
            sampling_params={},
            choice_tokens=5,
        )
    )
    assert aborted and result is None and len(decisions) == 3


def test_env_from_args_and_metadata_stripping():
    env = game_rollout.env_from_args(
        argparse.Namespace(
            civ_game="othello",
            civ_game_opp_depth=1,
            civ_game_opp_temp=60.0,
            civ_game_opp_exact_from=0,
        )
    )
    assert (env.opp_depth, env.opp_temp, env.opp_exact_from) == (1, 60.0, 0)
    start = {
        "position_id": "catan_train_000",
        "game": "catan",
        "focal_seat": 1,
        "save": {"state": "x" * 6000},
    }
    assert game_rollout.sample_metadata(start, 7) == {
        "position_id": "catan_train_000",
        "game": "catan",
        "focal_seat": 1,
        "game_opp_seed": 7,
    }


def test_assembled_samples_carry_reward_and_gae():
    pytest.importorskip("slime.utils.types")
    from civmarsh.train.rollout import assemble_decision_samples

    env = make_env("othello")
    decisions, result, _ = asyncio.run(
        game_rollout.run_episode(
            env,
            STANDARD,
            random.Random(9),
            tok=_FakeTok(),
            post_fn=_fake_post(lambda i, p: env.legal()[0]),
            sampling_params={},
            choice_tokens=5,
        )
    )
    samples = assemble_decision_samples(
        decisions,
        dict(STANDARD),
        result,
        episode_id="e0",
        source_index=3,
        source_group_index=1,
        over_cap_turns=[],
        gamma=1.0,
        lam=0.9,
        value_fn=game_values.score_now_value,
        value_fn_args=argparse.Namespace(),
        err_penalty=1.0,
    )
    assert len(samples) == len(decisions)
    assert {s.rollout_id for s in samples} == {3}
    assert samples[-1].train_metadata["done"] == 1
    assert samples[-1].train_metadata["reward"] == result["score_end"]
    assert all(s.train_metadata["advantage"] is not None for s in samples)
    assert samples[0].metadata["position_id"] == "std"


# ---------------------------------------------------------------------------
# potentials
# ---------------------------------------------------------------------------


def _decisions():
    return [
        {"turn": t, "focal": "Black", "value_input": f"R{t}", "score_now": float(s)}
        for t, s in ((6, -2), (8, 1), (14, 4), (40, -6))
    ]


RESULT = {
    "score_end": 10.0,
    "end_turn": 60,
    "eliminated": False,
    "control_lost": None,
    "elim_class": None,
    "position_id": "p0",
}


def _write_refset(path):
    rows = []
    for turn in (6, 14, 38):
        for k in range(4):
            rows.append(
                {
                    "turn": turn,
                    "focal": "Black",
                    "rendering": f"REF{turn}_{k}",
                    "end_score": -12.0 + 8.0 * k,
                    "position_id": f"r{turn}_{k}",
                }
            )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_score_now_value_returns_the_proxy_and_logs(tmp_path):
    args = argparse.Namespace(
        civ_value_log=str(tmp_path / "value.jsonl"),
        civ_value_harvest=str(tmp_path / "harvest.jsonl"),
    )
    values = game_values.score_now_value(_decisions(), RESULT, args)
    assert values == [-2.0, 1.0, 4.0, -6.0]
    log = json.loads((tmp_path / "value.jsonl").read_text())
    assert log["turns"] == [6, 8, 14, 40] and log["values"] == [-2.0, 1.0, 4.0, -6.0]
    assert log["score_end"] == 10.0 and log["value_fails"] == 0
    harvest = json.loads((tmp_path / "harvest.jsonl").read_text())
    assert harvest["position_id"] == "p0" and len(harvest["decisions"]) == 4
    assert harvest["decisions"][0] == {"turn": 6, "focal": "Black", "rendering": "R6"}


def test_telescope_value_interpolates_the_winrate(tmp_path, monkeypatch):
    refset = tmp_path / "refset.jsonl"
    _write_refset(refset)
    seen = []

    def fake(url, prompt):
        seen.append(prompt)
        assert url == "http://civtelescope/generate"
        if "RENDER_FAIL" in prompt:
            return None
        # the decision is position A (single order): P(A) rises with the reference index
        return [1.0, 1.0, 0.5, 0.0][int(re.search(r"REF\d+_(\d)", prompt).group(1))]

    monkeypatch.setattr(client, "prompt_preference", fake)
    args = argparse.Namespace(
        civ_value_url="http://civtelescope/generate",
        civ_value_refset=str(refset),
        civ_value_single_order=True,
        civ_value_log=str(tmp_path / "value.jsonl"),
        civ_value_harvest=str(tmp_path / "harvest.jsonl"),
    )
    decisions = _decisions() + [
        {"turn": 37, "focal": "Black", "value_input": "RENDER_FAIL", "score_now": 0.0}
    ]
    values = game_values.telescope_value(decisions, RESULT, args)
    # winrate (1 + 1 + 0.5 + 0) / 4 = 0.625 over sorted final scores [-12, -4, 4, 12]:
    # x = 0.625 * 3 = 1.875 -> -4 * 0.125 + 4 * 0.875 = 3.0
    assert values[:4] == pytest.approx([3.0, 3.0, 3.0, 3.0])
    # every call failed for the last decision -> winrate 0.5 -> midpoint 0.0
    assert values[4] == pytest.approx(0.0)
    # nearest bucket: turns 6, 8 -> 6; 14 -> 14; 40, 37 -> 38
    assert (
        sum("REF6_" in p for p in seen) == 8 and sum("REF38_" in p for p in seen) == 8
    )
    expected = game_values.game_pairwise_prompt("othello").format(
        focal_a="Black", rendering_a="R6", focal_b="Black", rendering_b="REF6_0"
    )
    assert expected in seen
    log = json.loads((tmp_path / "value.jsonl").read_text())
    assert log["value_fails"] == 1 and len(log["values"]) == 5
    harvest = json.loads((tmp_path / "harvest.jsonl").read_text())
    assert [d["turn"] for d in harvest["decisions"]] == [6, 14, 37]  # buckets + last


def test_refset_idx_both_orders_and_chatml_wrapper(tmp_path, monkeypatch):
    refset = tmp_path / "refset.jsonl"
    _write_refset(refset)
    seen = []

    def fake(url, prompt):
        seen.append(prompt)
        body = prompt.split("=== Position A")[1]
        decision_is_a = body.index("R6") < body.index("REF6_")
        return 0.9 if decision_is_a else 0.3  # P(shown A)

    monkeypatch.setattr(client, "prompt_preference", fake)
    args = argparse.Namespace(
        civ_value_url="u",
        civ_value_refset=str(refset),
        civ_value_refset_idx=[0, 3],
        civ_value_single_order=False,
        civ_game_value_chatml=True,
    )
    values = game_values.telescope_value(_decisions()[:1], RESULT, args)
    assert len(seen) == 4  # 2 references x 2 orders
    assert all(
        p.startswith("<|im_start|>system\n") and p.endswith("<|im_start|>assistant")
        for p in seen
    )
    # winrate = mean(0.9, 1 - 0.3) = 0.8 over the 2-reference set [-12, 12] -> 7.2
    assert values == pytest.approx([7.2])


def test_mix_value_blends_the_signals(tmp_path, monkeypatch):
    refset = tmp_path / "refset.jsonl"
    _write_refset(refset)
    monkeypatch.setattr(client, "prompt_preference", lambda url, prompt: 0.625)
    args = argparse.Namespace(
        civ_value_url="u",
        civ_value_refset=str(refset),
        civ_value_single_order=True,
        civ_value_log=str(tmp_path / "value_log.jsonl"),
        civ_game_mix_j0=0.0,
        civ_game_mix_j1=1.0,
        civ_game_mix_c=10.0,
        civ_game_mix_s=0.001,
        civ_game_mix_sb=0.5,
    )
    values = game_values.mix_value(_decisions(), RESULT, args)
    # before turn 10 half the weight on the scoreboard, half on the sparse
    # baseline; after it CivTelescope alone (its value is 3.0 in every bucket)
    assert values == pytest.approx([-1.0, 0.5, 3.0, 3.0])
    assert (tmp_path / "value_log_civtelescope.jsonl").is_file()
    none = argparse.Namespace(civ_game_mix_j0=0.0, civ_game_mix_j1=0.0)
    assert game_values.mix_value(_decisions(), RESULT, none) == [-2.0, 1.0, 4.0, -6.0]
    with pytest.raises(ValueError):
        game_values.mix_weights([1], argparse.Namespace(civ_game_mix_j0=2.0))


# ---------------------------------------------------------------------------
# scripts/games builders and readouts
# ---------------------------------------------------------------------------


def _bank(n_games, turns=(6, 10), black_to_move=True, id_base=0):
    rng = random.Random(id_base + 1)
    rows = []
    for g in range(n_games):
        for turn in turns:
            rows.append(
                {
                    "position_id": f"o{id_base + g:04d}-m{turn:03d}",
                    "game_id": f"o{id_base + g:04d}",
                    "turn": turn,
                    "focal_player": "Black",
                    "rendering": f"render {id_base + g} {turn}",
                    "score_now": rng.randint(-6, 6),
                    "oracle_mean": rng.uniform(-30, 30),
                    "oracle_se": 1.0,
                    "oracle_n": 24,
                    "black_to_move": black_to_move,
                }
            )
    return rows


def test_refset_takes_eight_per_bucket_spread_over_the_outcomes():
    build_reference_set = _script("build_reference_set")
    bank = (
        _bank(30)
        + _bank(3, turns=(14,), id_base=500)
        + _bank(40, turns=(18,), black_to_move=False, id_base=900)
    )
    refset, skipped = build_reference_set.build_refset(bank, per_bucket=8)
    assert set(skipped) == {14} and {r["turn"] for r in refset} == {6, 10}
    for turn in (6, 10):
        bucket = [r for r in refset if r["turn"] == turn]
        scores = [r["end_score"] for r in bucket]
        pool = sorted(r["oracle_mean"] for r in bank if r["turn"] == turn)
        assert len(bucket) == 8 and scores == sorted(scores)
        assert scores[0] == pool[0] and scores[-1] == pool[-1]
        assert len({r["position_id"] for r in bucket}) == 8
        assert set(bucket[0]) == {
            "turn",
            "focal",
            "rendering",
            "end_score",
            "position_id",
        }
    assert build_reference_set.spread_indices(8, 8) == list(range(8))
    assert build_reference_set.build_refset(bank, 8) == (refset, skipped)
    with_white, _ = build_reference_set.build_refset(bank, 8, black_only=False)
    assert {r["turn"] for r in with_white} == {6, 10, 18}


def test_refset_takes_one_seat_per_game_state():
    build_reference_set = _script("build_reference_set")
    rng = random.Random(3)
    bank = [
        {
            "position_id": f"catan-g{g:03d}-t020-{seat}",
            "game_id": f"catan-g{g:03d}",
            "turn": 20,
            "focal_player": seat,
            "rendering": f"r{g}{seat}",
            "score_now": 2.0,
            "oracle_mean": rng.uniform(3, 10),
            "oracle_se": 0.3,
            "oracle_n": 48,
        }
        for g in range(12)
        for seat in ("RED", "BLUE", "WHITE", "ORANGE")
    ]
    refset, skipped = build_reference_set.build_refset(bank, per_bucket=8)
    assert not skipped and len(refset) == 8
    assert len({r["position_id"].rsplit("-", 1)[0] for r in refset}) == 8
    scores = [r["end_score"] for r in refset]
    assert scores == sorted(scores) and scores[0] == min(r["oracle_mean"] for r in bank)
    few, skipped = build_reference_set.build_refset(bank[:28], per_bucket=8)  # 7 states
    assert few == [] and set(skipped) == {20}


def test_pairs_split_train_and_heldout_and_keep_the_trap_slices_clean():
    build_pairs = _script("build_pairs")
    train_bank, heldout_bank = _bank(40), _bank(20, id_base=5000)
    train, eval_trap, eval_nat, stats = build_pairs.build_all(
        train_bank, heldout_bank, n_train=200, trap_frac=0.5, n_eval=50, seed=47
    )
    assert stats["max_turn_gap"] == 8 and stats["train_bank"]["positions_paired"] == 80
    train_ids = {r["position_id"] for r in train_bank}
    heldout_ids = {r["position_id"] for r in heldout_bank}
    assert len(train) == 200 and sum(p["trap"] for p in train) == 100
    assert all(
        p["a"] in train_ids and p["b"] in train_ids and not p["score_tied"]
        for p in train
    )
    assert all(
        p["oracle_winner"] in ("A", "B") and p["pool"] == "replay" for p in train
    )
    assert len(eval_trap) == 50 and all(
        p["trap"] and p["a"] in heldout_ids for p in eval_trap
    )
    assert len(eval_nat) == 50 and not any(
        p["trap"] or p["score_tied"] for p in eval_nat
    )
    again = build_pairs.build_all(
        train_bank, heldout_bank, n_train=200, n_eval=50, seed=47
    )
    assert again[0] == train
    with pytest.raises(ValueError):
        build_pairs.build_all(train_bank, train_bank[:4])
    cut = build_pairs.build_all(
        train_bank, heldout_bank, n_train=20, n_eval=5, max_positions=30
    )
    assert cut[3]["train_bank"]["positions_paired"] == 30
    assert cut[3]["train_bank"]["positions"] == 80
    catan_like = [dict(r, meta={"max_turn_gap": 20}) for r in train_bank]
    assert build_pairs.bank_turn_gap(catan_like) == 20
    assert build_pairs.bank_turn_gap(train_bank) == 8


def test_pairs_write_a_dataset_the_trainer_loads(tmp_path, monkeypatch):
    from civmarsh.civtelescope.data import load_dataset

    build_pairs = _script("build_pairs")
    train_path, heldout_path = tmp_path / "train.jsonl", tmp_path / "heldout.jsonl"
    for path, rows in (
        (train_path, _bank(40)),
        (heldout_path, _bank(20, id_base=5000)),
    ):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_pairs.py",
            "--train-bank",
            str(train_path),
            "--heldout-bank",
            str(heldout_path),
            "--out-dir",
            str(tmp_path / "ds"),
            "--n-train",
            "100",
            "--n-eval",
            "20",
        ],
    )
    assert build_pairs.main() == 0
    ds = load_dataset(tmp_path / "ds")
    assert len(ds.train) == 100 and len(ds.positions) == 120
    held = ds.heldout_pairs()
    assert sum(p["trap"] for p in held) == len(ds.eval_trap) == 20
    for p in ds.train + held:
        assert p["a"] in ds.positions and p["b"] in ds.positions


def test_othello_starts_are_valid_disjoint_and_reproducible():
    build_starts = _script("build_starts")
    splits = build_starts.build_splits("othello")
    train, evals, val = splits["train"], splits["eval"], splits["val"]
    assert (len(train), len(evals), len(val)) == (40, 45, 40)
    boards = set()
    for row in train + evals + val:
        black, white = int(row["board_black"], 0), int(row["board_white"], 0)
        assert row["black_to_move"] is True and row["game"] == "othello"
        assert not black & white
        assert oth.popcount(black | white) - 4 == row["move_no"]
        assert oth.moves(black, white) and oth.moves(white, black)
        assert (black, white) not in boards
        boards.add((black, white))
        env = make_env("othello")
        env.reset(row, random.Random(0))
        assert env.legal() and env.turn() == row["move_no"]
    assert build_starts.build_splits("othello", ("eval",))["eval"] == evals


def _write_run(run, groups, decision_err=None, config=None):
    run.mkdir(parents=True)
    with (run / "reward_log.jsonl").open("w") as f:
        for i, (pid, scores) in enumerate(groups):
            episodes = [
                {
                    "episode_id": f"{pid}-{i}-{k}",
                    "position_id": pid,
                    "score_end": s,
                    "ok": s is not None,
                }
                for k, s in enumerate(scores)
            ]
            f.write(json.dumps({"ts": float(i), "episodes": episodes}) + "\n")
    if decision_err:
        with (run / "decision_log.jsonl").open("w") as f:
            for eid, errs in decision_err.items():
                row = {"episode_id": eid, "decisions": [{"err": e} for e in errs]}
                f.write(json.dumps(row) + "\n")
    if config:
        (run / "config.yaml").write_text(config)


def test_summarizer_reads_train_and_eval_logs(tmp_path):
    summarize = _script("summarize")
    _write_run(tmp_path / "entry", [("p0", [0, 2]), ("p1", [-4, None])])
    _write_run(tmp_path / "grad", [("p0", [4, 6]), ("p1", [-2, -2])])
    _write_run(tmp_path / "grad_b", [("p0", [2, 2]), ("p1", [0, 2])])
    cell = summarize.summarize_eval(tmp_path / "entry")
    assert cell["game"] == "othello"
    # Catan counts a win when the focal seat reached 10 victory points
    _write_run(
        tmp_path / "catan",
        [("c0", [10, 6]), ("c1", [11, 4])],
        config='civ_game: "catan"\n',
    )
    catan = summarize.summarize_eval(tmp_path / "catan")
    assert catan["game"] == "catan" and catan["win_rate"] == 0.5 and catan["S"] == 7.75
    assert (
        cell["S"] == pytest.approx((1.0 + -4.0) / 2) and cell["n_failed_episodes"] == 1
    )
    assert cell["n_valid_episodes"] == 3
    assert cell["win_rate"] == pytest.approx(1 / 3, abs=1e-3)
    gain = summarize.summarize_gain(
        tmp_path / "entry", tmp_path / "grad", tmp_path / "entry", tmp_path / "grad_b"
    )
    assert gain["G"] == pytest.approx((4.0 + 2.0) / 2)  # p0: 5 - 1, p1: -2 - -4
    assert gain["b"]["G"] == pytest.approx((1.0 + 5.0) / 2)
    assert gain["paired_win_rate_a_over_b"] == pytest.approx(0.5)

    _write_run(
        tmp_path / "train",
        [("p0", [1, 3]), ("p1", [5, None]), ("p2", [5, 7]), ("p3", [9, 9])],
        {
            "p0-0-0": ["bad", None],
            "p0-0-1": [None, None],
            "p2-2-0": [None],
            "p2-2-1": [None],
        },
    )
    run = summarize.summarize_train(tmp_path / "train", groups_per_iter=2)
    assert run["n_kept_groups"] == 3  # the group with a failure is skipped
    first, second = run["iterations"]
    assert first["mean_score_end"] == pytest.approx(4.0) and first["complete"]
    assert first["invalid_decision_rate"] == pytest.approx(1 / 6, abs=1e-3)
    assert second["mean_score_end"] == pytest.approx(9.0) and not second["complete"]


def test_calibration_picks_the_strongest_rung_the_base_policy_beats(tmp_path):
    calibrate = _script("calibrate_opponent")
    starts = tmp_path / "val.jsonl"
    starts.write_text(
        "".join(json.dumps({"position_id": p}) + "\n" for p in ("s0", "s1"))
    )

    def probe(name, scores):
        env = calibrate.LADDER[name]
        config = (
            'civ_game: "othello"\n'
            f"civ_game_opp_depth: {env['GAME_OPP_DEPTH']}\n"
            f"civ_game_opp_temp: {env['GAME_OPP_TEMP']}\n"
            f"civ_game_opp_exact_from: {env['GAME_OPP_EXACT_FROM']}\n"
        )
        _write_run(tmp_path / name, [("s0", scores), ("s1", scores)], config=config)
        return f"{name}={tmp_path / name}"

    args = argparse.Namespace(
        probe=[
            probe("d2t20x10", [-10] * 8),
            probe("d2t20x0", [-4] * 7 + [2]),  # 12.5% wins
            probe("d1t20x0", [-4] * 6 + [2, 2]),  # 25% wins
        ],
        starts=str(starts),
        threshold=0.15,
        games_per_start=8,
        out=str(tmp_path / "opponent.json"),
    )
    assert calibrate.cmd_select(args) == 0
    chosen = json.loads((tmp_path / "opponent.json").read_text())
    assert chosen["rung"] == "d1t20x0"
    assert chosen["launcher_args"] == calibrate.LADDER["d1t20x0"]
    # a probe played against another opponent than its rung is refused
    bad = argparse.Namespace(
        **{**vars(args), "probe": [f"d1t60x0={tmp_path / 'd1t20x0'}"]}
    )
    with pytest.raises(SystemExit):
        calibrate.cmd_select(bad)


# ---------------------------------------------------------------------------
# Catan (needs catanatron)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def catan_starts():
    build_starts = _script("build_starts")
    return build_starts.build_splits("catan", ("train", "eval"))


@needs_catan
def test_catan_starts_load_and_siblings_share_the_position(catan_starts):
    train, evals = catan_starts["train"], catan_starts["eval"]
    assert len(train) == 40 and len(evals) == 45
    assert len({r["save"]["state"] for r in train + evals}) == 85
    assert [r["focal_seat"] for r in train[:8]] == [0, 1, 2, 3, 0, 1, 2, 3]
    env, other = make_env("catan"), make_env("catan")
    for row in (train[0], train[5], evals[2]):
        env.load(row["save"])
        assert env.focal == row["focal_player"] and env.turn() == 0
        first = env.value_input()
        # two siblings: the same saved state, different engine seeds
        env.reset(row, random.Random(1))
        other.reset(row, random.Random(2))
        assert not env.done() and len(env.legal()) >= 2
        other.load(row["save"])
        assert other.value_input() == first
    seat0 = train[0]
    env.reset(seat0, random.Random(1))
    other.reset(seat0, random.Random(2))
    assert env.policy_prompt() == other.policy_prompt()  # seat 0 decides first


@needs_catan
def test_run_episode_plays_catan_through_the_rollout_loop(catan_starts):
    start = catan_starts["train"][1]
    args = argparse.Namespace(civ_game="catan", civ_game_max_turns=60)
    env = game_rollout.env_from_args(args)

    def replies(i, payload):
        return "nonsense" if i == 2 else env.legal()[-1]

    decisions, result, aborted = asyncio.run(
        game_rollout.run_episode(
            env,
            start,
            random.Random(5),
            tok=_FakeTok(),
            post_fn=_fake_post(replies),
            sampling_params={"temperature": 1.0},
            choice_tokens=5,
        )
    )
    assert not aborted and env.done() and len(decisions) > 5
    assert result["end_turn"] <= 60 and result["score_end"] == env.outcome() >= 2.0
    assert set(result) >= {"score_end", "end_turn", "eliminated", "won", "truncated"}
    assert (
        decisions[2]["error"]
        and decisions[2]["keys"] == []
        and decisions[2]["fallback"]
    )
    for d in decisions:
        assert d["focal"] == start["focal_player"] and d["n_legal"] >= 2
        assert d["value_input"].startswith("Settlers of Catan, turn ")
        assert f"prospects of {start['focal_player']}." in d["value_input"]
        if not d["error"]:
            assert re.fullmatch(r"a[0-9]+", d["keys"][0]) and d["action"]
    assert [d["turn"] for d in decisions] == sorted(d["turn"] for d in decisions)

    # same start, same engine seed, same replies -> the same episode
    env2 = game_rollout.env_from_args(args)
    again, result2, _ = asyncio.run(
        game_rollout.run_episode(
            env2,
            start,
            random.Random(5),
            tok=_FakeTok(),
            post_fn=_fake_post(lambda i, p: "nonsense" if i == 2 else env2.legal()[-1]),
            sampling_params={"temperature": 1.0},
            choice_tokens=5,
        )
    )
    assert result2 == result
    assert [d["action"] for d in again] == [d["action"] for d in decisions]


@needs_catan
def test_potentials_on_catan_decisions(tmp_path, monkeypatch, catan_starts):
    start = catan_starts["eval"][0]
    args = argparse.Namespace(civ_game="catan", civ_game_max_turns=40)
    env = game_rollout.env_from_args(args)
    decisions, result, _ = asyncio.run(
        game_rollout.run_episode(
            env,
            start,
            random.Random(3),
            tok=_FakeTok(),
            post_fn=_fake_post(lambda i, p: env.legal()[0]),
            sampling_params={},
            choice_tokens=5,
        )
    )
    values = game_values.score_now_value(decisions, result, argparse.Namespace())
    assert values == [d["score_now"] for d in decisions]
    assert set(values) <= {float(v) for v in range(11)}

    refset = tmp_path / "refset.jsonl"
    with refset.open("w") as f:
        for turn in (20, 40):
            for k in range(4):
                row = {
                    "turn": turn,
                    "focal": "RED",
                    "rendering": f"ref {turn} {k}",
                    "end_score": 4.0 + 2 * k,
                    "position_id": f"r{turn}{k}",
                }
                f.write(json.dumps(row) + "\n")
    prompts = []

    def fake(url, prompt):
        prompts.append(prompt)
        return 0.75

    monkeypatch.setattr(client, "prompt_preference", fake)
    args = argparse.Namespace(
        civ_value_url="http://x",
        civ_value_refset=str(refset),
        civ_value_single_order=True,
        civ_game="catan",
    )
    values = game_values.telescope_value(
        decisions, {**result, "position_id": "p"}, args
    )
    assert values == [pytest.approx(4.0 + 0.75 * 6)] * len(decisions)
    assert len(prompts) == 4 * len(decisions)
    assert "two different games of Settlers of Catan," in prompts[0]
    assert f"evaluate player '{start['focal_player']}'" in prompts[0]
