"""Server- and GPU-free tests for the prompting-scaffold baselines.

The engine side is faked at the handshake, not around it: the fake
``drive_episode`` runs in a real thread and does what the RL rollout's does
(clear ``reply_ready``, publish ``ep.obs``, set ``obs_ready`` through
``call_soon_threadsafe``, block on ``reply_ready``, then take the reply parsed
by the real action-bundle parser).  So these tests pin the timing contract the
driver must honor, without a freeciv server.

Covers: per-scaffold call structure (Mastaba >= 3 calls per decision, SAGA's
goal cache refreshing on the N-turn cadence, Reflexion injecting lessons only
from the second episode on), token bookkeeping against known mock counts, a
multi-decision loop reaching a terminal score, and the readout (valid
episodes, seat stalls excluded, one record per position and sample).
"""

import asyncio
import threading
import uuid

import pytest

from civharness.policy.client import BuildUnit, SetResearch
from civmarsh.baselines import driver, scaffolds
from civmarsh.env.menu import ActionMenu, MenuCandidate, MenuConfig

# ---------------------------------------------------------------------------
# fake engine: same handshake as civmarsh.train.episode
# ---------------------------------------------------------------------------

BUILD = BuildUnit(city_id=10, unit="Settlers")
RESEARCH = SetResearch(tech="Bronze Working")


def _menu():
    return ActionMenu(
        (
            MenuCandidate(
                key="city:10/build-unit:settlers",
                domain="city",
                entity="city:10",
                label="build Settlers",
                orders=(BUILD,),
            ),
            MenuCandidate(
                key="tech:bronze-working",
                domain="technology",
                entity="research",
                label="research Bronze Working",
                orders=(RESEARCH,),
            ),
        ),
        (),
        MenuConfig(),
    )


PICK = '{"city": ["city:10/build-unit:settlers"]}'
MENU_HEADER = f"ACTION_MENU {_menu().schema}"


class FakeEngine:
    """Stand-in for civmarsh.train.episode."""

    def __init__(
        self, turns, *, score_end=137.0, error=None, eliminated=False, control_lost=None
    ):
        self.turns = list(turns)
        self.score_end = score_end
        self.error = error
        self.eliminated = eliminated
        self.control_lost = control_lost
        self.orders = []  # orders the engine "executed", per decision

    class Episode:
        def __init__(self):
            self.uid = uuid.uuid4().hex[:8]
            self.loop = None
            self.obs_ready = None
            self.reply_ready = threading.Event()
            self.obs = None
            self.obs_raw = None
            self.reply = None
            self.raw = None
            self.done = False
            self.error = None
            self.result = None
            self.choices = []
            self.handed = False

    def drive_episode(self, ep, meta, args):
        menu = _menu()
        try:
            if self.error:
                raise RuntimeError(self.error)
            for turn in self.turns:
                ep.reply_ready.clear()
                ep.obs = (f"STATE turn={turn}\n{menu.render()}", menu, turn)
                ep.handed = True
                ep.loop.call_soon_threadsafe(ep.obs_ready.set)
                ep.reply_ready.wait()
                # the shape that matters: a bundle or an error object arrived
                reply = ep.reply
                if hasattr(reply, "orders"):
                    self.orders.append(list(reply.orders))
                else:
                    self.orders.append([])
                ep.choices.append({"turn": turn})
            ep.result = {
                "endpoint_save": "fake.sav",
                "end_turn": self.turns[-1] + 1,
                "score_end": self.score_end,
                "eliminated": self.eliminated,
                "control_lost": self.control_lost,
            }
        except Exception as exc:  # noqa: BLE001 -- the real thread's failure path
            ep.error = f"{type(exc).__name__}: {exc}"
        finally:
            ep.done = True
            ep.loop.call_soon_threadsafe(ep.obs_ready.set)


class Cfg:
    civ_endturn = 70
    civ_digest_window = 5


META = {
    "position_id": "rem120-s01",
    "save_path": "start.sav",
    "focal_player": "Alice",
    "start_turn": 1,
}


def mock_llm(replies, *, prompt_tokens=100, completion_tokens=7):
    """LLM stub with known token counts.  ``replies`` is a callable taking
    the prompt text, or a fixed string."""
    calls = []

    async def llm(messages):
        prompt = messages[0]["content"]
        calls.append(prompt)
        text = replies(prompt) if callable(replies) else replies
        return text, prompt_tokens, completion_tokens

    llm.calls = calls
    return llm


def run(scaffold, engine, llm, *, memory=None, cfg=None, meta=None):
    return asyncio.run(
        driver.run_episode(
            scaffold, meta or META, cfg or Cfg(), llm=llm, engine=engine, memory=memory
        )
    )


# ---------------------------------------------------------------------------
# 1. the loop: several decisions, handshake timing, terminal score
# ---------------------------------------------------------------------------


def test_driver_runs_three_decisions_and_reports_score():
    engine = FakeEngine([1, 2, 3, 4])
    llm = mock_llm(f"thinking...\n{PICK}")
    out = run(scaffolds.BaseLangStyle(), engine, llm)

    assert out["n_decisions"] == 4
    assert out["score_end"] == 137.0
    assert out["end_turn"] == 5
    assert out["eliminated"] is False
    assert [d["turn"] for d in out["decisions"]] == [1, 2, 3, 4]
    # every decision parsed and reached the engine as real orders
    assert all(d.get("error") is None for d in out["decisions"])
    assert [d["keys"] for d in out["decisions"]] == [
        ["city:10/build-unit:settlers"]
    ] * 4
    assert engine.orders == [[BUILD]] * 4


def test_prompt_carries_contract_digest_and_menu():
    engine = FakeEngine([1, 2])
    llm = mock_llm(PICK)
    run(scaffolds.BaseLangStyle(), engine, llm)

    first, second = llm.calls
    assert "Freeciv (civ2civ3 ruleset) as player 'Alice'" in first
    assert "Recent actions: none (first decision)." in first
    assert MENU_HEADER in first
    # the digest is deterministic and grows from our own executed keys
    assert "turn 1: executed city:10/build-unit:settlers" in second


def test_invalid_reply_is_a_logged_noop_not_a_crash():
    engine = FakeEngine([1, 2])
    llm = mock_llm("I refuse to answer in JSON.")
    out = run(scaffolds.BaseLangStyle(), engine, llm)

    assert out["n_decisions"] == 2
    assert all(d["error"] for d in out["decisions"])
    assert out["decisions"][0]["keys"] == []
    assert engine.orders == [[], []]
    assert "turn 1: no-op" in llm.calls[1]


def test_engine_failure_surfaces_as_error_without_score():
    engine = FakeEngine([1, 2], error="port race")
    out = run(scaffolds.BaseLangStyle(), engine, mock_llm(PICK))
    assert "port race" in out["error"]
    assert "score_end" not in out


def test_scaffold_exception_releases_the_engine_thread():
    # a scaffold blowing up must not leave the engine thread parked on
    # reply_ready (that leaks a live freeciv session)
    class Boom:
        name = "boom"

        async def reply(self, ctx):
            raise ValueError("boom")

    engine = FakeEngine([1, 2, 3])
    with pytest.raises(ValueError):
        run(Boom(), engine, mock_llm(PICK))
    assert engine.orders == [[]] * 3  # the session ran to its end, no-oped


# ---------------------------------------------------------------------------
# 2. per-scaffold call structure
# ---------------------------------------------------------------------------


def test_scaffold_registry_names():
    assert sorted(scaffolds.SCAFFOLDS) == [
        "baselang",
        "direct",
        "mastaba",
        "reflexion",
        "saga",
    ]


def test_baselang_is_one_call_per_decision():
    engine = FakeEngine([1, 2, 3])
    llm = mock_llm(PICK)
    out = run(scaffolds.BaseLangStyle(), engine, llm)
    assert len(llm.calls) == 3
    assert all(d["tokens"]["n_calls"] == 1 for d in out["decisions"])


def test_mastaba_takes_at_least_three_calls_per_decision():
    engine = FakeEngine([1, 2, 3])
    llm = mock_llm(PICK)
    out = run(scaffolds.MastabaStyle(), engine, llm)

    # 3 advisors + 1 president per decision
    assert len(llm.calls) == 12
    assert all(d["tokens"]["n_calls"] >= 3 for d in out["decisions"])
    tags = [c["tag"] for c in out["llm_calls"]]
    assert tags[:4] == [
        "advisor:military",
        "advisor:economy",
        "advisor:expansion",
        "president",
    ]
    # the president actually sees the advisors' text
    president = llm.calls[3]
    assert "[military]" in president and "[expansion]" in president


def test_saga_refreshes_the_goal_cache_every_n_turns():
    engine = FakeEngine(list(range(1, 26)))  # turns 1..25

    def replies(prompt):
        return "GOAL: expand north" if "SITUATION:" in prompt else PICK

    llm = mock_llm(replies)
    out = run(scaffolds.SagaStyle(refresh_every=10), engine, llm)

    refresh_turns = [c["turn"] for c in out["llm_calls"] if c["tag"] == "goal_refresh"]
    assert refresh_turns == [1, 11, 21]
    # 25 decide calls + 3 refreshes
    assert len(llm.calls) == 28
    # non-refresh decisions are one call and still carry the cached goal
    assert out["decisions"][1]["tokens"]["n_calls"] == 1
    assert "GOAL: expand north" in llm.calls[3]


def test_reflexion_injects_lessons_only_from_the_second_episode():
    memory = {}

    def replies(prompt):
        if "Write at most three short lessons" in prompt:
            return "- Build settlers earlier.\n- Do not idle workers."
        return PICK

    llm1 = mock_llm(replies)
    out1 = run(scaffolds.ReflexionStyle(), FakeEngine([1, 2, 3]), llm1, memory=memory)
    assert "this is your first game on this start" in llm1.calls[0]
    # 3 decide calls + one end-of-game reflection
    assert len(llm1.calls) == 4
    assert "finished with score 137.0" in llm1.calls[-1]
    # the reflection call is not attributed to any decision turn
    assert [c["tag"] for c in out1["llm_calls"]][-1] == "reflect"
    assert out1["decisions"][-1]["tokens"]["n_calls"] == 1
    assert memory["rem120-s01"] == [
        "- Build settlers earlier.",
        "- Do not idle workers.",
    ]

    llm2 = mock_llm(replies)
    run(scaffolds.ReflexionStyle(), FakeEngine([1, 2, 3]), llm2, memory=memory)
    assert "- Build settlers earlier." in llm2.calls[0]
    assert "first game on this start" not in llm2.calls[0]


def test_reflexion_lessons_do_not_cross_starts():
    memory = {"other-start": ["- irrelevant lesson"]}
    llm = mock_llm(
        lambda p: "- lesson" if "lessons" in p and "Write at most" in p else PICK
    )
    run(scaffolds.ReflexionStyle(), FakeEngine([1]), llm, memory=memory)
    assert "irrelevant lesson" not in llm.calls[0]
    assert memory["other-start"] == ["- irrelevant lesson"]
    assert memory["rem120-s01"] == ["- lesson"]


def test_reflexion_keeps_an_unformatted_lesson():
    memory = {}
    llm = mock_llm(lambda p: "Settle   more\ncities." if "Write at most" in p else PICK)
    run(scaffolds.ReflexionStyle(), FakeEngine([1]), llm, memory=memory)
    assert memory["rem120-s01"] == ["- Settle more cities."]


# ---------------------------------------------------------------------------
# 3. token bookkeeping
# ---------------------------------------------------------------------------


def test_token_account_reconciles_with_known_mock_counts():
    engine = FakeEngine([1, 2, 3])
    llm = mock_llm(PICK, prompt_tokens=1234, completion_tokens=17)
    out = run(scaffolds.MastabaStyle(), engine, llm)

    n_calls = 3 * 4  # 3 decisions x (3 advisors + president)
    assert out["tokens"] == {
        "n_calls": n_calls,
        "prompt_tokens": 1234 * n_calls,
        "completion_tokens": 17 * n_calls,
        "total_tokens": (1234 + 17) * n_calls,
    }
    # episode totals == sum of the per-decision totals (no unbooked calls)
    per_decision = out["decisions"]
    assert sum(d["tokens"]["n_calls"] for d in per_decision) == n_calls
    assert (
        sum(d["tokens"]["total_tokens"] for d in per_decision)
        == out["tokens"]["total_tokens"]
    )
    for d in per_decision:
        assert d["tokens"] == {
            "n_calls": 4,
            "prompt_tokens": 1234 * 4,
            "completion_tokens": 17 * 4,
            "total_tokens": (1234 + 17) * 4,
        }


def test_end_of_game_calls_are_in_the_episode_total_not_any_decision():
    engine = FakeEngine([1, 2])
    llm = mock_llm(
        lambda p: "- lesson" if "Write at most" in p else PICK,
        prompt_tokens=10,
        completion_tokens=3,
    )
    out = run(scaffolds.ReflexionStyle(), engine, llm, memory={})

    assert out["tokens"]["n_calls"] == 3  # 2 decisions + 1 reflection
    assert sum(d["tokens"]["n_calls"] for d in out["decisions"]) == 2
    assert out["tokens"]["total_tokens"] == 39


def test_scaffold_cost_ordering():
    # the multi-call scaffolds cost more than a single call per decision
    per_scaffold = {}
    for scaffold in (
        scaffolds.BaseLangStyle(),
        scaffolds.MastabaStyle(),
        scaffolds.SagaStyle(refresh_every=10),
    ):
        llm = mock_llm(
            lambda p: "GOAL: x" if "SITUATION:" in p else PICK,
            prompt_tokens=500,
            completion_tokens=50,
        )
        out = run(scaffold, FakeEngine(list(range(1, 21))), llm)
        per_scaffold[out["scaffold"]] = out["tokens"]["total_tokens"]

    assert per_scaffold["baselang"] < per_scaffold["saga"] < per_scaffold["mastaba"]


# ---------------------------------------------------------------------------
# 4. answer extraction
# ---------------------------------------------------------------------------


def test_last_json_object_is_extracted_from_reasoning_text():
    assert (
        scaffolds._last_json_object('I think {this} then\nfinally {"city": ["a"]}')
        == '{"city": ["a"]}'
    )
    # nested object: the whole outer block, not the inner tail
    assert scaffolds._last_json_object('x {"a": {"b": 1}}') == '{"a": {"b": 1}}'
    # no balanced block -> unchanged, so the strict parser reports the error
    assert scaffolds._last_json_object("no json here") == "no json here"


def test_extraction_does_not_repair_a_broken_bundle():
    engine = FakeEngine([1])
    # a well-formed object that violates the menu contract stays invalid
    llm = mock_llm('reasoning {"research": ["city:10/build-unit:settlers"]}')
    out = run(scaffolds.BaseLangStyle(), engine, llm)
    assert "unknown domains" in out["decisions"][0]["error"]


# ---------------------------------------------------------------------------
# 5. the task contract is the RL policy's system prompt
# ---------------------------------------------------------------------------


def test_task_contract_matches_the_rl_policy():
    from civmarsh.baselines.prompts import TASK_CONTRACT
    from civmarsh.env.prompts import SYSTEM_PROMPT

    assert TASK_CONTRACT == SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 6. reasoning persistence
# ---------------------------------------------------------------------------


def test_raw_reasoning_text_is_persisted_per_call():
    # the engine only ever sees the extracted JSON object, so the scaffold's
    # reasoning has to be logged on the call rows
    engine = FakeEngine([1, 2])
    llm = mock_llm(f"I should expand north because the coast is safe.\n{PICK}")
    out = run(scaffolds.BaseLangStyle(), engine, llm)

    assert out["decisions"][0]["reply_head"] == PICK  # engine-facing text
    texts = [c["text"] for c in out["llm_calls"]]
    assert len(texts) == 2
    assert all("expand north because the coast is safe" in t for t in texts)
    # a decision's reasoning is its turn's rows; every row is attributed
    assert [c["turn"] for c in out["llm_calls"]] == [1, 2]

    # Mastaba: the advisors' reasoning is kept too, tagged per call
    llm = mock_llm(lambda p: f"advice text\n{PICK}")
    out = run(scaffolds.MastabaStyle(), FakeEngine([1]), llm)
    rows = [(c["tag"], c["text"]) for c in out["llm_calls"]]
    assert [t for t, _ in rows] == [
        "advisor:military",
        "advisor:economy",
        "advisor:expansion",
        "president",
    ]
    assert all("advice text" in txt for _, txt in rows)


def test_persisted_text_is_capped():
    engine = FakeEngine([1])
    llm = mock_llm("x" * (driver.RESPONSE_TEXT_CAP + 500) + PICK)
    out = run(scaffolds.BaseLangStyle(), engine, llm)
    text = out["llm_calls"][0]["text"]
    assert len(text) < driver.RESPONSE_TEXT_CAP + 100
    assert text.endswith("chars total]")


def test_direct_arm_is_one_call_and_no_answer_extraction():
    # the floor arm mirrors the RL rollout: one call, no reasoning ask, and
    # the model's raw text goes to the strict parser (no JSON isolation)
    engine = FakeEngine([1, 2])
    llm = mock_llm(f"here you go {PICK}")
    out = run(scaffolds.DirectStyle(), engine, llm)

    assert out["scaffold"] == "direct"
    assert len(llm.calls) == 2
    assert "Think step by step" not in llm.calls[0]
    assert "Freeciv (civ2civ3 ruleset) as player 'Alice'" in llm.calls[0]
    assert MENU_HEADER in llm.calls[0]
    # unextracted prose reaches the parser and counts as an invalid decision,
    # exactly as it would in the RL rollout
    assert all(d["error"] for d in out["decisions"])
    assert out["decisions"][0]["reply_head"] == f"here you go {PICK}"


# ---------------------------------------------------------------------------
# 7. readout
# ---------------------------------------------------------------------------


def _at(pid, sample_idx, rec):
    return {**rec, "position_id": pid, "sample_idx": sample_idx}


def test_summarize_arm_counts_deaths_and_excludes_stalls_and_errors():
    llm = mock_llm(PICK)
    base = scaffolds.BaseLangStyle()
    dead = run(
        base,
        FakeEngine(
            [1, 2],
            score_end=0.0,
            eliminated=True,
            control_lost="GAME: lost control of 'Alice': player is out of the game",
        ),
        llm,
    )
    alive_a = run(base, FakeEngine([1, 2], score_end=20.0), llm)
    alive_b = run(base, FakeEngine([1, 2, 3], score_end=40.0), llm)
    stall = run(
        base,
        FakeEngine(
            [1],
            score_end=0.0,
            eliminated=True,
            control_lost="INFRA: lost control of 'Bob': no phase within 240s",
        ),
        llm,
    )
    unknown = run(
        base,
        FakeEngine(
            [1],
            score_end=0.0,
            eliminated=True,
            control_lost="UNKNOWN: lost control of 'Bob': no readable save",
        ),
        llm,
    )
    broken = run(base, FakeEngine([1], error="port race"), llm)
    records = [
        _at("pos_a", 0, dead),
        _at("pos_a", 1, alive_a),
        _at("pos_b", 0, alive_b),
        _at("pos_b", 1, stall),
        _at("pos_b", 2, unknown),
        _at("p3", 0, broken),
    ]

    s = driver.summarize_arm(records)
    assert s["n_episodes"] == 6
    assert s["n_valid"] == 3
    assert s["n_excluded"] == 3
    assert s["n_positions"] == 2
    # mean over positions of the per-position mean: (10 + 40) / 2
    assert s["position_means"] == {"pos_a": 10.0, "pos_b": 40.0}
    assert s["score"] == pytest.approx(25.0)
    # the GAME death counts as an elimination; the stalls do not
    assert s["eliminated_rate"] == pytest.approx(1 / 3)
    assert s["survivor_score"] == pytest.approx(30.0)
    assert s["control_lost_counts"] == {
        "GAME: lost control of 'X': player is out of the game": 1
    }
    assert s["n_decisions"] == 7
    assert s["decisions_per_game"] == pytest.approx(7 / 3)

    # restricting to a position subset (the excluded-games rule)
    s2 = driver.summarize_arm(records, positions=["pos_b"])
    assert s2["n_episodes"] == 3 and s2["n_valid"] == 1
    assert s2["score"] == pytest.approx(40.0)
    assert s2["eliminated_rate"] == 0.0


def test_stall_and_validity_predicates():
    assert driver.is_stall({"control_lost": "INFRA: seat stalled"})
    assert driver.is_stall({"control_lost": "UNKNOWN: no save"})
    assert not driver.is_stall({"control_lost": "GAME: dead"})
    assert not driver.is_stall({"control_lost": None})
    assert driver.is_valid({"score_end": 0, "control_lost": "GAME: dead"})
    assert not driver.is_valid({"score_end": 0, "control_lost": "INFRA: stall"})
    assert not driver.is_valid({"error": "port race"})
    assert not driver.is_valid({"score_end": None})


def test_final_records_prefers_a_finished_episode_over_its_failed_attempt():
    recs = [
        {"position_id": "pos_a", "sample_idx": 0, "error": "timeout"},
        {"position_id": "pos_a", "sample_idx": 0, "score_end": 5},
        {"position_id": "pos_a", "sample_idx": 0, "score_end": 9},
        {"position_id": "pos_a", "sample_idx": 1, "score_end": 3},
        {"position_id": "pos_a", "sample_idx": 2, "error": "a"},
        {"position_id": "pos_a", "sample_idx": 2, "error": "b"},
    ]
    kept = {(r["position_id"], r["sample_idx"]): r for r in driver.final_records(recs)}
    assert len(kept) == 3
    assert kept[("pos_a", 0)]["score_end"] == 5  # a finished episode is never replaced
    assert kept[("pos_a", 1)]["score_end"] == 3
    assert kept[("pos_a", 2)]["error"] == "b"  # the latest failure when none finished


def test_summarize_arm_reports_invalid_rate_and_token_cost():
    llm = mock_llm(
        lambda p: PICK if "turn=1" in p else "not json",
        prompt_tokens=100,
        completion_tokens=10,
    )
    rec = run(scaffolds.BaseLangStyle(), FakeEngine([1, 2, 3, 4]), llm)
    s = driver.summarize_arm([_at("pos_a", 0, rec)])

    assert s["n_decisions"] == 4
    assert s["invalid_decisions"] == 3  # turns 2..4 replied "not json"
    assert s["invalid_rate"] == pytest.approx(0.75)
    assert s["eliminated_rate"] == 0.0
    assert s["control_lost_counts"] == {}
    assert s["tokens_total"]["total_tokens"] == 4 * 110
    assert s["tokens_per_decision"] == pytest.approx(110)
    assert s["calls_per_decision"] == pytest.approx(1.0)


def test_summarize_arm_with_no_valid_episode():
    s = driver.summarize_arm([{"position_id": "pos_a", "sample_idx": 0, "error": "x"}])
    assert s["n_valid"] == 0 and s["score"] is None
    assert s["eliminated_rate"] is None and s["invalid_rate"] is None
