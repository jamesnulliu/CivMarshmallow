"""Episode driver helpers that need neither a server nor slime."""

from civmarsh.env.menu import ActionMenu, BundleValidationError, MenuConfig
from civmarsh.train.episode import ENV, parse_elim_class, parse_reply, resolve_reply


def test_elim_class_prefixes():
    assert parse_elim_class("GAME: player 'Alice' is out of the game") == "GAME"
    assert parse_elim_class("INFRA: no phase within 240s") == "INFRA"
    assert parse_elim_class("UNKNOWN: unreadable save") == "UNKNOWN"
    # no prefix, or an empty message, is never a proven death
    assert parse_elim_class("no phase within 240s") == "UNKNOWN"
    assert parse_elim_class(None) == "UNKNOWN"


def test_reply_resolution_is_a_logged_noop_when_invalid():
    menu = ActionMenu(candidates=(), skipped=(), config=MenuConfig())
    reply = parse_reply("not json", menu)
    assert isinstance(reply, BundleValidationError)
    orders, pick, error = resolve_reply(reply, menu)
    assert orders == [] and pick is None and "invalid JSON" in error
    assert resolve_reply(None, menu) == ([], None, "no reply")

    passed = parse_reply("{}", menu)  # an empty object is a legal pass
    orders, pick, error = resolve_reply(passed, menu)
    assert orders == [] and error is None and pick["keys"] == []


def test_policy_env_is_fog_of_war():
    assert ENV.observations["visibility"] == "player_visible"
    assert ENV.observations["players"] == "self"
