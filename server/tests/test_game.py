from app.advisors import _legal_laya_choice, _questions, random_action, rule_action
from app.game import Game


def test_seed_reproduces_map():
    assert Game.new(17).to_dict() == Game.new(17).to_dict()


def test_core_then_extract_wins():
    game = Game.new(7)
    game.x, game.y, game.has_core = 1, 6, True
    assert "extract" in game.legal_actions()
    game.step("extract")
    assert game.status == "escaped"
    assert game.reward >= 99


def test_invalid_action_rejected():
    game = Game.new(1)
    try:
        game.step("fly")
    except ValueError:
        pass
    else:
        raise AssertionError("illegal action was accepted")


def test_collect_core_unlocks_extraction_only_after_returning():
    game = Game.new(11)
    game.x, game.y, game.has_core = 1, 6, False
    assert "extract" not in game.legal_actions()


def test_rule_pilot_outperforms_random_on_fixed_seed_suite():
    wins = {"rule": 0, "random": 0}
    for seed in range(20):
        for pilot in wins:
            game = Game.new(seed)
            while game.status == "playing" and game.turn < 80:
                action = rule_action(game) if pilot == "rule" else random_action(game, seed)
                game.step(action)
            wins[pilot] += game.status == "escaped"
    assert wins["rule"] >= 15
    assert wins["rule"] > wins["random"]


def test_laya_question_schema_is_stable():
    questions = _questions()
    assert set(questions["action"]["criteria"]) == {
        "north", "south", "east", "west", "attack", "heal",
        "collect_core", "collect_loot", "extract",
    }


def test_laya_illegal_choice_is_masked_to_highest_scoring_legal_action():
    choice, details = _legal_laya_choice(
        {"choice": "attack", "probabilities": {"attack": 0.8, "east": 0.2}},
        ["east", "north"],
        "north",
    )
    assert choice == "east"
    assert details["constrained"] is True


def test_laya_without_probabilities_uses_visible_rule_fallback():
    choice, details = _legal_laya_choice({"choice": "attack"}, ["east", "north"], "north")
    assert choice == "north"
    assert details["fallback"] == "rule"

