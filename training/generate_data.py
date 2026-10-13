from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from app.advisors import rule_action
from app.game import Game, state_text

DESCRIPTIONS = {
    "north": "Move one tile north.", "south": "Move one tile south.", "west": "Move one tile west.", "east": "Move one tile east.",
    "attack": "Spend one ammo to attack an adjacent enemy.", "heal": "Use one medkit to restore health.",
    "collect_core": "Secure the reactor core at the objective.", "collect_loot": "Collect supplies on this tile.", "extract": "Finish the mission at the extraction tile.",
}

def rollout_value(start: Game, first_action: str, horizon: int = 8) -> float:
    g = Game.new(start.seed)
    g.__dict__.update({**start.__dict__, "enemies": [type(e)(e.x, e.y, e.hp) for e in start.enemies], "loot": dict(start.loot), "events": []})
    g.step(first_action)
    for _ in range(horizon - 1):
        if g.status != "playing":
            break
        g.step(rule_action(g))
    objective = (1, 6) if g.has_core else (6, 1)
    distance = abs(g.x - objective[0]) + abs(g.y - objective[1])
    survival = 3 * g.hp + 1.5 * g.ammo + 2 * g.medkits - 8 * len(g.enemies)
    progress = 12 if g.has_core else -1.5 * distance
    terminal = 40 if g.status == "escaped" else (-45 if g.status == "dead" else 0)
    return g.reward + survival + progress + terminal

def to_case(game: Game) -> dict:
    actions = game.legal_actions()
    values = [rollout_value(game, action) for action in actions]
    peak = max(values)
    weights = [math.exp(max(-20, (v - peak) / 5.0)) for v in values]
    z = sum(weights)
    probabilities = {action: weight / z for action, weight in zip(actions, weights)}
    return {
        "state": state_text(game),
        # Keep the choice schema identical in every row. Legal actions remain
        # state-specific in the state text and in the gold distribution.
        "questions": {"action": {"type": "choice", "instructions": "Choose only from the legal actions listed in the state. Prioritize survival and mission completion.", "criteria": DESCRIPTIONS}},
        "gold": {"action": {"probabilities": probabilities}},
    }

def make_dataset(episodes: int, seed: int, out: Path, stride: int = 2) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for episode in range(episodes):
            game = Game.new(seed + episode)
            while game.status == "playing" and game.turn < 50:
                if game.turn % stride == 0:
                    f.write(json.dumps(to_case(game), ensure_ascii=False) + "\n")
                    count += 1
                game.step(rule_action(game))
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    n = make_dataset(args.episodes, args.seed, args.out, args.stride)
    print(f"Wrote {n} reward-ranked Laya decisions to {args.out}")

if __name__ == "__main__":
    main()
