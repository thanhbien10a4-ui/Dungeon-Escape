from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from starlette.concurrency import run_in_threadpool

from .game import Game, state_text

ACTION_DESCRIPTIONS = {
    "north": "Move one tile north.", "south": "Move one tile south.",
    "west": "Move one tile west.", "east": "Move one tile east.",
    "attack": "Spend one ammo to attack an adjacent enemy.", "heal": "Use one medkit to restore health.",
    "collect_core": "Secure the reactor core at the objective.",
    "collect_loot": "Collect supplies on this tile.", "extract": "Finish the mission at the extraction tile.",
}

def rule_action(game: Game) -> str:
    legal = game.legal_actions()
    if "extract" in legal:
        return "extract"
    if "collect_core" in legal:
        return "collect_core"
    adjacent = [e for e in game.enemies if abs(e.x-game.x) + abs(e.y-game.y) == 1]
    if adjacent and "attack" in legal:
        return "attack"
    if game.hp <= 2 and "heal" in legal:
        return "heal"
    if "collect_loot" in legal:
        return "collect_loot"
    target = (1, 6) if game.has_core else (6, 1)
    candidates = [a for a in legal if a in {"north", "south", "east", "west"}]
    if not candidates:
        return legal[0]
    def dist_after(action: str) -> int:
        dxdy = {"north": (0,-1), "south": (0,1), "west": (-1,0), "east": (1,0)}[action]
        return abs(game.x + dxdy[0] - target[0]) + abs(game.y + dxdy[1] - target[1])
    return min(candidates, key=dist_after)

def random_action(game: Game, seed: int) -> str:
    from random import Random
    return Random(seed + game.turn * 997).choice(game.legal_actions())

@lru_cache(maxsize=2)
def _load_laya(checkpoint: str):
    try:
        import laya
    except ImportError as exc:
        raise RuntimeError("Install the Laya training/inference extra and set LAYA_CHECKPOINT.") from exc
    return laya.load(checkpoint)

def _questions() -> dict[str, Any]:
    # Laya training expects a stable question schema across examples. The
    # current legal set is included in state_text and enforced after prediction.
    return {"action": {"type": "choice", "instructions": "Choose only from the legal actions listed in the state.",
        "criteria": ACTION_DESCRIPTIONS}}

def _legal_laya_choice(answer: dict[str, Any], legal_actions: list[str], fallback_action: str) -> tuple[str, dict[str, Any]]:
    raw_choice = answer.get("choice")
    details = {"raw_choice": raw_choice, "probabilities": answer.get("probabilities")}
    if raw_choice in legal_actions:
        return raw_choice, details

    # Constrained decoding: if the model's top choice is illegal, keep its
    # learned ranking while masking actions that the game cannot execute.
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict):
        scored = []
        for action in legal_actions:
            try:
                score = float(probabilities[action])
            except (KeyError, TypeError, ValueError):
                continue
            if score == score and abs(score) != float("inf"):
                scored.append((score, action))
        if scored:
            choice = max(scored)[1]
            details["constrained"] = True
            return choice, details

    # Some Laya backends omit probabilities. Keep the game playable and make
    # the intervention visible to the benchmark/UI.
    return fallback_action, {**details, "constrained": True, "fallback": "rule"}

def choose_laya(game: Game) -> tuple[str, dict[str, Any]]:
    checkpoint = os.getenv("LAYA_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError("LAYA_CHECKPOINT is not set. Train or download a local Laya checkpoint first.")
    legal_actions = game.legal_actions()
    if not legal_actions:
        raise RuntimeError("No legal actions are available for the current game state.")
    result = _load_laya(checkpoint).predict(state_text(game), _questions())
    answers = result.get("answers", {}) if isinstance(result, dict) else {}
    answer = answers.get("action", {}) if isinstance(answers, dict) else {}
    if not isinstance(answer, dict):
        answer = {}
    choice, details = _legal_laya_choice(answer, legal_actions, rule_action(game))
    return choice, {"choice": choice, "confidence": answer.get("confidence"), **details}

async def choose_jev(game: Game) -> tuple[str, dict[str, Any]]:
    import httpx

    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        raise RuntimeError("TYPESAFE_API_KEY is not set; Jev mode requires a server-side API key.")
    legal = game.legal_actions()
    criteria = {action: ACTION_DESCRIPTIONS[action] for action in legal}
    payload = {"model": os.getenv("JEV_MODEL", "jev-latest"), "state": state_text(game),
        "questions": {"action": {"type": "choice", "instructions": "Choose the best legal action for this turn, prioritizing mission completion and survival.", "criteria": criteria}}}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post("https://api.typesafe.ai/v1/systemone", json=payload,
                headers={"Authorization": f"Bearer {api_key}"})
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Jev request failed ({type(exc).__name__}). Check the API key, quota, and network.") from exc
    answer = data.get("answers", {}).get("action", {})
    choice = answer.get("choice")
    if choice not in legal:
        raise RuntimeError(f"Jev selected an illegal action: {choice!r}")
    return choice, {"choice": choice, "confidence": answer.get("confidence"), "probabilities": answer.get("probabilities"), "model": data.get("model")}

async def decide(game: Game, mode: str, seed: int = 0) -> tuple[str, dict[str, Any]]:
    if mode == "rule":
        action = rule_action(game)
        return action, {"choice": action, "pilot": "rule"}
    if mode == "random":
        action = random_action(game, seed)
        return action, {"choice": action, "pilot": "random"}
    if mode == "laya":
        # Laya inference runs on CPU and is synchronous; keep it off FastAPI's event loop.
        return await run_in_threadpool(choose_laya, game)
    if mode == "jev":
        return await choose_jev(game)
    raise ValueError(f"Unknown pilot: {mode}")
