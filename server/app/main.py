from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .advisors import decide
from .game import Game

app = FastAPI(title="Dungeon Escape", version="0.1.0")
GAMES: dict[str, tuple[Game, str, dict]] = {}
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

class NewGame(BaseModel):
    seed: int | None = None
    pilot: Literal["manual", "rule", "random", "laya", "jev"] = "manual"

class MoveRequest(BaseModel):
    action: str

class AutoRequest(BaseModel):
    max_turns: int = Field(default=20, ge=1, le=80)

class BenchmarkRequest(BaseModel):
    pilot: Literal["rule", "random", "laya", "jev"] = "rule"
    episodes: int = Field(default=10, ge=1, le=100)
    start_seed: int = 2026

@app.get("/api/health")
async def health():
    return {"status": "ok", "pilots": {"laya": bool(os.getenv("LAYA_CHECKPOINT")), "jev": bool(os.getenv("TYPESAFE_API_KEY"))}}

@app.post("/api/games")
async def create_game(payload: NewGame):
    seed = payload.seed if payload.seed is not None else uuid.uuid4().int % 2_000_000_000
    game = Game.new(seed)
    game_id = str(uuid.uuid4())
    GAMES[game_id] = (game, payload.pilot, {})
    return {"game_id": game_id, "pilot": payload.pilot, "state": game.to_dict()}

def get_record(game_id: str):
    if game_id not in GAMES:
        raise HTTPException(404, "Game not found")
    return GAMES[game_id]

@app.get("/api/games/{game_id}")
async def read_game(game_id: str):
    game, pilot, last = get_record(game_id)
    return {"game_id": game_id, "pilot": pilot, "state": game.to_dict(), "decision": last}

@app.post("/api/games/{game_id}/move")
async def move(game_id: str, payload: MoveRequest):
    game, pilot, _ = get_record(game_id)
    try:
        state = game.step(payload.action)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    GAMES[game_id] = (game, pilot, {"choice": payload.action, "source": "manual"})
    return {"state": state, "decision": GAMES[game_id][2]}

@app.post("/api/games/{game_id}/auto")
async def auto_play(game_id: str, payload: AutoRequest):
    game, pilot, last = get_record(game_id)
    if pilot == "manual":
        raise HTTPException(400, "Select a pilot before auto play")
    trace = []
    try:
        for _ in range(payload.max_turns):
            if game.status != "playing":
                break
            action, last = await decide(game, pilot, game.seed)
            game.step(action)
            trace.append({"action": action, "state": game.to_dict(), "decision": last})
    except (RuntimeError, ValueError) as exc:
        GAMES[game_id] = (game, pilot, last)
        raise HTTPException(503, str(exc)) from exc
    GAMES[game_id] = (game, pilot, last)
    return {"state": game.to_dict(), "decision": last, "trace": trace}

async def run_episode(pilot: str, seed: int):
    game = Game.new(seed)
    for _ in range(80):
        if game.status != "playing":
            break
        action, _ = await decide(game, pilot, seed)
        game.step(action)
    return game

@app.post("/api/benchmark")
async def benchmark(payload: BenchmarkRequest):
    if payload.pilot == "jev" and payload.episodes > 3:
        raise HTTPException(400, "Jev benchmark is limited to 3 episodes to control hosted API calls.")
    results = []
    try:
        for i in range(payload.episodes):
            game = await run_episode(payload.pilot, payload.start_seed + i)
            results.append({"seed": game.seed, "status": game.status, "reward": round(game.reward, 2), "turns": game.turn, "hp": game.hp})
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    escaped = sum(r["status"] == "escaped" for r in results)
    return {"pilot": payload.pilot, "episodes": len(results), "escape_rate": escaped / len(results),
        "mean_reward": round(sum(r["reward"] for r in results) / len(results), 2),
        "mean_turns": round(sum(r["turns"] for r in results) / len(results), 2),
        "deaths": sum(r["status"] == "dead" for r in results), "results": results}

if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

@app.get("/")
async def home():
    return FileResponse(WEB_DIR / "index.html")
