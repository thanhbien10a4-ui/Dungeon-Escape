# Dungeon Escape

A small turn-based tactical game for studying decision models. The player must collect a reactor core, survive pursuing enemies, and reach the extraction tile.

The project separates three things that are often mixed together:

- The game engine owns legal actions, transitions, rewards, and evaluation.
- The rule pilot is a transparent baseline.
- Jev is a hosted, frozen inference baseline. It does not learn from a game by itself.
- Laya can be fine-tuned offline on game decisions. Training examples are generated from reward-ranked action rollouts and converted to Laya's typed-decision format.

## Run locally

```bash
cd server
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. The FastAPI server serves the browser game and API from one origin.

## Play

Create a run, choose a pilot (`manual`, `rule`, `random`, `laya`, or `jev`), then step one action at a time or press **Auto play**. Laya needs a local checkpoint; Jev needs a server-side `TYPESAFE_API_KEY`.

For Jev, copy the root `.env.example` to `.env` and set the key there. The server loads that file; never put the key in the browser code.

```bash
# Optional hosted Jev baseline; keep the key on the server.
set TYPESAFE_API_KEY=your_key       # PowerShell: $env:TYPESAFE_API_KEY="your_key"
```

For local Laya inference, fine-tune a checkpoint first and set `LAYA_CHECKPOINT` to its directory. See `training/README.md`.

## Train Laya on game outcomes

The training script runs the deterministic rule pilot across seeded maps, tests every legal action with short reward-based rollouts, turns the resulting action-value distribution into Laya's `gold` targets, and invokes the upstream `laya-train` command. This is offline reward-guided fine-tuning, not online learning during a single game. Keep map seeds split between training and evaluation.

```bash
pip install -e ".[train]"
python ../training/generate_data.py --episodes 300 --out ../training/data/train.jsonl --seed 2026
laya-train --data ../training/data/train.jsonl --base english --out ../training/checkpoints/laya-dungeon
```

The generator can run without a GPU. Fine-tuning needs the upstream Laya training dependencies; GPU training is recommended. Laya's [fine-tuning guide](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md) documents `laya-train`, JSONL `state/questions/gold` records, and checkpoint loading.

## Benchmark

```bash
cd server
pytest
```

The benchmark endpoint reports escape rate, mean reward, mean turns, and deaths across reproducible seeds. Treat Jev/Laya scores as experiments; compare them with the rule and random pilots on identical held-out maps before making claims.

## API

- `POST /api/games` — create a seeded game
- `GET /api/games/{game_id}` — read current state
- `POST /api/games/{game_id}/move` — apply one action
- `POST /api/games/{game_id}/auto` — run the selected pilot for a bounded number of turns
- `POST /api/benchmark` — compare pilots over fixed seeds
- `GET /api/health` — health check

## Limitations

- This is a compact educational environment, not a production game or a financial/real-world agent.
- Reward-ranked rollouts provide training targets; they do not guarantee a stronger fine-tuned model.
- Hosted Jev requests incur account usage and network latency. The application never sends an API key to the browser.
- A small benchmark can be noisy. Use multiple seeds and report all results, including losses.

## License

MIT for this project. Laya, Jev, and their model weights/services retain their own licenses and terms.
