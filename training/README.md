# Laya reward-guided fine-tuning

The generator creates Laya JSONL cases from seeded episodes. At each sampled state it scores all legal actions by running 12-turn rollouts with the built-in rule pilot, then converts the scores to a soft probability distribution (`gold`). The progress shaping rewards distance reduction toward the core before pickup and toward extraction after pickup. This creates a reproducible offline decision dataset; it is not online policy-gradient training directly against the full delayed mission reward.

Every training row uses the same nine-action question schema. The state text and gold distribution identify which actions are legal in that state. Inference masks illegal actions using Laya's returned probabilities, so an illegal top choice cannot break a game. If Laya returns no usable probabilities for an illegal choice, the API uses the Rule pilot and marks the fallback in the decision metadata. Retrain from newly generated data after updating the code; old checkpoints were trained with the conflicting state-dependent schema.

```bash
cd server
pip install -e ".[train]"
python ../training/generate_data.py --episodes 300 --out ../training/data/train.jsonl --seed 2026
laya-train --data ../training/data/train.jsonl --base english --out ../training/checkpoints/laya-dungeon-v2
```

Then point `LAYA_CHECKPOINT` in `server/.env` to `../training/checkpoints/laya-dungeon-v2` (or its absolute Windows path), restart the server, and benchmark Laya on the same seed range as Rule and Random. Keep the old checkpoint until the comparison is complete.

Use separate seed ranges for training and evaluation. Never report training accuracy as generalization. The upstream Laya project supports training from JSONL `state/questions/gold` records and loading the resulting checkpoint with `laya.load`.
