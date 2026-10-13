# Laya reward-guided fine-tuning

The generator creates Laya JSONL cases from seeded episodes. At each sampled state it scores all legal actions by running short rollouts with the built-in rule pilot, then converts the scores to a soft probability distribution (`gold`). This creates a reproducible offline decision dataset; it is not online policy-gradient training directly against the full delayed mission reward.

```bash
cd server
pip install -e ".[train]"
python ../training/generate_data.py --episodes 300 --out ../training/data/train.jsonl --seed 2026
laya-train --data ../training/data/train.jsonl --base english --out ../training/checkpoints/laya-dungeon
```

Use separate seed ranges for training and evaluation. Never report training accuracy as generalization. The upstream Laya project supports training from JSONL `state/questions/gold` records and loading the resulting checkpoint with `laya.load`.
