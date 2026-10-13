from __future__ import annotations

from dataclasses import asdict, dataclass, field
from random import Random
from typing import Any

SIZE = 8
MOVES = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}


@dataclass
class Enemy:
    x: int
    y: int
    hp: int = 1


@dataclass
class Game:
    seed: int
    x: int = 1
    y: int = 1
    hp: int = 5
    ammo: int = 5
    medkits: int = 2
    has_core: bool = False
    turn: int = 0
    reward: float = 0.0
    status: str = "playing"
    enemies: list[Enemy] = field(default_factory=list)
    loot: dict[str, tuple[int, int]] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    @classmethod
    def new(cls, seed: int) -> "Game":
        rng = Random(seed)
        reserved = {(1, 1), (6, 1), (6, 6), (1, 6)}
        candidates = [(x, y) for x in range(2, 7) for y in range(1, 7) if (x, y) not in reserved]
        rng.shuffle(candidates)
        enemies = [Enemy(*candidates[i]) for i in range(3)]
        loot = {"ammo": candidates[3], "medkit": candidates[4]}
        return cls(seed=seed, enemies=enemies, loot=loot)

    @staticmethod
    def inside(x: int, y: int) -> bool:
        return 0 <= x < SIZE and 0 <= y < SIZE

    def legal_actions(self) -> list[str]:
        if self.status != "playing":
            return []
        enemy_tiles = {(e.x, e.y) for e in self.enemies}
        actions = [name for name, (dx, dy) in MOVES.items()
                   if self.inside(self.x + dx, self.y + dy) and (self.x + dx, self.y + dy) not in enemy_tiles]
        if self.ammo > 0 and any(abs(e.x - self.x) + abs(e.y - self.y) == 1 for e in self.enemies):
            actions.append("attack")
        if self.medkits > 0 and self.hp <= 3:
            actions.append("heal")
        if (self.x, self.y) == (6, 1) and not self.has_core:
            actions.append("collect_core")
        if self.loot.get("ammo") == (self.x, self.y) or self.loot.get("medkit") == (self.x, self.y):
            actions.append("collect_loot")
        if self.has_core and (self.x, self.y) == (1, 6):
            actions.append("extract")
        return actions

    def step(self, action: str) -> dict[str, Any]:
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action: {action}")
        self.events = []
        self.turn += 1
        self.reward -= 0.2

        if action in MOVES:
            dx, dy = MOVES[action]
            self.x, self.y = self.x + dx, self.y + dy
            self.events.append(f"Moved {action}.")
        elif action == "attack":
            self.ammo -= 1
            target = min(self.enemies, key=lambda e: abs(e.x - self.x) + abs(e.y - self.y))
            target.hp -= 1
            if target.hp <= 0:
                self.enemies.remove(target)
                self.reward += 3
                self.events.append("Enemy defeated.")
            else:
                self.events.append("Enemy hit.")
        elif action == "heal":
            self.medkits -= 1
            self.hp = min(5, self.hp + 2)
            self.events.append("Recovered 2 HP.")
        elif action == "collect_core":
            self.has_core = True
            self.reward += 10
            self.events.append("Reactor core secured. Reach extraction at (1, 6).")
        elif action == "collect_loot":
            if self.loot.get("ammo") == (self.x, self.y):
                self.ammo += 2
                del self.loot["ammo"]
                self.events.append("Collected 2 ammo.")
            elif self.loot.get("medkit") == (self.x, self.y):
                self.medkits += 1
                del self.loot["medkit"]
                self.events.append("Collected a medkit.")
            self.reward += 1
        elif action == "extract":
            self.status = "escaped"
            self.reward += 100
            self.events.append("Mission complete: escaped with the core.")

        if self.status == "playing":
            self._enemies_act()
        if self.status == "playing" and self.turn >= 80:
            self.status = "timeout"
            self.reward -= 10
            self.events.append("Turn limit reached.")
        return self.to_dict()

    def _enemies_act(self) -> None:
        occupied = {(e.x, e.y) for e in self.enemies}
        for enemy in list(self.enemies):
            distance = abs(enemy.x - self.x) + abs(enemy.y - self.y)
            if distance <= 1:
                self.hp -= 1
                self.events.append("An enemy hit you for 1 HP.")
                if self.hp <= 0:
                    self.status = "dead"
                    self.reward -= 100
                    self.events.append("You were defeated.")
                    return
                continue
            choices = [(enemy.x + dx, enemy.y + dy) for dx, dy in MOVES.values()]
            choices = [p for p in choices if self.inside(*p) and p != (self.x, self.y) and p not in occupied]
            if choices:
                nx, ny = min(choices, key=lambda p: abs(p[0] - self.x) + abs(p[1] - self.y))
                occupied.discard((enemy.x, enemy.y))
                enemy.x, enemy.y = nx, ny
                occupied.add((nx, ny))

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed, "size": SIZE, "player": {"x": self.x, "y": self.y, "hp": self.hp},
            "ammo": self.ammo, "medkits": self.medkits, "has_core": self.has_core,
            "core": {"x": 6, "y": 1}, "exit": {"x": 1, "y": 6}, "turn": self.turn,
            "reward": round(self.reward, 2), "status": self.status,
            "enemies": [asdict(e) for e in self.enemies],
            "loot": [{"kind": k, "x": p[0], "y": p[1]} for k, p in self.loot.items()],
            "legal_actions": self.legal_actions(), "events": self.events,
        }


def state_text(game: Game) -> str:
    s = game.to_dict()
    return (
        f"Turn {s['turn']}; player=({game.x},{game.y}), HP={game.hp}/5, ammo={game.ammo}, "
        f"medkits={game.medkits}, core={'secured' if game.has_core else 'not secured'}; "
        f"enemies={[(e.x, e.y, e.hp) for e in game.enemies]}; loot={s['loot']}; "
        f"core=(6,1), extraction=(1,6). Legal actions: {', '.join(s['legal_actions'])}. "
        "Goal: collect the core, then reach extraction alive. Each turn, enemies move toward you and attack when adjacent."
    )
