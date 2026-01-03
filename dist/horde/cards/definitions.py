from __future__ import annotations
from typing import List
from .abilities import berserk, fear, regen, shield
from .types import Card, Rarity


# Starter pool: 10 cards total, longer fights, and 6 with abilities.
CARD_POOL: List[Card] = [
    # Commons: mostly stats, one light ability
    Card("Venolux", Rarity.COMMON, attack=7, health=24),
    Card("Mossit", Rarity.COMMON, attack=6, health=26),
    Card("Bronti", Rarity.COMMON, attack=8, health=22),
    Card("Skylux", Rarity.COMMON, attack=7, health=25),
    Card("Duskip", Rarity.COMMON, attack=6, health=23,
         ability=fear(1)),

    # Rares: single ability, slightly toned stats
    Card("Aquava", Rarity.RARE, attack=6, health=23,
         ability=regen(4)),
    Card("Florune", Rarity.RARE, attack=5, health=24,
         ability=regen(3)),
    Card("Aegor", Rarity.RARE, attack=6, health=22,
         ability=shield(10)),

    # Epics:
    Card("Ravvok", Rarity.EPIC, attack=8, health=26,
         ability=berserk(5)),
    Card("Nocturn", Rarity.EPIC, attack=7, health=24,
         ability=fear(2)),
]
