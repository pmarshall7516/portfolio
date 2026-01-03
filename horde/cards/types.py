from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict


class Rarity(str, Enum):
    COMMON = "COMMON"
    RARE = "RARE"
    EPIC = "EPIC"


class AbilityKey(str, Enum):
    REGEN = "REGEN"      # end of turn heal
    SHIELD = "SHIELD"    # start of round shield
    FEAR = "FEAR"        # reduce enemy attack after damage
    BERSERK = "BERSERK"  # gain attack once below 50% HP


@dataclass(frozen=True)
class AbilitySpec:
    key: AbilityKey
    params: Dict[str, int]


@dataclass(frozen=True)
class Card:
    name: str
    rarity: Rarity
    attack: int
    health: int
    ability: Optional[AbilitySpec] = None

    def short_ability(self) -> str:
        if self.ability is None:
            return "-"
        params = ",".join(f"{k}={v}" for k, v in self.ability.params.items())
        return f"{self.ability.key.value}({params})"
