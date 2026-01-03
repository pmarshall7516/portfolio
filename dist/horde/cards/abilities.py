from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

from .types import AbilityKey, AbilitySpec


@dataclass(frozen=True)
class AbilityDefinition:
    key: AbilityKey
    params: Tuple[str, ...]
    description: str

    def make(self, **values: int) -> AbilitySpec:
        extra = set(values) - set(self.params)
        if extra:
            raise ValueError(f"Unexpected params for {self.key.value}: {sorted(extra)}")
        missing = [p for p in self.params if p not in values]
        if missing:
            raise ValueError(f"Missing params for {self.key.value}: {missing}")
        return AbilitySpec(self.key, {p: int(values[p]) for p in self.params})


ABILITY_DEFS: Dict[AbilityKey, AbilityDefinition] = {
    AbilityKey.REGEN: AbilityDefinition(
        AbilityKey.REGEN,
        params=("H",),
        description="End of turn heal for +H HP.",
    ),
    AbilityKey.SHIELD: AbilityDefinition(
        AbilityKey.SHIELD,
        params=("S",),
        description="Start of round shield with S shield health.",
    ),
    AbilityKey.FEAR: AbilityDefinition(
        AbilityKey.FEAR,
        params=("W",),
        description="After damage, reduce enemy ATK by W for the round.",
    ),
    AbilityKey.BERSERK: AbilityDefinition(
        AbilityKey.BERSERK,
        params=("B",),
        description="First time HP drops below 50%, gain B ATK.",
    ),
}

ABILITY_ORDER = [
    AbilityKey.REGEN,
    AbilityKey.SHIELD,
    AbilityKey.FEAR,
    AbilityKey.BERSERK,
]

ABILITY_PARAM_KEYS = ["H", "S", "W", "B"]


def make_ability(key: AbilityKey, **values: int) -> AbilitySpec:
    return ABILITY_DEFS[key].make(**values)


def regen(h: int) -> AbilitySpec:
    return make_ability(AbilityKey.REGEN, H=h)


def shield(s: int) -> AbilitySpec:
    return make_ability(AbilityKey.SHIELD, S=s)


def fear(w: int) -> AbilitySpec:
    return make_ability(AbilityKey.FEAR, W=w)


def berserk(b: int) -> AbilitySpec:
    return make_ability(AbilityKey.BERSERK, B=b)
