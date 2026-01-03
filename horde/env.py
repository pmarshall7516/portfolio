from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Iterable, Callable
import random

from cards.abilities import ABILITY_ORDER, ABILITY_PARAM_KEYS
from cards.types import Card, AbilityKey, Rarity
from cards.definitions import CARD_POOL

RARITY_ORDER = [Rarity.COMMON, Rarity.RARE, Rarity.EPIC]
CARD_FEATURE_SIZE = 2 + len(RARITY_ORDER) + len(ABILITY_ORDER) + len(ABILITY_PARAM_KEYS)
OBS_VECTOR_SIZE = 3 * CARD_FEATURE_SIZE + 2


@dataclass
class Fighter:
    card: Card
    max_hp: int
    hp: int
    atk: int

    # Status
    shield: int = 0
    berserk_used: bool = False

    def alive(self) -> bool:
        return self.hp > 0


@dataclass(frozen=True)
class AbilityHooks:
    on_start_round: Optional[Callable[["HordeEnv", Fighter, str], None]] = None
    on_start_turn: Optional[Callable[["HordeEnv", Fighter, str], None]] = None
    on_after_damage: Optional[
        Callable[["HordeEnv", Fighter, Fighter, str, str, int], None]
    ] = None
    on_end_turn: Optional[Callable[["HordeEnv", Fighter, str], None]] = None
    on_threshold: Optional[Callable[["HordeEnv", Fighter, str], None]] = None


def _apply_shield_start_round(env: "HordeEnv", f: Fighter, owner: str) -> None:
    s = int(f.card.ability.params.get("S", 0))
    if s > 0:
        f.shield = s
        env.last_log.append(f"{owner} gains SHIELD({s}).")


def _apply_regen_end_turn(env: "HordeEnv", f: Fighter, owner: str) -> None:
    if not f.alive():
        return
    h = int(f.card.ability.params.get("H", 0))
    if h <= 0:
        return
    before = f.hp
    # Cap regen at the card's starting health.
    f.hp = min(f.card.health, f.hp + h)
    if f.hp != before:
        env.last_log.append(f"{owner} regenerates +{f.hp - before} HP (now {f.hp}).")


def _apply_fear_after_damage(
    env: "HordeEnv",
    attacker: Fighter,
    defender: Fighter,
    attacker_name: str,
    defender_name: str,
    _did_damage: int,
) -> None:
    w = int(attacker.card.ability.params.get("W", 0))
    if w <= 0:
        return
    original_floor = max(0, defender.card.attack // 2)
    new_atk = max(original_floor, defender.atk - w)
    if new_atk != defender.atk:
        defender.atk = new_atk
        env.last_log.append(
            f"{attacker_name} inflicts FEAR (-{w} ATK, now {defender.atk})."
        )


def _apply_berserk_threshold(env: "HordeEnv", f: Fighter, owner: str) -> None:
    if not f.alive() or f.berserk_used:
        return
    if f.hp * 2 < f.max_hp:
        b = int(f.card.ability.params.get("B", 0))
        if b > 0:
            f.atk += b
            f.berserk_used = True
            env.last_log.append(f"{owner} triggers BERSERK (+{b} ATK, now {f.atk}).")


# Ability behavior lives here; add new AbilityKey entries and register hooks.
ABILITY_HOOKS: Dict[AbilityKey, AbilityHooks] = {
    AbilityKey.SHIELD: AbilityHooks(on_start_round=_apply_shield_start_round),
    AbilityKey.REGEN: AbilityHooks(on_end_turn=_apply_regen_end_turn),
    AbilityKey.FEAR: AbilityHooks(on_after_damage=_apply_fear_after_damage),
    AbilityKey.BERSERK: AbilityHooks(on_threshold=_apply_berserk_threshold),
}


class HordeEnv:
    """
    Match:
      - First to target_wins points wins
      - Each round: each player receives their own offer of 3 cards; both choose; simulate combat; award point if non-draw
    """
    def __init__(self, seed: Optional[int] = None, target_wins: int = 3, max_combat_turns: int = 12):
        self.rng = random.Random(seed)
        self.target_wins = target_wins
        self.max_combat_turns = max_combat_turns

        self.pool: List[Card] = list(CARD_POOL)

        # Match state
        self.p0_points = 0
        self.p1_points = 0
        self.round_index = 0
        self.offer_p0: List[Card] = []
        self.offer_p1: List[Card] = []
        self.p0_deck: List[Card] = []
        self.p1_deck: List[Card] = []
        self.deck_refreshes = {"p0": 0, "p1": 0}

        # Last combat summary
        self.last_log: List[str] = []

    # -------- Public API --------
    def reset(self) -> Dict[str, Any]:
        self.p0_points = 0
        self.p1_points = 0
        self.round_index = 0
        self.deck_refreshes = {"p0": 0, "p1": 0}
        self._reset_decks()
        self._sample_offers()
        self.last_log = []
        return self._obs()

    def step(self, a0: int, a1: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        a0, a1 are indices in [0,1,2]
        Returns: (obs, reward_for_p0, done, info)
        """
        assert 0 <= a0 < 3 and 0 <= a1 < 3
        self.round_index += 1
        self.last_log = []
        self.last_log.append(f"=== ROUND {self.round_index} ===")
        self.last_log.append(
            f"Offer P0: [0]{self._card_line(self.offer_p0[0])} | "
            f"[1]{self._card_line(self.offer_p0[1])} | "
            f"[2]{self._card_line(self.offer_p0[2])}"
        )
        self.last_log.append(
            f"Offer P1: [0]{self._card_line(self.offer_p1[0])} | "
            f"[1]{self._card_line(self.offer_p1[1])} | "
            f"[2]{self._card_line(self.offer_p1[2])}"
        )

        c0 = self.offer_p0[a0]
        c1 = self.offer_p1[a1]

        a0, c0 = self._validate_pick(
            a0,
            c0,
            self.p0_deck,
            self.offer_p0,
            owner="P0",
        )
        a1, c1 = self._validate_pick(
            a1,
            c1,
            self.p1_deck,
            self.offer_p1,
            owner="P1",
        )

        self.last_log.append(f"P0 picks: {self._card_line(c0)}")
        self.last_log.append(f"P1 picks: {self._card_line(c1)}")

        self._consume_card(self.p0_deck, c0)
        self._consume_card(self.p1_deck, c1)

        r0, r1, outcome = self._resolve_round(c0, c1)

        if outcome == "P0_WIN":
            self.p0_points += 1
            reward = 1.0
            self.last_log.append("Round result: P0 wins (+1 point)")
        elif outcome == "P1_WIN":
            self.p1_points += 1
            reward = -1.0
            self.last_log.append("Round result: P1 wins (+1 point)")
        else:
            reward = 0.0
            self.last_log.append("Round result: Draw (no points)")

        done = (self.p0_points >= self.target_wins) or (self.p1_points >= self.target_wins)

        info = {
            "outcome": outcome,
            "p0_points": self.p0_points,
            "p1_points": self.p1_points,
            "combat_summary": {"p0_end_hp": r0.hp, "p1_end_hp": r1.hp},
            "log": list(self.last_log),
        }

        if not done:
            self._sample_offers()

        return self._obs(), reward, done, info

    def render(self) -> None:
        for line in self.last_log:
            print(line)

    def observation_size(self) -> int:
        return OBS_VECTOR_SIZE

    def action_size(self) -> int:
        return 3

    # -------- Internals --------
    def _obs(self) -> Dict[str, Any]:
        return self._obs_for("p0")

    def _obs_for(self, player: str) -> Dict[str, Any]:
        offer = self.offer_p1 if player == "p1" else self.offer_p0
        obs = {
            "offer": [self._card_to_dict(c) for c in offer],
            "offerP0": [self._card_to_dict(c) for c in self.offer_p0],
            "offerP1": [self._card_to_dict(c) for c in self.offer_p1],
            "score": {"p0": self.p0_points, "p1": self.p1_points},
            "round": self.round_index,
            "remaining": {
                "p0": len(self.p0_deck),
                "p1": len(self.p1_deck),
                "total": len(self.pool),
            },
            "deckRefreshes": {
                "p0": self.deck_refreshes["p0"],
                "p1": self.deck_refreshes["p1"],
                "total": self.deck_refreshes["p0"] + self.deck_refreshes["p1"],
            },
        }
        obs["obs_vector"] = self.encode_obs(obs)
        return obs

    def _sample_offer(self, deck: List[Card], owner: str) -> List[Card]:
        # Sample 3 distinct cards from the remaining offer pool.
        if len(deck) < 3:
            self._reset_deck(deck, owner, count_refresh=True)
        return self.rng.sample(deck, k=3)

    def _sample_offers(self) -> None:
        self.offer_p0 = self._sample_offer(self.p0_deck, "p0")
        self.offer_p1 = self._sample_offer(self.p1_deck, "p1")
        if self._offers_match(self.offer_p0, self.offer_p1):
            self.offer_p1 = self._resample_distinct(
                self.offer_p1,
                self.offer_p0,
                self.p1_deck,
                owner="p1",
            )

    def _offer_signature(self, offer: List[Card]) -> tuple[str, ...]:
        return tuple(sorted(c.name for c in offer))

    def _offers_match(self, left: List[Card], right: List[Card]) -> bool:
        return self._offer_signature(left) == self._offer_signature(right)

    def _resample_distinct(
        self,
        current: List[Card],
        target: List[Card],
        deck: List[Card],
        owner: str,
        attempts: int = 6,
    ) -> List[Card]:
        if len(deck) <= 3:
            return current
        target_sig = self._offer_signature(target)
        for _ in range(attempts):
            candidate = self._sample_offer(deck, owner)
            if self._offer_signature(candidate) != target_sig:
                return candidate
        return current

    def encode_obs(self, obs: Optional[Dict[str, Any]] = None) -> List[float]:
        if obs is None:
            offer_cards = self.offer_p0
            p0_points = self.p0_points
            p1_points = self.p1_points
            encoded_offer = self._encode_offer_cards(offer_cards)
        else:
            offer = obs.get("offer", [])
            score = obs.get("score", {})
            p0_points = score.get("p0", self.p0_points)
            p1_points = score.get("p1", self.p1_points)
            if offer and isinstance(offer[0], dict):
                encoded_offer = self._encode_offer_dicts(offer)
            else:
                encoded_offer = self._encode_offer_cards(offer)
        vector = list(encoded_offer)
        vector.extend([float(p0_points), float(p1_points)])
        return vector

    def _encode_offer_cards(self, cards: Iterable[Card]) -> List[float]:
        vector: List[float] = []
        for c in cards:
            vector.extend(self._encode_card(c))
        return vector

    def _encode_offer_dicts(self, cards: Iterable[Dict[str, Any]]) -> List[float]:
        vector: List[float] = []
        for c in cards:
            vector.extend(self._encode_card_dict(c))
        return vector

    def _offer_has_choices(self, offer: List[Card]) -> bool:
        return (
            any(self._card_available(self.p0_deck, c) for c in offer)
            and any(self._card_available(self.p1_deck, c) for c in offer)
        )

    def _card_available(self, deck: List[Card], card: Card) -> bool:
        return any(c.name == card.name for c in deck)

    def _consume_card(self, deck: List[Card], card: Card) -> None:
        for idx, c in enumerate(deck):
            if c.name == card.name:
                deck.pop(idx)
                return
        raise RuntimeError(f"Card {card.name} is not available in deck.")

    def _reset_deck(self, deck: List[Card], owner: str, count_refresh: bool) -> None:
        deck.clear()
        deck.extend(self.pool)
        if count_refresh:
            self.deck_refreshes[owner] += 1

    def _reset_decks(self) -> None:
        self._reset_deck(self.p0_deck, "p0", count_refresh=False)
        self._reset_deck(self.p1_deck, "p1", count_refresh=False)

    def _validate_pick(
        self,
        action: int,
        card: Card,
        deck: List[Card],
        offer: List[Card],
        owner: str,
    ) -> Tuple[int, Card]:
        if self._card_available(deck, card):
            return action, card
        available_indices = [i for i, c in enumerate(offer) if self._card_available(deck, c)]
        if not available_indices:
            raise RuntimeError(f"{owner} has no available cards in this offer.")
        new_action = self.rng.choice(available_indices)
        new_card = offer[new_action]
        self.last_log.append(
            f"{owner} attempted to pick an unavailable card; "
            f"auto-selected [{new_action}] {new_card.name}."
        )
        return new_action, new_card

    def _encode_card(self, c: Card) -> List[float]:
        vector: List[float] = [float(c.attack), float(c.health)]
        vector.extend([1.0 if c.rarity == r else 0.0 for r in RARITY_ORDER])
        if c.ability is None:
            ability_key = None
            params: Dict[str, int] = {}
        else:
            ability_key = c.ability.key
            params = c.ability.params
        vector.extend([1.0 if ability_key == k else 0.0 for k in ABILITY_ORDER])
        vector.extend([float(params.get(k, 0)) for k in ABILITY_PARAM_KEYS])
        return vector

    def _encode_card_dict(self, c: Dict[str, Any]) -> List[float]:
        vector: List[float] = [float(c["attack"]), float(c["health"])]
        rarity = c.get("rarity")
        vector.extend([1.0 if rarity == r.value else 0.0 for r in RARITY_ORDER])
        ability = c.get("ability")
        if ability is None:
            ability_key = None
            params: Dict[str, int] = {}
        else:
            ability_key = ability.get("key")
            params = ability.get("params", {})
        vector.extend([1.0 if ability_key == k.value else 0.0 for k in ABILITY_ORDER])
        vector.extend([float(params.get(k, 0)) for k in ABILITY_PARAM_KEYS])
        return vector

    def _make_fighter(self, c: Card, owner: str) -> Fighter:
        f = Fighter(card=c, max_hp=c.health, hp=c.health, atk=c.attack)
        self._trigger_start_round(f, owner)
        return f

    def _resolve_round(self, c0: Card, c1: Card) -> Tuple[Fighter, Fighter, str]:
        f0 = self._make_fighter(c0, owner="P0")
        f1 = self._make_fighter(c1, owner="P1")

        self.last_log.append(f"Start: P0 HP={f0.hp} ATK={f0.atk} SH={f0.shield} | P1 HP={f1.hp} ATK={f1.atk} SH={f1.shield}")

        for t in range(1, self.max_combat_turns + 1):
            self.last_log.append(f"-- Combat Turn {t} --")

            # 1) Start-of-turn effects
            self._start_of_turn(f0, owner="P0")
            self._start_of_turn(f1, owner="P1")

            # KO check after start-of-turn effects
            if not f0.alive() or not f1.alive():
                return self._end_by_ko(f0, f1)

            # 2) Simultaneous attack
            dmg0 = f0.atk
            dmg1 = f1.atk
            hp_dmg0 = self._deal_damage(attacker="P0", defender=f1, dmg=dmg0)
            hp_dmg1 = self._deal_damage(attacker="P1", defender=f0, dmg=dmg1)

            # Threshold checks after taking attack damage
            self._trigger_threshold(f0, owner="P0")
            self._trigger_threshold(f1, owner="P1")

            # 3) KO check after attack
            if not f0.alive() or not f1.alive():
                return self._end_by_ko(f0, f1)

            # 4) After-damage abilities (on-hit)
            self._after_damage(
                attacker=f0,
                defender=f1,
                attacker_name="P0",
                defender_name="P1",
                did_damage=hp_dmg0,
            )
            self._after_damage(
                attacker=f1,
                defender=f0,
                attacker_name="P1",
                defender_name="P0",
                did_damage=hp_dmg1,
            )

            # 5) KO check after abilities
            if not f0.alive() or not f1.alive():
                return self._end_by_ko(f0, f1)

            # End-of-turn effects
            self._end_of_turn(f0, owner="P0")
            self._end_of_turn(f1, owner="P1")

            self.last_log.append(
                f"Status: P0 HP={f0.hp} ATK={f0.atk} SH={f0.shield} | "
                f"P1 HP={f1.hp} ATK={f1.atk} SH={f1.shield}"
            )

        # Turn cap reached: decide by remaining HP, else draw
        self.last_log.append("Turn cap reached.")
        if f0.hp > f1.hp:
            return f0, f1, "P0_WIN"
        elif f1.hp > f0.hp:
            return f0, f1, "P1_WIN"
        else:
            return f0, f1, "DRAW"

    def _start_of_turn(self, f: Fighter, owner: str) -> None:
        self._trigger_start_turn(f, owner)
        self._trigger_threshold(f, owner)

    def _deal_damage(self, attacker: str, defender: Fighter, dmg: int) -> int:
        if dmg <= 0 or not defender.alive():
            return 0
        absorbed = 0
        if defender.shield > 0:
            absorbed = min(defender.shield, dmg)
            defender.shield -= absorbed
            dmg -= absorbed
        if absorbed > 0:
            self.last_log.append(f"{attacker} hits shield for {absorbed} (defender shield now {defender.shield}).")
        if dmg > 0:
            defender.hp -= dmg
            self.last_log.append(f"{attacker} deals {dmg} damage (defender HP now {defender.hp}).")
            return dmg
        return 0

    def _after_damage(
        self,
        attacker: Fighter,
        defender: Fighter,
        attacker_name: str,
        defender_name: str,
        did_damage: int,
    ) -> None:
        if did_damage <= 0:
            return
        if not attacker.card.ability:
            return
        if not attacker.alive():
            return
        if not defender.alive():
            return
        hooks = ABILITY_HOOKS.get(attacker.card.ability.key)
        if hooks and hooks.on_after_damage:
            hooks.on_after_damage(
                self,
                attacker,
                defender,
                attacker_name,
                defender_name,
                did_damage,
            )

    def _end_of_turn(self, f: Fighter, owner: str) -> None:
        if not f.card.ability:
            return
        hooks = ABILITY_HOOKS.get(f.card.ability.key)
        if hooks and hooks.on_end_turn:
            hooks.on_end_turn(self, f, owner)

    def _trigger_start_round(self, f: Fighter, owner: str) -> None:
        if not f.card.ability or not f.alive():
            return
        hooks = ABILITY_HOOKS.get(f.card.ability.key)
        if hooks and hooks.on_start_round:
            hooks.on_start_round(self, f, owner)

    def _trigger_start_turn(self, f: Fighter, owner: str) -> None:
        if not f.card.ability or not f.alive():
            return
        hooks = ABILITY_HOOKS.get(f.card.ability.key)
        if hooks and hooks.on_start_turn:
            hooks.on_start_turn(self, f, owner)

    def _trigger_threshold(self, f: Fighter, owner: str) -> None:
        if not f.card.ability or not f.alive():
            return
        hooks = ABILITY_HOOKS.get(f.card.ability.key)
        if hooks and hooks.on_threshold:
            hooks.on_threshold(self, f, owner)

    def _end_by_ko(self, f0: Fighter, f1: Fighter) -> Tuple[Fighter, Fighter, str]:
        # Determine outcome given current HP values
        if f0.alive() and not f1.alive():
            return f0, f1, "P0_WIN"
        if f1.alive() and not f0.alive():
            return f0, f1, "P1_WIN"
        return f0, f1, "DRAW"

    def _card_line(self, c: Card) -> str:
        return f"{c.name} [{c.rarity.value}] ATK={c.attack} HP={c.health} AB={c.short_ability()}"

    def _card_to_dict(self, c: Card) -> Dict[str, Any]:
        return {
            "name": c.name,
            "rarity": c.rarity.value,
            "attack": c.attack,
            "health": c.health,
            "ability": None if c.ability is None else {"key": c.ability.key.value, "params": dict(c.ability.params)},
        }
