use std::collections::HashMap;
use std::sync::LazyLock;

use crate::{
    card_ids::CardId,
    database::get_card_by_enum,
    models::{Card, PlayedCard, TrainerCard, TrainerType},
    State,
};

pub(crate) fn ensure_tool_card(card: &Card) -> &TrainerCard {
    match card {
        Card::Trainer(trainer_card) => ensure_tool_trainer(trainer_card),
        _ => panic!("Expected TrainerCard of subtype Tool, got non-trainer card"),
    }
}

pub(crate) fn ensure_tool_trainer(trainer_card: &TrainerCard) -> &TrainerCard {
    if trainer_card.trainer_card_type != TrainerType::Tool {
        panic!(
            "Expected TrainerCard of subtype Tool, got {:?}",
            trainer_card.trainer_card_type
        );
    }
    trainer_card
}

fn tool_effect_text_from_card_id(tool_card_id: CardId) -> String {
    let card = get_card_by_enum(tool_card_id);
    let trainer_card = ensure_tool_card(&card);
    trainer_card.effect.clone()
}

static GIANT_CAPE_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::A2147GiantCape));
static ROCKY_HELMET_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::A2148RockyHelmet));
static POISON_BARB_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::A3146PoisonBarb));
static LEAF_CAPE_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::A3147LeafCape));
static ELECTRICAL_CORD_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::A3a065ElectricalCord));
static LEFTOVERS_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::A3b067Leftovers));
static INFLATABLE_BOAT_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::A4a067InflatableBoat));
static STEEL_APRON_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::A4153SteelApron));
static HEAVY_HELMET_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::B1219HeavyHelmet));
static PROTECTIVE_PONCHO_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::B2147ProtectivePoncho));
static METAL_CORE_BARRIER_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::B2148MetalCoreBarrier));
static BIG_AIR_BALLOON_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::B2a087BigAirBalloon));
static LUCKY_EGG_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::B3148LuckyEgg));
static ANCIENT_BOOSTER_ENERGY_CAPSULE_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::B3a069AncientBoosterEnergyCapsule));
static FUTURE_BOOSTER_ENERGY_CAPSULE_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::B3a070FutureBoosterEnergyCapsule));
static SMALL_BALLOON_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::B3b064SmallBalloon));
static ELEGANT_CAPE_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::B3b065ElegantCape));
static DECEPTIVE_NEEDLE_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::B4148DeceptiveNeedle));
static LUM_BERRY_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::A2149LumBerry));
static BEASTITE_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::A3a066Beastite));
static DARK_PENDANT_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::A4154DarkPendant));
static RESCUE_SCARF_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::A4155RescueScarf));
static MEMORY_LIGHT_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::A4a068MemoryLight));
static SITRUS_BERRY_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::B1218SitrusBerry));
static LUCKY_MITTENS_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::B1220LuckyMittens));
static CLEAR_VEIL_EFFECT: LazyLock<String> =
    LazyLock::new(|| tool_effect_text_from_card_id(CardId::B4149ClearVeil));

/// Effect text of the Tools the engine checks for, built once. `has_tool` runs on every HP
/// check, and building the reference card from the database each time (every string cloned)
/// was about a third of the engine's time in self-play.
static TOOL_EFFECTS: LazyLock<HashMap<CardId, &'static str>> = LazyLock::new(|| {
    [
        (CardId::A2147GiantCape, &GIANT_CAPE_EFFECT),
        (CardId::A2148RockyHelmet, &ROCKY_HELMET_EFFECT),
        (CardId::A3146PoisonBarb, &POISON_BARB_EFFECT),
        (CardId::A3147LeafCape, &LEAF_CAPE_EFFECT),
        (CardId::A3a065ElectricalCord, &ELECTRICAL_CORD_EFFECT),
        (CardId::A3b067Leftovers, &LEFTOVERS_EFFECT),
        (CardId::A4a067InflatableBoat, &INFLATABLE_BOAT_EFFECT),
        (CardId::A4153SteelApron, &STEEL_APRON_EFFECT),
        (CardId::B1219HeavyHelmet, &HEAVY_HELMET_EFFECT),
        (CardId::B2147ProtectivePoncho, &PROTECTIVE_PONCHO_EFFECT),
        (CardId::B2148MetalCoreBarrier, &METAL_CORE_BARRIER_EFFECT),
        (CardId::B2a087BigAirBalloon, &BIG_AIR_BALLOON_EFFECT),
        (CardId::B3148LuckyEgg, &LUCKY_EGG_EFFECT),
        (
            CardId::B3a069AncientBoosterEnergyCapsule,
            &ANCIENT_BOOSTER_ENERGY_CAPSULE_EFFECT,
        ),
        (
            CardId::B3a070FutureBoosterEnergyCapsule,
            &FUTURE_BOOSTER_ENERGY_CAPSULE_EFFECT,
        ),
        (CardId::B3b064SmallBalloon, &SMALL_BALLOON_EFFECT),
        (CardId::B3b065ElegantCape, &ELEGANT_CAPE_EFFECT),
        (CardId::B4148DeceptiveNeedle, &DECEPTIVE_NEEDLE_EFFECT),
        (CardId::A2149LumBerry, &LUM_BERRY_EFFECT),
        (CardId::A3a066Beastite, &BEASTITE_EFFECT),
        (CardId::A4154DarkPendant, &DARK_PENDANT_EFFECT),
        (CardId::A4155RescueScarf, &RESCUE_SCARF_EFFECT),
        (CardId::A4a068MemoryLight, &MEMORY_LIGHT_EFFECT),
        (CardId::B1218SitrusBerry, &SITRUS_BERRY_EFFECT),
        (CardId::B1220LuckyMittens, &LUCKY_MITTENS_EFFECT),
        (CardId::B4149ClearVeil, &CLEAR_VEIL_EFFECT),
    ]
    .into_iter()
    .map(|(id, effect)| (id, effect.as_str()))
    .collect()
});

/// Whether `effect` is the effect text of the referenced Tool (cached for the Tools above).
fn is_tool_effect(effect: &str, reference_tool_id: CardId) -> bool {
    match TOOL_EFFECTS.get(&reference_tool_id) {
        Some(reference) => effect == *reference,
        None => effect == tool_effect_text_from_card_id(reference_tool_id),
    }
}

pub fn tool_effects_equal(trainer_card: &TrainerCard, reference_tool_id: CardId) -> bool {
    ensure_tool_trainer(trainer_card);
    is_tool_effect(&trainer_card.effect, reference_tool_id)
}

/// Whether *any* Tool attached to `played_card` is the referenced Tool. A Pokémon with two Tool
/// slots (Revavroom's Dual Customization) answers `true` for either of them, so every
/// "if this Pokémon has X attached" hook keeps working unchanged.
pub fn has_tool(played_card: &PlayedCard, reference_tool_id: CardId) -> bool {
    tool_position(played_card, reference_tool_id).is_some()
}

/// The slot of the referenced Tool among `played_card.attached_tools`, for effects that have to
/// remove *that* Tool specifically (Lum Berry and Sitrus Berry discard themselves; Metal Core
/// Barrier is discarded at the end of the opponent's turn) rather than all of them.
pub(crate) fn tool_position(played_card: &PlayedCard, reference_tool_id: CardId) -> Option<usize> {
    played_card.attached_tools.iter().position(|attached_tool| {
        is_tool_effect(&ensure_tool_card(attached_tool).effect, reference_tool_id)
    })
}

pub(crate) fn enumerate_tool_choices<'a>(
    trainer_card: &TrainerCard,
    state: &'a State,
    actor: usize,
) -> Vec<(usize, &'a PlayedCard)> {
    ensure_tool_trainer(trainer_card);
    // Pokémon Tools can be attached to ANY Pokémon — the game never restricts attachment by
    // type or stage. Tools whose effect is type/stage-specific (Leaf Cape [G] +30 HP, Big Air
    // Balloon Stage-2 free retreat, Steel Apron [M] −10, etc.) gate the *effect* at its
    // application site, not the attachment.
    //
    // Two attachment rules remain, both in `PlayedCard::can_attach_tool`:
    //   * capacity — one Tool per Pokémon, or two while Revavroom's Dual Customization is live;
    //   * no duplicates — a Pokémon may not hold two copies of the same Tool.
    //
    // The duplicate rule is an ENGINE RULING, not printed text. "This Pokémon may have up to 2
    // Pokémon Tool cards attached to it" says nothing about same-name Tools, and the rules corpus
    // is silent (`rules/edge-cases.md` §8 only covers the one-Tool default). We block duplicates
    // because every Tool effect in this engine is keyed off `has_tool`, i.e. a boolean "is this
    // Tool attached" — a second Giant Cape would grant +20 HP, not +40, and a second Rocky Helmet
    // 20 recoil, not 40. Offering a move that the engine then silently under-resolves is worse
    // than not offering it: it would let the search burn a card for literally nothing. If Tool
    // effects are ever made to stack, lift this restriction with them.
    let card = Card::Trainer(trainer_card.clone());
    state
        .enumerate_in_play_pokemon(actor)
        .filter(move |(_, x)| x.can_attach_tool(&card))
        .collect()
}

pub fn is_tool_effect_implemented(trainer_card: &TrainerCard) -> bool {
    let trainer_card = ensure_tool_trainer(trainer_card);
    let effect = trainer_card.effect.as_str();
    matches!(
        effect,
        e if e == GIANT_CAPE_EFFECT.as_str()
            || e == ROCKY_HELMET_EFFECT.as_str()
            || e == POISON_BARB_EFFECT.as_str()
            || e == LEAF_CAPE_EFFECT.as_str()
            || e == ELECTRICAL_CORD_EFFECT.as_str()
            || e == LEFTOVERS_EFFECT.as_str()
            || e == INFLATABLE_BOAT_EFFECT.as_str()
            || e == STEEL_APRON_EFFECT.as_str()
            || e == HEAVY_HELMET_EFFECT.as_str()
            || e == PROTECTIVE_PONCHO_EFFECT.as_str()
            || e == METAL_CORE_BARRIER_EFFECT.as_str()
            || e == BIG_AIR_BALLOON_EFFECT.as_str()
            || e == LUCKY_EGG_EFFECT.as_str()
            || e == ANCIENT_BOOSTER_ENERGY_CAPSULE_EFFECT.as_str()
            || e == FUTURE_BOOSTER_ENERGY_CAPSULE_EFFECT.as_str()
            || e == SMALL_BALLOON_EFFECT.as_str()
            || e == ELEGANT_CAPE_EFFECT.as_str()
            || e == DECEPTIVE_NEEDLE_EFFECT.as_str()
            || e == LUM_BERRY_EFFECT.as_str()
            || e == BEASTITE_EFFECT.as_str()
            || e == DARK_PENDANT_EFFECT.as_str()
            || e == RESCUE_SCARF_EFFECT.as_str()
            || e == MEMORY_LIGHT_EFFECT.as_str()
            || e == SITRUS_BERRY_EFFECT.as_str()
            || e == LUCKY_MITTENS_EFFECT.as_str()
            || e == CLEAR_VEIL_EFFECT.as_str()
    )
}
