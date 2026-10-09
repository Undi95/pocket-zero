//! Regression tests for the Trainer audit (lab/README.md): each test fails on the code before
//! the fix.
use deckgym::{
    actions::{Action, SimpleAction},
    card_ids::CardId,
    database::get_card_by_enum,
    models::{Card, EnergyType, PlayedCard},
    test_support::{get_test_game_with_board, play_trainer, trainer_from_id},
};

fn stack_choices(game: &deckgym::Game<'static>) -> Vec<Action> {
    let (_, actions) = game.get_state_clone().generate_possible_actions();
    actions
}

/// Quick-Grow Extract: "Choose 1 of your [G] Pokémon in play." The player picks which Pokémon
/// evolves; only the evolution card is random.
#[test]
fn test_quick_grow_extract_lets_player_choose_the_target() {
    let mut game = get_test_game_with_board(
        vec![
            PlayedCard::from_id(CardId::B3005Treecko),
            PlayedCard::from_id(CardId::A1001Bulbasaur),
        ],
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
    );
    let mut state = game.get_state_clone();
    state.decks[0].cards = vec![
        get_card_by_enum(CardId::B3006Grovyle),
        get_card_by_enum(CardId::A1002Ivysaur),
    ];
    state.hands[0] = vec![get_card_by_enum(CardId::B1a067QuickGrowExtract)];
    game.set_state(state);

    play_trainer(
        &mut game,
        0,
        trainer_from_id(CardId::B1a067QuickGrowExtract),
    );
    let choices = stack_choices(&game);
    let targets: Vec<usize> = choices
        .iter()
        .filter_map(|a| match a.action {
            SimpleAction::EvolveRandomFromDeck { in_play_idx, .. } => Some(in_play_idx),
            _ => None,
        })
        .collect();
    assert_eq!(
        targets,
        vec![0, 1],
        "both Grass Pokémon should be offered as targets"
    );

    let pick_treecko = choices
        .into_iter()
        .find(|a| {
            matches!(
                a.action,
                SimpleAction::EvolveRandomFromDeck { in_play_idx: 0, .. }
            )
        })
        .unwrap();
    game.apply_action(&pick_treecko);
    let state = game.get_state_clone();
    assert_eq!(state.get_active(0).get_name(), "Grovyle");
    assert_eq!(
        state.in_play_pokemon[0][1].as_ref().unwrap().get_name(),
        "Bulbasaur"
    );
}

/// Mythical Slab: "If that card is a [P] Pokémon, put it into your hand." A Basic [G] Pokémon
/// goes to the bottom, a [P] Pokémon goes to hand.
#[test]
fn test_mythical_slab_takes_psychic_pokemon_not_any_basic() {
    for (top, expect_in_hand) in [(CardId::A1001Bulbasaur, false), (CardId::A1128Mewtwo, true)] {
        let mut game = get_test_game_with_board(
            vec![PlayedCard::from_id(CardId::A1128Mewtwo)],
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        );
        let mut state = game.get_state_clone();
        let top_card = get_card_by_enum(top);
        state.decks[0].cards = vec![top_card.clone(), get_card_by_enum(CardId::A1002Ivysaur)];
        state.hands[0] = vec![get_card_by_enum(CardId::A1a065MythicalSlab)];
        game.set_state(state);

        play_trainer(&mut game, 0, trainer_from_id(CardId::A1a065MythicalSlab));
        let state = game.get_state_clone();
        assert_eq!(
            state.hands[0].contains(&top_card),
            expect_in_hand,
            "{top:?}"
        );
        if !expect_in_hand {
            assert_eq!(state.decks[0].cards.last(), Some(&top_card));
        }
    }
}

/// Koga: "Put your Muk or Weezing in the Active Spot into your hand." Works on every print.
#[test]
fn test_koga_works_on_any_weezing_print() {
    let mut game = get_test_game_with_board(
        vec![
            PlayedCard::from_id(CardId::B3102Weezing),
            PlayedCard::from_id(CardId::A1001Bulbasaur),
        ],
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
    );
    let mut state = game.get_state_clone();
    state.hands[0] = vec![get_card_by_enum(CardId::A1222Koga)];
    game.set_state(state);

    let koga = trainer_from_id(CardId::A1222Koga);
    assert!(stack_choices(&game).iter().any(
        |a| matches!(&a.action, SimpleAction::Play { trainer_card } if trainer_card.name == koga.name)
    ));
}

/// Dragalge ex's Poison Point works on its later prints too.
#[test]
fn test_dragalge_ex_reprint_poisons_attacker() {
    let mut game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)
            .with_energy(vec![EnergyType::Grass, EnergyType::Colorless])],
        vec![PlayedCard::from_id(CardId::B3231DragalgeEx)],
    );
    game.apply_action(&Action {
        actor: 0,
        action: deckgym::test_support::attack_action(CardId::A1001Bulbasaur, 0),
        is_stack: false,
    });
    let state = game.get_state_clone();
    let Card::Pokemon(_) = &state.get_active(0).card else {
        panic!()
    };
    assert!(state.get_active(0).is_poisoned());
}

fn can_play(game: &deckgym::Game<'static>, id: CardId) -> bool {
    let name = trainer_from_id(id).name;
    stack_choices(game).iter().any(
        |a| matches!(&a.action, SimpleAction::Play { trainer_card } if trainer_card.name == name),
    )
}

/// Only 1 Stadium card can be played per turn; the second one replaces the first next turn.
#[test]
fn test_only_one_stadium_per_turn() {
    let mut game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
    );
    let mut state = game.get_state_clone();
    state.hands[0] = vec![
        get_card_by_enum(CardId::B2154StartingPlains),
        get_card_by_enum(CardId::B2153TrainingArea),
    ];
    game.set_state(state);

    assert!(can_play(&game, CardId::B2153TrainingArea));
    play_trainer(&mut game, 0, trainer_from_id(CardId::B2154StartingPlains));
    assert!(
        !can_play(&game, CardId::B2153TrainingArea),
        "a second Stadium can't be played this turn"
    );
}

/// Brock: "attach it to Golem or Onix" can't be played without one of them in play.
#[test]
fn test_brock_needs_golem_or_onix() {
    for (board, playable) in [(CardId::A1001Bulbasaur, false), (CardId::A1150Onix, true)] {
        let mut game = get_test_game_with_board(
            vec![PlayedCard::from_id(board)],
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        );
        let mut state = game.get_state_clone();
        state.hands[0] = vec![get_card_by_enum(CardId::A1224Brock)];
        game.set_state(state);
        assert_eq!(can_play(&game, CardId::A1224Brock), playable, "{board:?}");
    }
}
