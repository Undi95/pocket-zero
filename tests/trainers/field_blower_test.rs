use deckgym::{
    actions::{Action, SimpleAction},
    card_ids::CardId,
    database::get_card_by_enum,
    models::{Card, PlayedCard, TrainerCard},
    test_support::get_initialized_game,
};

fn make_field_blower_trainer_card() -> TrainerCard {
    match get_card_by_enum(CardId::B3147FieldBlower) {
        Card::Trainer(tc) => tc,
        _ => panic!("Expected trainer card"),
    }
}

fn make_rocky_helmet_card() -> Card {
    get_card_by_enum(CardId::A2148RockyHelmet)
}

#[test]
fn test_field_blower_discards_opponents_tool() {
    let mut game = get_initialized_game(0);
    let mut state = game.get_state_clone();
    state.current_player = 0;

    state.set_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![PlayedCard::from_id(CardId::A1033Charmander).with_tool(make_rocky_helmet_card())],
    );
    state.hands[0].clear();
    let trainer_card = make_field_blower_trainer_card();
    state.hands[0].push(Card::Trainer(trainer_card.clone()));
    game.set_state(state);

    let play_action = Action {
        actor: 0,
        action: SimpleAction::Play { trainer_card },
        is_stack: false,
    };
    game.apply_action(&play_action);

    // Player must choose what to discard
    let state = game.get_state_clone();
    let (actor, choices) = state.generate_possible_actions();
    assert_eq!(actor, 0);
    assert!(
        choices.iter().any(|a| matches!(
            a.action,
            SimpleAction::DiscardToolFromPokemon {
                player: 1,
                in_play_idx: 0,
                tool_idx: 0
            }
        )),
        "Should offer discarding opponent's tool"
    );

    // Apply the discard
    let discard_action = choices
        .iter()
        .find(|a| {
            matches!(
                a.action,
                SimpleAction::DiscardToolFromPokemon {
                    player: 1,
                    in_play_idx: 0,
                    tool_idx: 0
                }
            )
        })
        .unwrap()
        .clone();
    game.apply_action(&discard_action);

    let state = game.get_state_clone();
    assert!(
        state.in_play_pokemon[1][0]
            .as_ref()
            .unwrap()
            .attached_tools
            .is_empty(),
        "Opponent's tool should be discarded"
    );
    assert!(
        state.discard_piles[1].contains(&make_rocky_helmet_card()),
        "Tool should be in opponent's discard pile"
    );
}

#[test]
fn test_field_blower_discards_active_stadium() {
    let mut game = get_initialized_game(0);
    let mut state = game.get_state_clone();
    state.current_player = 0;

    state.set_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![PlayedCard::from_id(CardId::A1033Charmander)],
    );
    // Place a stadium
    let stadium_card = get_card_by_enum(CardId::B2155PeculiarPlaza);
    state.active_stadium = Some(stadium_card.clone());

    state.hands[0].clear();
    let trainer_card = make_field_blower_trainer_card();
    state.hands[0].push(Card::Trainer(trainer_card.clone()));
    game.set_state(state);

    let play_action = Action {
        actor: 0,
        action: SimpleAction::Play { trainer_card },
        is_stack: false,
    };
    game.apply_action(&play_action);

    let state = game.get_state_clone();
    let (actor, choices) = state.generate_possible_actions();
    assert_eq!(actor, 0);
    assert!(
        choices
            .iter()
            .any(|a| matches!(a.action, SimpleAction::DiscardActiveStadium)),
        "Should offer discarding the active stadium"
    );

    let discard_action = choices
        .iter()
        .find(|a| matches!(a.action, SimpleAction::DiscardActiveStadium))
        .unwrap()
        .clone();
    game.apply_action(&discard_action);

    let state = game.get_state_clone();
    assert!(state.active_stadium.is_none(), "Stadium should be gone");
    assert!(
        state.discard_piles[0].contains(&stadium_card),
        "Stadium should be in player's discard pile"
    );
}

/// Field Blower on Starting Plains: once the Stadium is gone, every Basic loses its +20 HP, and a
/// Basic whose damage now reaches its printed HP is Knocked Out (its owner promotes).
#[test]
fn test_field_blower_on_starting_plains_removes_hp_bonus_and_knocks_out() {
    use deckgym::test_support::{get_test_game_with_board, play_trainer, trainer_from_id};

    let mut game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![
            PlayedCard::from_id(CardId::A1001Bulbasaur),
            PlayedCard::from_id(CardId::A1053Squirtle),
        ],
    );
    let mut state = game.get_state_clone();
    state.hands[0] = vec![
        get_card_by_enum(CardId::B2154StartingPlains),
        get_card_by_enum(CardId::B3147FieldBlower),
    ];
    game.set_state(state);
    play_trainer(&mut game, 0, trainer_from_id(CardId::B2154StartingPlains));

    // 80 damage on the opponent's 70-HP Bulbasaur (90 HP with Starting Plains).
    game.apply_action(&Action {
        actor: 0,
        action: SimpleAction::ApplyDamage {
            attacking_ref: (0, 0),
            targets: vec![(80, 1, 0)],
            is_from_active_attack: false,
        },
        is_stack: false,
    });
    assert_eq!(game.get_state_clone().get_active(1).get_remaining_hp(), 10);
    assert_eq!(game.get_state_clone().get_active(0).get_remaining_hp(), 90);

    play_trainer(&mut game, 0, trainer_from_id(CardId::B3147FieldBlower));
    let (actor, choices) = game.get_state_clone().generate_possible_actions();
    assert_eq!(actor, 0);
    let discard_stadium = choices
        .iter()
        .find(|a| matches!(a.action, SimpleAction::DiscardActiveStadium))
        .expect("Field Blower should offer the Stadium")
        .clone();
    game.apply_action(&discard_stadium);

    let state = game.get_state_clone();
    assert!(state.active_stadium.is_none());
    assert_eq!(
        state.get_active(0).get_remaining_hp(),
        70,
        "Bulbasaur should be back to its printed 70 HP"
    );
    assert!(
        state.in_play_pokemon[1][0].is_none(),
        "the 80-damage Bulbasaur has only 70 HP without Starting Plains: Knocked Out"
    );
    assert_eq!(state.points[0], 1);
    let (actor, choices) = state.generate_possible_actions();
    assert_eq!(actor, 1, "the opponent promotes a new Active");
    assert!(choices
        .iter()
        .any(|a| matches!(a.action, SimpleAction::Activate { player: 1, .. })));
}
