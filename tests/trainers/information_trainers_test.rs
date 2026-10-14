use deckgym::{
    actions::{Action, SimpleAction},
    card_ids::CardId,
    database::get_card_by_enum,
    models::{Card, PlayedCard},
    test_support::{get_test_game_with_board, play_trainer, trainer_from_id},
};

fn cards(ids: &[CardId]) -> Vec<Card> {
    ids.iter().map(|id| get_card_by_enum(*id)).collect()
}

/// Pokédex, Hand Scope, Looker, Hiker and Morty move no card between zones (what the player sees
/// is recorded in `state.knowledge`). They must be playable and leave the board as it was.
#[test]
fn test_information_only_trainers_are_playable_and_change_nothing() {
    for card_id in [
        CardId::PA003HandScope,
        CardId::PA004PokedEx,
        CardId::PA008PokedEx,
        CardId::A3a068Looker,
        CardId::A4161Hiker,
        CardId::A4a071Morty,
    ] {
        let mut game = get_test_game_with_board(
            vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
            vec![PlayedCard::from_id(CardId::A1033Charmander)],
        );
        let mut state = game.get_state_clone();
        let trainer = trainer_from_id(card_id);
        state.hands[0] = vec![Card::Trainer(trainer.clone())];
        let deck_before = state.decks[0].cards.len();
        let opponent_hand_before = state.hands[1].len();
        game.set_state(state);

        play_trainer(&mut game, 0, trainer);

        let state = game.get_state_clone();
        assert_eq!(
            state.decks[0].cards.len(),
            deck_before,
            "{card_id:?} should not move any card out of the deck"
        );
        assert!(
            state.hands[0].is_empty(),
            "{card_id:?} should leave the hand empty after being played"
        );
        assert_eq!(
            state.hands[1].len(),
            opponent_hand_before,
            "{card_id:?} should not touch the opponent's hand"
        );
        assert_eq!(state.get_active(0).get_name(), "Bulbasaur");
    }
}

/// Hiker: "For each of your [F] Pokémon in play, look at that many cards from the top of your
/// deck and put them back in any order."
#[test]
fn test_hiker_puts_the_top_cards_back_in_the_chosen_order() {
    let mut game = get_test_game_with_board(
        vec![
            PlayedCard::from_id(CardId::A1143Machop),
            PlayedCard::from_id(CardId::A1143Machop),
        ],
        vec![PlayedCard::from_id(CardId::A1033Charmander)],
    );
    let mut state = game.get_state_clone();
    let hiker = trainer_from_id(CardId::A4161Hiker);
    state.hands[0] = vec![Card::Trainer(hiker.clone())];
    state.decks[0].cards = cards(&[
        CardId::A1001Bulbasaur,
        CardId::A1033Charmander,
        CardId::A1053Squirtle,
    ]);
    game.set_state(state);

    play_trainer(&mut game, 0, hiker);

    // Two [F] Pokémon: the top 2 cards are put back in any order.
    let (actor, choices) = game.get_state_clone().generate_possible_actions();
    assert_eq!(actor, 0);
    let charmander_first = choices
        .iter()
        .find(|choice| {
            matches!(&choice.action, SimpleAction::ArrangeTopOfDeck { player: 0, position_idx: 0, count: 2, card }
                if card.get_name() == "Charmander")
        })
        .expect("Charmander can be put on top")
        .clone();
    assert_eq!(choices.len(), 2);
    game.apply_action(&charmander_first);

    let state = game.get_state_clone();
    assert_eq!(
        state.decks[0].cards,
        cards(&[
            CardId::A1033Charmander,
            CardId::A1001Bulbasaur,
            CardId::A1053Squirtle
        ])
    );
    assert!(
        state.move_generation_stack.is_empty(),
        "nothing left to choose"
    );
    let known = state.knowledge[0].top_of_deck[0]
        .as_ref()
        .expect("Hiker shows the top cards");
    assert_eq!(known.1, 2);
}

/// Morty: "For each of your [P] Pokémon in play, look at that many cards from the top of your
/// opponent's deck and put them back in any order."
#[test]
fn test_morty_arranges_the_top_of_the_opponents_deck() {
    let mut game = get_test_game_with_board(
        vec![
            PlayedCard::from_id(CardId::A1120Gastly),
            PlayedCard::from_id(CardId::A1120Gastly),
            PlayedCard::from_id(CardId::A1120Gastly),
        ],
        vec![PlayedCard::from_id(CardId::A1033Charmander)],
    );
    let mut state = game.get_state_clone();
    let morty = trainer_from_id(CardId::A4a071Morty);
    state.hands[0] = vec![Card::Trainer(morty.clone())];
    state.decks[1].cards = cards(&[
        CardId::A1001Bulbasaur,
        CardId::A1033Charmander,
        CardId::A1053Squirtle,
        CardId::PA001Potion,
    ]);
    game.set_state(state);

    play_trainer(&mut game, 0, morty);

    // Three [P] Pokémon: player 0 orders the opponent's top 3 cards, one position at a time.
    for name in ["Squirtle", "Charmander"] {
        let (actor, choices) = game.get_state_clone().generate_possible_actions();
        assert_eq!(actor, 0);
        let choice = choices
            .iter()
            .find(|choice| {
                matches!(&choice.action, SimpleAction::ArrangeTopOfDeck { player: 1, count: 3, card, .. }
                    if card.get_name() == name)
            })
            .expect("each remaining card can go next")
            .clone();
        game.apply_action(&choice);
    }

    let state = game.get_state_clone();
    assert_eq!(
        state.decks[1].cards,
        cards(&[
            CardId::A1053Squirtle,
            CardId::A1033Charmander,
            CardId::A1001Bulbasaur,
            CardId::PA001Potion
        ]),
        "the 4th card is not touched"
    );
    assert!(state.move_generation_stack.is_empty());
}

/// Unown's CHECK: "Once during your turn, you may choose either player. Look at the top card of
/// that player's deck."
#[test]
fn test_unown_check_looks_at_the_top_card_of_the_chosen_deck() {
    let mut game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A2a034Unown)],
        vec![PlayedCard::from_id(CardId::A1033Charmander)],
    );

    game.apply_action(&Action {
        actor: 0,
        action: SimpleAction::UseAbility { in_play_idx: 0 },
        is_stack: false,
    });

    let (actor, choices) = game.get_state_clone().generate_possible_actions();
    assert_eq!(actor, 0);
    let choices: Vec<_> = choices.into_iter().map(|choice| choice.action).collect();
    assert_eq!(
        choices,
        vec![
            SimpleAction::LookAtTopOfDeck { player: 0 },
            SimpleAction::LookAtTopOfDeck { player: 1 }
        ]
    );
    game.apply_action(&Action {
        actor: 0,
        action: choices[1].clone(),
        is_stack: true,
    });

    let state = game.get_state_clone();
    assert_eq!(
        state.knowledge[0].top_of_deck[1].as_ref().map(|k| k.1),
        Some(1)
    );
    assert!(state.knowledge[0].top_of_deck[0].is_none());
}

/// Rotom Dex: "Look at the top card of your deck. Then, you may shuffle your deck." The optional
/// shuffle is offered as a real choice (decline = Noop).
#[test]
fn test_rotom_dex_offers_optional_shuffle() {
    let mut game = get_test_game_with_board(
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
        vec![PlayedCard::from_id(CardId::A1033Charmander)],
    );
    let mut state = game.get_state_clone();
    let rotom_dex = trainer_from_id(CardId::A3145RotomDEx);
    state.hands[0] = vec![Card::Trainer(rotom_dex.clone())];
    let deck_size_before = state.decks[0].cards.len();
    game.set_state(state);

    play_trainer(&mut game, 0, rotom_dex);

    let (actor, actions) = game.get_state_clone().generate_possible_actions();
    assert_eq!(actor, 0);
    assert!(
        actions
            .iter()
            .any(|a| matches!(a.action, SimpleAction::ShuffleOwnDeck)),
        "Rotom Dex should offer to shuffle the deck"
    );
    assert!(
        actions
            .iter()
            .any(|a| matches!(a.action, SimpleAction::Noop)),
        "Rotom Dex should offer to decline the shuffle"
    );

    let shuffle = actions
        .iter()
        .find(|a| matches!(a.action, SimpleAction::ShuffleOwnDeck))
        .expect("shuffle option")
        .clone();
    game.apply_action(&shuffle);

    assert_eq!(
        game.get_state_clone().decks[0].cards.len(),
        deck_size_before,
        "Shuffling should not change the number of cards in the deck"
    );
}
