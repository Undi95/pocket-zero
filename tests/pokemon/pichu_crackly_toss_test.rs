use deckgym::{
    actions::{Action, SimpleAction},
    card_ids::CardId,
    models::PlayedCard,
    test_support::{attack_action, get_test_game_with_board},
};

/// Pichu's Crackly Toss: "Take a [L] Energy from your Energy Zone and attach it to 1 of your
/// Benched Basic Pokémon." An evolved Benched Pokémon (Raichu) must not be offered.
#[test]
fn test_pichu_crackly_toss_only_targets_benched_basic() {
    let mut game = get_test_game_with_board(
        vec![
            PlayedCard::from_id(CardId::A4066Pichu),
            PlayedCard::from_id(CardId::A1094Pikachu),
            PlayedCard::from_id(CardId::A1095Raichu),
        ],
        vec![PlayedCard::from_id(CardId::A1001Bulbasaur)],
    );

    game.apply_action(&Action {
        actor: 0,
        action: attack_action(CardId::A4066Pichu, 0),
        is_stack: false,
    });

    let (actor, choices) = game.get_state_clone().generate_possible_actions();
    assert_eq!(actor, 0);
    let targets: Vec<usize> = choices
        .iter()
        .filter_map(|choice| match &choice.action {
            SimpleAction::Attach { attachments, .. } => Some(attachments[0].2),
            _ => None,
        })
        .collect();
    assert_eq!(
        targets,
        vec![1],
        "only the Benched Basic Pikachu can receive the Energy"
    );
}
