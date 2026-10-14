use serde::{Deserialize, Serialize};

use crate::{models::Card, State};

/// What a player has seen of hidden cards besides their own hand (Pokédex, Hiker, Morty, Hand
/// Scope, Looker, Misdreavus's Infiltrating Inspection), so that resampling the hidden
/// information from that player's point of view (`determinize`) stays consistent with it.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Knowledge {
    /// Per deck owner: the deck as it was when its top `usize` cards were seen. Those cards stay
    /// known while that deck only loses cards from its top; anything else (a shuffle, a card put
    /// into it) makes the order unknown again.
    pub top_of_deck: [Option<(Vec<Card>, usize)>; 2],
    /// Cards seen in the opponent's hand that they still hold.
    pub opponent_hand: Vec<Card>,
    /// Looker: the opponent's hand holds no Supporter besides the ones in `opponent_hand`. Holds
    /// until they draw.
    pub opponent_supporters_known: bool,
}

impl State {
    /// The top cards of `owner`'s deck that `knower` knows, in draw order.
    pub(crate) fn known_top_of_deck(&self, knower: usize, owner: usize) -> &[Card] {
        let deck = &self.decks[owner].cards;
        let Some((seen, known)) = &self.knowledge[knower].top_of_deck[owner] else {
            return &[];
        };
        match seen.len().checked_sub(deck.len()) {
            Some(drawn) if seen[drawn..] == deck[..] => &deck[..known.saturating_sub(drawn)],
            _ => &[],
        }
    }

    /// `knower` looks at the top `count` cards of `owner`'s deck.
    pub(crate) fn see_top_of_deck(&mut self, knower: usize, owner: usize, count: usize) {
        let known = self.known_top_of_deck(knower, owner).len();
        let deck = self.decks[owner].cards.clone();
        let count = count.min(deck.len()).max(known);
        self.knowledge[knower].top_of_deck[owner] = (count > 0).then_some((deck, count));
    }

    /// `knower` sees the cards of the opponent's hand that `revealed` selects.
    pub(crate) fn see_opponent_hand(&mut self, knower: usize, revealed: impl Fn(&Card) -> bool) {
        let mut seen = self.known_opponent_hand(knower);
        let mut rest = self.hands[(knower + 1) % 2].clone();
        for card in &seen {
            if let Some(at) = rest.iter().position(|c| c == card) {
                rest.swap_remove(at);
            }
        }
        seen.extend(rest.into_iter().filter(|card| revealed(card)));
        self.knowledge[knower].opponent_hand = seen;
    }

    /// `knower` sees one card of the opponent's hand. A copy of a card they already know to be
    /// there tells them nothing new: they cannot tell which copy it was.
    pub(crate) fn see_opponent_card(&mut self, knower: usize, card: &Card) {
        let mut known = self.known_opponent_hand(knower);
        if !known.contains(card) {
            known.push(card.clone());
        }
        self.knowledge[knower].opponent_hand = known;
    }

    /// The cards `knower` has seen in the opponent's hand that are still there.
    pub(crate) fn known_opponent_hand(&self, knower: usize) -> Vec<Card> {
        if self.knowledge[knower].opponent_hand.is_empty() {
            return vec![];
        }
        let mut hand = self.hands[(knower + 1) % 2].clone();
        let mut known = vec![];
        for card in &self.knowledge[knower].opponent_hand {
            if let Some(at) = hand.iter().position(|c| c == card) {
                known.push(hand.swap_remove(at));
            }
        }
        known
    }

    /// `owner` is about to draw the top card of their deck: the other player learns it if they
    /// knew it was there, and no longer knows which Supporters that hand holds.
    pub(crate) fn note_draw(&mut self, owner: usize) {
        let knower = (owner + 1) % 2;
        if let Some(card) = self.known_top_of_deck(knower, owner).first().cloned() {
            self.knowledge[knower].opponent_hand.push(card);
        }
        self.knowledge[knower].opponent_supporters_known = false;
    }

    /// `card` left `owner`'s hand in plain sight.
    pub(crate) fn note_left_hand(&mut self, owner: usize, card: &Card) {
        let known = &mut self.knowledge[(owner + 1) % 2].opponent_hand;
        if let Some(at) = known.iter().position(|c| c == card) {
            known.swap_remove(at);
        }
    }

    /// `owner`'s whole hand went back into their deck: nothing is known about it anymore.
    pub(crate) fn note_hand_shuffled_away(&mut self, owner: usize) {
        let knowledge = &mut self.knowledge[(owner + 1) % 2];
        knowledge.opponent_hand.clear();
        knowledge.opponent_supporters_known = false;
    }
}
