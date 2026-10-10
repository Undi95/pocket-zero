//! pocket-zero: an AlphaZero-style player for this engine.
//!
//! - `encode`: what a player can see, as network inputs.
//! - `net`: the value/policy network (trained in `zero/` with PyTorch, evaluated here).
//! - `mcts`: search with hidden information and chance.
//! - `selfplay`: games of the network against itself, written as training samples.
//! - `player`: the `z` player code, usable anywhere the engine takes a player.

pub mod batch;
pub mod encode;
pub mod mcts;
pub mod net;
mod player;
#[cfg(feature = "python")]
pub mod py;
pub mod selfplay;

pub use player::{load_net, NetSearchPlayer, ZeroPlayer};
