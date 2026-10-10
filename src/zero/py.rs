//! Python access to batched self-play, so the network can run on GPU (see zero/gpu_selfplay.py).
//! Arrays cross the boundary as little-endian bytes; Python wraps them with numpy.frombuffer.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::batch::BatchSelfPlay;
use super::mcts::MctsConfig;
use super::selfplay::SelfPlayConfig;
use crate::Deck;

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[pyclass(name = "BatchSelfPlay")]
pub struct PyBatchSelfPlay {
    inner: BatchSelfPlay,
}

#[pymethods]
impl PyBatchSelfPlay {
    #[new]
    #[pyo3(signature = (deck_paths, games, sims, out, seed=0, concurrency=1024, temperature_moves=20, noise=0.25, games_log=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        deck_paths: Vec<String>,
        games: usize,
        sims: usize,
        out: String,
        seed: u64,
        concurrency: usize,
        temperature_moves: usize,
        noise: f32,
        games_log: Option<String>,
    ) -> PyResult<Self> {
        let (decks, names): (Vec<Deck>, Vec<String>) = deck_paths
            .iter()
            .filter_map(|p| {
                let d = Deck::from_file(p).ok().filter(|d| d.is_valid())?;
                let name = std::path::Path::new(p)
                    .file_stem()
                    .map_or_else(|| p.clone(), |s| s.to_string_lossy().into_owned());
                Some((d, name))
            })
            .unzip();
        if decks.is_empty() {
            return Err(PyRuntimeError::new_err("no valid deck"));
        }
        let cfg = SelfPlayConfig {
            mcts: MctsConfig {
                simulations: sims,
                noise_frac: noise,
                gumbel_noise: noise > 0.0,
                ..MctsConfig::default()
            },
            temperature_moves,
        };
        let inner = BatchSelfPlay::new(
            cfg,
            decks,
            names,
            games,
            concurrency,
            seed,
            &out,
            games_log.as_deref(),
        )
        .map_err(|e| PyRuntimeError::new_err(format!("{out}: {e}")))?;
        Ok(PyBatchSelfPlay { inner })
    }

    fn done(&self) -> bool {
        self.inner.done()
    }

    /// (games_done, games_target, samples_written, first_player_wins, ties)
    fn progress(&self) -> (usize, usize, usize, usize, usize) {
        self.inner.progress()
    }

    /// Returns (n_requests, dense, tokens, action_dense, action_tokens, owner) as bytes.
    #[allow(clippy::type_complexity)]
    fn collect<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        usize,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
    )> {
        let inner = &mut self.inner;
        let b = py
            .detach(|| inner.collect())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            b.n,
            PyBytes::new(py, &bytes_f32(&b.dense)),
            PyBytes::new(py, &bytes_i32(&b.tokens)),
            PyBytes::new(py, &bytes_f32(&b.action_dense)),
            PyBytes::new(py, &bytes_i32(&b.action_tokens)),
            PyBytes::new(py, &bytes_i32(&b.owner)),
        ))
    }

    /// values: one f32 per request, logits: one f32 per action (little-endian bytes).
    fn respond(&mut self, py: Python<'_>, values: &[u8], logits: &[u8]) {
        let (v, l) = (f32s(values), f32s(logits));
        let inner = &mut self.inner;
        py.detach(|| inner.respond(&v, &l));
    }
}
