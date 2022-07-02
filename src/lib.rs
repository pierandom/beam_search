use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::{prelude::*, types::PySequence};

pub mod core;

#[pyfunction(beam_width="10", topk_paths="1", charset="None")]
fn decode(
    py: Python,
    probs: PyReadonlyArray2<f32>,
    alphabet: &PySequence,
    beam_width: usize,
    topk_paths: usize,
    charset: Option<&PySequence>,
) -> PyResult<Vec<(String, f32)>> {
    let probs = probs.as_array();
    let alphabet: Vec<String> = alphabet.tuple()?.iter().map(|x| x.to_string()).collect();
    let charset: Vec<String> = match charset {
        Some(cs) => cs.tuple()?.iter().map(|x| x.to_string()).collect(),
        None => Vec::new()
    };
    let predictions =
        py.allow_threads(move || core::decode(probs, &alphabet, beam_width, topk_paths, &charset));
    Ok(predictions)
}

#[pyfunction(beam_width="10", topk_paths="1", charset="None")]
fn decode_batch(
    py: Python,
    probs: PyReadonlyArray3<f32>,
    alphabet: &PySequence,
    beam_width: usize,
    topk_paths: usize,
    charset: Option<&PySequence>,
) -> PyResult<Vec<Vec<(String, f32)>>> {
    let probs = probs.as_array();
    let alphabet: Vec<String> = alphabet.tuple()?.iter().map(|x| x.to_string()).collect();
    let charset: Vec<String> = match charset {
        Some(cs) => cs.tuple()?.iter().map(|x| x.to_string()).collect(),
        None => Vec::new()
    };
    let predictions = py.allow_threads(move || {
        core::decode_batch(probs, &alphabet, beam_width, topk_paths, &charset)
    });
    Ok(predictions)
}

#[pymodule]
fn beam_search(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(decode_batch, m)?)?;
    Ok(())
}
