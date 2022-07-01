use crossbeam;
use ndarray::{ArrayView2, ArrayView3, Axis};
use std::collections::HashMap;

#[derive(Clone)]
struct Beam {
    p_blank: f32,
    p_non_blank: f32,
    p_tot: f32,
    label: Vec<usize>,
}

impl Beam {
    fn new() -> Beam {
        Beam {
            p_blank: 1.,
            p_non_blank: 0.,
            p_tot: 1.,
            label: Vec::new(),
        }
    }
}

fn sort(registry: &HashMap<Vec<usize>, Beam>) -> Vec<Beam> {
    let mut beams: Vec<Beam> = registry.values().cloned().collect();
    beams.sort_unstable_by(|a, b| b.p_tot.partial_cmp(&a.p_tot).unwrap());
    beams
}

pub fn decode(
    probs: ArrayView2<f32>,
    alphabet: &[String],
    beam_width: usize,
    topk_paths: usize,
    charset: &Vec<String>,
) -> Vec<(String, f32)> {
    let mut last: HashMap<Vec<usize>, Beam> = HashMap::new();
    let root_beam = Beam::new();
    let key: Vec<usize> = root_beam.label.clone();
    last.insert(key, root_beam);
    let blank_idx = alphabet.len();
    for p in probs.axis_iter(Axis(0)) {
        let mut curr: HashMap<Vec<usize>, Beam> = HashMap::new();
        for beam in sort(&last).iter().take(beam_width) {
            let p_non_blank = if !beam.label.is_empty() {
                beam.p_non_blank * p[*beam.label.last().unwrap()]
            } else {
                0.
            };
            let p_blank = beam.p_tot * p[blank_idx];
            let p_tot = p_blank + p_non_blank;

            match curr.get_mut(&beam.label) {
                Some(beam) => {
                    beam.p_blank += p_blank;
                    beam.p_non_blank += p_non_blank;
                    beam.p_tot += p_tot
                }
                None => {
                    let new_beam = Beam {
                        p_blank,
                        p_non_blank,
                        p_tot: p_tot,
                        label: beam.label.clone(),
                    };
                    curr.insert(beam.label.clone(), new_beam);
                }
            }

            let new_idxs = match charset.get(beam.label.len()) {
                Some(characters) => {
                    let new_idxs: Vec<usize> = (0..alphabet.len())
                        .filter(|&i| characters.contains(&alphabet[i]))
                        .collect();
                    new_idxs
                }
                None => {
                    let new_idxs: Vec<usize> = (0..alphabet.len()).collect();
                    new_idxs
                }
            };
            for i in new_idxs {
                let p_non_blank = if !beam.label.is_empty() && *beam.label.last().unwrap() == i {
                    let p_non_blank = last.get(&beam.label).unwrap().p_blank * p[i];
                    p_non_blank
                } else {
                    let p_non_blank = last.get(&beam.label).unwrap().p_tot * p[i];
                    p_non_blank
                };

                let mut label = beam.label.clone();
                label.push(i);

                match curr.get_mut(&label) {
                    Some(beam) => {
                        beam.p_non_blank += p_non_blank;
                        beam.p_tot += p_non_blank;
                    }
                    None => {
                        let new_beam = Beam {
                            p_blank: 0.,
                            p_non_blank,
                            p_tot: p_non_blank,
                            label: label.clone(),
                        };
                        curr.insert(label.clone(), new_beam);
                    }
                }
            }
        }
        last = curr;
    }

    let mut predictions: Vec<(String, f32)> = Vec::new();
    for beam in sort(&last).iter().take(topk_paths) {
        let mut word = String::new();
        for &i in beam.label.iter() {
            word.push_str(&alphabet[i]);
        }
        predictions.push((word, beam.p_tot));
    }
    predictions
}

pub fn decode_batch(
    probs: ArrayView3<f32>,
    alphabet: &[String],
    beam_width: usize,
    topk_paths: usize,
    charset: &Vec<String>,
) -> Vec<Vec<(String, f32)>> {
    let mut predictions: Vec<Vec<(String, f32)>> = vec![];
    crossbeam::scope(|scope| {
        let mut handles = vec![];
        for example_probs in probs.axis_iter(Axis(0)) {
            handles.push(
                scope.spawn(move |_| {
                    decode(example_probs, alphabet, beam_width, topk_paths, charset)
                }),
            );
        }
        for handle in handles {
            predictions.push(handle.join().unwrap());
        }
    }).unwrap();
    predictions
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_decode() {
        let probs = array![
            [0.2, 0.0, 0.8],
            [0.4, 0.0, 0.6],
        ];
        let alphabet = vec![String::from("A"), String::from("B")];
        let beam_width = 5;
        let topk_paths = 2;
        let charset: Vec<String> = Vec::new();
        let predictions = decode(probs.view(), &alphabet, beam_width, topk_paths, &charset);
        let beam_1 = predictions.iter().nth(0).unwrap();
        let beam_2 = predictions.iter().nth(1).unwrap();
        assert_eq!(beam_1.0, "A");
        assert!((beam_1.1 - 0.52).abs() < 0.00001);
        assert_eq!(beam_2.0, "");
        assert!((beam_2.1 - 0.48).abs() < 0.00001);
    }
}