use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array;
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use beam_search::core::{decode, decode_batch};


pub fn decode_benchmark(c: &mut Criterion) {
    let probs = black_box(
        Array::random((256,37), Uniform::<f32>::new(0., 1.)));
    let alphabet: Vec<String> = black_box(vec![
        "0","1","2","3","4","5","6","7","8","9",
        "A","B","C","D","E","F","G","H","I","J",
        "K","L","M","N","O","P","Q","R","S","T",
        "U","V","W","X","Y","Z"
    ].iter().map(|&c| c.to_string()).collect());
    let beam_width = black_box(10);
    let topk_paths = black_box(5);
    let charset: Vec<String> = black_box(Vec::new());
    c.bench_function(
        "decode 256x37",
        |b| b.iter(
            || decode(probs.view(), &alphabet, beam_width, topk_paths, &charset)));
}

pub fn decode_batch_benchmark(c: &mut Criterion) {
    let probs = black_box(
        Array::random((8,256,37), Uniform::<f32>::new(0., 1.)));
    let alphabet: Vec<String> = black_box(vec![
        "0","1","2","3","4","5","6","7","8","9",
        "A","B","C","D","E","F","G","H","I","J",
        "K","L","M","N","O","P","Q","R","S","T",
        "U","V","W","X","Y","Z"
    ].iter().map(|&c| c.to_string()).collect());
    let beam_width = black_box(10);
    let topk_paths = black_box(5);
    let charset: Vec<String> = black_box(Vec::new());
    c.bench_function(
        "decode_batch 8x256x37",
        |b| b.iter(
            || decode_batch(probs.view(), &alphabet, beam_width, topk_paths, &charset)));
}

criterion_group!(benches, decode_benchmark, decode_batch_benchmark);
criterion_main!(benches);