# Beam Search Decoding Algorithm

*Beam Search* decoder for *Text Recognition* Neural Networks trained with *Connectionist Temporal Classification* (CTC) loss. Python :snake: extension implemented in Rust :crab: for improved performances.

## Features
- **Format Specification**: if you need to decode strings with a given format like *IBANs*, *etc*..
- **Batch Support**: multithreaded decoding.

## Installation
- Install [Rust](https://www.rust-lang.org/tools/install)
- Install maturin `pip install maturin`
- Build Python wheel `maturin build --release`
- Install wheel `pip install ./target/wheels/<wheel>`

If you encounter problems with the wheel installation you may need to upgrade pip.

Optional
- Run tests `cargo test`
- Run benchmarks `cargo bench`


## Usage
```python
>>> import beam_search
>>> import numpy as np
>>> 
>>> alphabet = "ABC"
>>> logits = np.arange(5*len(alphabet)+1).reshape(5, 4)
>>> logits = logits.astype(np.float32)
>>> probs = logits/logits.sum(axis=-1, keepdims=True)
>>> beam_search.decode(probs, alphabet, topk_paths=3)
[('CB', 0.06815903633832932), ('CA', 0.06014531850814819),
('BC', 0.05005190148949623)]

>>> format_ = ["AB", "AC", "BC"]
>>> beam_search.decode(probs, alphabet, topk_paths=3, format=format_)
[('BC', 0.05005190148949623), ('BA', 0.03964773565530777),
('BCB', 0.02974703535437584)]

>>> logits_batch = np.arange(2*5*4).reshape(2,5,4)
>>> logits_batch = logits_batch.astype(np.float32)
>>> probs_batch = logits_batch/logits_batch.sum(axis=-1, keepdims=True)
>>> beam_search.decode_batch(probs_batch, alphabet, topk_paths=3)
[[('CB', 0.06815903633832932), ('CA', 0.06014531850814819),
('BC', 0.05005190148949623)],
[('CB', 0.03784028813242912), ('BC', 0.03726544231176376),
('CA', 0.03605256229639053)]]
```

## Acknowledgements
- Reference implementation [https://github.com/githubharald/CTCDecoder](https://github.com/githubharald/CTCDecoder)
- Reference for Python extension in Rust [https://github.com/nanoporetech/fast-ctc-decode](https://github.com/nanoporetech/fast-ctc-decode)