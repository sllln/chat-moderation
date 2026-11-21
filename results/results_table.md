Test set: OffensEval-TR 2020 official test split (3528 tweets). Latency: mean over 200 single-message runs. Hardware: 11th Gen Intel(R) Core(TM) i7-11370H @ 3.30GHz, CPU only, 4 PyTorch threads.

| Model | Numeric precision | Macro-F1 | Size (MB) | Latency mean (ms) | Latency p95 (ms) | <= 50 ms |
|---|---|---|---|---|---|---|
| Lexicon + edit distance | rules | 0.713 | 0.1 | 10.11 | 26.78 | yes |
| tf-idf + logistic regression | FP64 | 0.769 | 4.4 | 1.37 | 1.73 | yes |
| word2vec + feedforward NN | FP32 | 0.663 | 23.3 | 0.23 | 0.36 | yes |
| BERTurk | FP32 | 0.813 | 422.0 | 31.58 | 44.15 | yes |
| BERTurk (quantized) | INT8 (dynamic) | 0.810 | 177.4 | 17.55 | 25.11 | yes |
