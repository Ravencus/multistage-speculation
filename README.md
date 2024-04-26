# multistage-speculation
evaluation and analysis of multistage speculation decoder, implemented based on hugging face transformers
## benchmark based on spec-bench: https://github.com/hemingkx/Spec-Bench


## TODO LIST

### Profiling

- logger.INFO: print candidate time, verifier time, candidate token length, accepted token length
  - need to attach model identifier(name/size) to log
- logger.INFO: need baseline gen_time for each prompt
    - baselines: tinyllama, llama-7b, llama-13b, vicuna-7B, 13B, 33B
- func def script_generator()
  - input args: --
  - output str
- Spec-Bench/evaluation/inference_assisted.py
  - encapsulate assisted_forward
  - eval() compatibility check
  - logger file handler
- Token Length control
  - disable scheduler for fix length baseline(exp scale settings: 1, 2, 4, 8, ...)
  - enable scheduler for comparison
- need to run multiple times to eliminate variation

- Data cropper: crop valid data from log file
- Data Analyzer: get critical metrics: acceptance rate; speed up, ...
- Comparator: compare candiate_time + verifier_time with timer from Spec-Bench

### Fusing

- generationMixin add methods
  - medusa_decode
  - lookahead_decode
  - https://github.com/Ravencus/transformers.git
  - (): medusa with lookahead


- Problems:
  - 1. KV cache management
    - Oracle will instruct candidate_generator to crop kv cache using _crop_past_key_values
    - may not be compatible with medusa heads or Lookahead decoding
  
  - 2. output len control:
    - input kwargs need to be compatible with scheduler: candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)

- Multistage:
  - 1. need manager class and code refactor
  - 2. need more advanced scheduler
  - 