import copy
import torch
from tqdm import tqdm
from tensorrt_llm.evaluate.lm_eval import LmEvalWrapper, profiler, logger
from tensorrt_llm.sampling_params import SamplingParams


def generate_until(self, requests, disable_tqdm: bool = False):
    config["n_samples_actual"] = len(requests)
    profiler.start("trtllm exec")
    results = []
    for request in tqdm(requests, desc="Submitting requests", disable=disable_tqdm):
        prompt, gen_kwargs = request.args
        sampling_params = self._get_sampling_params(gen_kwargs)
        output = self.llm.generate_async(prompt, sampling_params=sampling_params, streaming=self.streaming)
        results.append(output)

    outputs = []
    for output in tqdm(results, desc="Fetching responses", disable=disable_tqdm):
        outputs.append(output.result())

    profiler.stop("trtllm exec")
    logger.info(f"TRTLLM execution time: {profiler.elapsed_time_in_sec('trtllm exec'):.3f} seconds.")
    profiler.reset("trtllm exec")

    texts = [output.outputs[0].text for output in outputs]
    n = min(config["print_outputs"], len(texts))
    for req, text in zip(requests[:n], texts[:n]):
        print(f"\n[DEBUG] Input: ...{req.args[0][-300:]}")
        print(f"[DEBUG] Output: {text}")
    return texts


def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
    """
    Full MMLU task will crash on certain hardware (HPC will work, allocate a lot of ram). Divide the tasks as much as possible for stable evaluation. Making big requests with gather logits enabled
    seems to have some kind of memory accumulation in the engine.
    """
    # Count unique prompts to deduplicate MC choices (e.g. 4-choice MMLU: len/4 = num docs)
    config["n_samples_actual"] = len(set(tuple(r[1]) for r in requests))
    profiler.start("trtllm exec")

    sp = copy.deepcopy(self.sampling_params) if self.sampling_params else SamplingParams()
    sp.max_tokens = 1
    sp.temperature = 0.0
    sp.return_context_logits = True

    results = []
    for _, prompt_tokens, target_tokens in tqdm(requests, desc="Processing requests", disable=disable_tqdm):
        output = self.llm.generate(prompt_tokens + target_tokens[:-1], sampling_params=sp, use_tqdm=False).result()
        start = len(prompt_tokens) - 1
        logits_f = output.context_logits[start:start + len(target_tokens)].clone()

        target = torch.tensor(target_tokens, device=logits_f.device)
        # log-softmax of the correct token at each position, summed as the overall sequence score
        logprob_sum = float((logits_f[torch.arange(len(target_tokens)), target] - torch.logsumexp(logits_f, dim=-1)).sum())
        # true if the model's top-1 prediction matched the target at every position
        is_greedy = bool((logits_f.argmax(dim=-1) == target).all())
        results.append((logprob_sum, is_greedy))

    profiler.stop("trtllm exec")
    logger.info(f"TRTLLM execution time: {profiler.elapsed_time_in_sec('trtllm exec'):.3f} seconds.")
    profiler.reset("trtllm exec")

    return results


# Perplexity tasks
def loglikelihood_rolling(self, requests, disable_tqdm=False):
    from lm_eval import utils as lm_utils

    max_len = 2048

    ppl_stats["logprob_sum"] = 0.0
    ppl_stats["num_tokens"] = 0

    config["n_samples_actual"] = len(requests)  # overwrite: 1 request = 1 doc for perplexity
    loglikelihoods = []
    for request in tqdm(requests, desc="Processing rolling loglikelihoods", disable=disable_tqdm):
        (string,) = request.args

        token_list = self.tok_encode(string)
        rolling_windows = list(map(
            lm_utils.make_disjoint_window,
            lm_utils.get_rolling_token_windows(
                token_list=token_list,
                prefix_token=self.prefix_token_id,
                max_seq_len=max_len,
                context_len=1,
            ),
        ))

        # _loglikelihood_tokens expects (_, prompt_tokens, target_tokens) tuples
        windows = [(None,) + w for w in rolling_windows]
        window_results = _loglikelihood_tokens(self, windows, disable_tqdm=True)
        doc_logprob = sum(lp for lp, _ in window_results)

        loglikelihoods.append(doc_logprob)
        ppl_stats["logprob_sum"] += doc_logprob
        ppl_stats["num_tokens"] += len(token_list)

    return loglikelihoods


LmEvalWrapper._loglikelihood_tokens = _loglikelihood_tokens
LmEvalWrapper.loglikelihood_rolling = loglikelihood_rolling
LmEvalWrapper.generate_until = generate_until

config = {
    "print_outputs": 0,
    "n_samples_actual": None,
}

# Accumulated during loglikelihood_rolling for token-level perplexity.
# Reset at the start of each rolling evaluation so re-runs are clean.
ppl_stats = {
    "logprob_sum": 0.0,
    "num_tokens": 0,
}


_CODE_EVAL_TASKS = {"humaneval", "mbpp"}

def patch_evaluator(task: str):
    import lm_eval.evaluator as _lm_evaluator
    _orig_eval = _lm_evaluator.evaluate
    if task in _CODE_EVAL_TASKS:
        _lm_evaluator.evaluate = lambda *a, **kw: _orig_eval(*a, confirm_run_unsafe_code=True, log_samples=False, **kw)
