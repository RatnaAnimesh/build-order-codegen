"""Chat generation for Stage A: vLLM on Kaggle's two T4s, or a scripted stand-in for the CPU dry run.

generate(reqs) takes [(messages, n, max_tokens)] and returns ([[(text, generated tokens)] * n per request],
[prompt tokens per request]). Sampling is DecompRL's evaluation setting: temperature 1.0, top-p 1.0.
A request whose prompt plus max_tokens exceeds the context returns n empty replies with 0 tokens (counted as
`too_long`), so one long prompt can't fail a whole batch.
"""
import time


class VLLMChat:
    def __init__(self, model, tp=2, max_model_len=8192, seed=0, gpu_mem=0.90, max_num_seqs=128, log=print):
        from vllm import LLM, SamplingParams
        self.SP, self.max_len, self.log = SamplingParams, max_model_len, log
        kw = dict(model=model, tensor_parallel_size=tp, dtype="half", max_model_len=max_model_len, seed=seed,
                  gpu_memory_utilization=gpu_mem, max_num_seqs=max_num_seqs, disable_custom_all_reduce=True)
        try:
            self.llm = LLM(enable_prefix_caching=True, **kw)
        except Exception as e:                            # prefix caching only saves prefill; don't lose the run
            log(f"[llm] prefix caching failed ({type(e).__name__}: {str(e)[:200]}); retrying without it")
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
            self.llm = LLM(enable_prefix_caching=False, **kw)
        self.tok = self.llm.get_tokenizer()
        self.gen_tokens = self.prompt_tokens = self.too_long = self.calls = 0
        self.seconds = 0.0

    def generate(self, reqs):
        ids = [self.tok.apply_chat_template(m, tokenize=True, add_generation_prompt=True) for m, _, _ in reqs]
        ok = [len(i) + mt <= self.max_len for i, (_, _, mt) in zip(ids, reqs)]
        sub = [{"prompt_token_ids": i} for i, k in zip(ids, ok) if k]
        sps = [self.SP(n=n, temperature=1.0, top_p=1.0, max_tokens=mt) for (_, n, mt), k in zip(reqs, ok) if k]
        t0 = time.time()
        outs = iter(self.llm.generate(sub, sps, use_tqdm=False) if sub else [])
        dt = time.time() - t0
        res, gen = [], 0
        for (m, n, mt), k in zip(reqs, ok):
            if not k:
                res.append([("", 0)] * n)
                self.too_long += 1
                continue
            o = next(outs)
            res.append([(c.text, len(c.token_ids)) for c in o.outputs])
            gen += sum(len(c.token_ids) for c in o.outputs)
        self.calls += 1
        self.seconds += dt
        self.gen_tokens += gen
        self.prompt_tokens += sum(len(i) for i, k in zip(ids, ok) if k)
        self.log(f"[llm] {len(reqs)} requests, {sum(n for _, n, _ in reqs)} seqs, {gen} tokens in {dt:.0f}s "
                 f"({gen / max(dt, 1e-9):.0f} tok/s); total {self.gen_tokens} in {self.seconds / 60:.1f} min")
        return res, [len(i) for i in ids]


def preflight(model=None):
    """Import what vLLM's engine imports, build Triton's CUDA stub and, given a model, run the chat template through
    vLLM's tokenizer wrapper, so a broken environment fails in seconds instead of during engine start."""
    import torch, transformers, triton, vllm, xformers
    import vllm.attention.backends.xformers, vllm.model_executor.models.qwen2  # noqa: F401
    from triton.runtime.driver import driver
    print("torch", torch.__version__, "gpus", torch.cuda.device_count(), "transformers", transformers.__version__,
          "vllm", vllm.__version__, "xformers", xformers.__version__, "triton", triton.__version__,
          driver.active.get_current_target(), flush=True)
    if model:
        from vllm.transformers_utils.tokenizer import get_tokenizer
        ids = get_tokenizer(model).apply_chat_template([{"role": "user", "content": "hi"}], tokenize=True,
                                                       add_generation_prompt=True)
        print("chat template ok:", len(ids), "tokens", flush=True)


class ScriptedChat:
    """Stand-in model: reply(messages, j) -> text. Tokens are counted as whitespace-separated words."""

    def __init__(self, reply):
        self.reply = reply
        self.gen_tokens = self.prompt_tokens = self.too_long = self.calls = 0
        self.seconds = 0.0

    def generate(self, reqs):
        res = [[(t, len(t.split())) for t in (self.reply(m, j) for j in range(n))] for m, n, _ in reqs]
        self.calls += 1
        self.gen_tokens += sum(nt for r in res for _, nt in r)
        lens = [sum(len(x["content"].split()) for x in m) for m, _, _ in reqs]
        self.prompt_tokens += sum(lens)
        return res, lens
