"""Stage B training: one LoRA per arm on its own passing samples (README "Stage B, round 1"). GPU, on Kaggle.

    python train_lora.py --model /tmp/qwen7b --data train_idea.jsonl.gz --out /kaggle/working/lora_idea
    python train_lora.py --merge --model /tmp/qwen7b --adapter lora_idea --out /tmp/qwen_idea   # fp16, for vLLM

Fixed before any Stage B sample: the first 4 passing attempts per problem; shuffled passes over their examples until
--budget tokens (prompt and reply, 1M) or 2 passes, whichever comes first, so every arm gets the same training
compute at most. The base in 4 bits (QLoRA: nf4, double quantization, fp16 compute; T4s have no bf16), LoRA rank 16,
alpha 32, dropout 0.05 on every linear projection, AdamW at 2e-4 with 10 warm-up steps then cosine to 0, 8 examples
per step, gradient norm clipped at 1, loss on the reply tokens only (the end-of-turn token included). The prompt is
the chat template exactly as vLLM applied it when sampling. An example longer than --max_len is skipped and counted.
Every arm stops at --max_hours of training (1.8) even if its budget isn't used up, and the adapter is saved
either way, so a slow GPU cuts all arms alike. <out>/train_stats.json: examples, tokens, steps, seconds, loss.
"""
import argparse, gzip, json, math, random, time
from collections import Counter

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


def examples(path, tok, per_problem=4, max_len=8192):
    """-> [(token ids, prompt length)], skipped, problems. Raises if the template doesn't end the prompt cleanly."""
    seen, out, skipped = Counter(), [], 0
    for line in gzip.open(path, "rt"):
        rec = json.loads(line)
        if seen[rec["id"]] >= per_problem:
            continue
        seen[rec["id"]] += 1
        for e in rec["examples"]:
            prompt = tok.apply_chat_template(e["messages"], add_generation_prompt=True, tokenize=True)
            full = tok.apply_chat_template(e["messages"] + [{"role": "assistant", "content": e["reply"]}],
                                           tokenize=True)
            if full[:len(prompt)] != prompt or not tok.decode(full[len(prompt):]).startswith(e["reply"].strip()[:50]):
                raise ValueError(f"{rec['id']}: the reply tokens don't follow the sampling prompt")
            if len(full) > max_len:
                skipped += 1
                continue
            out.append((full, len(prompt)))
    return out, skipped, len(seen)


def schedule(data, budget, passes=2, seed=0):
    """Example indices in training order: shuffled passes until `budget` tokens or `passes` passes."""
    rng, order, used = random.Random(seed), [], 0
    for _ in range(passes):
        idx = list(range(len(data)))
        rng.shuffle(idx)
        for i in idx:
            if used + len(data[i][0]) > budget:
                return order, used
            order.append(i)
            used += len(data[i][0])
    return order, used


def train(a):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    data, skipped, problems = examples(a.data, tok, max_len=a.max_len)
    order, tokens = schedule(data, a.budget)
    steps = math.ceil(len(order) / a.batch)
    print(f"{problems} problems, {len(data)} examples ({skipped} too long); {len(order)} in training, {tokens} tokens, "
          f"{steps} steps", flush=True)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=q, torch_dtype=torch.float16,
                                                 device_map={"": 0}, attn_implementation="sdpa")
    model = get_peft_model(prepare_model_for_kbit_training(model, use_gradient_checkpointing=True), LoraConfig(**LORA))
    base = model.get_base_model()                         # logits only where a reply token is predicted
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    losses, t1, done = [], time.time(), 0
    for s in range(steps):
        if time.time() - t1 > a.max_hours * 3600:
            print(f"stopped at --max_hours after {s} of {steps} steps", flush=True)
            break
        done = s + 1
        batch = order[s * a.batch:(s + 1) * a.batch]
        total = 0.0
        for i in batch:
            ids, n = data[i]
            x = torch.tensor([ids], device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                h = base.model(input_ids=x).last_hidden_state[0, n - 1:-1]
                logits = base.lm_head(h)
            loss = torch.nn.functional.cross_entropy(logits.float(), x[0, n:]) / len(batch)
            scaler.scale(loss).backward()
            total += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        losses.append(total)
        if s % 5 == 0 or s == steps - 1:
            el = time.time() - t1
            print(f"step {s + 1}/{steps} loss {total:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{sum(len(data[i][0]) for i in order[:(s + 1) * a.batch]) / el:.0f} tok/s", flush=True)
    model.save_pretrained(a.out)
    seen = order[:done * a.batch]
    stats = {"problems": problems, "examples": len(data), "too_long": skipped, "trained_examples": len(seen),
             "tokens": sum(len(data[i][0]) for i in seen), "planned_tokens": tokens,
             "reply_tokens": sum(len(data[i][0]) - data[i][1] for i in seen), "steps": done, "planned_steps": steps,
             "train_seconds": time.time() - t1, "total_seconds": time.time() - t0, "loss_first": losses[:5],
             "loss_last": losses[-5:], "settings": {**{k: v for k, v in LORA.items() if k != "task_type"},
                                                    "lr": a.lr, "batch": a.batch, "budget": a.budget}}
    json.dump(stats, open(f"{a.out}/train_stats.json", "w"), indent=1)
    print(json.dumps(stats), flush=True)


def merge(a):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map="cpu")
    PeftModel.from_pretrained(m, a.adapter).merge_and_unload().save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
    print("merged into", a.out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--adapter")
    ap.add_argument("--budget", type=int, default=1_000_000)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_hours", type=float, default=1.8)
    a = ap.parse_args()
    merge(a) if a.merge else train(a)
