# Build order in hierarchical code generation

Code and run record for **Where Did the Tokens Go? Build Order in Hierarchical Code Generation, and a Blind Spot in
Token-Budget Evaluation** (Animesh Ratna, BITS Pilani, 2026). Paper: [`paper/main.pdf`](paper/main.pdf); arXiv link
to follow.

On the benchmark, model and metric of DecompRL (Meta FAIR, arXiv 2607.02390), with no training, the paper compares
three ways for Qwen2.5-7B-Instruct to spend a token budget on 279 LiveCodeBench problems: sampling whole programs,
DecompRL's inference (split the problem into functions, write each one independently from stubs), and an ordered
variant (build the simplest functions first and show each later function the finished code of the ones it calls).
The code uses shorter names than the paper:

| paper | code (`arms.py`) |
|---|---|
| Plain | `plain` |
| Independent (DecompRL's inference) | `decomp` |
| Ordered (ours) | `idea` |

## What's here

- `prompts.py`, `decomp.py`, `arms.py`: the three methods (DecompRL's prompts are verbatim from its Appendix I).
- `llm.py`: the vLLM wrapper. `lcb_data.py`, `lcb_exec.py`: the problems and the sandboxed grader.
- `run_stage_a.py`, `report.py`, `rescore.py`: the main run and its analysis.
- `cc_data.py`, `collect.py`, `train_lora.py`, `dev_report.py`: the self-training round (in progress).
- `kaggle_*.py`: the Kaggle notebook jobs (2x T4); `kaggle_watch.py` pushed them from a laptop.
- `tests_qwen.py`: tests, with hand-written programs only.
- `results/stage_a/`: the main run. `state.json` holds every attempt's token counts, generation time and test
  results (not the program text); `report.md` is the pre-registered report written at the end of the run.
- `paper/`: LaTeX source, figures and PDF.

`python3 rescore.py results/stage_a/state.json` recomputes the paper's re-scoring and GPU-time tables.

The rest of this README is the lab log, written as the work went: each dated decision was fixed before the runs it governs.

## Lab log: Qwen2.5-7B, ordered decomposition against DecompRL

The base is Qwen/Qwen2.5-7B-Instruct, the model DecompRL reports on.

## Why "equal compute"

One DecompRL RL run took 80 H100s for 2-3 days (Appendix H), about 4,000-6,000 H100-hours, and the method needs
two (decomposer, then implementer). Kaggle gives about 30 T4-hours a week, roughly 3 H100-hours. Training the idea
on that and comparing with their trained numbers would only measure who had more compute. So both methods run on
the same base, the same problems, the same metric and the same budget:

- **Stage A (this code), no training:** plain sampling vs DecompRL's hierarchical inference vs the idea, at matched
  generated tokens. Their Table 4 has the untrained point for two of these (standard 14.0%, hierarchical no-RL
  11.0% at about 4k tokens), and Table 1 has the untrained instruct curve, so our harness can be checked against theirs.
- **Stage B:** LoRA self-training per format on Qwen's own test-passing samples from CodeContests training
  problems (never LCB), equal collection GPU time per arm, evaluated on a dev set first ("Stage B, round 1").

## Protocol (DecompRL's, where they state it)

| | DecompRL paper | here |
|---|---|---|
| problems | LiveCodeBench 2024-08-01..2025-02-01, 279 | the same window from `livecodebench/code_generation_lite` |
| sampling | temperature 1.0, top-p 1.0 | the same |
| metric | pass@tokens, generated tokens | the same; prompt tokens reported too |
| decomposition | ≤ 6 functions, 8 implementations each | the same; > 6 fails the attempt (both arms) |
| evaluations | up to 4,096 combinations per decomposition | **128** (4 CPUs here vs their CPU cluster), same cap both arms |
| time limit | 2T+1 s for T tests | 2T+1 **CPU** s (vLLM shares the 4 vCPUs) + wall caps 4 s/public test, 12 s/private test |
| grading | sandbox like Li et al. 2022 | lcb_runner's comparison rules, forked sandboxed child per program |
| prompts | Appendix I | verbatim; `{context}` gets the LeetCode starter code or a stdin note (not stated in the paper) |
| precision | not stated (H100, likely bf16) | fp16 (T4 has no bf16), vLLM 0.8.5.post1 V0, tensor parallel 2 |
| max tokens | not stated | 2,048 plain and decomposition, 1,024 per implementation |

## The three arms (`arms.py`)

- **plain:** LCB's own prompt, sampled up to what a 10k-token budget buys (the paper's 1k-10k points and the
  4k comparison; going to 50k would cost about 4 more GPU-hours).
- **decomp:** DecompRL's prompts. Every function is implemented independently from the stubs, 8 each. Programs are
  random combinations.
- **idea:** the same prompts plus the idea's parts. Each docstring gets a `Uses:` line and functions come basics
  first. Functions are built level by level. Each of a function's 8 samples sees one random already-built
  implementation of each function it uses. It may reply `MISSING: def …`; that helper is built first and then the
  function is asked again. A checker also builds helpers for calls to functions nobody defined. Programs: first the
  8 coherent ones (the top function with exactly what it was shown), then random combinations, the same 128 cap.

## Budget rule and schedule

pass@tokens follows Appendix K: a budget B buys k = max(1, floor(B / c)) attempts, c being the arm's mean generated
tokens per attempt, and a problem scores the unbiased pass@k of its graded attempts. The paper doesn't state the
rounding; floor with at least one attempt fits Table 1, whose DecompRL row (4,000 tokens per attempt) is flat at
0.18 from 1k to 5k. The stricter exact-token rule (an attempt counts once all its tokens fit) is reported beside it
as a check.

Rounds raise the budget per problem: 5k, 10k, 20k, 30k, 40k, 50k. In each round every 48-problem chunk is topped up,
arm by arm, to the k that budget buys at the tokens per attempt measured so far, plain only up to 10k. After the
last round, coverage passes top up any problem left short when that mean moved. At the pilot's costs (plain 207,
decomp 14.0k, idea 5.8k) each problem gets 48 plain, 3 decomp and 8 idea attempts, about 98k generated tokens, so
27M for the 279 problems: 10 GPU-hours at 750 tok/s, 15 at the pilot's 512. A session that runs out stops before a
chunk-round whose predicted time doesn't fit, so every problem stays at the same budget, give or take one chunk.

## Decisions fixed before the full run

Set on 2026-11-11, after the 12-problem pilot and before any full-run sample. Two changes from the first version,
both made to follow the paper's protocol and give each arm the same tokens, not because of which arm led: the budget
rule is Appendix K's (the exact-token rule had read DecompRL as 0 below 14k), and plain stops at 10k. The pilot's
samples are not pooled with the full run.

1. **Harness check:** plain within 0.05 of Table 1's instruct row at 5k and 10k tokens (0.16, 0.21). If it fails,
   the absolute numbers aren't comparable with the paper's; the paired comparisons below still stand.
2. **Primary:** idea − decomp at the largest budget both decomposition arms cover (≥ 95% of problems have the k
   attempts it buys, graded), paired bootstrap over problems, 95% CI. The idea's inference beats DecompRL's
   inference only if the CI is above 0. The schedule aims at 50k.
3. **Secondary:** idea − plain and decomp − plain at the largest budget all three cover (at most 10k); decomp −
   plain at 4k against Table 4's −3 points. The same idea − decomp under the exact-token rule, as a check.
4. A run cut by time is reported at the budget it covers. Nothing is re-run to change a result.

**Change on 2026-11-18, after session 1's report and before session 2's:** session 1 generated at 456 tok/s, not
the 512-750 planned for. It covered 30k for decomp and idea, and the 13.5 h cap would have stopped the run short of
40k. We added a session 3 and raised `PLAN_HOURS` from 13.5 to 21, so the run reaches the 40k and 50k rounds
the schedule aims at. Nothing else changed. Rule 2 still picks the budget: if 40k or 50k gets covered, the primary
is reported there, whatever it shows, and the 30k number from session 1 is not the result.

## Stage A result (2026-11-18)

Three sessions: 9.67 + 2.61 + 3.78 h, every round done, all 279 problems paired, grader left nothing pending.
24.7M generated and 101.4M prompt tokens. Tokens per attempt: plain 260, decomp 17,032, idea 5,573 generated
(prompt: 33, 5,980, 43,779). Decomp and idea cover 50k; all three arms cover 10k. Report:
`results/stage_a/report.md`.

| tokens | plain | decomp | idea | k plain/decomp/idea |
|---|---|---|---|---|
| 1k-10k | 0.183-0.312 | 0.113 | 0.095 | 3-38 / 1 / 1 |
| 20k | | 0.113 | 0.173 | - / 1 / 3 |
| 30k | | 0.113 | 0.208 | - / 1 / 5 |
| 40k | | 0.165 | 0.229 | - / 2 / 7 |
| 50k | | 0.165 | 0.237 | - / 2 / 8 |

1. **Harness check fails.** Plain is 0.183 / 0.283 / 0.312 at 1k / 5k / 10k against Table 1's 0.06 / 0.16 / 0.21.
   The absolute numbers aren't comparable with the paper's. The gap is in plain: one decomp attempt scores 0.113
   against Table 4's 11.0% for untrained hierarchical inference.
2. **Primary passes.** idea − decomp at 50k = **+0.072 [+0.032, +0.111]** (8 idea attempts against 2 decomp).
3. **Secondary.** At 10k, idea − plain = −0.217 [−0.261, −0.173] and decomp − plain = −0.199 [−0.247, −0.152].
   decomp − plain at 4k is −0.159 [−0.202, −0.119], not Table 4's −3 points, because plain is higher here.
   Exact-token check: idea − decomp at 10k = +0.115 [+0.085, +0.147].

Read with it, none of it changing the rules above:

- One attempt each, the idea is no better than decomp: −0.018 [−0.041, +0.004] at ≤10k. The win at 50k comes from
  attempts costing a third of the generated tokens, so the idea gets 8 where decomp gets 2. Both write about 29
  implementations per attempt; the idea's are about 190 tokens against decomp's 580, because each one sees the
  bodies it uses instead of re-writing them. That moves the cost into the prompt: 43.8k prompt tokens per idea
  attempt against 6.0k. Counting prompt and generated tokens together (49.4k against 23.0k), a 50k budget buys 1
  idea attempt (0.095) and 2 decomp (0.165), and the order flips. The paper's metric, used here, counts only
  generated tokens.
- Plain at 10k (0.312) beats both decomposition arms at 50k. Untrained, decomposing is a worse use of tokens than
  sampling whole programs.
- 197 of the idea's 212 passing attempts passed with one of their 8 coherent programs, the first program built in
  the median; random combinations added little.
- At 50k by difficulty, idea against decomp: easy 0.731 vs 0.537 (67), medium 0.157 vs 0.112 (89), hard 0.024 vs 0
  (123).
- Failed decompositions: idea 287 of 2,232 attempts (149 over 6 functions, 138 with none), decomp 45 of 558.
  The idea wrote `Uses:` lines in 97% of its decompositions, with 2.4 levels on average.

## After Stage A: harness and compute checks (2026-11-18, exploratory)

`python3 -I rescore.py <state.json>` re-scores the saved attempts. None of this was fixed before the run, and none of
it changes the result above. Its "ours" row and its k=2 / k=3 cells reproduce `report.md`.

**Why the harness check fails.** The paper never says how its instruct row was computed, and no code is out. Plain
re-scored under other rules:

| plain | 1k | 5k | 10k | 50k |
|---|---|---|---|---|
| ours, 260 generated tokens per attempt | 0.183 | 0.283 | 0.312 | 0.312* |
| Table 3's 387 tokens per attempt | 0.160 | 0.260 | 0.295 | 0.312* |
| prompt counted in every attempt (1,496 per attempt) | 0.122 | 0.183 | 0.223 | 0.306 |
| LeetCode problems all failed | 0.117 | 0.176 | 0.194 | 0.194* |
| paper instruct | 0.06 | 0.16 | 0.21 | 0.30 |
| paper GRPO | 0.18 | 0.29 | 0.32 | 0.38 |

- The paper's stated rule (generated tokens, 387 per attempt) doesn't reproduce its row: our attempts pass more often.
  LeetCode (104) and AtCoder (175) problems score alike, so platform doesn't explain it.
- Counting the prompt in every attempt fits within 0.05 at 5k, 10k and 50k but not at 1k, and the paper says it
  counts generated tokens. Only the authors can settle it.
- Our untrained plain sits on their GRPO row at 1k-10k. Without their prompt and accounting, Table 1's absolute
  numbers can't be targeted; Table 4's untrained decomposition (11.0% for one attempt, ours 11.3%) is the one
  number that lines up.
- Table 4's standard 14.0% is the score at a 4k-token budget, not one attempt.

**GPU time instead of generated tokens.** Generation wall time per arm, prefill included (`timing` in `state.json`):
plain 0.4 s, decomp 26.8 s, idea 16.4 s per attempt. The idea generates at 340 tok/s against decomp's 636, because
it reads 7x the prompt and builds level by level in smaller batches. By GPU time an idea attempt is 1.6x cheaper,
not 3x.

| GPU-s per problem | plain | decomp | idea |
|---|---|---|---|
| 10 | 0.297 (k=26) | 0.113 (1) | 0.095 (1) |
| 40 | | 0.113 (1) | 0.143 (2) |
| 60 | | 0.165 (2) | 0.173 (3) |
| 80 | | 0.165 (2) | 0.193 (4) |

At 80 GPU-s, the most both decomposition arms cover: idea − decomp = +0.028 [−0.006, +0.062], a CI that spans 0.
Plain at 10 GPU-s beats both at 80. The idea's Stage A win holds on generated tokens, the paper's metric, but not
clearly on compute. Its batching is this code's; a better scheduler could close part of the speed gap.

**What beating DecompRL would take** (Table 1, Qwen2.5-7B, LiveCodeBench): above DecompRL's 0.18 / 0.18 / 0.25 /
0.40 / 0.44 / 0.48 at 1k / 5k / 10k / 50k / 100k / 500k, with a CI above it, in a harness that reproduces their
instruct row. GRPO beats DecompRL up to 10k; the paper claims its win only from 100k up.

**Dev set for everything that follows:** LiveCodeBench release_v6 (`test6.jsonl`, 134 MB) problems dated from
2025-02-02, none of the 279: `lcb_data.py --raw test6.jsonl --start 2025-02-02 --end 2100-01-01`, built with
Stage B's prep and checked against the 279's ids. All tuning happens there; a frozen method meets the 279 once.

## Stage B, round 1 (fixed 2026-11-18, before any Stage B sample)

One round of self-training per format, the cheap relative of DecompRL's RL: sample, keep what passes, fine-tune on
it. Every result is reported in generated tokens and in GPU time.

- **Training pool:** CodeContests train (`deepmind/code_contests`), Codeforces problems rated 800-1600, stdin,
  untranslated statements, with a Python 3 solution. A problem stays only if its first Python 3 solution passes our
  grader on its tests (public, private, then generated, the first 30 under 1 MB each). A seeded 1,500 of those (`cc_data.py`). The
  newest is from 2021, so none can be a LiveCodeBench problem.
- **Collection** (`collect.py`, untrained Qwen, Stage A's prompts and settings): every arm gets the same
  generation GPU time, 1.5 h, measured as Stage A's `timing`. Arms take turns chunk by chunk, the one with the
  least time used going next, until each has 1.5 h; one attempt per problem per pass through the shuffled pool
  (plain 4, since its attempts are tiny).
- **Training examples:** passing attempts only, the first 4 per problem and arm. Plain: its prompt and reply.
  decomp and idea: the decomposition prompt and reply, plus, for each function in the first passing program, the
  prompt that implementation was sampled with and its reply. The idea's helper replies (MISSING, checker) are left
  out this round.
- **Training** (`train_lora.py`): one LoRA per arm on its own examples, the same settings for all: 4-bit base
  (QLoRA, the T4s have no room for fp16 training), rank 16, alpha 32, dropout 0.05, every linear projection,
  learning rate 2e-4 cosine, 8 examples per step, loss on replies only. Every arm gets the same budget: shuffled
  passes until 1M tokens (prompt and reply) or 2 passes, and at most 1.8 h each. Merged into fp16 weights for
  evaluation. Examples, tokens and GPU time are reported per arm.
  Changed on 2026-11-19, before any Stage B sample: this said 1e-4 and 2 epochs over everything. At QLoRA's
  150-300 tokens/s on a T4, plain's ~2M tokens a pass would have taken 4-7 GPU-h alone; an equal token budget also
  makes training compute equal, like collection.
- **Evaluation, dev set only:** the untrained model on all three arms, each trained model on its own arm, with
  Stage A's schedule (decomposition arms to 50k generated tokens, plain to 10k). Read on dev, for steering only:
  trained minus untrained per arm; trained idea minus trained decomp at the largest budget both cover, on tokens
  and on GPU-seconds; each against plain. The 279 stay untouched until a method is frozen and its test rules are
  written here.
  Made exact on 2026-11-19, before any dev sample (`dev_report.py`): problems graded in every series; "each against
  plain" means trained decomp and trained idea minus trained plain; every comparison at the largest budget both
  series cover, paired bootstrap 95% CI over problems. Token budgets are report.py's (1k-500k); GPU-second budgets
  are 5, 10, 15, 20, 40, 60, 80, 120 and 160 per problem, with cost per attempt = generation seconds over attempts.
  A trained model runs as its LoRA merged into fp16 weights (`train_lora.py --merge`), so it generates as fast as
  the base; the merge is onto the fp16 base, not the 4-bit one it was trained against, a small mismatch.
- **Compute:** about 6 GPU-h to collect, at most 4.4 to train (two GPUs side by side), about 15 for the dev
  evaluations.
- **Notebooks:** `qwen7b-stage-b-prep` (CPU: `kaggle_stage_b_prep_job.py`, dev set and pool),
  `qwen7b-stage-b-collect` (`kaggle_stage_a_job.py` with `RUNNER=collect.py`, `DATA=cc_train`, chunk 24),
  `qwen7b-stage-b-train` (`kaggle_stage_b_train_job.py`). The dev evaluation needs `run_stage_a.py` to run one arm
  alone: `--arms`, built on 2026-11-19 (`report.py` then reports whichever arms a state holds, unchanged for three).
  Dev notebooks: `qwen7b-stage-b-dev-base` (untrained, all three arms) and `qwen7b-stage-b-dev-<arm>` (sets
  `ADAPTER=lora_<arm>` and `EXTRA="--arms <arm>"`; input: `qwen7b-stage-b-train`'s output), each `DATA=lcb_dev`,
  chunk 24, a 9.6 h generation cap. Stage A's GPU time scaled to 119 problems gives about 0.5 h for plain, 1.8 for
  decomp and 4.3 for the idea per model. Then on the laptop: `python3 -I dev_report.py --base dev_base/state.json
  --trained dev_plain/state.json dev_decomp/state.json dev_idea/state.json`.
- **Collection result (notebook v1, 2026-11-18 18:54 to 23:34 UTC, 4.67 h, all rounds done, nothing left
  ungraded):**

  | arm | GPU-h | attempts | passed | generated tokens |
  |---|---|---|---|---|
  | plain | 1.50 | 8,928 | 1,478 (16.6%) | 1.80M |
  | decomp | 1.51 | 216 | 22 (10.2%) | 3.43M |
  | idea | 1.58 | 312 | 16 (5.1%) | 1.67M |

  One attempt per problem per pass, so decomp and idea train on 22 and 16 problems; plain went 1.5 times through
  the pool. Equal GPU time buys the decomposition arms very little training data. That is the honest cost of
  their format, and it is what round 1 tests.
- **Training result (notebook v1, 2026-11-19 13:21 to about 15:10 UTC, 1.78 h, all three adapters saved):**

  | arm | problems | examples | trained (passes) | tokens | reply tokens | steps | GPU-h | loss, first 5 -> last 5 steps |
  |---|---|---|---|---|---|---|---|---|
  | plain | 534 | 1,275 | 1,460 (1.15, the 1M budget) | 999,578 | 208,380 | 183 | 1.31 | 0.20 -> 0.13 |
  | decomp | 22 | 118 | 236 (2) | 397,392 | 126,274 | 30 | 0.42 | 0.21 -> 0.14 |
  | idea | 16 | 88 | 176 (2) | 259,460 | 33,322 | 22 | 0.26 | 0.05 -> 0.06 |

  No example was too long; QLoRA ran at 212-274 tokens/s. The losses start low because these are the model's own
  samples. With 30 and 22 optimizer steps, the decomposition arms get a light nudge at most; the dev evaluation will
  say whether it moves anything.

## Running

- `python tests_qwen.py`: everything, with hand-written programs only (the grader refuses model output outside Kaggle).
- Kaggle: `kaggle_prep_job.py` (CPU, internet on) builds `lcb/` and runs the tests. Then
  `kaggle_stage_a_job.py` (2×T4, internet on) installs vLLM, downloads the model and runs `run_stage_a.py`.
  The hours and chunk size are set by `GEN_HOURS` / `TOTAL_HOURS` / `CHUNK` in the job; a run still going
  0.1 h past `TOTAL_HOURS` is killed and the report is rebuilt from `state.json`.
- Full run, from 2026-11-14 (30 GPU-h that week, shared with other notebooks): session 1 is notebook
  `qwen7b-stage-a` (10 h of generation, hard stop at 10.7 h; Kaggle's GPU limit is 12 h). Session 2 is notebook
  `qwen7b-stage-a-s2`, the same job with session 1's output as an input. The job sees that input's `state.json`,
  copies it in and passes `--resume`: finished chunk-rounds need nothing, ungraded attempts are sampled again, and
  the session number seeds vLLM so session 2 doesn't repeat session 1's samples. All sessions together stop at
  `PLAN_HOURS`: 13.5 for sessions 1 and 2, then 21 when session 3 (`qwen7b-stage-a-s3`, input: session 2's output
  only, so it sees one `state.json`) was added. A session whose input already finished every round exits before
  installing anything.
  `stage_a/job_status.json` says how each session ended.
- `kaggle_watch.py` (on the laptop, never on Kaggle) does the pushes. Session 1 goes once the quota has reset and
  11.5 GPU-h are left. Each later session goes only after the one before was seen running, then showed the same final status on
  two checks 10 min apart, lists a `state.json` from that run, and its `job_status.json` says rounds are left and
  `run_stage_a` ended by itself or by the hard stop, with the plan's remaining hours still in the quota. Anything
  else stops it with NEEDS USER. Only `job_status.json` and the reports are downloaded. Its steps are kept in
  `~/slm-runs/qwen7b/full_run/state.json`, so `--watch` can be stopped and started again without pushing twice.
- Pilot (2026-11-10): `CHUNK` 12, all problems in the shuffled order, so the chunks that fit are a random subset.
  It checks vLLM on T4, throughput and the harness; its numbers are not the result. Try 1 (`GEN_HOURS` 0.55,
  `TOTAL_HOURS` 0.7) stopped at the CPU tests after 6.8 min: one correct stdin program timed out 0.1 s after the
  15 GB model download ended (it passed on the CPU notebook and locally), so the job now syncs the disk first and
  retries the tests once. Try 2 (`GEN_HOURS` 0.45, `TOTAL_HOURS` 0.58) passed setup and the tests in 3.1 min, then
  vLLM failed to start: uv had picked transformers 5.22, whose tokenizers lack `all_special_tokens_extended`. Both
  jobs now pin `transformers==4.51.3`. Try 3 (`GEN_HOURS` 0.42, `TOTAL_HOURS` 0.55) got into engine start, then
  Triton failed on `import setuptools` (vLLM lists it only above Python 3.11; uv venvs come without it). The jobs now
  install `setuptools==80.9.0`, and `llm.preflight` imports the engine's modules, builds Triton's CUDA stub and runs
  the chat template right after the download. Try 4 (`GEN_HOURS` 0.40, `TOTAL_HOURS` 0.52): the preflight found
  no `llm.preflight`, because the job had copied the 15:15 code/ folder in the prep notebook's output instead of
  the code dataset (tries 1-3 ran those copies too, the same code then apart from the job file). The job now takes
  code only from `qwen7b-decomp-code`. Try 5 (`GEN_HOURS` 0.38, `TOTAL_HOURS` 0.50) ran end to end in 25 min
  (0.42 GPU-h): setup 9.4 min (the uv install took 1-8 min across tries), model ready at 10.7 min, one chunk-round
  of 12 problems in 14.1 min (429k generated tokens in 14 min of generation: 512 tok/s on average, 600-800 tok/s
  in large batches; 660k prompt tokens, 572k of them the idea's), and grading kept up (3,048 programs, none left).

  Pilot findings, 12 problems, not the result:
  * Tokens per attempt: plain 207 (paper 387), decomp 14.0k (paper's RL-trained DecompRL 4.0k), idea 5.8k
    generated plus 47.6k prompt (dependency bodies in context). 2 of 12 decomp attempts had no decomposition.
  * Solved: plain 4 problems, decomp 1 (abc373_a), idea 0. No assembly bug found: the idea's abc373_a decomposition
    specified the wrong count (any string of length i, not the i-th), and its abc382_b `parse_input` dropped D, so
    every combination inherited the error. One decomposition per attempt makes these all-or-nothing.
  * The report's budget rule differs from the paper's. Appendix K converts a budget into attempts with each method's
    mean tokens per attempt, and Table 1's flat DecompRL 0.18 at 1k and 5k fits at least one whole attempt; this
    report counts an attempt only once its tokens fit, so decomp reads 0 below 14k. Rescored the paper's way: plain
    0.149 / 0.256 / 0.247 at 1k / 5k / 10k (paper 0.06 / 0.16 / 0.21), decomp 0.083, idea 0.
  * One attempt per arm per round gives the idea about 40% of decomp's tokens, so the idea's covered budget grows
    slowest and caps the primary comparison.
- Outputs: `state.json` (every attempt), `raw_*.jsonl.gz` (all replies, first two programs), `report.md/json`.
