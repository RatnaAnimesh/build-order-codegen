# Stage A: Qwen2.5-7B-Instruct on LiveCodeBench 2024-08-01..2025-02-01

Paired problems: 279 of 279, 3 session(s). Tokens per attempt: plain 260, decomp 17032, idea 5573. Largest budget covered (95% of problems) by decomp and idea: 50000; by all three: 10000.

Budget rule: the paper's (Appendix K), k = max(1, floor(B / tokens per attempt)) attempts, unbiased pass@k.

| tokens | plain | decomp | idea | k plain/decomp/idea | paper instruct | paper DecompRL (RL) |
|---|---|---|---|---|---|---|
| 1000 | 0.183 | 0.113 | 0.095 | 3/1/1 | 0.06 | 0.18 |
| 2000 | 0.231 | 0.113 | 0.095 | 7/1/1 |  |  |
| 4000 | 0.272 | 0.113 | 0.095 | 15/1/1 |  |  |
| 5000 | 0.283 | 0.113 | 0.095 | 19/1/1 | 0.16 | 0.18 |
| 10000 | 0.312 | 0.113 | 0.095 | 38/1/1 | 0.21 | 0.25 |
| 20000 | 0.312* | 0.113 | 0.173 | 76/1/3 |  |  |
| 30000 | 0.312* | 0.113 | 0.208 | 115/1/5 |  |  |
| 40000 | 0.312* | 0.165 | 0.229 | 153/2/7 |  |  |
| 50000 | 0.312* | 0.165 | 0.237 | 192/2/8 | 0.3 | 0.4 |

`*` fewer than 95% of problems have k graded attempts (pass@n with n < k, biased low).

**Primary, idea - decomp at 50000 tokens:** +0.072 [+0.032, +0.111] -> passes (CI above 0).

- idea_minus_plain at 10000: -0.217 [-0.261, -0.173]
- decomp_minus_plain at 10000: -0.199 [-0.247, -0.152]
- at 4000: idea_minus_decomp -0.018 [-0.041, +0.004], idea_minus_plain -0.177 [-0.216, -0.139], decomp_minus_plain -0.159 [-0.202, -0.119]
- at 5000: idea_minus_decomp -0.018 [-0.041, +0.004], idea_minus_plain -0.188 [-0.228, -0.149], decomp_minus_plain -0.170 [-0.213, -0.129]
- at 10000: idea_minus_decomp -0.018 [-0.041, +0.004], idea_minus_plain -0.217 [-0.261, -0.174], decomp_minus_plain -0.199 [-0.246, -0.152]

Harness check against Table 1 instruct: 1000: 0.183 vs 0.06 (off), 5000: 0.283 vs 0.16 (off), 10000: 0.312 vs 0.21 (off)

Check, exact-token rule (an attempt counts once all its tokens fit):

| tokens | plain | decomp | idea |
|---|---|---|---|
| 1000 | 0.210 | 0.000 | 0.000 |
| 2000 | 0.247 | 0.000 | 0.006 |
| 4000 | 0.279* | 0.005 | 0.052 |
| 5000 | 0.288* | 0.005 | 0.075 |
| 10000 | 0.308* | 0.023 | 0.139 |
| 20000 | 0.312* | 0.105* | 0.195 |
| 30000 | 0.312* | 0.147* | 0.221* |
| 40000 | 0.312* | 0.161* | 0.231* |

idea - decomp at 10000 (exact-token rule): +0.115 [+0.085, +0.147]

```
{
 "arms": {
  "plain": {
   "attempts": 10602,
   "graded": 10602,
   "attempts_per_problem": 38.0,
   "tokens_per_attempt": 260.3292774948123,
   "prompt_tokens_per_attempt": 32.508017355216104,
   "prompt_tokens_per_problem": 1235.3046594982118,
   "gen_tokens_per_problem": 9892.512544802867,
   "programs_per_attempt": 0.9581211092246746,
   "attempt_pass_rate": 0.12205244293529523,
   "statuses": {
    "wrong": 6505,
    "error": 1676,
    "pass": 1294,
    "compile": 203,
    "timeout": 471,
    "crash": 9
   }
  },
  "decomp": {
   "attempts": 558,
   "graded": 558,
   "attempts_per_problem": 2.0,
   "tokens_per_attempt": 17031.706093189965,
   "prompt_tokens_per_attempt": 5979.569892473119,
   "prompt_tokens_per_problem": 11959.139784946237,
   "gen_tokens_per_problem": 34063.41218637993,
   "programs_per_attempt": 113.81720430107526,
   "attempt_pass_rate": 0.11290322580645161,
   "statuses": {
    "wrong": 36093,
    "pass": 102,
    "error": 15044,
    "compile": 3791,
    "timeout": 1378,
    "crash": 3
   },
   "units_mean": 3.96415770609319,
   "reasons": {
    "no_decomposition": 27,
    "too_many_units": 18
   },
   "parsed_impls_per_attempt": 29.308243727598565,
   "uses_line_share": 0.0,
   "levels_mean": 0.9516129032258065,
   "first_pass_index_median": 1
  },
  "idea": {
   "attempts": 2232,
   "graded": 2232,
   "attempts_per_problem": 8.0,
   "tokens_per_attempt": 5573.33288530466,
   "prompt_tokens_per_attempt": 43779.28315412186,
   "prompt_tokens_per_problem": 350234.2652329749,
   "gen_tokens_per_problem": 44586.66308243728,
   "programs_per_attempt": 107.78360215053763,
   "attempt_pass_rate": 0.09498207885304659,
   "statuses": {
    "wrong": 144483,
    "error": 54782,
    "timeout": 8101,
    "pass": 2421,
    "crash": 90,
    "compile": 8659
   },
   "units_mean": 4.209229390681004,
   "reasons": {
    "too_many_units": 149,
    "no_decomposition": 138
   },
   "parsed_impls_per_attempt": 29.45878136200717,
   "uses_line_share": 0.9737094199042043,
   "levels_mean": 2.4086021505376345,
   "missing_explicit": 36,
   "missing_checker": 201,
   "helpers_built": 174,
   "resumed": 36,
   "coherent": 15503,
   "first_pass_index_median": 0,
   "passed_by_coherent_program": 197,
   "passed_attempts": 212
  }
 },
 "by_difficulty": {
  "budget": 50000,
  "easy": {
   "n": 67,
   "decomp": 0.5373134328358209,
   "idea": 0.7313432835820896
  },
  "medium": {
   "n": 89,
   "decomp": 0.11235955056179775,
   "idea": 0.15730337078651685
  },
  "hard": {
   "n": 123,
   "decomp": 0.0,
   "idea": 0.024390243902439025
  }
 },
 "by_difficulty_all_arms": {
  "budget": 10000,
  "easy": {
   "n": 67,
   "plain": 0.8059701492537313,
   "decomp": 0.3805970149253731,
   "idea": 0.31716417910447764
  },
  "medium": {
   "n": 89,
   "plain": 0.29213483146067415,
   "decomp": 0.06741573033707865,
   "idea": 0.054775280898876406
  },
  "hard": {
   "n": 123,
   "plain": 0.056910569105691054,
   "decomp": 0.0,
   "idea": 0.003048780487804878
  }
 },
 "llm": {
  "gen_tokens": 24703382,
  "prompt_tokens": 101396610,
  "seconds": 54698.981751441956,
  "too_long": 0
 },
 "grader": {
  "seconds": 15525.349382074986,
  "counts": {
   "error": 14087,
   "wrong": 33466,
   "compile": 2801,
   "timeout": 1601,
   "pass": 67
  },
  "pending": 0,
  "errors": []
 }
}
```