# LLM Entropy/Surprise for Realtime Hallucination Detection
> Exploring LLM hallucination from an algorithmic trading perspective.

As explored in previous work (HALT 2026), LLM **surprise** and **entropy** can act as a signal for  hallucination. LLMs are sampling machines at heart, predicting next token probabiltities. From this rises the notion of **surpise**,

$$-\log(P(x))$$

and **entropy**,

$$-\sum P(x_i)\log P(x_i)$$

In simple terms, these measure _"how unexpected a single specific token was"_ and _"the models total uncertainty"_.

## Idea
Prior work has explored the concept of identifying trends within these values in order to determine if an LLM is __hallucinating__. This is striking similar to the problem formulation of **algorthim trading**, and all time series analysis problems. 

In this work, we apply ideas from algorithmic trading in the hopes of identifing patterns in LLM uncertainty for hallucination detection.

## Method

The repo is structured like an algorithmic trading research repo — a signal
is a "strategy", a labelled response is a "price history", and classification
quality (AUROC) is the PnL:

```
signals/          strategies: consume (surprisal, entropy) token by token,
                  causally, and maintain a running hallucination score
backtest/         engine (streaming replay + AUROC/AP/F1) and grid optimiser
mcpt/             Monte Carlo permutation test (token-shuffle null)
configs/spaces/   search spaces; optimised configs land in configs/
scripts/          run_backtest.py, optimize.py, run_mcpt.py
```

### Sample gt

`data.py` samples N=1000 TriviaQA questions against Llama-3.1-8B, storing the
answer's per-token surprisal and (top-20 truncated) entropy. An LLM judge
labels each response `succ` / `fail` (hallucinated) / `null` (refusal).
Refusals are excluded; `succ` → label 0, `fail` → label 1.

### Identify trends

Each signal in `signals/` is one hypothesis about the SHAPE of uncertainty,
each ported from a trading idea:

| signal | trading analog | hypothesis |
|---|---|---|
| `baseline` | buy-and-hold / raw feature | order-free summary stat (mean/max/sum) — the "raw ML" strawman every temporal idea must beat |
| `spike` | fixed-move event study | hallucinations are tokens above an absolute surprisal threshold |
| `bollinger` | Bollinger band breaches | spikes *relative to the response's own rolling baseline* — a reach after calm is the tell |
| `crossover` | EWMA crossover / momentum | sustained regime shifts into high entropy (confabulated multi-token spans) |

Signals are streaming and causal (`update(surprisal, entropy) → score`), so
any of them can run in realtime during generation.

### Backtest

```bash
python scripts/run_backtest.py spike                    # DEFAULTS params
python scripts/optimize.py configs/spaces/spike.json    # grid search
python scripts/run_backtest.py configs/spike_<stamp>.json
```

Stratified 70/30 train/test split (fixed seed). The optimiser selects on
train AUROC only; test metrics are reported once for the winner.

### MCPT

Trading MCPT shuffles returns in time to destroy temporal patterns while
preserving the return distribution. The analog here shuffles **token order
within each response**: every marginal statistic (mean, max, total
surprisal, length, label) survives, only the temporal shape dies. The null
hypothesis is therefore sharp:

> *the signal's edge comes only from the marginal distribution of
> uncertainty, not from **when** the uncertainty happens.*

```bash
python scripts/run_mcpt.py configs/spike_<stamp>.json --sample out -n 200
```

`out` permutes held-out responses under fixed params (fast); `in` refits the
whole search space per permutation (slow, guards against the optimiser
manufacturing temporal structure from noise). Small p ⇒ order genuinely
matters. Plot + JSON land in `reports/`.

### Results so far (N=788 graded responses, base rate 0.21)

Tuned on the 551-response train split; validation = 237 held-out responses
(188 true / 49 hallucinated) touched once per winner. Full table via
`python scripts/leaderboard.py`.

| signal | validation AUROC | in/out gap | temporal? |
|---|---|---|---|
| **recency** (EWMA entropy, α=0.2, floor 0.5) | **0.703** | +0.08 | **yes** — token-shuffling drops it to 0.66 (1/50 shuffles ≥ real) |
| spike (entropy, thr 0.75, weighted, len-norm) | 0.678 | +0.09 | no (order-free) |
| baseline (mean entropy) | 0.674 | +0.09 | no (order-free) |
| ensemble (mean entropy + 0.5 × bollinger) | 0.674 | +0.10 | timing sleeve adds nothing |
| bollinger (adaptive z-breach) | 0.661 | +0.10 | weakly (shuffle p ≈ 0.16) |
| streak (worst 16-token burst) | 0.658 | +0.11 | no better than level |
| crossover (EWMA regimes) | 0.627 | +0.10 | no |
| *length-only confound check* | *0.584* | — | — |

Honest read: uncertainty robustly predicts hallucination out-of-sample
(AUROC ≈ 0.67 ≫ 0.5, and ≫ the 0.58 length confound). Marginal-level
signals cluster at ≈ 0.67, and most temporal ideas fail to beat them — but
**recency-weighted entropy is a genuinely temporal edge**: it leads the
board at 0.703 with the smallest overfit gap, and shuffling token order
destroys its advantage back down to the marginal level. WHERE the
uncertainty happens (late, as the model commits to the answer) carries
information beyond how much of it there is.

### Realtime analysis

TODO: running-score trajectories — flag a response the moment its score
crosses the threshold, mid-generation, and measure detection latency in
tokens.