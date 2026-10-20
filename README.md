# GENCO TO THE MOON

**Unlocking the social value of grid foundation models in electricity markets with AI agents.**

This repository is the minimal code release for our GENCO Challenge entry. It reproduces the main results of the write-up:

- one generator bidding under a deadline (Section 4), both from the true scenario distribution and end to end from news bulletins read by an LLM;
- three generators bidding against each other (Section 5);
- market-power regulation (Section 6);
- transmission outage screening by the grid operator (Section 7).

Each actor has to estimate what the grid will do in many uncertain scenarios before a deadline. We compare exact AC-OPF, DC-OPF, GENCO (the grid foundation model of [gridfm](https://huggingface.co/gridfm)), and multi-fidelity combinations of AC with a cheap model. All of them run inside the same wall-clock budget, and every decision is scored against AC-OPF.

Links: write-up [LINK: post] · [interactive demo](https://claude.ai/artifact/CPDyJnDgaoNLra9358dLrG) · fine-tuned checkpoints and heads [LINK: HF model]

## What is in here

```
genco_market/        the library
  env/               IEEE 118 market simulator: AC-OPF clearing (PowerModels/Ipopt via gridfm-datakit), DC LP,
                     ERCOT-2024 load baseline, event physics, solver process pool, outage requests (Layer 4)
  semantics/         world and event generator, bulletin text at levels L0-L3
  agent/             LLM perception (bulletins -> scenario distribution) and the bidding agent
  surrogate/         GENCO service, price head (LMPHead), profit models, social-cost head (operator GENCO)
  estimators/        single- and multi-fidelity (control-variate) estimators and budget split
  eval/              oracle tables (true distribution, AC) for scoring
configs/             market, events (effects.yaml), firms, bulletin templates, Layer 4 settings
experiments/         one script per experiment (below) and make_tables.py
data/                llm_judgments_L3.jsonl: the 239 LLM judgments used in the end-to-end runs
results/             reference outputs of our runs (Exp 1 summary, three-firm and regulation tables)
tests/               fast unit tests (no Julia, no GPU)
```

| Write-up | Result | Scripts (in order) | Table printed by `make_tables.py --only` |
|---|---|---|---|
| §3 | Price head on GENCO embeddings (LMP error 1.9 vs. 4.6 $/MWh for DC, 9.3 without embeddings) | `exp1_fidelity.py` | `price_head` |
| §3 | Fine-tuned GENCO (`small_ft2`) and its price head | `genco_finetune.py` | – |
| §4 | Bidding regret at 0.5–5 s, 110 states, and by scenario type | `exp2_budget.py` → `exp2_genco_ft.py` | `bidding` |
| §4 | End to end from L3 bulletins at 0.5 s (240 decisions) | `run_episode.py` | `end_to_end` |
| §5 | Nash gap of three strategic firms at 2–10 s | `l3_abd.py` → `l3_abd_fidelity.py` | `three_firms` |
| §6 | Five regulation rules, firms anticipate the rule | `l3_abd.py --stage pivotal/rules` | `regulation` |
| §7 | Outage screening at 0.5–2 s, 200 requests | `l4_offline.py` → `l4_op_data.py` → `l4_op_train.py` → `l4_offline.py --op` | `operator` |

## Setup

Tested on Linux with Python 3.12, one NVIDIA A100 and 16 CPU cores for the solver pool.

1. **PyTorch and GENCO.** GENCO runs through `gridfm-graphkit`. We use the paper-reproduction branch, so that the official checkpoints give the published metrics.

   ```bash
   conda create -n genco python=3.12 -y && conda activate genco
   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
   pip install torch-geometric==2.7.0
   pip install torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
   git clone -b genco-paper-repro https://github.com/gridfm/gridfm-graphkit.git   # we used commit 4e7fbcf
   pip install -e gridfm-graphkit
   ```

2. **AC-OPF solver and this package.** `gridfm-datakit` brings Julia, PowerModels and Ipopt.

   ```bash
   pip install -r requirements.txt
   pip install -e .
   gridfm_datakit setup_pm          # installs Julia + PowerModels + Ipopt (a few minutes, once)
   ```

3. **GENCO checkpoints.** Download the OPF checkpoints for IEEE 118 into `checkpoints/genco-opf-datakit` (or point `GENCO_OPF_CKPT` elsewhere).

   ```bash
   hf download gridfm/genco-opf-datakit --include "case118_ieee/small/*" "case118_ieee/base/*" \
       --local-dir checkpoints/genco-opf-datakit
   ```

   Fine-tuning (below) writes `case118_ieee/small_ft2` next to them. To skip it, download our `small_ft2` checkpoint and the trained heads from [LINK: HF model]. Put the checkpoint under `checkpoints/genco-opf-datakit/case118_ieee/small_ft2/seed0/` and the `*.pt` heads under `work/models/`.

4. **Environment variables** (optional; see `.env.example`):
   - `GENCO_WORKDIR` holds intermediate data, caches, trained heads and parquet results. The default is `./work`; put it on a fast local disk.
   - `MLFLOW_ALLOW_FILE_STORE=true` is needed by graphkit.
   - `CUDA_VISIBLE_DEVICES` picks the GPU.
   - An `OPENAI_API_KEY` is only needed for live LLM runs.

Check the install:

```bash
pytest -q -m "not slow"      # ~10 s
```

## Reproducing the results

All commands run from the repository root. Run times are for our machine: 16 solver processes, where one AC-OPF takes about 20 ms amortized, and one A100. Every script caches its expensive phase under `GENCO_WORKDIR` and can be restarted.

**Note on budgets.** The compute budgets are wall-clock, and each estimator's cost per evaluation is measured during the run. On different hardware the number of scenarios each method affords changes, so the regrets move somewhat. The ranking of the methods should not change.

### 1. Price head and fine-tuned GENCO (§3)

```bash
python experiments/exp1_fidelity.py --stage all --workers 16        # AC labels, LMP heads, fidelity table
python experiments/genco_finetune.py --stage label --data all --workers 32   # 72k AC labels (~20 min)
python experiments/genco_finetune.py --stage train --data all --epochs 20    # fine-tune GENCO small (~1 h, 1 GPU)
python experiments/genco_finetune.py --stage head --data all                 # price head on the fine-tuned model
```

`exp1_fidelity.py` writes `results/exp1_summary.md` and the price heads (`lmp_head_small.pt`, `lmp_head_base.pt`, `lmp_head_noemb.pt`) to `work/models/`. Fine-tuning starts from the official `small` checkpoint. It writes `small_ft2` and `lmp_head_small_ft2.pt`, and leaves the original untouched.

### 2. One strategic firm, true distribution (§4)

There are 110 states: 40 random, 30 with a forced heatwave or thunderstorm, and 40 with one of the eight extension events. Each state has 400 scenarios. Estimators draw from the first half; the second half defines the true optimum.

```bash
python experiments/exp2_budget.py --states 40 --tag heatv4
python experiments/exp2_budget.py --states 30 --tag heatv4_hs --events heatwave,thunderstorm
python experiments/exp2_budget.py --states 40 --tag ext2 \
    --events cold_snap,wildfire_psps,high_wind,solar_drop,demand_response,market_power_review,geomagnetic_storm,cyber_physical
python experiments/exp2_genco_ft.py --src heatv4,heatv4_hs --name all --size small_ft2
python experiments/exp2_genco_ft.py --src ext2 --name ext2 --size small_ft2
python experiments/make_tables.py --only bidding
```

The first three commands take about 1.5 h together. `exp2_genco_ft.py` re-evaluates the same scenarios with the fine-tuned GENCO and checks that they match. To check for leakage, run `python experiments/ft_leak_check.py`: it compares test scenarios to the fine-tuning set.

### 3. End to end from news (§4)

Each of the 60 episodes has 4 hourly decisions with L3 bulletins and a 0.5 s compute budget. All LLM agents share one LLM judgment per decision. By default these judgments are replayed from `data/llm_judgments_L3.jsonl`, so no API calls are made:

```bash
export OPENAI_API_KEY=disabled-for-replay LLM_CACHE_MODE=read
for o in 0 20 40; do
  python experiments/run_episode.py --config exp.yaml --budget 0.5 --episodes 20 --episode-offset $o \
      --tag e2e_ft2_b05_$o --genco-size small_ft2 --replay-beliefs data/llm_judgments_L3.jsonl \
      --agents llm/ac llm/mf_dc llm/genco llm/mf_genco
done
python experiments/make_tables.py --only end_to_end
```

`make_tables.py` drops the very first decision (episode 0, hour 0), because the solver pool is still cold there. That leaves 239 decisions.

To let a live LLM read the bulletins instead, drop `--replay-beliefs` and add `--shared-belief`. This needs an OpenAI key (we used `gpt-5.4-mini`). The L3 bulletins then use template text where our paraphrase cache is missing, so the judgments will differ from ours.

### 4. Three strategic firms (§5) and regulation (§6)

```bash
python experiments/l3_abd.py --stage tensor --normal 27 --scarce 13 --n 80   # 40 states x 80 scenarios x 216 profiles, AC (~4 h)
python experiments/l3_abd.py --stage game
python experiments/l3_abd_fidelity.py --stage head                          # price head for A, B, D on GENCO embeddings
python experiments/l3_abd_fidelity.py --stage cheap --n 400                 # DC and GENCO on 400 scenarios x 216 profiles
python experiments/l3_abd_fidelity.py --stage play --reps 20                # each firm estimates and bids within a budget
python experiments/l3_abd.py --stage pivotal
python experiments/l3_abd.py --stage rules
python experiments/make_tables.py --only three_firms regulation
```

To speed up the tensor stage, run a second copy with `--reverse`. The AC cost per evaluation used for the budgets is fixed at our measured 19.65 ms (`--ac-cost-ms`).

### 5. Outage screening (§7)

There are 200 test requests (episodes 0–199). The operator GENCO is trained on 600 separate days (episodes 1000–1599).

```bash
python experiments/l4_offline.py --episodes 0 200 --no-phase2       # truth: 8 h x 300 scenarios, line in/out, AC + DC + GENCO (~5 h)
python experiments/l4_op_data.py --episodes 1000 1600 --n 4          # 19,200 training triplets with AC social cost (~20 min)
python experiments/l4_op_train.py                                    # social-cost head on fine-tuned GENCO embeddings (~1 min)
python experiments/l4_offline.py --episodes 0 200 --genco-size small_ft2 --op --tag op_200
python experiments/make_tables.py --only operator
```

The truth step parallelizes over episode ranges: run several copies with disjoint `--episodes`.

### All tables

```bash
python experiments/make_tables.py
```

It prints every table whose inputs exist and skips the others.

## Released data

- `data/llm_judgments_L3.jsonl` holds the structured LLM judgments of the end-to-end runs: one line per decision, with the extraction and its latency. One of the 240 calls failed; that decision falls back to the prior, as it did in our run.
- `results/` holds reference outputs from our runs, for comparison: the Exp 1 summary, the Nash-gap table and the regulation tables.

## Notes

- Market model: IEEE 118 (PGLib), five firms, pay-as-clear LMPs from AC-OPF with load shedding at VOLL $2,000/MWh, and stylized events. See the write-up for the limitations.
- The fine-tuned GENCO and the operator GENCO are tested on scenarios from the same generator with disjoint seeds, so their results are in-distribution. `ft_leak_check.py` measures how close the bidding test scenarios are to the fine-tuning set. No test scenario duplicates a training one; the median relative load distance to the nearest training scenario is 3.2%. For outage screening we found the same 3.2%, against 3.1% between training scenarios themselves.

## License

Apache 2.0, see `LICENSE`.
