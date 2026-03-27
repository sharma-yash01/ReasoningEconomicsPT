# ReasoningEconomicsPT

Post-training code for LLM/LRMs against the deployed `ReasoningEconomicsEnv` OpenEnv environment.

## Install

Install core training dependencies:

```bash
pip install -r requirements.txt
```

### Production (recommended on CARC/hyperscalers)

Install the environment client artifact from your Hugging Face Space:

```bash
pip install "git+https://huggingface.co/spaces/<owner>/<space_repo>"
```

Set base URL to your deployed Space endpoint:

```bash
export ENV_BASE_URL="https://<owner>-<space_repo>.hf.space"
```

### Development fallback (local sibling repo)

```bash
pip install -e "../ReasoningEconomicsEnv"
```

## Run GRPO training

```bash
python -m training.grpo_train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --env_base_url "$ENV_BASE_URL" \
  --alpha 1.0 \
  --beta 0.0 \
  --default_budget_mode hard \
  --num_train_epochs 1 \
  --output_dir runs/grpo_train
```

You can also provide a Space URL and let the script derive the base URL:

```bash
python -m training.grpo_train \
  --space_url "https://huggingface.co/spaces/<owner>/<space_repo>"
```

## Metadata contract for token-cap mode

`grpo_train.py` reads `observation.metadata["budget_mode"]` each step:

- `hard`: cap generation by remaining budget (plus minimum floor).
- `soft`: use `max_tokens_per_step` without remaining-budget cap.

If metadata is missing, training falls back to `--default_budget_mode` unless `--strict_budget_mode_metadata` is set.

## Reward topology used in PT (current)

- PT trains on the **incoming total reward signal** from the environment (`StepResult.reward`).
- In prod env behavior, terminal-step reward already includes episode bonus.
- `alpha` scales total signal; `beta` is reserved for future decomposed reward wiring.

## Reward logging

By default, reward logs are emitted:

- Console logs every `--log_every_n_steps`.
- JSONL file at `runs/<output_dir>/reward_logs.jsonl` (or `--reward_log_path`).

Disable logs with:

```bash
python -m training.grpo_train --no_log_rewards ...
```

## Notes

- Training uses remote OpenEnv `reset/step` calls instead of importing `ReasonBudgetEnvironment` in-process.
- Hosted Spaces may have limited concurrency; for sustained RL training, use your own duplicated Space or a local Docker deployment as recommended in OpenEnv docs.
