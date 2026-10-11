#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

python scripts/check_environment.py --require-gpu --require-verl
test -s artifacts/verl_data/train.parquet
test -s artifacts/verl_data/validation.parquet

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

TRAIN_FILE="${REPO_DIR}/artifacts/verl_data/train.parquet"
VAL_FILE="${REPO_DIR}/artifacts/verl_data/validation.parquet"
REWARD_FILE="${REPO_DIR}/verl_reward/math_reward.py"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=16 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="${REWARD_FILE}" \
  custom_reward_function.name=compute_score \
  'trainer.logger=["console"]' \
  trainer.project_name=math-rlvr-lab \
  trainer.experiment_name=qwen3-0.6b-grpo-smoke \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=3 \
  2>&1 | tee artifacts/grpo_smoke.log

grep -Eq 'step:[[:space:]]*[012]|global_step' artifacts/grpo_smoke.log
python scripts/validate_artifacts.py grpo-log
