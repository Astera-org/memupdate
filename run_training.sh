#!/bin/bash

# MemUpdate Training Script
# Runs GRPO training for memory update RL using verl framework

set -e

export WANDB_API_KEY=5fb2c3eb35cb3bc0124a02069ce91eedc6570e5a

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/configs"

echo "Starting MemUpdate training..."
echo "Project directory: $PROJECT_DIR"
echo "Config path: $CONFIG_PATH"

# Check if data is preprocessed
if [ ! -f "$PROJECT_DIR/data/locomo/train.parquet" ]; then
    echo "Training data not found. Running preprocessing..."
    uv run python -m memupdate.data.preprocess_locomo
fi

# Check if verl is available
if ! uv run python -c "import verl" 2>/dev/null; then
    echo "Error: verl not found. Please run 'uv sync' first."
    exit 1
fi

# Run verl training
echo "Starting GRPO training with verl..."

uv run python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='simple_memory_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=30 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/configs/tool_config/memory_tools.yaml" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.trace.backend=wandb \
    actor_rollout_ref.rollout.trace.token2text=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='memupdate-rl' \
    trainer.experiment_name='qwen2.5-3b-memory-grpo' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_training_steps=10 \
    data.train_files="$PROJECT_DIR/data/locomo/train.parquet" \
    data.val_files="$PROJECT_DIR/data/locomo/test.parquet" \
    trainer.total_epochs=1

echo "Training completed!"
echo "Check outputs directory for results and logs."