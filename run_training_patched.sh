#!/bin/bash

# MemUpdate Training Script with Dataset Patch
# Run from within the verl Docker container

set -e

export WANDB_API_KEY=5fb2c3eb35cb3bc0124a02069ce91eedc6570e5a

# Container paths
PROJECT_DIR="/workspace/verl"
MEMUPDATE_DIR="/workspace/memupdate"

cd $PROJECT_DIR

echo "Starting MemUpdate training with dataset patch..."
echo "Project directory: $PROJECT_DIR"
echo "MemUpdate directory: $MEMUPDATE_DIR"

# Apply dataset patch before training
echo "Applying dataset patch..."
python3 -c "
import sys
sys.path.append('/workspace/memupdate')
from patch_verl_dataset import patch_rl_dataset
patch_rl_dataset()
"

# Check if data is preprocessed
if [ ! -f "$MEMUPDATE_DIR/data/locomo/train.parquet" ]; then
    echo "Training data not found. Running preprocessing..."
    cd $MEMUPDATE_DIR && python3 regenerate_data.py
    cd $PROJECT_DIR
fi

# Run verl training using existing GSM8K config as base
echo "Starting GRPO training with verl..."

python3 -c "
import sys
sys.path.append('/workspace/memupdate')
from patch_verl_dataset import patch_rl_dataset
patch_rl_dataset()

# Now import and run the training
from verl.trainer.main_ppo import main
import sys
sys.argv = [
    'main_ppo.py',
    '--config-path=/workspace/verl/examples/sglang_multiturn/config',
    '--config-name=gsm8k_multiturn_grpo',
    'algorithm.adv_estimator=grpo',
    'data.train_batch_size=16',
    'data.max_prompt_length=1024',
    'data.max_response_length=1024',
    'data.filter_overlong_prompts=True',
    'data.truncation=error',
    'data.return_raw_chat=True',
    'actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct',
    'actor_rollout_ref.actor.optim.lr=1e-6',
    'actor_rollout_ref.model.use_remove_padding=True',
    'actor_rollout_ref.actor.ppo_mini_batch_size=16',
    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8',
    'actor_rollout_ref.actor.use_kl_loss=True',
    'actor_rollout_ref.actor.kl_loss_coef=0.001',
    'actor_rollout_ref.actor.kl_loss_type=low_var_kl',
    'actor_rollout_ref.actor.entropy_coeff=0',
    'actor_rollout_ref.model.enable_gradient_checkpointing=True',
    'actor_rollout_ref.rollout.name=sglang',
    'actor_rollout_ref.rollout.mode=async',
    'actor_rollout_ref.rollout.gpu_memory_utilization=0.5',
    'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8',
    'actor_rollout_ref.rollout.tensor_model_parallel_size=1',
    'actor_rollout_ref.rollout.n=1',
    'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8',
    'actor_rollout_ref.rollout.multi_turn.enable=True',
    'actor_rollout_ref.rollout.multi_turn.max_assistant_turns=30',
    'actor_rollout_ref.rollout.multi_turn.tool_config_path=/workspace/memupdate/configs/tool_config/memory_tools.yaml',
    'algorithm.use_kl_in_reward=False',
    'trainer.critic_warmup=0',
    'trainer.logger=[\"console\", \"wandb\"]',
    'trainer.project_name=memupdate-rl',
    'trainer.experiment_name=qwen2.5-3b-memory-grpo-patched',
    'trainer.n_gpus_per_node=1',
    'trainer.nnodes=1',
    'trainer.save_freq=-1',
    'trainer.test_freq=5',
    'trainer.total_training_steps=3',
    'data.train_files=/workspace/memupdate/data/locomo/train.parquet',
    'data.val_files=/workspace/memupdate/data/locomo/test.parquet',
    'trainer.total_epochs=1'
]
main()
"

echo "Training completed!"
echo "Check outputs directory for results and logs."