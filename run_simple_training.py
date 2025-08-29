#!/usr/bin/env python3

import os
import sys
import ray

# Add memupdate to Python path
sys.path.insert(0, '/workspace/memupdate')
sys.path.insert(0, '/workspace/verl')

# Set environment variables
os.environ['PYTHONPATH'] = '/workspace/verl:/workspace/memupdate:' + os.environ.get('PYTHONPATH', '')
os.environ['WANDB_API_KEY'] = '5fb2c3eb35cb3bc0124a02069ce91eedc6570e5a'

# Initialize Ray with runtime environment that includes memupdate
ray.init(runtime_env={
    "py_modules": ["/workspace/memupdate"],
    "env_vars": {
        "PYTHONPATH": "/workspace/verl:/workspace/memupdate",
        "WANDB_API_KEY": "5fb2c3eb35cb3bc0124a02069ce91eedc6570e5a",
        "TOKENIZERS_PARALLELISM": "true", 
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
    }
})

print("Ray initialized with memupdate package distribution")

# Change to verl directory for proper config paths
os.chdir('/workspace/verl')

# Now import and run verl training 
from verl.trainer.main_ppo import main

# Set up command line arguments for training
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
    'trainer.logger=["console", "wandb"]',
    'trainer.project_name=memupdate-rl',
    'trainer.experiment_name=memupdate-distributed-success',
    'trainer.n_gpus_per_node=1',
    'trainer.nnodes=1',
    'trainer.save_freq=-1',
    'trainer.test_freq=5',
    'trainer.total_training_steps=3',
    'data.train_files=/workspace/memupdate/data/locomo/train.parquet',
    'data.val_files=/workspace/memupdate/data/locomo/test.parquet',
    'trainer.total_epochs=1',
    # MEMUPDATE: Add our custom reward manager (+ prefix to add new config section)
    '+reward.manager_class=memory_rag',
    '+reward.config.max_total_memories=100',
    '+reward.config.evaluator_model=openai:gpt-4o-mini',
    '+reward.config.use_llm_judge=true'
]

print("Starting MemUpdate RL training with distributed Ray setup...")
try:
    main()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {e}")
    import traceback
    traceback.print_exc()
finally:
    ray.shutdown()