# MemUpdate: Self-Refining Memory via Reinforcement Learning

MemUpdate is an experimental project that explores self-refining memory in LLMs via Reinforcement Learning. It uses DAPO (improved GRPO) RL methods to train a model for updating any existing memory database to maximize rewards for future question-answering tasks.

## Overview

**Core Concept**: Train an agent to iteratively improve memory database through tool use, optimizing for better performance on ANY questions tomorrow.

**Package Strategy**: Use LoCoMo, LangMem, and verl as external packages to minimize local dependencies and leverage existing implementations.

## Features

- 🧠 **Memory Tools**: 6 memory management tools (search, manage, delete, sample, merge, split)
- 🔄 **RL Training**: GRPO-based reinforcement learning with verl framework  
- 📊 **LoCoMo Dataset**: Evaluation on long-term conversational memory benchmark
- 🎯 **Multi-turn Episodes**: Up to 30 memory operations per training episode
- 📈 **Custom Rewards**: Performance delta × memory efficiency reward function

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd memupdate

# Install dependencies (requires Python 3.10+)
pip install -e .

# Install external dependencies
pip install verl langmem openai anthropic
```

### Data Preprocessing

```bash
# Preprocess LoCoMo dataset to training format
python -m memupdate.data.preprocess_locomo
```

### Training

```bash
# Run GRPO training
./run_training.sh

# Or run directly with Python
python -m verl.trainer.main_ppo --config-path=configs --config-name=locomo_memory_grpo
```

## Architecture

### Dataset (LoCoMo)
- **10 conversations** with 1,986 QA pairs (198.6 avg per conversation)  
- **Train/Test Split**: 7 train conversations (1,347 QA pairs) / 3 test conversations (639 QA pairs)
- **Question Categories**: Identity (1), Temporal (2), Inference (3), Preference (4)

### Memory Tools
1. **search_memory**: Retrieve relevant memories via similarity search
2. **manage_memory**: Create/update memories with metadata
3. **delete_memory**: Remove specific memories with optional reasoning
4. **sample_memory**: Random/diverse/recent memory sampling 
5. **merge_memory**: Consolidate related memories (summarize/concatenate/extract_common)
6. **split_memory**: Decompose complex memories (temporal/thematic/speaker)

### Training Process
1. **Sample**: Pick 1 conversation + 1 question from training data
2. **Initialize**: Create initial memory M from conversation facts
3. **Agent Loop**: Agent receives memory state + target question
   - Tool selection via function calling
   - Tool execution updates memory → M'
   - Repeat for max 30 steps or until agent stops
4. **Evaluate**: RAG-based QA performance on target question only
5. **Reward**: Compute reward = performance_delta × memory_efficiency
6. **Update**: Policy gradient step

### Reward Function

```python
reward = performance_delta * memory_efficiency

where:
- performance_delta = evaluate_qa(M_new, question) - evaluate_qa(M_old, question)
- memory_efficiency = size_penalty * density_bonus * change_factor
```

## Configuration

### Training Parameters
- **Batch Size**: 64 episodes per batch
- **Total Steps**: 1,347 (matching number of training QA pairs)
- **Evaluation**: Every 25 training steps using test.parquet validation data
- **Multi-turn**: Up to 15 assistant turns (30 memory operations)
- **Model**: Qwen/Qwen2.5-3B-Instruct

### Tool Configuration
See `configs/tool_config/memory_tools.yaml` for memory tool settings.

### Training Configuration  
See `configs/locomo_memory_grpo.yaml` for full GRPO training configuration.

## Project Structure

```
memupdate/
├── agents/           # Simple local agent (no LangGraph complexity)
├── tools/            # Memory management tools (verl BaseTool wrappers)
│   ├── search_memory.py    # Memory retrieval
│   ├── manage_memory.py    # Memory creation/update  
│   ├── delete_memory.py    # Memory deletion
│   ├── sample_memory.py    # Memory sampling
│   ├── merge_memory.py     # Memory consolidation
│   └── split_memory.py     # Memory decomposition
├── rewards/          # Custom reward manager for verl
│   └── memory_reward.py    # Memory rag reward function
├── data/             # Data processing and storage
│   ├── preprocess_locomo.py    # LoCoMo to parquet conversion
│   └── locomo/       # Training data (parquet format)
├── configs/          # Training configurations
│   ├── locomo_memory_grpo.yaml     # Base config for verl training
│   └── tool_config/
│       └── memory_tools.yaml       # Tool definitions for verl
├── outputs/          # Training outputs and logs
├── logs/             # Detailed execution logs
├── pyproject.toml    # Package configuration
└── run_training.sh   # Training script
```

## Dependencies

### Core Dependencies
- **verl**: RL training framework
- **langmem**: Memory management tools  
- **pandas/pyarrow**: Data processing
- **openai/anthropic**: LLM evaluation

### External Data
- **LoCoMo**: Available at `/data/users/alan/locomo/` (conversational memory dataset)

## Usage Examples

### Data Preprocessing
```python
from memupdate.data import LoCoMoProcessor

processor = LoCoMoProcessor()
stats = processor.process_full_pipeline()
print(f"Created {stats['total_training_examples']} training examples")
```

### Memory Tools
```python
from memupdate.tools import SearchMemoryTool

tool = SearchMemoryTool(config={})
result = await tool.execute(
    instance_id="test",
    parameters={"query": "What did Caroline do?", "top_k": 3}
)
```

### Reward Computation
```python
from memupdate.rewards import MemoryRewardManager

reward_manager = MemoryRewardManager(config={
    "evaluator_model": "openai:gpt-4o-mini"
})

reward = await reward_manager.compute_reward(
    memory_old=old_memories,
    memory_new=updated_memories, 
    target_question="When did Caroline attend the support group?",
    target_answer="7 May 2023"
)
```

## Monitoring

Training progress is automatically logged to:
- **Console**: Real-time training metrics
- **WandB**: Experiment tracking with charts and metrics
- **Files**: Local logs in `outputs/` and `logs/` directories

Key metrics tracked:
- QA accuracy improvement
- Memory efficiency scores
- Tool usage patterns  
- Episode lengths and rewards

## Contributing

This is an experimental research project. See the progress log in `progress_log.md` for current implementation status.

## License

[Add your license here]