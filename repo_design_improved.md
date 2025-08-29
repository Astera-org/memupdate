# MemUpdate - Self-Refining Memory via RL

## Overview

MemUpdate is an experimental project that explores the possibility of self-refining memory in LLMs via Reinforcement Learning. It uses DAPO (improved GRPO) RL methods to train a model for updating any existing memory database to maximize rewards for future question-answering tasks.

**Core Concept**: Train an agent to iteratively improve memory database through tool use, optimizing for better performance on ANY questions tomorrow.

**Package Strategy**: Use LoCoMo, LangMem, and verl as external packages to minimize local dependencies and leverage existing implementations. Use APIs until we really have to have a local copy to work on.

## Data Pipeline

### Input Data Structure (LoCoMo Dataset)

The LoCoMo dataset (`~/locomo/data/locomo10.json`) contains conversation pairs with the following structure:

```json
{
  "qa": [
    {
      "question": "When did Caroline go to the LGBTQ support group?",
      "answer": "7 May 2023",
      "evidence": ["D1:3"],  // References to conversation segments
      "category": 2  // Question type (1=identity, 2=temporal, 3=inference, 4=preference)
    }
  ],
  "conversations": {
    "D1": [
      {
        "speaker": "Caroline",
        "dia_id": "D1:3",
        "text": "I attended an LGBTQ support group on May 7th...",
        "img_url": ["..."],  // Optional image sharing
        "blip_caption": "..."  // Image descriptions
      }
    ],
    "D2": [...],  // Additional conversation sessions
  },
  "session_summary": {
    "session_1_summary": "Caroline and Melanie had a conversation on 8 May 2023...",
    // Summaries for each conversation session
  },
  "facts": {
    "Caroline": [
      [
        "Caroline attended an LGBTQ support group and was inspired by transgender stories.",
        "D1:3"
      ]
    ]
  },
  "sample_id": "conv-26"
}
```

### Memory Database Format (LangMem)

LangMem handles all memory storage automatically:

- **Built-in Tools**: `create_manage_memory_tool()` and `create_search_memory_tool()`
- **Automatic**: Embedding generation, vector storage, similarity search all handled internally
- **Agent Integration**: Tools can be directly used by LangGraph agents

We just wrap LangMem's tools in verl's `BaseTool` interface for integration.

### Data Splits and Processing

- **Dataset Scale**: 10 conversations total, 1,986 QA pairs (198.6 avg per conversation)
- **Train/Test Split**: 7 train conversations (1,347 QA pairs) / 3 test conversations (639 QA pairs)
- **Training Trials**: Each trial = 1 conversation + 1 question → **1,347 unique training trials**
- **Memory Initialization**: Convert conversation facts to initial LangMem entries (~20-40 memories per conversation). Use Mem0's conversion script (https://github.com/mem0ai/mem0/blob/main/evaluation/src/langmem.py) as reference, implement custom if needed.
- **Episode Structure**: One episode = one memory update session for one specific question

## Agent Architecture

### Decision Flow

**Training Episode Flow:**

1. **Sample**: Pick 1 conversation + 1 question from training data
2. **Initialize**: Create initial memory M from conversation facts
3. **Agent Loop**: Agent receives memory state + target question
   - Tool selection via function calling
   - Tool execution updates memory → M'
   - Repeat for max 30 steps or until agent stops
4. **Evaluate**: RAG-based QA performance on target question only
5. **Reward**: Compute reward = performance_delta × memory_efficiency
6. **Update**: Policy gradient step

### Tool Specifications

All tools use verl's `BaseTool` interface with the following specifications:

#### 1. `search_memory.py` - Memory Retrieval

```python
{
  "operation": "search",
  "query": str,  # Natural language query
  "top_k": int,  # Default: 5
  "memory_type": str,  # episodic/semantic/procedural (optional)
  "threshold": float  # Similarity threshold (optional)
}
# Returns: List of relevant memory entries with scores
```

#### 2. `create_memory.py` - Memory Creation

```python
{
  "operation": "create",
  "content": str,  # Memory content
  "memory_type": str,  # episodic/semantic/procedural
  "metadata": dict,  # Additional context
  "source": str  # Reference to original information
}
# Returns: New memory ID and confirmation
```

#### 3. `delete_memory.py` - Memory Deletion

```python
{
  "operation": "delete",
  "memory_id": str,  # Target memory ID
  "reason": str  # Reason for deletion (optional)
}
# Returns: Deletion confirmation
```

#### 4. `sample_memory.py` - Memory Sampling

```python
{
  "operation": "sample",
  "k": int,  # Number of memories to sample
  "memory_type": str,  # Filter by type (optional)
  "strategy": str  # random/diverse/recent (default: random)
}
# Returns: List of sampled memories
```

#### 5. `merge_memory.py` - Memory Consolidation

```python
{
  "operation": "merge",
  "memory_ids": List[str],  # 2-5 memories to merge
  "strategy": str  # summarize/concatenate/extract_common
}
# Returns: New consolidated memory ID
```

#### 6. `split_memory.py` - Memory Decomposition

```python
{
  "operation": "split",
  "memory_id": str,  # Memory to split
  "split_criteria": str,  # temporal/thematic/speaker
  "max_parts": int  # Maximum number of parts (default: 3)
}
# Returns: List of new memory IDs
```

### Agent Decision Process

The agent uses multi-turn generation with SGLang backend:

1. **Context**: Current memory state summary + **target question** for this episode
2. **Tool Selection**: Function calling to select appropriate memory operations
3. **Execution Feedback**: Tool execution results guide next actions
4. **Termination**: Agent can end episode early or hit max_steps (30)
5. **Evaluation**: QA performance tested only on the **target question** for this episode

## Training Configuration

### Training Script (Following verl Pattern)

```bash
#!/bin/bash

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/configs"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='locomo_memory_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=15 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/configs/tool_config/memory_tools.yaml" \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='locomo-memory-update' \
    trainer.experiment_name='qwen2.5-3b-memory-rl' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=25 \
    trainer.total_training_steps=1347 \
    data.train_files="$PROJECT_DIR/data/locomo/train.parquet" \
    data.val_files="$PROJECT_DIR/data/locomo/test.parquet" \
    trainer.total_epochs=1
```

### Key Training Parameters

- **Batch Size**: 64 (conversation, question) pairs per batch
- **Total Steps**: 1,347 (matching number of training QA pairs)
- **Evaluation**: Every 25 training steps using test.parquet validation data
- **Multi-turn**: Up to 15 assistant turns (30 memory operations)
- **Data Format**: Parquet files following verl pattern
- **Logging**: WandB integration for training and validation metrics

## Reward Function

### Mathematical Definition

```python
def compute_reward(M_old, M_new, target_question, target_answer, evaluator_model):
    """
    Args:
        M_old: Original memory database
        M_new: Updated memory database
        target_question: Single question for this episode
        target_answer: Ground truth answer for this episode
        evaluator_model: Fixed LLM for evaluation (same as agent model)
    """
    # 1. Compute QA performance difference for target question only
    p_old = evaluate_single_qa(M_old, target_question, target_answer, evaluator_model)
    p_new = evaluate_single_qa(M_new, target_question, target_answer, evaluator_model)
    performance_delta = p_new - p_old

    # 2. Memory efficiency penalty
    size_penalty = 1 - (len(M_new) / max_total_memories)

    # 3. Final reward
    reward = performance_delta * size_penalty
    return reward

def evaluate_single_qa(memory_db, question, answer, model):
    """
    Evaluate QA performance for a single question:
    1. Retrieve top-k memories using RAG (LangMem search tool)
    2. Generate answer using model
    3. Compare against ground truth using exact match + LLM judge
    4. Return 0-1 score
    """
    context = rag_retrieve(memory_db, question, top_k=5)
    generated_answer = model.generate(question, context)
    score = evaluate_answer(generated_answer, answer)  # 0-1 score
    return score
```

## Technical Implementation

### Project Structure

```
memupdate/
├── agents/
│   └── memory_agent.py          # Simple local agent (no LangGraph complexity)
├── tools/
│   ├── search_memory.py         # Memory retrieval (simple LangMem wrapper)
│   ├── manage_memory.py         # Memory creation/update (simple LangMem wrapper)
│   ├── delete_memory.py         # Memory deletion
│   ├── sample_memory.py         # Memory sampling
│   ├── merge_memory.py          # Memory consolidation
│   └── split_memory.py          # Memory decomposition
├── rewards/
│   └── memory_reward.py         # Custom reward manager for verl
├── data/
│   ├── preprocess_locomo.py     # LoCoMo to parquet conversion for verl
│   ├── locomo/
│   │   ├── train.parquet        # Training data (1,347 QA pairs)
│   │   └── test.parquet         # Validation data (639 QA pairs)
├── configs/
│   ├── tool_config/
│   │   └── memory_tools.yaml    # Tool definitions for verl
│   └── locomo_memory_grpo.yaml  # Base config for verl training
├── pyproject.toml               # uv package management
└── run_training.sh              # Bash training script (following verl pattern)
```

### Package Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "verl",  # Main RL framework
    "langmem",  # Memory tools
    "openai",  # For embeddings/LLM
    "pandas",  # For parquet conversion
]
```

### Memory Storage Backend (LangMem)

**LangMem handles everything automatically** - we don't manage storage directly:

- **Tools**: `create_manage_memory_tool()` and `create_search_memory_tool()`
- **Integration**: Wrap LangMem tools in verl's `BaseTool` interface
- **Agent Access**: Tools called via function calling during multi-turn generation

### Integration with External Packages

#### 1. LoCoMo Dataset Integration

```python
# Use LoCoMo as package if available, otherwise direct file access
try:
    from locomo import load_dataset
    conversations = load_dataset('locomo10')
except ImportError:
    # Fallback to direct JSON loading
    import json
    with open('path/to/locomo10.json') as f:
        conversations = json.load(f)
```

#### 2. LangMem Integration

```python
# Import LangMem tools
from langmem import create_manage_memory_tool, create_search_memory_tool
from verl.tools.base_tool import BaseTool

# Simple LangMem wrappers - no LangGraph complexity
class SearchMemoryTool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.langmem_search = create_search_memory_tool()

    async def execute(self, instance_id, parameters, **kwargs):
        # Simple delegation to LangMem
        result = await self.langmem_search.invoke(parameters)
        return ToolResponse(text=str(result)), 0.1, {}

class ManageMemoryTool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.langmem_manage = create_manage_memory_tool()

    async def execute(self, instance_id, parameters, **kwargs):
        result = await self.langmem_manage.invoke(parameters)
        return ToolResponse(text=str(result)), 0.1, {}
```

#### 3. verl Integration

```python
# Register custom components with verl's extension system
from verl.tools.utils.tool_registry import register_tool
from verl.workers.reward_manager import register as register_reward
from verl.interactions.utils.interaction_registry import register_interaction

# Register simple LangMem wrapper tools
register_tool("search_memory", SearchMemoryTool)  # Simple LangMem search wrapper
register_tool("manage_memory", ManageMemoryTool)  # Simple LangMem manage wrapper
register_tool("sample_memory", SampleMemoryTool)  # Custom sampling logic
# ... etc
register_reward("memory_rag")(MemoryRewardManager)
```

### Evaluation Metrics (Built into verl Training)

**Automatic Evaluation Every 25 Steps:**
1. **Primary**: QA accuracy improvement on validation set (EM + F1 score)
2. **Memory Metrics**: Tool usage patterns, memory operations per episode
3. **Training Metrics**: Reward curves, policy gradient losses, KL divergence
4. **Efficiency**: Episode length, convergence speed
5. **Logging**: All metrics automatically logged to WandB during training

## Implementation Notes

### Handling Edge Cases

- **Empty Memory**: Initialize with minimal conversation facts
- **Tool Failures**: Return error messages to agent for recovery
- **Memory Conflicts**: Use timestamps and metadata for resolution
- **Large Conversations**: Chunk into multiple episodes if needed

### Performance Optimizations

- **Batch Processing**: Process multiple conversations in parallel
- **Embedding Caching**: Cache embeddings for repeated content
- **Memory Pruning**: Remove low-relevance memories automatically
- **Incremental Updates**: Only re-embed changed memories

### Package Management Strategy

1. **Primary Dependencies**: Install via pip/conda (verl, chromadb, transformers)
2. **LoCoMo Data**: Direct file access from local copy
3. **LangMem Compatibility**: Use Mem0's conversion script as reference, implement custom if needed
4. **Fallback Implementations**: Custom implementations for any missing package functionality

### Future Extensions

1. **Multi-modal Memory**: Support for image/video memories from LoCoMo
2. **Hierarchical Memory**: Multiple levels of abstraction
3. **Collaborative Memory**: Multi-agent memory sharing
4. **Continual Learning**: Memory updates across multiple conversations

## Questions for Clarification

1. **Memory Size Constraints**: What should be the maximum memory database size per conversation?
2. **Evaluation Strategy**: Should we evaluate on all LoCoMo question categories or focus on specific types?
3. **Tool Complexity**: Should merge/split tools use LLM-based processing or rule-based heuristics?
4. **Baseline Comparison**: What specific baseline memory update strategies should we compare against?
5. **Embedding Model**: Any preference for embedding model (sentence-transformers vs. model-specific embeddings)?
