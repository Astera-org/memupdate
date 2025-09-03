# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Rules

1. Do NOT git commit or push unless explicitly told to do so.
2. Python execution must be done in Docker container - local Python lacks dependencies.
3. File examination and editing should be done locally (files are mounted to Docker).

## Project Overview

MemUpdate is a reinforcement learning system that trains LLM agents to optimize memory databases through multi-turn tool interactions. The system uses GRPO (Generalized Reward Preference Optimization) with Ray + SGLang + FSDP for distributed training on the LoCoMo dataset.

**Core Architecture**: LLM agents discover and modify memory through function calls rather than hardcoded prompts, with a custom reward system that measures performance improvements.

## Development Environment

**Critical**: This project requires the verl Docker container to run. All development and testing must be done inside the container. By default you should assume there is already a docker container running. Below are the commands to start and enter the container but you can usually skip these and just use the commands at the bottom (Execute Python commands via Docker) to run things inside the container.

```bash
# Start container (from host)
docker run --name verl_container -d --gpus all \
  -v /path/to/memupdate:/workspace/memupdate \
  -v /path/to/verl:/workspace/verl \
  --shm-size=10g \
  verlai/verl:app-verl0.5-transformers4.55.4-sglang0.4.10.post2-mcore0.13.0-te2.2 \
  sleep infinity

# Enter container for development
docker exec -it verl_container bash

# Execute Python commands via Docker
docker exec verl_container bash -c "cd /workspace/memupdate && python3 script.py"
```

## Key Commands

### Training

Note: usually you shouldn't run this, I manually start this in docker and tell you the results.

```bash
# Main training command (run inside Docker container)
docker exec verl_container bash -c "cd /workspace/memupdate && bash run_training_container.sh"

# Test with limited steps
# Edit run_training_container.sh and set: trainer.total_training_steps=1

# Monitor training
# Check WandB dashboard at project: memupdate-rl
```

### Data Processing

```bash
# Regenerate training data (inside container)
docker exec verl_container bash -c "cd /workspace/memupdate && python3 memupdate/data/preprocess_locomo.py --input /workspace/locomo/data/locomo10.json"

```

## Architecture Deep Dive

### Memory Management System

The core innovation is a **Ray Actor-based memory broker** that enables cross-process memory sharing between rollout workers and reward workers:

```python
@ray.remote
class MemoryBrokerActor:
    """Shared memory broker across all Ray workers in verl cluster"""
```

**Key Components:**

- `MemoryStoreManager` (class-level singleton): Manages Ray Actor connections
- `MemoryBrokerActor` (Ray Actor): Handles actual InMemoryStore instances per namespace
- **Namespace Isolation**: Each trajectory gets unique ID like `conv-26-qa159-4f01449a`

### Tool Architecture

Six memory management tools inherit from verl's `BaseTool`:

1. **SearchMemoryTool**: RAG-based memory retrieval using LangMem
2. **ManageMemoryTool**: Create/update memories with episodic/semantic/procedural types
3. **DeleteMemoryTool**: Remove memories with optional reasoning
4. **SampleMemoryTool**: Random/diverse/recent sampling for analysis
5. **MergeMemoryTool**: Consolidate memories (summarize/concatenate/extract)
6. **SplitMemoryTool**: Decompose memories (temporal/thematic/speaker)

**Critical Implementation Pattern:**

```python
# WRONG: Creates local store copies (serialization issue)
store = self.store_manager.get_or_create_store(namespace)
result = await langmem_tool.ainvoke(params)

# CORRECT: Direct Ray Actor operations
result = self.store_manager.create_memory_via_actor(namespace, params)
```

### Training Flow

1. **Data Processing**: `preprocess_locomo.py` creates unique trajectory IDs for each QA pair
2. **Tool Initialization**: Tools receive `create_kwargs` with namespace and `initial_memories`
3. **Multi-turn Episodes**: SGLang manages tool calling with `max_assistant_turns=30`
4. **Reward Computation**: `MemoryRewardManager` compares initial vs final memory states
5. **Policy Updates**: GRPO updates using custom memory-aware rewards

### Reward System

```python
reward = performance_delta * memory_efficiency

where:
- performance_delta = QA_score(new_memory) - QA_score(old_memory)
- memory_efficiency = size_factor * density_factor * change_factor
```

Uses RAG retrieval + context-answer overlap as QA performance proxy (no external LLM needed).

## Critical Implementation Details

### Namespace Isolation Fix

**Problem**: Multiple QA pairs from same conversation shared memory banks, causing unrealistic memory growth (169→332 memories).

**Solution**: Generate unique `trajectory_id` per QA pair:

```python
trajectory_id = f"{sample_id}-qa{qa_idx}-{str(uuid.uuid4())[:8]}"
```

### Ray Actor Serialization Fix

**Problem**: `ray.get(actor.get_store())` returns serialized copies, not references to shared store.

**Solution**: Direct Ray Actor methods for all memory operations:

```python
# Tools call these instead of local store operations
self.store_manager.create_memory_via_actor(namespace, data)
self.store_manager.search_memory_via_actor(namespace, query)
```

### LangMem Python 3.10 Compatibility

The container uses Python 3.10 but LangMem expects 3.11+ features:

```bash
# Auto-applied patch in setup
sed -i 's/typing.NotRequired/typing_extensions.NotRequired/g' langmem/knowledge/extraction.py
```

## File Organization

### Core Implementation

- `memupdate/tools/base_memory_tool.py`: Ray Actor memory broker
- `memupdate/tools/[tool_name].py`: Individual tool implementations
- `memupdate/rewards/memory_reward.py`: Custom reward manager
- `memupdate/data/preprocess_locomo.py`: Dataset preprocessing with trajectory isolation

### Configuration

- `configs/tool_config/memory_tools.yaml`: Tool definitions for verl
- `run_training_container.sh`: Complete training configuration
- `configs/locomo_memory_grpo.yaml`: GRPO algorithm parameters

### Utilities

- `patch_reward_loading.py`: Registers custom reward manager with verl
- `test_memory_integration.py`: Integration testing for memory system
- `check_data_format.py`: Validates training data format

## Common Issues

### "Ray Actor not found" errors

Ensure Ray is initialized in the verl container before accessing MemoryBrokerActor.

### Tool import failures

Verify memupdate package is installed with `--no-deps` to avoid version conflicts with container dependencies.

### Memory isolation problems

Check that each trajectory has a unique namespace in logs. Should see `conv-XX-qaYY-ZZZZZZ` format, not repeated `conv-XX`.

### Training hangs on tool initialization

Usually indicates missing LangMem dependencies or import errors. Check tool import logs during SGLang startup.

## Success Indicators

A working training run shows:

- ✅ Unique trajectory namespaces in logs: `conv-26-qa159-4f01449a`
- ✅ Realistic memory growth: `184→190` memories (not `169→332`)
- ✅ Ray Actor operations: `Store now contains X memories after Y operation`
- ✅ Tool execution: `SearchMemoryTool.execute called with query: ...`
- ✅ Reward computation: `Retrieved X final memories from namespace ...`

**Key Debug Pattern**: Memory count consistency between tool operations and reward computation indicates proper cross-process sharing.
