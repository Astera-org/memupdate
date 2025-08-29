# Corrected Memory Architecture for Verl

## The Problem You Identified
You correctly pointed out that the LLM should **discover memory state via function calls**, not have it hardcoded in prompts. The previous approach was flawed.

## ❌ Previous Wrong Approach:
1. Memory hardcoded in system/user prompts as text
2. LLM sees memory as static information
3. Tools operate on separate hidden state
4. No way for LLM to discover current memory

## ✅ Corrected Architecture:

### 1. **Memory Initialization via `create_kwargs`**
```python
"tools_kwargs": {
    "search_memory": {
        "create_kwargs": {"initial_memories": initial_memories, "namespace": conversation_id},
        "execute_kwargs": {"namespace": conversation_id}
    }
}
```
- Tools receive `initial_memories` during creation
- Memory lives in tool state, isolated by `namespace`
- Each conversation gets separate memory space

### 2. **LLM Discovers Memory via Function Calls**
Instead of showing memory in prompt:
```python
user_content = """Target question: What did John do yesterday?

IMPORTANT: Start by calling search_memory() to see what information is currently stored, then optimize as needed."""
```

LLM must call functions to discover state:
```json
{
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "search_memory", 
        "arguments": {"query": "John activities", "limit": 10}
      }
    }
  ]
}
```

### 3. **Tool Returns Current Memory State**
```python
# SearchMemoryTool.execute() returns:
ToolResponse(text="Found 5 memories: 1. John went to store 2. John called mom 3. ...")
```

### 4. **LLM Uses Discovered State to Take Actions**
```json
{
  "tool_calls": [
    {
      "type": "function", 
      "function": {
        "name": "manage_memory",
        "arguments": {
          "operation": "create", 
          "content": "John completed his workout at 6pm yesterday",
          "memory_type": "episodic"
        }
      }
    }
  ]
}
```

## Key Components:

### **Tool State Management**
- `MemoryStoreManager` provides namespace-isolated memory stores
- `create()` methods initialize memory from `create_kwargs`
- `execute()` methods operate on namespaced memory state

### **Memory Flow**
1. **Initialization**: `create_kwargs` loads initial memories into tool state
2. **Discovery**: LLM calls `search_memory()` to see current state  
3. **Analysis**: LLM decides what needs to be changed
4. **Modification**: LLM calls other tools to update memory
5. **Verification**: LLM can search again to verify changes

### **Reward Computation**
```python
# MemoryRewardManager compares:
initial_memories = extra_info["initial_memories"]  # From create_kwargs
final_memories = MemoryStoreManager.get_current_memories(namespace)  # From tool state
reward = compute_improvement(initial_memories, final_memories, target_question)
```

## Why This Is Correct:

✅ **LLM must actively discover memory state** (not passive recipient)  
✅ **Memory persists across tool calls** within episode  
✅ **Each conversation isolated** via namespace  
✅ **Standard verl tool interface** - no framework modifications needed  
✅ **Parquet compatibility** - no complex Python objects in data  

## Files Changed:
- `memupdate/data/preprocess_locomo.py`: Added `create_kwargs` with initial_memories
- `memupdate/tools/search_memory.py`: Added `create()` method for memory initialization  
- `memupdate/tools/manage_memory.py`: Added `create()` method for memory initialization
- `run_training_container.sh`: Use corrected data files

This follows **exactly** the architecture you described - memory in tool state, LLM discovers via function calls! 🎉