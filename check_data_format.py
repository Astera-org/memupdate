import datasets
import json

print("=== CHECKING NEW DATA FORMAT ===")
ds = datasets.load_dataset("parquet", data_files="data/locomo/train.parquet")["train"]
row = ds[0]

print("\n1. FIELD TYPES (what verl receives):")
for key in row.keys():
    value = row[key]
    print(f"  {key}: {type(value).__name__}")

print("\n2. MESSAGES FIELD (should be 'prompt' for verl):")
if "messages" in row:
    print(f"  messages type: {type(row['messages'])}")
    print(f"  First 100 chars: {row['messages'][:100]}...")
if "prompt" in row:
    print(f"  prompt type: {type(row['prompt'])}")
    print(f"  Content: {row['prompt'][:100]}...")

if "messages" in row:
    messages = json.loads(row["messages"])
    print("\n3. PARSED MESSAGES (what verl needs as 'prompt'):")
    for i, msg in enumerate(messages):
        print(f"  Message {i+1}:")
        print(f"    role: {msg.get('role')}")
        print(f"    content (first 200 chars): {msg.get('content', '')[:200]}...")

print("\n4. EXTRA_INFO FIELD:")
extra_info = json.loads(row["extra_info"])
print(f"  Type after parsing: {type(extra_info)}")
print(f"  Keys: {list(extra_info.keys())}")
print(f"  tools_kwargs keys: {list(extra_info.get('tools_kwargs', {}).keys())}")
print(f"  Has initial_memories: {'initial_memories' in extra_info}")
print(f"  Number of initial_memories: {len(extra_info.get('initial_memories', []))}")

print("\n=== VERL rl_dataset.py EXPECTATIONS ===")
print("\nVERL expects (line 196: messages = example.pop(self.prompt_key)):")
print("  - Field name: 'prompt' (or configured prompt_key)")
print("  - Type: native Python list of dicts")
print("  - Current: We have 'messages' as JSON string AND 'prompt' as string")

print("\nVERL expects (line 328: row_dict.get('extra_info', {}).get('index', 0)):")
print("  - Field name: 'extra_info'")
print("  - Type: native Python dict")
print("  - Current: We have JSON string")

print("\n❌ CRITICAL ISSUES:")
print("1. We're storing 'messages' but verl expects 'prompt' with list type")
print("2. Both 'messages' and 'extra_info' are JSON strings, not native objects")
print("3. Verl cannot parse JSON - it expects native Python objects in parquet")

print("\n✅ SOLUTION NEEDED:")
print("1. Rename 'messages' to 'prompt' in preprocessing")  
print("2. Store as native list, not JSON string")
print("3. Store extra_info as native dict, not JSON string")