import datasets
import json

ds = datasets.load_dataset("parquet", data_files="data/locomo/train.parquet")["train"]
row = ds[0]

print("=== FINAL DATA FORMAT ===")
print("\nFields in parquet:")
for key in row.keys():
    value = row[key]
    print(f"  {key}: type={type(value).__name__}")

print("\n=== FIELD DETAILS ===")

print("\n1. PROMPT field (verl expects list here):")
prompt = row["prompt"]
print(f"  Type: {type(prompt)}")
if isinstance(prompt, list):
    print(f"  Length: {len(prompt)}")
    for i, msg in enumerate(prompt):
        print(f"  Message {i+1}: role={msg.get('role')}, content_len={len(msg.get('content', ''))}")

print("\n2. EXTRA_INFO field (needs JSON parsing):")
extra_info_str = row["extra_info"]
print(f"  Raw type: {type(extra_info_str)}")
if isinstance(extra_info_str, str):
    extra_info = json.loads(extra_info_str)
    print(f"  After parsing: {type(extra_info)}")
    print(f"  Keys: {list(extra_info.keys())}")
    print(f"  tools_kwargs present: {'tools_kwargs' in extra_info}")
    print(f"  initial_memories count: {len(extra_info.get('initial_memories', []))}")

print("\n=== WHAT VERL EXPECTS ===")
print("✅ prompt field: list of message dicts - WE HAVE THIS")
print("❌ extra_info field: dict object - WE HAVE JSON STRING (needs fix in verl)")

print("\n=== SOLUTION ===")
print("We need to add JSON deserialization in verl for extra_info field only.")
print("The prompt field is already in the correct format!")