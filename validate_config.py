import yaml

# Check our tool config file has all required tools
with open("configs/tool_config/memory_tools.yaml", "r") as f:
    config = yaml.safe_load(f)

print("=== TOOL CONFIG VALIDATION ===")
print(f"Number of tools defined: {len(config['tools'])}")

# Extract tool names from class names
defined_tools = []
for tool in config["tools"]:
    class_name = tool["class_name"]
    # Extract tool name from class path like "memupdate.tools.search_memory.SearchMemoryTool"
    tool_name = class_name.split(".")[-2]  # Gets "search_memory"
    defined_tools.append(tool_name)
    print(f"  - {tool_name}: {class_name}")

print(f"\nDefined tools: {defined_tools}")

# Check if all required tools from our data are present
required_tools = ["search_memory", "manage_memory", "delete_memory", "sample_memory", "merge_memory", "split_memory"]
print(f"Required tools: {required_tools}")

missing = set(required_tools) - set(defined_tools)
extra = set(defined_tools) - set(required_tools)

print(f"\n✅ All required tools present: {len(missing) == 0}")
if missing:
    print(f"❌ Missing tools: {missing}")
if extra:
    print(f"ℹ️  Extra tools: {extra}")

# Check tool configs have required fields
print(f"\n=== TOOL CONFIG STRUCTURE ===")
for tool in config["tools"]:
    name = tool["class_name"].split(".")[-2]
    has_type = "type" in tool.get("config", {})
    print(f"  - {name}: has 'type' config = {has_type}")
    if not has_type:
        print(f"    ❌ Missing required 'type' field in config")