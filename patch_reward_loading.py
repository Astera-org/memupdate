#!/usr/bin/env python3
"""
Patch verl reward loading to ensure memupdate is imported.
"""
import os
import sys

# Patch the reward loading function
reward_file = "/workspace/verl/verl/trainer/ppo/reward.py"
backup_file = reward_file + ".backup"

print(f"Patching reward loading in {reward_file}")

# Backup original file
if not os.path.exists(backup_file):
    with open(reward_file, 'r') as f:
        original_content = f.read()
    with open(backup_file, 'w') as f:
        f.write(original_content)
    print("✅ Created backup of original file")

# Read current content
with open(reward_file, 'r') as f:
    content = f.read()

# Add memupdate import before get_reward_manager_cls call
patch = """    # MEMUPDATE: Ensure reward manager registration
    try:
        import sys
        sys.path.insert(0, '/workspace/memupdate')
        import memupdate
        print(f"✅ MemoryRewardManager registered in process {os.getpid()}")
    except Exception as e:
        print(f"⚠️  Failed to import memupdate in process {os.getpid()}: {e}")
    
"""

# Insert the patch before reward_manager_cls = get_reward_manager_cls(reward_manager_name)
if "# MEMUPDATE: Ensure reward manager registration" not in content:
    content = content.replace(
        "    reward_manager_cls = get_reward_manager_cls(reward_manager_name)",
        patch + "    reward_manager_cls = get_reward_manager_cls(reward_manager_name)"
    )
    
    with open(reward_file, 'w') as f:
        f.write(content)
    print("✅ Applied reward loading patch")
else:
    print("✅ Patch already applied")