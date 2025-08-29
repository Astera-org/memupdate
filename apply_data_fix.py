#!/usr/bin/env python3
"""
Apply a surgical fix to verl's RLHFDataset to handle JSON serialized fields.
This is a minimal patch that only modifies the __getitem__ method.
"""

import os
import shutil

def apply_surgical_fix():
    """Apply minimal fix to the __getitem__ method in rl_dataset.py"""
    
    original_file = "/workspace/verl/verl/utils/dataset/rl_dataset.py"
    backup_file = "/workspace/verl/verl/utils/dataset/rl_dataset.py.backup"
    
    # Backup original if not already done
    if not os.path.exists(backup_file):
        shutil.copy(original_file, backup_file)
        print(f"✅ Backed up original to: {backup_file}")
    
    # Read the original file
    with open(original_file, 'r') as f:
        lines = f.readlines()
    
    # Find the __getitem__ method and add our fix
    modified = False
    for i, line in enumerate(lines):
        if line.strip() == 'def __getitem__(self, item):':
            # Find where row_dict is assigned
            for j in range(i, min(i+10, len(lines))):
                if 'row_dict: dict = self.dataframe[item]' in lines[j] or 'row_dict = self.dataframe[item]' in lines[j]:
                    # Insert our JSON deserialization code after this line
                    indent = '        '  # 8 spaces for method body
                    fix_code = [
                        '\n',
                        f'{indent}# MEMUPDATE FIX: Handle JSON deserialization for parquet-stored fields\n',
                        f'{indent}import json\n',
                        f'{indent}if "extra_info" in row_dict and isinstance(row_dict["extra_info"], str):\n',
                        f'{indent}    try:\n',
                        f'{indent}        row_dict["extra_info"] = json.loads(row_dict["extra_info"])\n',
                        f'{indent}    except (json.JSONDecodeError, TypeError):\n',
                        f'{indent}        row_dict["extra_info"] = {{}}\n',
                        f'{indent}if "messages" in row_dict and isinstance(row_dict["messages"], str):\n',
                        f'{indent}    try:\n',
                        f'{indent}        row_dict["messages"] = json.loads(row_dict["messages"])\n',
                        f'{indent}    except (json.JSONDecodeError, TypeError):\n',
                        f'{indent}        row_dict["messages"] = []\n',
                        f'{indent}# Ensure extra_info has required fields\n',
                        f'{indent}if not isinstance(row_dict.get("extra_info", {{}}), dict):\n',
                        f'{indent}    row_dict["extra_info"] = {{}}\n',
                        f'{indent}if "index" not in row_dict.get("extra_info", {{}}):\n',
                        f'{indent}    row_dict["extra_info"]["index"] = item\n',
                        '\n'
                    ]
                    
                    # Insert the fix after the row_dict assignment
                    lines[j+1:j+1] = fix_code
                    modified = True
                    print(f"✅ Inserted JSON deserialization fix at line {j+1}")
                    break
            break
    
    if modified:
        # Write the modified file
        with open(original_file, 'w') as f:
            f.writelines(lines)
        print(f"✅ Successfully patched {original_file}")
        return True
    else:
        print("❌ Could not find the location to apply the fix")
        return False

if __name__ == "__main__":
    if apply_surgical_fix():
        print("\n🎯 Data format fix applied successfully!")
        print("The training should now work without JSON deserialization errors.")
    else:
        print("\n❌ Failed to apply the fix. Manual intervention may be needed.")