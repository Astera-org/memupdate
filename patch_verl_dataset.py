"""Patch verl's RLDataset to handle JSON serialized fields."""

import json
from typing import Dict, Any

def patch_rl_dataset():
    """Apply monkey patch to verl's RLDataset."""
    try:
        from verl.utils.dataset.rl_dataset import RLDataset
        
        # Store original __getitem__ method
        original_getitem = RLDataset.__getitem__
        
        def patched_getitem(self, index: int) -> Dict[str, Any]:
            """Patched __getitem__ that handles JSON deserialization."""
            row_dict = self.df.iloc[index].to_dict()
            
            # Deserialize JSON strings back to objects
            if 'extra_info' in row_dict and isinstance(row_dict['extra_info'], str):
                try:
                    row_dict['extra_info'] = json.loads(row_dict['extra_info'])
                except (json.JSONDecodeError, TypeError):
                    row_dict['extra_info'] = {}
            
            if 'messages' in row_dict and isinstance(row_dict['messages'], str):
                try:
                    row_dict['messages'] = json.loads(row_dict['messages'])
                except (json.JSONDecodeError, TypeError):
                    row_dict['messages'] = []
            
            # Ensure extra_info is a dict and has required fields
            if not isinstance(row_dict.get('extra_info', {}), dict):
                row_dict['extra_info'] = {}
            
            # Set default index if missing (this was the original error)
            if 'index' not in row_dict['extra_info']:
                row_dict['extra_info']['index'] = index
                
            return row_dict
        
        # Apply the patch
        RLDataset.__getitem__ = patched_getitem
        print("✅ Successfully patched RLDataset to handle JSON fields")
        return True
        
    except ImportError as e:
        print(f"❌ Could not import verl components: {e}")
        return False
    except Exception as e:
        print(f"❌ Error patching RLDataset: {e}")
        return False

if __name__ == "__main__":
    patch_rl_dataset()