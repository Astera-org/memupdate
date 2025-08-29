"""Custom dataset loader that handles JSON deserialization for verl."""

import json
from typing import Dict, Any
import pandas as pd
import torch
from verl.utils.dataset.rl_dataset import RLDataset


class MemUpdateRLDataset(RLDataset):
    """Custom RL dataset that deserializes JSON fields."""
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item with proper JSON deserialization."""
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
        
        # Ensure extra_info has required fields
        if not isinstance(row_dict.get('extra_info', {}), dict):
            row_dict['extra_info'] = {}
        
        # Set default index if missing
        if 'index' not in row_dict['extra_info']:
            row_dict['extra_info']['index'] = index
            
        return row_dict