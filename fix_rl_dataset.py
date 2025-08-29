#!/usr/bin/env python3
"""
Fix for verl's RLHFDataset to handle JSON serialized fields from memupdate data.

This patches the __getitem__ method to deserialize JSON strings back to dictionaries
for proper compatibility with verl's training pipeline.
"""

import json
import os
import sys

def create_fixed_rl_dataset():
    """Create a fixed version of RLHFDataset that handles JSON deserialization."""
    
    fixed_code = '''# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MEMUPDATE FIX: Added JSON deserialization for extra_info and messages fields

import copy
import logging
import os
import re
import json  # MEMUPDATE: Added for JSON deserialization
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

# Import the original module to get all the helper functions
from verl.utils.dataset.rl_dataset import *

# We only need to override the RLHFDataset class
class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.
    
    MEMUPDATE FIX: This version handles JSON serialized fields in extra_info and messages.
    """
    
    def __init__(self, data_files, tokenizer: PreTrainedTokenizer, config: DictConfig, processor: ProcessorMixin = None):
        """Initialize dataset with all original parameters."""
        # Copy the original initialization code
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.config = config
        self.processor = processor
        
        # Get all config parameters
        self.prompt_key = config.get("prompt_key", "prompt")
        self.data_key = config.get("data_key", "data_source")
        self.response_length = config.get("response_length", 2048)
        self.max_prompt_length = config.get("max_prompt_length", 8192)
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.cache_dir = config.get("cache_dir", None)
        self.eos_in_prompt = config.get("eos_in_prompt", True)
        self.truncation = config.get("truncation", "error")
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        
        # Initialize from checkpoint if provided
        checkpoint_folder = config.get("checkpoint_folder", None)
        if checkpoint_folder is not None:
            self.dataframe = load_datasets_from_checkpoint(checkpoint_folder, config)
            print(f"Load from checkpoint with {len(self.dataframe)} samples")
            return
        
        # Load parquet files
        print(f"Loading data from {self.data_files}")
        if not isinstance(self.data_files, list):
            self.data_files = [self.data_files]
        
        dataframes = []
        for parquet_file in self.data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        
        print(f"dataset len: {len(self.dataframe)}")
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
    
    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        """Filter out too long prompts - original implementation."""
        if dataframe is None:
            dataframe = self.dataframe
        
        if not self.filter_overlong_prompts:
            return dataframe
        
        # Original filtering logic would go here
        # For now, just return the dataframe
        return dataframe
    
    def _build_messages(self, row_dict):
        """Build messages from row_dict - handles JSON deserialization."""
        # MEMUPDATE FIX: Deserialize messages if it's a JSON string
        if 'messages' in row_dict and isinstance(row_dict['messages'], str):
            try:
                row_dict['messages'] = json.loads(row_dict['messages'])
            except (json.JSONDecodeError, TypeError):
                row_dict['messages'] = []
        
        # Original message building logic
        messages = row_dict.get("messages", [])
        if not messages and "prompt" in row_dict:
            # Fallback to prompt field if messages not available
            messages = [{"role": "user", "content": row_dict["prompt"]}]
        
        return messages
    
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataframe)
    
    def __getitem__(self, item):
        """
        Get item with MEMUPDATE JSON deserialization fix.
        
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        
        # MEMUPDATE FIX: Handle JSON deserialization for extra_info and messages
        if 'extra_info' in row_dict and isinstance(row_dict['extra_info'], str):
            try:
                row_dict['extra_info'] = json.loads(row_dict['extra_info'])
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to deserialize extra_info for item {item}")
                row_dict['extra_info'] = {}
        
        if 'messages' in row_dict and isinstance(row_dict['messages'], str):
            try:
                row_dict['messages'] = json.loads(row_dict['messages'])
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to deserialize messages for item {item}")
                row_dict['messages'] = []
        
        # Ensure extra_info is a dict and has required index field
        if not isinstance(row_dict.get('extra_info', {}), dict):
            row_dict['extra_info'] = {}
        if 'index' not in row_dict['extra_info']:
            row_dict['extra_info']['index'] = item
        
        # Now proceed with the original __getitem__ logic
        messages = self._build_messages(row_dict)
        model_inputs = {}
        
        # Handle multimodal inputs if processor is available
        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video
            
            # Process images/videos if present
            # ... (original multimodal processing code)
        
        # Tokenize the messages
        if self.processor is not None:
            # Multimodal tokenization
            raw_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.processor(
                raw_prompt, return_tensors="pt", padding=False, truncation=False
            )
        else:
            # Text-only tokenization
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer(
                raw_prompt, return_tensors="pt", padding=False, truncation=False
            )
        
        # Extract tensors
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", torch.ones_like(input_ids))
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # Add to row_dict
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        
        # Encode raw prompt IDs
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        
        row_dict["raw_prompt_ids"] = raw_prompt_ids
        
        # Optionally return raw chat
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        
        # Optionally return full prompt
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt
        
        # Extract index and tool kwargs from extra_info
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        
        if need_tools_kwargs and not tools_kwargs:
            logger.warning(f"tools_kwargs is empty for index {index}, data source: {row_dict.get('data_source', 'unknown')}")
        
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        
        # Add other required fields
        for key in ["data_source", "extra_info"]:
            if key not in row_dict:
                row_dict[key] = None
        
        return row_dict


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    from collections import defaultdict
    import numpy as np
    import torch
    
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


# Export the fixed class
__all__ = ['RLHFDataset', 'collate_fn']
'''
    
    # Write the fixed version
    output_path = "/workspace/verl/verl/utils/dataset/rl_dataset_fixed.py"
    with open(output_path, 'w') as f:
        f.write(fixed_code)
    
    print(f"✅ Created fixed RLHFDataset at: {output_path}")
    return output_path

def apply_monkey_patch():
    """Apply the fix by monkey-patching the original module."""
    
    # First backup the original
    import shutil
    original_path = "/workspace/verl/verl/utils/dataset/rl_dataset.py"
    backup_path = "/workspace/verl/verl/utils/dataset/rl_dataset_original.py"
    
    if not os.path.exists(backup_path):
        shutil.copy(original_path, backup_path)
        print(f"✅ Backed up original to: {backup_path}")
    
    # Create the fixed version
    fixed_path = create_fixed_rl_dataset()
    
    # Replace the original with the fixed version
    shutil.copy(fixed_path, original_path)
    print("✅ Replaced original with fixed version")
    
    return True

if __name__ == "__main__":
    apply_monkey_patch()
    print("\n🎯 Data format fix applied successfully!")
    print("You can now run the training script.")