#!/usr/bin/env python3
"""Script to examine the train.parquet dataset with breakpoints."""

import pandas as pd
import json
import sys
from pathlib import Path

def examine_parquet_dataset():
    """Examine the structure and content of train.parquet."""
    
    # Load the dataset
    parquet_path = "/workspace/memupdate/data/locomo/test.parquet"
    
    if not Path(parquet_path).exists():
        print(f"❌ File not found: {parquet_path}")
        return
    
    print(f"📖 Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    # Add breakpoint here to examine df
    breakpoint()  # <-- BREAKPOINT 1: Examine basic dataset structure
    
    # Look at first few rows
    print("\n" + "="*80)
    print("FIRST FEW ROWS:")
    for i in range(min(3, len(df))):
        print(f"\n--- ROW {i} ---")
        row = df.iloc[i]
        for col in df.columns:
            value = row[col]
            if isinstance(value, str) and len(value) > 100:
                print(f"{col}: {value[:100]}...")
            else:
                print(f"{col}: {value}")
    
    # Add breakpoint here to examine specific rows
    breakpoint()  # <-- BREAKPOINT 2: Examine row contents
    
    # Focus on extra_info column structure
    print("\n" + "="*80)
    print("EXAMINING EXTRA_INFO STRUCTURE:")
    
    sample_extra_infos = []
    for i in range(min(5, len(df))):
        extra_info = df.iloc[i]['extra_info']
        if pd.notna(extra_info):
            try:
                parsed = json.loads(extra_info) if isinstance(extra_info, str) else extra_info
                sample_extra_infos.append((i, parsed))
                print(f"\nRow {i} extra_info keys: {list(parsed.keys())}")
            except Exception as e:
                print(f"Row {i} failed to parse: {e}")
    
    # Add breakpoint here to examine extra_info structure
    breakpoint()  # <-- BREAKPOINT 3: Examine extra_info structure
    
    # Extract and examine tools_kwargs
    print("\n" + "="*80)
    print("EXAMINING TOOLS_KWARGS:")
    
    for row_idx, extra_info in sample_extra_infos:
        if 'tools_kwargs' in extra_info:
            tools_kwargs = extra_info['tools_kwargs']
            print(f"\nRow {row_idx} tools_kwargs keys: {list(tools_kwargs.keys())}")
            
            for tool_name, tool_data in tools_kwargs.items():
                print(f"  {tool_name} keys: {list(tool_data.keys())}")
                if 'create_kwargs' in tool_data:
                    create_kwargs = tool_data['create_kwargs']
                    print(f"    create_kwargs keys: {list(create_kwargs.keys())}")
                    if 'initial_memories' in create_kwargs:
                        memories = create_kwargs['initial_memories']
                        print(f"    initial_memories count: {len(memories)}")
                        if memories:
                            first_mem = memories[0]
                            print(f"    first memory keys: {list(first_mem.keys()) if isinstance(first_mem, dict) else 'not dict'}")
    
    # Add breakpoint here to examine tools_kwargs
    breakpoint()  # <-- BREAKPOINT 4: Examine tools_kwargs and initial_memories
    
    # Extract some actual memory content
    print("\n" + "="*80)
    print("SAMPLE MEMORY CONTENTS:")
    
    memory_samples = []
    for row_idx, extra_info in sample_extra_infos:
        if ('tools_kwargs' in extra_info and 
            len(extra_info['tools_kwargs']) > 0):
            
            tools_kwargs = extra_info['tools_kwargs']
            for tool_name, tool_data in tools_kwargs.items():
                if ('create_kwargs' in tool_data and 
                    'initial_memories' in tool_data['create_kwargs']):
                    
                    memories = tool_data['create_kwargs']['initial_memories']
                    for i, mem in enumerate(memories[:3]):  # First 3 memories
                        if isinstance(mem, dict) and 'content' in mem:
                            memory_samples.append({
                                'row': row_idx,
                                'tool': tool_name,
                                'memory_idx': i,
                                'content': mem['content'],
                                'metadata': mem.get('metadata', {}),
                                'id': mem.get('id', 'no_id')
                            })
    
    for i, sample in enumerate(memory_samples[:10]):  # Show first 10
        print(f"\nMemory {i+1}:")
        print(f"  From row {sample['row']}, tool {sample['tool']}, memory #{sample['memory_idx']}")
        print(f"  ID: {sample['id']}")
        print(f"  Content: {sample['content'][:150]}...")
        print(f"  Metadata: {sample['metadata']}")
    
    # Final breakpoint to examine extracted memory samples
    breakpoint()  # <-- BREAKPOINT 5: Examine extracted memory samples
    
    print(f"\n🎯 Total memory samples extracted: {len(memory_samples)}")
    print("✅ Examination complete!")

if __name__ == "__main__":
    examine_parquet_dataset()