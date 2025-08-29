#!/usr/bin/env python3

import pandas as pd
import json

def fix_data_format():
    """Fix the data format by deserializing JSON strings back to dictionaries"""
    
    # Fix train data
    print("Fixing training data...")
    train_df = pd.read_parquet("/workspace/memupdate/data/locomo/train.parquet")
    
    # Convert JSON strings back to dictionaries
    train_df['extra_info'] = train_df['extra_info'].apply(json.loads)
    train_df['messages'] = train_df['messages'].apply(json.loads)
    
    train_df.to_parquet("/workspace/memupdate/data/locomo/train.parquet")
    print(f"Fixed {len(train_df)} training samples")
    
    # Fix test data
    print("Fixing test data...")
    test_df = pd.read_parquet("/workspace/memupdate/data/locomo/test.parquet")
    
    # Convert JSON strings back to dictionaries  
    test_df['extra_info'] = test_df['extra_info'].apply(json.loads)
    test_df['messages'] = test_df['messages'].apply(json.loads)
    
    test_df.to_parquet("/workspace/memupdate/data/locomo/test.parquet")
    print(f"Fixed {len(test_df)} test samples")
    
    print("Data format fixed successfully!")

if __name__ == "__main__":
    fix_data_format()