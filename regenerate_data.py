#!/usr/bin/env python3

import sys
import os
sys.path.append('/workspace/memupdate')

from memupdate.data.preprocess_locomo import LoCoMoProcessor
import pandas as pd
import json

def safe_to_parquet(df, filepath):
    """Save DataFrame to parquet with proper handling of complex types."""
    # Make a copy and handle complex fields
    df_copy = df.copy()
    
    # For verl compatibility, keep extra_info as dict but make sure it's serializable
    for idx, row in df_copy.iterrows():
        if 'extra_info' in row and isinstance(row['extra_info'], dict):
            # Ensure all values in extra_info are JSON serializable
            try:
                json.dumps(row['extra_info'])
            except:
                # If serialization fails, convert to string
                df_copy.at[idx, 'extra_info'] = str(row['extra_info'])
        
        if 'messages' in row and isinstance(row['messages'], list):
            # Ensure messages list is properly formatted
            try:
                json.dumps(row['messages'])
            except:
                df_copy.at[idx, 'messages'] = str(row['messages'])
    
    # Save with simpler format
    df_copy.to_parquet(filepath, index=False, engine='pyarrow')
    

def regenerate_data():
    """Regenerate data files without JSON string serialization."""
    processor = LoCoMoProcessor(locomo_data_path="/workspace/locomo/data/locomo10.json")
    
    # Load and split data
    train_conversations, test_conversations = processor.create_train_test_split()
    
    # Extract QA pairs
    train_trials = processor.extract_qa_pairs(train_conversations)
    test_trials = processor.extract_qa_pairs(test_conversations)
    
    print(f"Extracted {len(train_trials)} training trials, {len(test_trials)} test trials")
    
    # Create training data WITHOUT serialization
    train_data = processor.create_verl_training_data(train_trials)
    test_data = processor.create_verl_training_data(test_trials)
    
    # Create DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data) 
    
    # Save with proper handling
    output_dir = "/workspace/memupdate/data/locomo"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        safe_to_parquet(train_df, f"{output_dir}/train.parquet")
        print(f"✅ Saved {len(train_data)} training samples")
    except Exception as e:
        print(f"❌ Error saving training data: {e}")
        # Fallback: serialize to JSON strings
        train_df['extra_info'] = train_df['extra_info'].apply(json.dumps)
        train_df['messages'] = train_df['messages'].apply(json.dumps)
        train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
        print(f"✅ Saved training data with JSON serialization fallback")
    
    try:
        safe_to_parquet(test_df, f"{output_dir}/test.parquet")
        print(f"✅ Saved {len(test_data)} test samples")
    except Exception as e:
        print(f"❌ Error saving test data: {e}")
        # Fallback: serialize to JSON strings
        test_df['extra_info'] = test_df['extra_info'].apply(json.dumps)
        test_df['messages'] = test_df['messages'].apply(json.dumps)
        test_df.to_parquet(f"{output_dir}/test.parquet", index=False)
        print(f"✅ Saved test data with JSON serialization fallback")


if __name__ == "__main__":
    regenerate_data()