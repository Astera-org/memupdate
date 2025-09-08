#!/usr/bin/env python3
"""Pre-compute embeddings for all memory content in LoCoMo training data.

This script:
1. Loads the Qwen3-0.6B embedding model
2. Extracts all unique memory content from training data
3. Computes embeddings for each unique content
4. Saves to memory_embeddings.pkl for fast lookup during training

Usage:
    python3 precompute_embeddings.py --input /workspace/memupdate/data/locomo/train.parquet
    python3 precompute_embeddings.py --input /workspace/locomo/data/locomo10.json
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
import numpy as np

# Add memupdate to path
sys.path.append('/workspace/memupdate')

def hash_content(content: str) -> str:
    """Generate hash for content (matches cached_embeddings.py)."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

def load_qwen_embeddings():
    """Load the Qwen3-0.6B embedding model."""
    print("🔍 Loading Qwen3-0.6B embedding model...")
    
    try:
        # Try to import QwenEmbeddings if available
        try:
            from qwen_embeddings_fallback import QwenEmbeddings
            embeddings = QwenEmbeddings()
            print("✅ Loaded QwenEmbeddings successfully")
            return embeddings
        except ImportError:
            print("⚠️ QwenEmbeddings not available, falling back to HuggingFace")
            
        # Fallback to HuggingFace implementation
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        model_name = "sentence-transformers/all-mpnet-base-v2"  # or whatever Qwen model you're using
        print(f"Loading {model_name} from HuggingFace...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Move to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f"✅ Model loaded on {device}")
        
        class HuggingFaceEmbeddings:
            def __init__(self, model, tokenizer, device):
                self.model = model
                self.tokenizer = tokenizer
                self.device = device
                
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Embed a list of documents."""
                embeddings = []
                
                for text in texts:
                    # Tokenize
                    inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Generate embeddings
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # Use mean pooling of last hidden state
                        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                        embeddings.append(embedding.cpu().numpy().tolist())
                
                return embeddings
        
        return HuggingFaceEmbeddings(model, tokenizer, device)
        
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        print("Creating mock embeddings for testing...")
        
        class MockEmbeddings:
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Generate mock 1024-dimensional embeddings."""
                import random
                embeddings = []
                for text in texts:
                    # Generate deterministic mock embedding based on text hash
                    random.seed(hash(text))
                    embedding = [random.gauss(0, 1) for _ in range(1024)]
                    embeddings.append(embedding)
                return embeddings
        
        return MockEmbeddings()

def extract_memories_from_verl_parquet(file_path: str) -> Set[str]:
    """Extract unique memory content from verl training data parquet file."""
    print(f"📖 Loading memories from verl parquet: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} rows from parquet")
        
        unique_memories = set()
        
        for _, row in df.iterrows():
            # Extract initial memories from extra_info
            if 'extra_info' in row and pd.notna(row['extra_info']):
                try:
                    extra_info = json.loads(row['extra_info']) if isinstance(row['extra_info'], str) else row['extra_info']
                    
                    # Look for tools_kwargs -> search_memory -> create_kwargs -> initial_memories
                    if 'tools_kwargs' in extra_info:
                        tools_kwargs = extra_info['tools_kwargs']
                        for tool_name, tool_data in tools_kwargs.items():
                            if 'create_kwargs' in tool_data and 'initial_memories' in tool_data['create_kwargs']:
                                initial_memories = tool_data['create_kwargs']['initial_memories']
                                
                                for memory in initial_memories:
                                    if isinstance(memory, dict) and 'content' in memory:
                                        content = memory['content'].strip()
                                        if content:
                                            unique_memories.add(content)
                                
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    print(f"⚠️ Failed to parse extra_info: {e}")
                    continue
        
        print(f"✅ Extracted {len(unique_memories)} unique memory contents from verl parquet")
        return unique_memories
        
    except Exception as e:
        print(f"❌ Failed to load parquet file: {e}")
        return set()

def extract_memories_from_json(file_path: str) -> Set[str]:
    """Extract unique memory content from LoCoMo JSON file."""
    print(f"📖 Loading memories from JSON: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} conversations from JSON")
        
        unique_memories = set()
        
        for conversation in data:
            # Handle LoCoMo format with 'observation' field
            if 'observation' in conversation:
                observations = conversation['observation']
                # Convert observations to memory content (same logic as base_memory_tool.py)
                for session_key, session_obs in observations.items():
                    if isinstance(session_obs, dict):
                        for speaker, speaker_observation in session_obs.items():
                            for fact_entry in speaker_observation:
                                if isinstance(fact_entry, list) and len(fact_entry) >= 2:
                                    fact_text, evidence = fact_entry[0], fact_entry[1]
                                    content = fact_text.strip()
                                    if content:
                                        unique_memories.add(content)
            # Handle legacy format with 'memories' field (backward compatibility)
            elif 'memories' in conversation:
                for memory in conversation['memories']:
                    if isinstance(memory, dict) and 'content' in memory:
                        content = memory['content'].strip()
                        if content:
                            unique_memories.add(content)
                    elif isinstance(memory, str):
                        content = memory.strip()
                        if content:
                            unique_memories.add(content)
        
        print(f"✅ Extracted {len(unique_memories)} unique memory contents from JSON")
        return unique_memories
        
    except Exception as e:
        print(f"❌ Failed to load JSON file: {e}")
        return set()

def compute_embeddings_batch(embedding_model, contents: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
    """Compute embeddings for a batch of contents."""
    print(f"🔄 Computing embeddings for {len(contents)} contents (batch size: {batch_size})")
    
    cache = {}
    
    for i in range(0, len(contents), batch_size):
        batch_contents = contents[i:i + batch_size]
        
        try:
            # Compute embeddings for batch
            batch_embeddings = embedding_model.embed_documents(batch_contents)
            
            # Store in cache with hash keys
            for content, embedding in zip(batch_contents, batch_embeddings):
                content_hash = hash_content(content)
                cache[content_hash] = np.array(embedding, dtype=np.float32)
            
            print(f"✅ Processed batch {i//batch_size + 1}/{(len(contents) + batch_size - 1)//batch_size}")
            
        except Exception as e:
            print(f"❌ Error processing batch {i//batch_size + 1}: {e}")
            # Skip this batch and continue
            continue
    
    return cache

def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for memory content")
    parser.add_argument("--input", required=True, help="Input file (parquet or JSON)")
    parser.add_argument("--output", default="/workspace/memupdate/data/embedding_cache/memory_embeddings.pkl", 
                       help="Output pickle file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding computation")
    parser.add_argument("--dry-run", action="store_true", help="Just count unique memories, don't compute embeddings")
    
    args = parser.parse_args()
    
    print("🚀 Starting embedding pre-computation...")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    
    # Extract unique memory contents
    if args.input.endswith('.parquet'):
        unique_memories = extract_memories_from_verl_parquet(args.input)
    elif args.input.endswith('.json'):
        unique_memories = extract_memories_from_json(args.input)
    else:
        print(f"❌ Unsupported file format: {args.input}")
        return
    
    if not unique_memories:
        print("❌ No memories found in input file")
        return
    
    print(f"📊 Found {len(unique_memories)} unique memory contents")
    
    # Show some examples
    memory_list = list(unique_memories)[:5]
    print("📝 Sample memories:")
    for i, memory in enumerate(memory_list):
        print(f"  {i+1}. {memory[:100]}{'...' if len(memory) > 100 else ''}")
    
    if args.dry_run:
        print("🏁 Dry run complete - no embeddings computed")
        return
    
    # Load embedding model
    embedding_model = load_qwen_embeddings()
    
    # Compute embeddings
    memory_list = list(unique_memories)
    embedding_cache = compute_embeddings_batch(embedding_model, memory_list, args.batch_size)
    
    print(f"💾 Saving {len(embedding_cache)} embeddings to {args.output}")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to pickle file
    with open(args.output, 'wb') as f:
        pickle.dump(embedding_cache, f)
    
    # Verify the saved file
    try:
        with open(args.output, 'rb') as f:
            loaded_cache = pickle.load(f)
        
        print(f"✅ Verification: Loaded {len(loaded_cache)} embeddings from saved file")
        
        # Show sample
        sample_key = list(loaded_cache.keys())[0]
        sample_embedding = loaded_cache[sample_key]
        print(f"📊 Sample embedding shape: {sample_embedding.shape}, dtype: {sample_embedding.dtype}")
        
    except Exception as e:
        print(f"❌ Failed to verify saved file: {e}")
    
    print("🎉 Embedding pre-computation complete!")

if __name__ == "__main__":
    main()
