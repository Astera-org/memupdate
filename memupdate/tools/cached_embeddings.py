"""Hybrid embedding wrapper: cached for known content, GPU for queries."""

import torch
import hashlib
import pickle
import numpy as np
from typing import List
from langchain_core.embeddings import Embeddings


class HybridCachedEmbeddings(Embeddings):
    """Hybrid embeddings: use cache for known content, GPU model for search queries."""
    
    def __init__(self, gpu_embeddings, cache_file: str = "/workspace/memupdate/data/embedding_cache/memory_embeddings.pkl"):
        self.gpu_embeddings = gpu_embeddings  # QwenEmbeddings instance for queries
        self.cache_file = cache_file
        self._cache = None
        self._load_cache()
    
    def _load_cache(self):
        """Load pre-computed embeddings from cache."""
        try:
            with open(self.cache_file, 'rb') as f:
                self._cache = pickle.load(f)
            print(f"🚀 Loaded {len(self._cache)} cached embeddings for hybrid mode")
        except Exception as e:
            print(f"❌ Failed to load embedding cache: {e}")
            self._cache = {}
    
    def _hash_content(self, content: str) -> str:
        """Get hash for content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Hybrid embedding: use cache when available, GPU for new content."""
        if not texts:
            return []
        
        # Separate cached vs uncached content
        cached_results = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            content_hash = self._hash_content(text)
            cached_embedding = self._cache.get(content_hash)
            
            if cached_embedding is not None:
                # Use cached embedding
                if isinstance(cached_embedding, np.ndarray):
                    cached_results[i] = cached_embedding.tolist()
                elif isinstance(cached_embedding, torch.Tensor):
                    cached_results[i] = cached_embedding.cpu().numpy().tolist()
                else:
                    cached_results[i] = cached_embedding
            else:
                # Need to compute with GPU
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Compute uncached embeddings with GPU model
        uncached_embeddings = []
        if uncached_texts:
            print(f"🔄 Computing {len(uncached_texts)} uncached embeddings with GPU...")
            uncached_embeddings = self.gpu_embeddings.embed_documents(uncached_texts)
        
        # Combine results
        results = [None] * len(texts)
        
        # Fill cached results
        for i, embedding in cached_results.items():
            results[i] = embedding
        
        # Fill uncached results
        for idx, i in enumerate(uncached_indices):
            results[i] = uncached_embeddings[idx]
        
        return results
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text (usually not cached, so uses GPU)."""
        return self.embed_documents([text])[0]
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version - just calls sync version."""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version - just calls sync version."""
        return self.embed_query(text)


class InstantCachedEmbeddings(Embeddings):
    """Embedding wrapper that returns cached embeddings instantly (no GPU needed)."""
    
    def __init__(self, cache_file: str = "/workspace/memupdate/data/embedding_cache/memory_embeddings.pkl"):
        self.cache_file = cache_file
        self._cache = None
        self._load_cache()
    
    def _load_cache(self):
        """Load pre-computed embeddings from cache."""
        try:
            with open(self.cache_file, 'rb') as f:
                self._cache = pickle.load(f)
        except Exception as e:
            print(f"❌ Failed to load embedding cache: {e}")
            self._cache = {}
    
    def _hash_content(self, content: str) -> str:
        """Get hash for content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return cached embeddings instantly (no computation)."""
        if not texts:
            return []
        
        results = []
        cache_hits = 0
        
        for text in texts:
            content_hash = self._hash_content(text)
            cached_embedding = self._cache.get(content_hash)
            
            if cached_embedding is not None:
                # Convert to list if needed
                if isinstance(cached_embedding, np.ndarray):
                    results.append(cached_embedding.tolist())
                elif isinstance(cached_embedding, torch.Tensor):
                    results.append(cached_embedding.cpu().numpy().tolist())
                else:
                    results.append(cached_embedding)
                cache_hits += 1
            else:
                # If not cached, return zero vector (should not happen with pre-computed data)
                results.append([0.0] * 1024)  # Qwen3-0.6B has 1024 dims
        return results
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.embed_documents([text])[0]
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version - just calls sync version."""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version - just calls sync version."""
        return self.embed_query(text)