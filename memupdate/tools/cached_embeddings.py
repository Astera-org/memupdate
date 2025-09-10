"""Smart cached embeddings: uses pre-computed cache, generates new ones on-demand."""

import torch
import hashlib
import pickle
import numpy as np
from typing import List, Dict, Optional, Any
from langchain_core.embeddings import Embeddings


class SmartCachedEmbeddings(Embeddings):
    """Unified embeddings class that:
    1. Uses pre-computed embeddings from cache
    2. Filters by conversation when provided
    3. Generates embeddings for new content using GPU if available, else CPU
    4. Can optionally cache newly generated embeddings
    """
    
    def __init__(
        self, 
        cache: Optional[Dict[str, Any]] = None,
        cache_file: Optional[str] = None,
        sample_id: Optional[str] = None,
        embedding_model = None,
        device: Optional[str] = None
    ):
        """Initialize smart cached embeddings.
        
        Args:
            cache: Pre-loaded embedding cache (if already loaded by broker)
            cache_file: Path to cache file (if not pre-loaded)
            sample_id: Optional conversation ID to filter embeddings
            embedding_model: Optional embedding model for new content
            device: Device for embedding model ('cuda', 'cpu', or None for auto)
        """
        self._cache = cache or {}
        self.sample_id = sample_id
        self.embedding_model = embedding_model
        self._new_embeddings = {}  # Cache for newly generated embeddings
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load cache from file if not pre-loaded
        if not self._cache and cache_file:
            self._load_cache(cache_file)
            
        # Filter cache by conversation if sample_id provided
        if self.sample_id and self._cache:
            self._filter_cache_by_conversation()
            
        # Initialize embedding model only if not provided
        if self.embedding_model is None:
            self._init_embedding_model()
        else:
            print(f"✅ Using shared embedding model (no per-sample initialization)")
            
        print(f"🧠 SmartCachedEmbeddings initialized: {len(self._cache)} cached, device={self.device}, sample_id={self.sample_id}")
    
    def _load_cache(self, cache_file: str):
        """Load pre-computed embeddings from cache file."""
        try:
            with open(cache_file, 'rb') as f:
                self._cache = pickle.load(f)
                    
            print(f"💾 Loaded {len(self._cache)} embeddings from {cache_file}")
        except Exception as e:
            print(f"⚠️ Failed to load embedding cache: {e}")
            self._cache = {}
    
    def _filter_cache_by_conversation(self):
        """Filter cache to only include embeddings for the specified conversation."""
        if not self.sample_id:
            return
            
        filtered_cache = {}
        for key, value in self._cache.items():
            if value.get('sample_id') == self.sample_id:
                filtered_cache[key] = value
                
        original_size = len(self._cache)
        self._cache = filtered_cache
        print(f"🎯 Filtered cache from {original_size} to {len(self._cache)} embeddings for {self.sample_id}")
    
    def _init_embedding_model(self):
        """Initialize embedding model for generating new embeddings."""
        try:
            # Try to import and initialize QwenEmbeddings
            from memupdate.data.qwen_embeddings import QwenEmbeddings
            
            # QwenEmbeddings automatically detects device - no device parameter needed
            self.embedding_model = QwenEmbeddings()
            print(f"✅ Initialized QwenEmbeddings (auto-detected device)")
                
        except Exception as e:
            print(f"⚠️ Failed to initialize embedding model: {e}")
            self.embedding_model = None
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def _get_cached_embedding(self, content_hash: str) -> Optional[List[float]]:
        """Get embedding from cache (either pre-computed or newly generated)."""
        # Check pre-computed cache
        if content_hash in self._cache:
            cached_data = self._cache[content_hash]
            embedding = cached_data.get('embedding')
                
            # Convert to list if needed
            if isinstance(embedding, np.ndarray):
                return embedding.tolist()
            elif isinstance(embedding, torch.Tensor):
                return embedding.cpu().numpy().tolist()
            else:
                return embedding
                
        # Check newly generated cache
        if content_hash in self._new_embeddings:
            return self._new_embeddings[content_hash]
            
        return None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using cache when available, generate for new content."""
        if not texts:
            return []
        
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # First pass: use cached embeddings
        for i, text in enumerate(texts):
            content_hash = self._hash_content(text)
            cached_embedding = self._get_cached_embedding(content_hash)
            
            if cached_embedding is not None:
                results.append(cached_embedding)
            else:
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Second pass: generate embeddings for uncached content
        if uncached_texts:
            if self.embedding_model:
                print(f"🔄 Generating {len(uncached_texts)} new embeddings on {self.device}...")
                try:
                    new_embeddings = self.embedding_model.embed_documents(uncached_texts)
                    
                    # Cache the new embeddings and fill results
                    for idx, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                        content_hash = self._hash_content(text)
                        self._new_embeddings[content_hash] = embedding
                        results[uncached_indices[idx]] = embedding
                    
                    if new_embeddings:
                        print(f"✅ Generated {len(new_embeddings)} new embeddings, sample: new_embeddings[0][:10]...")
                    else:
                        print(f"⚠️ No new embeddings were generated.")
                    
                except Exception as e:
                    print(f"❌ Failed to generate embeddings: {e}, returning zero vectors")
                    # Fill with zero vectors as fallback
                    for idx in uncached_indices:
                        results[idx] = [0.0] * 1024  # Qwen3-0.6B dimension
            else:
                print(f"⚠️ No embedding model available, using zero vectors for {len(uncached_texts)} texts")
                # Fill with zero vectors
                for idx in uncached_indices:
                    results[idx] = [0.0] * 1024
        
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the embeddings cache."""
        return {
            'cached_embeddings': len(self._cache),
            'new_embeddings': len(self._new_embeddings),
            'sample_id': self.sample_id,
            'device': self.device,
            'has_model': self.embedding_model is not None
        }